//! One render target shared by many blended groups.
//!
//! A display object with a blend mode is rendered through a temporary target so
//! that its children composite as a unit before the blend is applied. Giving
//! each of them a target of its own is what a crowded room pays for: at eight
//! hundred blended objects the frame begins eight hundred render passes, takes
//! eight hundred targets out of the pool and hands them back, for objects that
//! are a fraction of a megabyte each. The size of those targets is no longer the
//! problem - content-bounded targets fixed that - but their *number* is, and
//! that cost is fixed per target rather than proportional to its area.
//!
//! Nothing about a blended group requires a target all to itself. What it
//! requires is a rectangle of transparent pixels that only its own commands
//! draw into, and a way to sample exactly that rectangle afterwards. A page
//! gives it a region of a bigger texture instead: every group on the page is
//! drawn in one render pass, scissored to its own region, and composited from
//! the page by the same quad it would have used for its own target, with the
//! region's texture coordinates.
//!
//! # Why the picture does not change
//!
//! * **Nothing writes outside its region.** The scissor rectangle is set to the
//!   region before each group's draws, so rasterisation - multisampled or not -
//!   cannot touch a neighbour. The group's own commands are bounded by the
//!   region anyway; the scissor is what makes that a guarantee rather than a
//!   property of [`content_bounds`](crate::bounds::content_bounds).
//!
//! * **Nothing reads outside its region.** Pages are a power of two on a side
//!   and a region composites through a quad exactly as many pixels wide as the
//!   region is, so the fragment at the region's `i`-th pixel samples the page at
//!   `(x + i + 0.5) / page`, which is the centre of page texel `x + i` exactly -
//!   the division is by a power of two, so it is exact in binary. That is the
//!   same texel the group's own target would have handed back. A gutter of
//!   [`GUTTER`] pixels between regions covers the rest: filtering slack, the
//!   multisample resolve at a region's edge, and anything that rounds outwards.
//!
//! * **Nothing sees another frame's pixels.** The page is cleared to
//!   transparent by the load operation of its one render pass, so a region is
//!   transparent everywhere its group did not draw, exactly as a freshly taken
//!   target is. Regions are handed out in one direction and never reissued
//!   within a frame.
//!
//! * **Sub-pixel phase is kept.** A region starts on a whole pixel and the
//!   commands drawn into it have the region's origin subtracted from their
//!   translation, so an object at `x = 100.37` lands at `.37` into its region
//!   just as it landed `.37` into its own target.
//!
//! A group only takes a region if it is a plain run of draws - no nested blend,
//! no mask, no alpha mask, nothing that would need a pass or a target of its
//! own inside the shared one. Everything else keeps the target it had.

use crate::PageFallback;
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::{AlwaysCompatible, PoolEntry, PooledTexture, TexturePool};
use crate::descriptors::Descriptors;
use crate::dynamic_transforms::DynamicTransforms;
use crate::globals::Globals;
use crate::surface::commands::{CommandRenderer, DrawCommand};
use crate::surface::target::PoolOrArcTexture;
use crate::{MaskState, Pipelines, PosUvVertex};
use std::mem;
use std::sync::Arc;
use wgpu_profiler::Scope;

/// Blank pixels kept between regions, and around the edge of a page.
///
/// A region is sampled at its own texel centres, so this is not needed for the
/// composite itself; it is there so that anything which rounds outwards by a
/// pixel - a multisample resolve at the region's edge, a filter's slack, a
/// driver's rasterisation rule - finds transparent pixels rather than a
/// neighbour's.
const GUTTER: u32 = 1;

/// The largest region a page will hand out.
///
/// Pages exist for the many small targets a room full of avatars asks for. A
/// group bigger than this would take a large part of a page for itself and get
/// none of the benefit, so it keeps a target of its own - where it is also the
/// only thing paying for the render pass it starts.
const MAX_REGION: u32 = 512;

/// How big each page a surface opens is.
///
/// The first two are small, so that a quiet scene with a handful of blended
/// objects does not take sixteen megabytes to hold three of them; a scene that
/// has filled two of those has shown it has enough objects to fill a large one.
/// A 2048-pixel page holds about ninety of the client's avatar-with-equipment
/// groups at 98% of its area, so even the worst rooms reported need single
/// figures of them, and the memory works out below what a size-classed target
/// per group came to.
///
/// A multisampled surface keeps to the small size throughout: a page carries
/// `sample_count` copies of every pixel in its attachment on top of the resolve
/// target it is read from, so a 2048-pixel page at four samples is eighty
/// megabytes. That is more pages, but a page is still shared by thirty groups
/// where a target is shared by none.
const PAGE_SIDES: &[u32] = &[1024, 1024, 2048];
const MSAA_PAGE_SIDES: &[u32] = &[1024];

/// Where one blended group's pixels sit on a page.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// The page's own size. Never assume it is the region's.
    pub page_width: u32,
    pub page_height: u32,
}

impl PageRegion {
    /// `[u0, v0, du, dv]`: where this region is in the page's texture
    /// coordinates.
    ///
    /// A whole texture is `[0.0, 0.0, 1.0, 1.0]`, which is what the consumers
    /// of a group that kept its own target pass.
    pub fn uv(&self) -> [f32; 4] {
        [
            self.x as f32 / self.page_width as f32,
            self.y as f32 / self.page_height as f32,
            self.width as f32 / self.page_width as f32,
            self.height as f32 / self.page_height as f32,
        ]
    }

    /// The quad that composites this region back, in the same shape as the one
    /// a whole target is composited by: the unit square, carrying the texture
    /// coordinates of the region at its corners.
    pub fn quad(&self) -> [PosUvVertex; 4] {
        let [u0, v0, du, dv] = self.uv();
        let (u1, v1) = (u0 + du, v0 + dv);
        [
            PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
            PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
            PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
            PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
        ]
    }
}

/// Shelf packing: a row of regions grows to the right, and a new row starts
/// above the tallest region of the last one.
///
/// Cheap, allocation-free and near-perfect for the case that matters, which is
/// a room of objects that are all about one size - the client's rooms pack at
/// 98% of a page's area. A cleverer packer would win a few per cent of that for
/// a per-object cost this exists to remove.
#[derive(Debug)]
struct Shelf {
    width: u32,
    height: u32,
    y: u32,
    row_height: u32,
    cursor_x: u32,
}

impl Shelf {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            y: GUTTER,
            row_height: 0,
            cursor_x: GUTTER,
        }
    }

    /// Finds room for a `width` x `height` region, or says the page is full.
    fn allocate(&mut self, width: u32, height: u32) -> Option<PageRegion> {
        if self.cursor_x + width + GUTTER > self.width {
            self.y += self.row_height + GUTTER;
            self.row_height = 0;
            self.cursor_x = GUTTER;
        }
        if self.cursor_x + width + GUTTER > self.width || self.y + height + GUTTER > self.height {
            return None;
        }
        let region = PageRegion {
            x: self.cursor_x,
            y: self.y,
            width,
            height,
            page_width: self.width,
            page_height: self.height,
        };
        self.cursor_x += width + GUTTER;
        self.row_height = self.row_height.max(height);
        Some(region)
    }
}

/// One group's draws, waiting for the page's render pass.
struct PageRun {
    region: PageRegion,
    commands: Vec<DrawCommand>,
}

/// A page and everything queued to be drawn into it.
struct Page {
    /// The multisampled attachment, when the surface is multisampled. The
    /// page's pixels are always *read* from `color`.
    msaa: Option<PoolEntry<PooledTexture, AlwaysCompatible>>,
    color: Arc<PoolOrArcTexture>,
    /// Samples the whole page. Kept with the pooled texture, so a page that
    /// comes back out of the pool does not build one again.
    binds: wgpu::BindGroup,
    globals: Arc<Globals>,
    shelf: Shelf,

    transforms: BufferBuilder,
    vertices: BufferBuilder,
    runs: Vec<PageRun>,
}

impl Page {
    fn new(
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        side: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
        dynamic_transforms: &DynamicTransforms,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        };
        // The same usage a `CommandTarget`'s resolve buffer asks for, so pages
        // and targets of the same size share a pool key rather than each
        // keeping its own free list.
        let color = pool.get_texture(
            descriptors,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            format,
            1,
        );
        let msaa = (sample_count > 1).then(|| {
            pool.get_texture(
                descriptors,
                size,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                sample_count,
            )
        });
        crate::render_stats::page_taken(
            side as usize
                * side as usize
                * 4
                * (1 + msaa.as_ref().map_or(0, |_| sample_count as usize)),
        );

        let color = Arc::new(PoolOrArcTexture::Pool(color));
        let binds = color.bitmap_bind_group(descriptors).clone();
        let globals = pool.get_globals(descriptors, side, side);

        let mut transforms = BufferBuilder::new_for_uniform(&descriptors.limits);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        let mut vertices = BufferBuilder::new_for_vertices(&descriptors.limits);
        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());

        Self {
            msaa,
            color,
            binds,
            globals,
            shelf: Shelf::new(side, side),
            transforms,
            vertices,
            runs: Vec::new(),
        }
    }

    /// Draws everything queued on this page, in one render pass.
    fn flush<'global>(
        &mut self,
        descriptors: &Descriptors,
        pipelines: &Pipelines,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        encoder: &mut Scope<'global, wgpu::CommandEncoder>,
    ) {
        if self.runs.is_empty() {
            return;
        }
        let runs = mem::take(&mut self.runs);

        let mut transforms = BufferBuilder::new_for_uniform(&descriptors.limits);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        let mut vertices = BufferBuilder::new_for_vertices(&descriptors.limits);
        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());
        let transforms = mem::replace(&mut self.transforms, transforms);
        let vertices = mem::replace(&mut self.vertices, vertices);
        transforms.copy_to(staging_belt, encoder, &dynamic_transforms.buffer);
        vertices.copy_to(staging_belt, encoder, &dynamic_transforms.vertex_buffer);

        crate::render_stats::record_render_pass();
        let mut render_pass = encoder.scoped_render_pass(
            "Blend page",
            wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: match &self.msaa {
                        Some(msaa) => &msaa.1,
                        None => self.color.view(),
                    },
                    resolve_target: self.msaa.as_ref().map(|_| self.color.view()),
                    ops: wgpu::Operations {
                        // The whole page, every frame: a region must not see
                        // what the last scene left in the pooled texture.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            },
        );
        render_pass.set_bind_group(0, self.globals.bind_group(), &[]);

        let mut renderer = CommandRenderer::new(
            pipelines,
            descriptors,
            dynamic_transforms,
            0,
            MaskState::NoMask,
            false,
        );
        for run in &runs {
            render_pass.set_scissor_rect(
                run.region.x,
                run.region.y,
                run.region.width,
                run.region.height,
            );
            for command in &run.commands {
                renderer.execute(&mut render_pass.scope(command.name()), command);
            }
        }
    }
}

/// What a group got when it asked for a region.
pub struct Placement {
    pub region: PageRegion,
    /// The page's pixels, kept alive for as long as anything might sample them.
    pub source: Arc<PoolOrArcTexture>,
    /// Samples the whole page; the region is selected by the quad's texture
    /// coordinates, not by a bind group of its own.
    pub binds: wgpu::BindGroup,
}

/// The pages one surface's blended groups are sharing.
///
/// One of these lives for the length of a [`chunk_blends`] walk. Every page it
/// opens is drawn before that walk returns, and so before any of the passes
/// that composite from them; the pages themselves are held until the surface
/// has finished with them, so the pool cannot hand a page's texture out again
/// while a composite that reads it is still to be encoded.
///
/// [`chunk_blends`]: crate::surface::commands::chunk_blends
pub struct BlendPages {
    format: wgpu::TextureFormat,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    max_side: u32,
    open: Option<Page>,
    opened: u32,
    /// Pages whose pass has been encoded, kept alive.
    held: Vec<Arc<PoolOrArcTexture>>,
}

impl BlendPages {
    pub fn new(descriptors: &Descriptors, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        Self {
            format,
            sample_count,
            pipelines: descriptors.pipelines(sample_count, format),
            max_side: descriptors.limits.max_texture_dimension_2d,
            open: None,
            opened: 0,
            held: Vec::new(),
        }
    }

    /// The largest region any page will hand out.
    pub fn max_region(&self) -> u32 {
        MAX_REGION.min(self.page_side(0)).saturating_sub(GUTTER * 2)
    }

    fn page_side(&self, opened: u32) -> u32 {
        let sides = if self.sample_count > 1 {
            MSAA_PAGE_SIDES
        } else {
            PAGE_SIDES
        };
        let wanted = sides[(opened as usize).min(sides.len() - 1)];
        wanted.min(self.max_side)
    }

    /// Puts a `width` x `height` region on a page, opening one - or closing the
    /// one that is full and opening the next - as needed.
    ///
    /// `transforms` and `vertices` are what the group's draws will need at
    /// worst; a group whose draws would not fit alongside what a page is
    /// already holding starts a fresh one, because a run split across two
    /// buffers would be split across two render passes and half of it would
    /// land on the wrong page.
    #[expect(clippy::too_many_arguments)]
    pub fn place<'global>(
        &mut self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        encoder: &mut Scope<'global, wgpu::CommandEncoder>,
        width: u32,
        height: u32,
        transforms: usize,
        vertices: usize,
    ) -> Result<Placement, PageFallback> {
        if width > self.max_region() || height > self.max_region() {
            return Err(PageFallback::Size);
        }

        for attempt in 0..2 {
            if self.open.is_none() {
                let side = self.page_side(self.opened);
                self.open = Some(Page::new(
                    descriptors,
                    pool,
                    side,
                    self.format,
                    self.sample_count,
                    dynamic_transforms,
                ));
                self.opened += 1;
            }
            let page = self.open.as_mut().expect("a page was just opened");
            if !page.has_room(transforms, vertices) && page.runs.is_empty() {
                // Not even an empty page's buffers can hold this group's draws.
                return Err(PageFallback::Capacity);
            }
            if page.has_room(transforms, vertices)
                && let Some(region) = page.shelf.allocate(width, height)
            {
                return Ok(Placement {
                    region,
                    source: page.color.clone(),
                    binds: page.binds.clone(),
                });
            }
            // The page is out of room, for pixels or for buffer space. Draw it
            // and try once more on a fresh one.
            debug_assert!(attempt == 0, "a region fits an empty page or no page");
            self.close_open_page(descriptors, staging_belt, dynamic_transforms, encoder);
        }
        Err(PageFallback::NoPage)
    }

    /// Hands a group's draws to the page it was placed on.
    ///
    /// Called after [`place`](Self::place) with the same group; the commands
    /// were built against the region that call returned.
    pub fn add_run(&mut self, region: PageRegion, commands: Vec<DrawCommand>) {
        if let Some(page) = self.open.as_mut() {
            page.runs.push(PageRun { region, commands });
        }
    }

    /// Exchanges the caller's buffers for the open page's, so that a group's
    /// draws are written into the page's chunk rather than the surface's.
    ///
    /// Called once on the way in and once on the way out, which puts each pair
    /// back where it came from. Does nothing when no page is open, which cannot
    /// happen between a successful [`place`](Self::place) and its run.
    pub fn swap_builders(&mut self, transforms: &mut BufferBuilder, vertices: &mut BufferBuilder) {
        if let Some(page) = self.open.as_mut() {
            mem::swap(&mut page.transforms, transforms);
            mem::swap(&mut page.vertices, vertices);
        }
    }

    /// Draws every page that still has work queued.
    ///
    /// Must be called before anything composites from a page, which for a
    /// surface means before its chunks are executed.
    pub fn finish<'global>(
        &mut self,
        descriptors: &Descriptors,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        encoder: &mut Scope<'global, wgpu::CommandEncoder>,
    ) {
        self.close_open_page(descriptors, staging_belt, dynamic_transforms, encoder);
    }

    /// The pages this walk drew, which have to outlive the passes that read
    /// them.
    pub fn take_held(&mut self) -> Vec<Arc<PoolOrArcTexture>> {
        mem::take(&mut self.held)
    }

    fn close_open_page<'global>(
        &mut self,
        descriptors: &Descriptors,
        staging_belt: &mut wgpu::util::StagingBelt,
        dynamic_transforms: &DynamicTransforms,
        encoder: &mut Scope<'global, wgpu::CommandEncoder>,
    ) {
        if let Some(mut page) = self.open.take() {
            page.flush(
                descriptors,
                &self.pipelines,
                staging_belt,
                dynamic_transforms,
                encoder,
            );
            self.held.push(page.color.clone());
        }
    }
}

impl Page {
    fn has_room(&self, transforms: usize, vertices: usize) -> bool {
        self.transforms
            .has_room_for(transforms, mem::size_of::<crate::Transforms>())
            && self
                .vertices
                .has_room_for(vertices, mem::size_of::<PosUvVertex>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The one thing a packer must never do.
    #[test]
    fn regions_never_overlap() {
        let mut shelf = Shelf::new(512, 512);
        let mut regions = Vec::new();
        // Sizes that vary the way a room's objects do, so shelves end up with
        // different heights and the rows do not line up.
        for i in 0..1000u32 {
            let width = 20 + (i * 37) % 90;
            let height = 24 + (i * 53) % 110;
            match shelf.allocate(width, height) {
                Some(region) => regions.push(region),
                None => break,
            }
        }
        assert!(regions.len() > 10, "only {} regions fitted", regions.len());
        for (i, a) in regions.iter().enumerate() {
            assert!(
                a.x + a.width <= 512 && a.y + a.height <= 512,
                "{a:?} leaves the page"
            );
            for b in &regions[i + 1..] {
                let apart = a.x + a.width <= b.x
                    || b.x + b.width <= a.x
                    || a.y + a.height <= b.y
                    || b.y + b.height <= a.y;
                assert!(apart, "{a:?} and {b:?} overlap");
            }
        }
    }

    /// And the gutter has to be there, or a neighbour is a texel away.
    #[test]
    fn regions_keep_a_gutter_between_them() {
        let mut shelf = Shelf::new(256, 256);
        let mut regions = Vec::new();
        while let Some(region) = shelf.allocate(40, 30) {
            regions.push(region);
        }
        for (i, a) in regions.iter().enumerate() {
            assert!(
                a.x >= GUTTER && a.y >= GUTTER,
                "{a:?} touches the page edge"
            );
            for b in &regions[i + 1..] {
                let apart = a.x + a.width + GUTTER <= b.x
                    || b.x + b.width + GUTTER <= a.x
                    || a.y + a.height + GUTTER <= b.y
                    || b.y + b.height + GUTTER <= a.y;
                assert!(apart, "{a:?} and {b:?} are closer than a gutter");
            }
        }
    }

    /// A page of objects that are all one size should be nearly full, or pages
    /// would cost more memory than the targets they replace.
    #[test]
    fn a_room_of_one_size_fills_its_page() {
        let mut shelf = Shelf::new(2048, 2048);
        let mut covered = 0u32;
        let mut count = 0;
        while shelf.allocate(150, 200).is_some() {
            covered += 150 * 200;
            count += 1;
        }
        let filled = covered as f64 / (2048.0 * 2048.0);
        assert!(
            filled > 0.9,
            "{count} avatar-sized regions covered only {:.0}% of a page",
            filled * 100.0
        );
    }

    /// A region is sampled at its own texel centres, exactly.
    ///
    /// This is what makes a page indistinguishable from a target of the
    /// region's own size: the fragment covering the region's `i`-th pixel asks
    /// for page texel `x + i` and no other, so nothing a neighbour holds can
    /// reach it whatever the filter is.
    #[test]
    fn a_region_samples_its_own_texels_and_no_others() {
        for (x, y, width, height, page) in [
            (1u32, 1u32, 150u32, 200u32, 1024u32),
            (777, 513, 61, 37, 1024),
            (1, 1, 510, 510, 2048),
            (1900, 1900, 100, 100, 2048),
        ] {
            let region = PageRegion {
                x,
                y,
                width,
                height,
                page_width: page,
                page_height: page,
            };
            let [u0, v0, du, dv] = region.uv();
            let mut texels = HashSet::new();
            for i in 0..width {
                // Where the composite quad's `i`-th pixel centre lands.
                let u = u0 + du * (i as f32 + 0.5) / width as f32;
                let texel = (u * page as f32).floor() as i64;
                assert_eq!(
                    texel,
                    (x + i) as i64,
                    "pixel {i} of a {width}x{height} region at ({x}, {y}) on a {page} page \
                     sampled texel {texel}"
                );
                texels.insert(texel);
            }
            assert_eq!(texels.len(), width as usize);
            for i in 0..height {
                let v = v0 + dv * (i as f32 + 0.5) / height as f32;
                assert_eq!((v * page as f32).floor() as i64, (y + i) as i64);
            }
        }
    }
}
