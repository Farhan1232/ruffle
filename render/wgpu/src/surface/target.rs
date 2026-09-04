use crate::Transforms;
use crate::backend::RenderTargetMode;
use crate::bind_cache::BindGroupCache;
use crate::bounds::TargetRect;
use crate::buffer_pool::{AlwaysCompatible, PoolEntry, PooledTexture, TexturePool};
use crate::descriptors::Descriptors;
use crate::globals::Globals;
use crate::utils::create_buffer_with_data;
use crate::utils::run_copy_pipeline;
use std::cell::OnceCell;
use std::sync::Arc;

#[derive(Debug)]
pub struct ResolveBuffer {
    texture: PoolOrArcTexture,
}

impl ResolveBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, 1);
        Self {
            texture: PoolOrArcTexture::Pool(texture),
        }
    }

    pub fn new_manual(texture: wgpu::Texture) -> Self {
        Self {
            texture: PoolOrArcTexture::Manual(Box::new(ManualTexture::new(texture))),
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.1,
            PoolOrArcTexture::Manual(ref texture) => &texture.view,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.0,
            PoolOrArcTexture::Manual(ref texture) => &texture.texture,
        }
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }
}

#[derive(Debug)]
pub struct FrameBuffer {
    texture: PoolOrArcTexture,
}

#[derive(Debug)]
/// Holds either a `PoolEntry` texture, or an `Arc`-wrapped texture.
/// This is used to select between using a texture pool for our framebuffer/resolve-buffer
/// (when rendering to the main screen), or rendering to a non-pooled `Texture`
/// (when doing an offscreen render to a BitmapData texture)
pub enum PoolOrArcTexture {
    Pool(PoolEntry<PooledTexture, AlwaysCompatible>),
    Manual(Box<ManualTexture>),
}

/// A target the renderer owns outright rather than borrowing from a pool: a
/// `BitmapData`'s own texture, or a `cacheAsBitmap` backing store.
#[derive(Debug)]
pub struct ManualTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    binds: BindGroupCache,
}

impl ManualTexture {
    fn new(texture: wgpu::Texture) -> Self {
        let view = texture.create_view(&Default::default());
        Self {
            texture,
            view,
            binds: BindGroupCache::default(),
        }
    }
}

impl PoolOrArcTexture {
    pub fn texture(&self) -> &wgpu::Texture {
        match self {
            PoolOrArcTexture::Pool(texture) => &texture.0,
            PoolOrArcTexture::Manual(texture) => &texture.texture,
        }
    }
    pub fn view(&self) -> &wgpu::TextureView {
        match self {
            PoolOrArcTexture::Pool(texture) => &texture.1,
            PoolOrArcTexture::Manual(texture) => &texture.view,
        }
    }

    /// The bind groups kept with this texture, so that compositing it does not
    /// build a new one every frame.
    pub fn binds(&self) -> &BindGroupCache {
        match self {
            PoolOrArcTexture::Pool(texture) => &texture.2,
            PoolOrArcTexture::Manual(texture) => &texture.binds,
        }
    }

    /// The bind group that samples this whole texture as a bitmap, which is how
    /// a blended group's target is composited back.
    pub fn bitmap_bind_group(&self, descriptors: &Descriptors) -> &wgpu::BindGroup {
        self.binds().whole(|| {
            descriptors
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &descriptors.bind_layouts.bitmap,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(self.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                descriptors.bitmap_samplers.get_sampler(false, false),
                            ),
                        },
                    ],
                    label: create_debug_label!("Composite blended target").as_deref(),
                })
        })
    }
}

impl FrameBuffer {
    pub fn new(
        descriptors: &Descriptors,
        sample_count: u32,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, sample_count);

        Self {
            texture: PoolOrArcTexture::Pool(texture),
        }
    }

    pub fn new_manual(texture: wgpu::Texture) -> Self {
        Self {
            texture: PoolOrArcTexture::Manual(Box::new(ManualTexture::new(texture))),
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.1,
            PoolOrArcTexture::Manual(ref texture) => &texture.view,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        match self.texture {
            PoolOrArcTexture::Pool(ref texture) => &texture.0,
            PoolOrArcTexture::Manual(ref texture) => &texture.texture,
        }
    }

    pub fn take_texture(self) -> PoolOrArcTexture {
        self.texture
    }
}

#[derive(Debug)]
pub struct BlendBuffer {
    texture: PoolEntry<PooledTexture, AlwaysCompatible>,
}

impl BlendBuffer {
    pub fn new(
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(descriptors, size, usage, format, 1);

        Self { texture }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.texture.1
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture.0
    }

    /// Identifies the texture behind this buffer, so a bind group naming it can
    /// be cached against it.
    pub fn binds_id(&self) -> u64 {
        self.texture.2.id()
    }
}

#[derive(Debug)]
pub struct StencilBuffer {
    texture: PoolEntry<PooledTexture, AlwaysCompatible>,
}

impl StencilBuffer {
    pub fn new(
        descriptors: &Descriptors,
        msaa_sample_count: u32,
        size: wgpu::Extent3d,
        pool: &mut TexturePool,
    ) -> Self {
        let texture = pool.get_texture(
            descriptors,
            size,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureFormat::Stencil8,
            msaa_sample_count,
        );

        Self { texture }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.texture.1
    }
}

pub struct CommandTarget {
    frame_buffer: FrameBuffer,
    blend_buffer: OnceCell<BlendBuffer>,
    resolve_buffer: Option<ResolveBuffer>,
    depth: OnceCell<StencilBuffer>,
    globals: Arc<Globals>,
    /// Where this target sits in the coordinates of the commands drawn into it.
    ///
    /// A blend composited onto this target needs it to know its own position:
    /// the blend samples this target's pixels through the rectangle it covers,
    /// and the two rectangles are given in the same space.
    rect: TargetRect,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    sample_count: u32,
    whole_frame_bind_group: OnceCell<(wgpu::Buffer, wgpu::BindGroup)>,
    color_needs_clear: OnceCell<bool>,
    render_target_mode: RenderTargetMode,
}

impl CommandTarget {
    pub fn new(
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        rect: TargetRect,
        format: wgpu::TextureFormat,
        sample_count: u32,
        render_target_mode: RenderTargetMode,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Self {
        let size = rect.extent();
        let globals = pool.get_globals(descriptors, size.width, size.height);

        let mut make_pooled_frame_buffer = || {
            FrameBuffer::new(
                descriptors,
                sample_count,
                size,
                format,
                if sample_count > 1 {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING
                },
                pool,
            )
        };

        let whole_frame_bind_group = OnceCell::new();

        let (frame_buffer, resolve_buffer) =
            if let RenderTargetMode::ExistingWithColor(texture, _) = &render_target_mode {
                if sample_count > 1 {
                    (
                        make_pooled_frame_buffer(),
                        Some(ResolveBuffer::new_manual(texture.clone())),
                    )
                } else {
                    (FrameBuffer::new_manual(texture.clone()), None)
                }
            } else if sample_count > 1 {
                (
                    make_pooled_frame_buffer(),
                    Some(ResolveBuffer::new(
                        descriptors,
                        size,
                        format,
                        wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                        pool,
                    )),
                )
            } else {
                (make_pooled_frame_buffer(), None)
            };

        if let RenderTargetMode::FreshWithTexture(texture) = &render_target_mode {
            if let Some(resolve_buffer) = &resolve_buffer {
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    resolve_buffer.texture().as_image_copy(),
                    size,
                );
            }

            if sample_count > 1 {
                // Both our frame buffer and resolve buffer need to start out
                // in the same state, so copy our existing texture to the freshly
                // allocated frame buffer. We cannot use `copy_texture_to_texture`,
                // since the sample counts are different.
                run_copy_pipeline(
                    descriptors,
                    format,
                    frame_buffer.texture.view(),
                    &texture.create_view(&Default::default()),
                    get_whole_frame_bind_group(&whole_frame_bind_group, descriptors, size),
                    &globals,
                    sample_count,
                    encoder,
                );
            } else {
                encoder.copy_texture_to_texture(
                    texture.as_image_copy(),
                    frame_buffer.texture().as_image_copy(),
                    size,
                );
            }
        }

        Self {
            frame_buffer,
            blend_buffer: OnceCell::new(),
            resolve_buffer,
            depth: OnceCell::new(),
            globals,
            rect,
            size,
            format,
            sample_count,
            whole_frame_bind_group,
            color_needs_clear: OnceCell::new(),
            render_target_mode,
        }
    }

    pub fn rect(&self) -> TargetRect {
        self.rect
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn ensure_cleared(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.color_needs_clear.get().is_some() {
            return;
        }
        // If we aren't clearing with a color (eg a texture instead)
        // the there's no point in creating a new render pass that does nothing.
        if self.render_target_mode.color().is_some() {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: create_debug_label!("Clearing command target").as_deref(),
                color_attachments: &[self.color_attachments()],
                ..Default::default()
            });
        }
    }

    pub fn take_color_texture(self) -> PoolOrArcTexture {
        self.resolve_buffer
            .map(|b| b.take_texture())
            .unwrap_or_else(|| self.frame_buffer.take_texture())
    }

    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    pub fn whole_frame_bind_group(&self, descriptors: &Descriptors) -> &wgpu::BindGroup {
        get_whole_frame_bind_group(&self.whole_frame_bind_group, descriptors, self.size)
    }

    pub fn color_attachments(&self) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let mut load = wgpu::LoadOp::Load;
        if self.color_needs_clear.set(false).is_ok()
            && let Some(clear_color) = self.render_target_mode.color()
        {
            load = wgpu::LoadOp::Clear(clear_color);
        }
        Some(wgpu::RenderPassColorAttachment {
            view: self.frame_buffer.view(),
            resolve_target: self.resolve_buffer.as_ref().map(|b| b.view()),
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn stencil_attachment(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        let new_buffer = self.depth.get().is_none();
        let stencil = self
            .depth
            .get_or_init(|| StencilBuffer::new(descriptors, self.sample_count, self.size, pool));
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: stencil.view(),
            depth_ops: None,
            stencil_ops: Some(wgpu::Operations {
                load: if new_buffer {
                    wgpu::LoadOp::Clear(0)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            }),
        })
    }

    /// Takes a copy of this target's pixels for a blend to read as its
    /// destination, refreshing `region` of it.
    ///
    /// `region` is in this target's own pixels. A blend only samples the
    /// destination underneath the rectangle it covers, so only that part has to
    /// be up to date - and copying the whole screen for each of a crowded
    /// scene's hundreds of blends is most of what made them expensive. The rest
    /// of the buffer holds whatever an earlier blend left there and is never
    /// read.
    pub fn update_blend_buffer(
        &self,
        descriptors: &Descriptors,
        pool: &mut TexturePool,
        encoder: &mut wgpu::CommandEncoder,
        region: TargetRect,
    ) -> &BlendBuffer {
        let blend_buffer = self.blend_buffer.get_or_init(|| {
            BlendBuffer::new(
                descriptors,
                self.size,
                self.format,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                pool,
            )
        });
        self.ensure_cleared(encoder);

        // Clamp to the target: `region` comes from a blend's own bounds, which
        // are rounded out and may be a pixel past the edge.
        let x = region.x.clamp(0, self.size.width as i32) as u32;
        let y = region.y.clamp(0, self.size.height as i32) as u32;
        let width = (region.right().clamp(0, self.size.width as i32) as u32).saturating_sub(x);
        let height = (region.bottom().clamp(0, self.size.height as i32) as u32).saturating_sub(y);
        if width == 0 || height == 0 {
            return blend_buffer;
        }
        crate::render_stats::record_destination_copy(u64::from(width) * u64::from(height));

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self
                    .resolve_buffer
                    .as_ref()
                    .map(|b| b.texture())
                    .unwrap_or_else(|| self.frame_buffer.texture()),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: Default::default(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: blend_buffer.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: Default::default(),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        blend_buffer
    }

    /// The snapshot of this target's pixels, if a blend has already asked for
    /// one.
    ///
    /// [`update_blend_buffer`](Self::update_blend_buffer) both refreshes the
    /// snapshot and hands it back; a batch of blends refreshes each of their
    /// regions first and then reads the one buffer they all share, which is
    /// what this is for.
    pub fn blend_buffer(&self) -> Option<&BlendBuffer> {
        self.blend_buffer.get()
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        self.resolve_buffer
            .as_ref()
            .map(|b| b.view())
            .unwrap_or_else(|| self.frame_buffer.view())
    }

    pub fn color_texture(&self) -> &wgpu::Texture {
        self.resolve_buffer
            .as_ref()
            .map(|b| b.texture())
            .unwrap_or_else(|| self.frame_buffer.texture())
    }
}

fn get_whole_frame_bind_group<'a>(
    once_cell: &'a OnceCell<(wgpu::Buffer, wgpu::BindGroup)>,
    descriptors: &Descriptors,
    size: wgpu::Extent3d,
) -> &'a wgpu::BindGroup {
    &once_cell
        .get_or_init(|| {
            let transform = Transforms {
                world_matrix: [
                    [size.width as f32, 0.0, 0.0, 0.0],
                    [0.0, size.height as f32, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                mult_color: [1.0, 1.0, 1.0, 1.0],
                add_color: [0.0, 0.0, 0.0, 0.0],
            };
            let transforms_buffer = create_buffer_with_data(
                &descriptors.device,
                bytemuck::cast_slice(&[transform]),
                wgpu::BufferUsages::UNIFORM,
                create_debug_label!("Whole-frame transforms buffer"),
            );
            let whole_frame_bind_group =
                descriptors
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: &descriptors.bind_layouts.transforms,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: transforms_buffer.as_entire_binding(),
                        }],
                        label: create_debug_label!("Whole-frame transforms bind group").as_deref(),
                    });
            (transforms_buffer, whole_frame_bind_group)
        })
        .1
}
