use super::target::PoolOrArcTexture;
use crate::backend::RenderTargetMode;
use crate::blend::TrivialBlend;
use crate::blend::{BlendType, ComplexBlend};
use crate::bounds::{TargetRect, content_bounds, region_rect_for, target_rect_for};
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::mesh::{DrawType, Mesh, as_mesh};
use crate::surface::Surface;
use crate::surface::page::{BlendPages, PageRegion};
use crate::surface::target::CommandTarget;
use crate::{Descriptors, MaskState, Pipelines, PosUvVertex, Transforms, as_texture};
use ruffle_render::backend::ShapeHandle;
use ruffle_render::bitmap::{BitmapHandle, PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandHandler, CommandList, RenderBlendMode};
use ruffle_render::lines::{emulate_line, emulate_line_rect};
use ruffle_render::matrix::Matrix;
use ruffle_render::pixel_bender::PixelBenderShaderHandle;
use ruffle_render::quality::StageQuality;
use ruffle_render::transform::Transform;
use std::mem;
use std::sync::Arc;
use swf::{BlendMode, Color, ColorTransform, Twips};
use wgpu::Backend;
use wgpu_profiler::Scope;

pub struct CommandRenderer<'encoder> {
    pipelines: &'encoder Pipelines,
    descriptors: &'encoder Descriptors,
    num_masks: u32,
    mask_state: MaskState,
    needs_stencil: bool,
    dynamic_transforms: &'encoder DynamicTransforms,
}

impl<'encoder> CommandRenderer<'encoder> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pipelines: &'encoder Pipelines,
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
        num_masks: u32,
        mask_state: MaskState,
        needs_stencil: bool,
    ) -> Self {
        Self {
            pipelines,
            num_masks,
            mask_state,
            descriptors,
            needs_stencil,
            dynamic_transforms,
        }
    }

    pub fn execute(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        command: &'encoder DrawCommand,
    ) {
        if self.needs_stencil {
            match self.mask_state {
                MaskState::NoMask => {}
                MaskState::DrawMaskStencil => {
                    render_pass.set_stencil_reference(self.num_masks - 1);
                }
                MaskState::DrawMaskedContent => {
                    render_pass.set_stencil_reference(self.num_masks);
                }
                MaskState::ClearMaskStencil => {
                    render_pass.set_stencil_reference(self.num_masks);
                }
            }
        }

        match command {
            DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                vertex_offset,
                smoothing,
                blend_mode,
                render_stage3d,
            } => self.render_bitmap(
                render_pass,
                bitmap,
                *transform_buffer,
                *smoothing,
                *blend_mode,
                *render_stage3d,
                *vertex_offset,
            ),
            DrawCommand::RenderTexture {
                _texture,
                binds,
                transform_buffer,
                vertex_offset,
                blend_mode,
            } => self.render_texture(
                render_pass,
                *transform_buffer,
                binds,
                *blend_mode,
                *vertex_offset,
            ),
            DrawCommand::RenderShape {
                shape,
                transform_buffer,
            } => self.render_shape(render_pass, shape, *transform_buffer),
            DrawCommand::DrawRect { transform_buffer } => {
                self.draw_rect(render_pass, *transform_buffer)
            }
            DrawCommand::DrawLine { transform_buffer } => {
                self.draw_lines::<false>(render_pass, *transform_buffer)
            }
            DrawCommand::DrawLineRect { transform_buffer } => {
                self.draw_lines::<true>(render_pass, *transform_buffer)
            }
            DrawCommand::PushMask => self.push_mask(render_pass),
            DrawCommand::ActivateMask => self.activate_mask(render_pass),
            DrawCommand::DeactivateMask => self.deactivate_mask(render_pass),
            DrawCommand::PopMask => self.pop_mask(render_pass),
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                transform_buffer,
            } => self.render_alpha_mask(render_pass, maskee, mask, binds, *transform_buffer),
        }
    }

    pub fn prep_color(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.color.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.color.stencilless_pipeline());
        }
    }

    pub fn prep_lines(&self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.lines.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.lines.stencilless_pipeline());
        }
    }

    pub fn prep_gradient(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.gradients.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.gradients.stencilless_pipeline());
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn prep_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    ) {
        match (self.needs_stencil, render_stage3d) {
            (true, true) => {
                render_pass.set_pipeline(&self.pipelines.bitmap_opaque_dummy_stencil);
            }
            (true, false) => {
                render_pass
                    .set_pipeline(self.pipelines.bitmap[blend_mode].pipeline_for(self.mask_state));
            }
            (false, true) => {
                render_pass.set_pipeline(&self.pipelines.bitmap_opaque);
            }
            (false, false) => {
                render_pass.set_pipeline(self.pipelines.bitmap[blend_mode].stencilless_pipeline());
            }
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn prep_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bind_group: &'encoder wgpu::BindGroup,
    ) {
        if self.needs_stencil {
            render_pass.set_pipeline(self.pipelines.alpha_mask.pipeline_for(self.mask_state));
        } else {
            render_pass.set_pipeline(self.pipelines.alpha_mask.stencilless_pipeline());
        }

        render_pass.set_bind_group(2, bind_group, &[]);
    }

    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        vertices: wgpu::BufferSlice<'encoder>,
        indices: wgpu::BufferSlice<'encoder>,
        num_indices: u32,
    ) {
        render_pass.set_vertex_buffer(0, vertices);
        render_pass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);

        render_pass.draw_indexed(0..num_indices, 0, 0..1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_bitmap(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        bitmap: &'encoder BitmapHandle,
        transform_buffer: wgpu::DynamicOffset,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
        vertex_offset: Option<wgpu::BufferAddress>,
    ) {
        let texture = as_texture(bitmap);

        let descriptors = self.descriptors;
        let bind = texture.bind_group(
            false,
            smoothing,
            &descriptors.device,
            &descriptors.bind_layouts.bitmap,
            bitmap.clone(),
            &descriptors.bitmap_samplers,
        );
        self.prep_bitmap(render_pass, &bind.bind_group, blend_mode, render_stage3d);
        render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

        let vertex_slice = if let Some(vertex_offset) = vertex_offset {
            self.dynamic_transforms.vertex_buffer.slice(vertex_offset..)
        } else {
            self.descriptors.quad.vertices_pos_uv.slice(..)
        };

        self.draw(
            render_pass,
            vertex_slice,
            self.descriptors.quad.indices.slice(..),
            6,
        );
    }

    pub fn render_texture(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        transform_buffer: wgpu::DynamicOffset,
        bind_group: &'encoder wgpu::BindGroup,
        blend_mode: TrivialBlend,
        vertex_offset: Option<wgpu::BufferAddress>,
    ) {
        self.prep_bitmap(render_pass, bind_group, blend_mode, false);

        render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

        // A group that had a target to itself samples all of it, which is what
        // the shared quad's texture coordinates say. One that took a region of
        // a page brings its own, naming the region.
        let vertices = match vertex_offset {
            Some(offset) => self.dynamic_transforms.vertex_buffer.slice(offset..),
            None => self.descriptors.quad.vertices_pos_uv.slice(..),
        };

        self.draw(
            render_pass,
            vertices,
            self.descriptors.quad.indices.slice(..),
            6,
        );
    }

    pub fn render_shape(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        shape: &'encoder ShapeHandle,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        let mesh = as_mesh(shape);
        for draw in &mesh.draws {
            let num_indices = if self.mask_state != MaskState::DrawMaskStencil
                && self.mask_state != MaskState::ClearMaskStencil
            {
                draw.num_indices
            } else {
                // Omit strokes when drawing a mask stencil.
                draw.num_mask_indices
            };
            if num_indices == 0 {
                continue;
            }

            match &draw.draw_type {
                DrawType::Color => {
                    self.prep_color(render_pass);
                }
                DrawType::Gradient { bind_group, .. } => {
                    self.prep_gradient(render_pass, bind_group);
                }
                DrawType::Bitmap { binds, .. } => {
                    self.prep_bitmap(render_pass, &binds.bind_group, TrivialBlend::Normal, false);
                }
            }
            render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

            self.draw(
                render_pass,
                mesh.vertex_buffer.slice(draw.vertices.clone()),
                mesh.index_buffer.slice(draw.indices.clone()),
                num_indices,
            );
        }
    }

    pub fn render_alpha_mask(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        _maskee: &PoolOrArcTexture,
        _mask: &PoolOrArcTexture,
        bind_group: &'encoder wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        if cfg!(feature = "render_debug_labels") {
            render_pass.push_debug_group("render_alpha_mask");
        }

        self.prep_alpha_mask(render_pass, bind_group);

        render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
        );

        if cfg!(feature = "render_debug_labels") {
            render_pass.pop_debug_group();
        }
    }

    pub fn draw_rect(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        self.prep_color(render_pass);

        render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            self.descriptors.quad.indices.slice(..),
            6,
        );
    }

    pub fn draw_lines<const RECT: bool>(
        &self,
        render_pass: &mut wgpu::RenderPass<'encoder>,
        transform_buffer: wgpu::DynamicOffset,
    ) {
        self.prep_lines(render_pass);

        render_pass.set_bind_group(1, &self.dynamic_transforms.bind_group, &[transform_buffer]);

        self.draw(
            render_pass,
            self.descriptors.quad.vertices_pos_color.slice(..),
            if RECT {
                self.descriptors.quad.indices_line_rect.slice(..)
            } else {
                self.descriptors.quad.indices_line.slice(..)
            },
            if RECT { 5 } else { 2 },
        );
    }

    pub fn push_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(
            self.mask_state == MaskState::NoMask || self.mask_state == MaskState::DrawMaskedContent
        );
        self.num_masks += 1;
        self.mask_state = MaskState::DrawMaskStencil;
        render_pass.set_stencil_reference(self.num_masks - 1);
    }

    pub fn activate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskStencil);
        self.mask_state = MaskState::DrawMaskedContent;
        render_pass.set_stencil_reference(self.num_masks);
    }

    pub fn deactivate_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::DrawMaskedContent);
        self.mask_state = MaskState::ClearMaskStencil;
        render_pass.set_stencil_reference(self.num_masks);
    }

    pub fn pop_mask(&mut self, render_pass: &mut wgpu::RenderPass<'encoder>) {
        debug_assert!(self.num_masks > 0 && self.mask_state == MaskState::ClearMaskStencil);
        self.num_masks -= 1;
        render_pass.set_stencil_reference(self.num_masks);
        if self.num_masks == 0 {
            self.mask_state = MaskState::NoMask;
        } else {
            self.mask_state = MaskState::DrawMaskedContent;
        };
    }

    pub fn num_masks(&self) -> u32 {
        self.num_masks
    }

    pub fn mask_state(&self) -> MaskState {
        self.mask_state
    }
}

pub enum Chunk {
    Draw {
        chunk: Vec<DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
    },
    Blend {
        /// The blended group's pixels. Shared, because a page holds the pixels
        /// of many groups and each of them composites separately.
        texture: Arc<PoolOrArcTexture>,
        /// `[u0, v0, du, dv]`: which part of `texture` is this group's.
        /// `[0.0, 0.0, 1.0, 1.0]` for a group that had a target to itself.
        source_uv: [f32; 4],
        blend_mode: ChunkBlendMode,
        needs_stencil: bool,
        /// Where the blended group's target belongs, in the coordinates the
        /// commands of the target it is composited into are expressed in.
        rect: TargetRect,
    },
}

#[derive(Debug)]
pub enum ChunkBlendMode {
    Complex(ComplexBlend),
    Shader(PixelBenderShaderHandle),
}

#[derive(Debug)]
pub enum DrawCommand {
    RenderBitmap {
        bitmap: BitmapHandle,
        transform_buffer: wgpu::DynamicOffset,
        vertex_offset: Option<wgpu::BufferAddress>,
        smoothing: bool,
        blend_mode: TrivialBlend,
        render_stage3d: bool,
    },
    RenderTexture {
        /// The target this was rendered through, when it had one to itself.
        /// A region of a page is kept alive by the page instead.
        _texture: Option<PoolOrArcTexture>,
        binds: wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
        /// The quad naming the part of `binds` to sample; the whole of it when
        /// this is `None`.
        vertex_offset: Option<wgpu::BufferAddress>,
        blend_mode: TrivialBlend,
    },
    RenderAlphaMask {
        maskee: PoolOrArcTexture,
        mask: PoolOrArcTexture,
        binds: wgpu::BindGroup,
        transform_buffer: wgpu::DynamicOffset,
    },
    RenderShape {
        shape: ShapeHandle,
        transform_buffer: wgpu::DynamicOffset,
    },
    DrawRect {
        transform_buffer: wgpu::DynamicOffset,
    },
    DrawLine {
        transform_buffer: wgpu::DynamicOffset,
    },
    DrawLineRect {
        transform_buffer: wgpu::DynamicOffset,
    },
    PushMask,
    ActivateMask,
    DeactivateMask,
    PopMask,
}

impl DrawCommand {
    pub fn name(&self) -> &'static str {
        match self {
            DrawCommand::RenderBitmap { .. } => "render bitmap",
            DrawCommand::RenderShape { .. } => "render shape",
            DrawCommand::RenderTexture { .. } => "render texture",
            DrawCommand::DrawRect { .. } => "draw rect",
            DrawCommand::DrawLine { .. } => "draw line",
            DrawCommand::DrawLineRect { .. } => "draw line rect",
            DrawCommand::PushMask => "push mask",
            DrawCommand::ActivateMask => "activate mask",
            DrawCommand::DeactivateMask => "deactivate mask",
            DrawCommand::PopMask => "pop mask",
            DrawCommand::RenderAlphaMask { .. } => "render alpha mask",
        }
    }
}

/// The matrix that maps the unit square onto `rect`.
///
/// Compositing a sub-target back is a quad of exactly that shape, so this puts
/// its texture where its contents were.
fn rect_matrix(rect: TargetRect) -> Matrix {
    Matrix {
        a: rect.width as f32,
        b: 0.0,
        c: 0.0,
        d: rect.height as f32,
        tx: Twips::from_pixels(rect.x as f64),
        ty: Twips::from_pixels(rect.y as f64),
    }
}

#[derive(Copy, Clone)]
pub enum LayerRef<'a> {
    None,
    Current,
    Parent(&'a CommandTarget),
}

/// The passes a command list has been broken into, and the pages they read.
pub struct ChunkedCommands {
    pub chunks: Vec<Chunk>,
    /// The pages blended groups were rendered onto.
    ///
    /// Their passes are already encoded; these keep the pooled textures out of
    /// circulation until the chunks that composite from them have been encoded
    /// too, so that a later target cannot be handed a page and clear it first.
    pub pages: Vec<Arc<PoolOrArcTexture>>,
}

/// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
/// Every complex blend will be its own item, but every other draw will be chunked together
#[expect(clippy::too_many_arguments)]
pub fn chunk_blends<'encoder, 'global: 'encoder>(
    commands: CommandList,
    descriptors: &'encoder Descriptors,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    meshes: &'encoder Vec<Mesh>,
    quality: StageQuality,
    rect: TargetRect,
    nearest_layer: LayerRef,
    texture_pool: &'encoder mut TexturePool,
) -> ChunkedCommands {
    WgpuCommandHandler::new(
        descriptors,
        staging_belt,
        dynamic_transforms,
        draw_encoder,
        meshes,
        quality,
        rect,
        nearest_layer,
        texture_pool,
    )
    .chunk_blends(commands)
}

struct WgpuCommandHandler<'encoder, 'global: 'encoder> {
    descriptors: &'encoder Descriptors,
    quality: StageQuality,
    /// The target these commands are being drawn into, in the space the
    /// commands themselves are expressed in.
    ///
    /// Commands always carry the coordinates they would have on the surface
    /// that started this draw, so a target that covers only part of it has to
    /// take its own origin back off every matrix it hands to the shaders.
    rect: TargetRect,
    nearest_layer: LayerRef<'encoder>,
    meshes: &'encoder Vec<Mesh>,
    staging_belt: &'encoder mut wgpu::util::StagingBelt,
    dynamic_transforms: &'encoder DynamicTransforms,
    draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
    texture_pool: &'encoder mut TexturePool,
    emulate_lines: bool,
    /// The pages this walk's blended groups are sharing.
    pages: BlendPages,

    result: Vec<Chunk>,
    current: Vec<DrawCommand>,
    transforms: BufferBuilder,
    vertices: BufferBuilder,
    needs_stencil: bool,
    num_masks: i32,
    /// Whether the draws being built belong to a page rather than to the
    /// surface.
    paging: bool,
}

impl<'encoder, 'global: 'encoder> WgpuCommandHandler<'encoder, 'global> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        descriptors: &'encoder Descriptors,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'encoder Vec<Mesh>,
        quality: StageQuality,
        rect: TargetRect,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> Self {
        let transforms = Self::new_transforms(descriptors, dynamic_transforms);
        let vertices = Self::new_vertices(descriptors, dynamic_transforms);

        // DirectX does support drawing lines, but it's very inconsistent.
        // With MSAA, lines have 1.4px thickness, which makes them too thick.
        // Without MSAA, lines have 1px thickness, but their placement is sometimes off.
        let emulate_lines = descriptors.backend == Backend::Dx12;

        // Blended groups are always rendered through `Rgba8Unorm` at the
        // surface's quality, whatever the surface's own format is, so a page
        // has to match that and not the surface.
        let pages = BlendPages::new(
            descriptors,
            BLEND_TARGET_FORMAT,
            crate::utils::supported_sample_count(
                &descriptors.adapter,
                quality.sample_count(),
                BLEND_TARGET_FORMAT,
            ),
        );

        Self {
            descriptors,
            quality,
            rect,
            nearest_layer,
            meshes,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            texture_pool,
            emulate_lines,
            pages,

            result: vec![],
            current: vec![],
            transforms,
            vertices,
            needs_stencil: false,
            num_masks: 0,
            paging: false,
        }
    }

    fn new_transforms(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut transforms = BufferBuilder::new_for_uniform(&descriptors.limits);
        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
        transforms
    }

    fn new_vertices(
        descriptors: &'encoder Descriptors,
        dynamic_transforms: &'encoder DynamicTransforms,
    ) -> BufferBuilder {
        let mut vertices = BufferBuilder::new_for_vertices(&descriptors.limits);
        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());
        vertices
    }

    /// Replaces every blend with a RenderBitmap, with the subcommands rendered out to a temporary texture
    /// Every complex blend will be its own item, but every other draw will be chunked together
    fn chunk_blends(&mut self, commands: CommandList) -> ChunkedCommands {
        commands.execute(self);

        // Every page has to be drawn before anything composites from it, and
        // every composite is in a chunk the caller has yet to execute.
        self.pages.finish(
            self.descriptors,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
        );

        let current = mem::take(&mut self.current);
        let mut result = mem::take(&mut self.result);
        let needs_stencil = mem::take(&mut self.needs_stencil);
        let transforms = mem::replace(
            &mut self.transforms,
            Self::new_transforms(self.descriptors, self.dynamic_transforms),
        );
        let vertices = mem::replace(
            &mut self.vertices,
            Self::new_vertices(self.descriptors, self.dynamic_transforms),
        );

        if !current.is_empty() {
            result.push(Chunk::Draw {
                chunk: current,
                needs_stencil,
                transforms,
                vertices,
            });
        }

        ChunkedCommands {
            chunks: result,
            pages: self.pages.take_held(),
        }
    }

    /// Draws a bitmap with an explicit blend state.
    ///
    /// `render_bitmap` is this with `Normal`; a blended group of one bitmap is
    /// this with the group's blend state, which is what lets that group skip
    /// its render target entirely.
    fn render_bitmap_with_blend(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
        blend_mode: TrivialBlend,
    ) {
        let texture = as_texture(&bitmap);

        let mut matrix = transform.matrix;
        pixel_snapping.apply(&mut matrix);
        matrix *= Matrix::scale(region.width() as f32, region.height() as f32);

        let vertices: &[PosUvVertex] = {
            let (u0, u1, v0, v1) = (
                region.x_min as f32 / texture.texture.width() as f32,
                region.x_max as f32 / texture.texture.width() as f32,
                region.y_min as f32 / texture.texture.height() as f32,
                region.y_max as f32 / texture.texture.height() as f32,
            );
            &[
                PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
                PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
                PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
                PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
            ]
        };

        self.add_to_current_with_vertices(
            matrix,
            transform.color_transform,
            Some(vertices),
            |transform_buffer, vertex_offset| DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                vertex_offset,
                smoothing,
                blend_mode,
                render_stage3d: false,
            },
        );
    }

    /// Renders a blended group onto a shared page and composites it from there.
    ///
    /// Gives the group's commands back if it could not: it is then rendered
    /// through a target of its own, exactly as before.
    fn blend_through_page(
        &mut self,
        commands: CommandList,
        blend_type: &BlendType,
    ) -> Result<(), CommandList> {
        if !crate::tuning::blend_pages_enabled() {
            return Err(commands);
        }
        if let Err(reason) = page_eligible(&commands, blend_type) {
            crate::render_stats::record_batch(false, Some(reason));
            return Err(commands);
        }

        // A page region is not a pool key, so it is the content's own size
        // rather than a size class - a third less area for an avatar.
        let rect = region_rect_for(content_bounds(&commands), self.rect);
        let (transforms, vertices) = page_reserve(&commands, self.emulate_lines);
        let placement = match self.pages.place(
            self.descriptors,
            self.texture_pool,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            rect.width,
            rect.height,
            transforms,
            vertices,
        ) {
            Ok(placement) => placement,
            Err(reason) => {
                crate::render_stats::record_batch(false, Some(reason));
                return Err(commands);
            }
        };
        crate::render_stats::record_batch(true, None);
        let region = placement.region;

        self.draw_into_region(commands, rect, region);

        match blend_type {
            BlendType::Trivial(blend_mode) => {
                let blend_mode = *blend_mode;
                let binds = placement.binds;
                let quad = region.quad();
                self.add_to_current_with_vertices(
                    rect_matrix(rect),
                    Default::default(),
                    Some(&quad),
                    |transform_buffer, vertex_offset| DrawCommand::RenderTexture {
                        _texture: None,
                        binds,
                        transform_buffer,
                        vertex_offset,
                        blend_mode,
                    },
                );
            }
            BlendType::Complex(complex) => {
                self.flush_current();
                self.result.push(Chunk::Blend {
                    texture: placement.source,
                    source_uv: region.uv(),
                    blend_mode: ChunkBlendMode::Complex(*complex),
                    needs_stencil: self.num_masks > 0,
                    rect,
                });
                self.needs_stencil = self.num_masks > 0;
            }
            BlendType::Shader(_) => unreachable!("a shader blend is never page-eligible"),
        }
        Ok(())
    }

    /// Draws a blended group's commands into its region of a page.
    ///
    /// The handler's own state is lent to the group for the walk: its draws go
    /// into the page's buffers rather than the surface's, and its world
    /// matrices have the region's origin put back on, so that an object at
    /// `x = 100.37` lands `.37` into the region just as it landed `.37` into a
    /// target of its own.
    fn draw_into_region(&mut self, commands: CommandList, rect: TargetRect, region: PageRegion) {
        let saved_rect = mem::replace(
            &mut self.rect,
            TargetRect {
                x: rect.x - region.x as i32,
                y: rect.y - region.y as i32,
                width: region.page_width,
                height: region.page_height,
            },
        );
        let saved_current = mem::take(&mut self.current);
        let saved_needs_stencil = mem::replace(&mut self.needs_stencil, false);
        self.pages
            .swap_builders(&mut self.transforms, &mut self.vertices);
        let saved_paging = mem::replace(&mut self.paging, true);

        commands.execute(self);

        self.paging = saved_paging;
        self.pages
            .swap_builders(&mut self.transforms, &mut self.vertices);
        let run = mem::replace(&mut self.current, saved_current);
        self.rect = saved_rect;
        self.needs_stencil = saved_needs_stencil;
        self.pages.add_run(region, run);
    }

    /// Closes the chunk being built, if there is anything in it.
    fn flush_current(&mut self) {
        if self.current.is_empty() {
            return;
        }
        self.result.push(Chunk::Draw {
            chunk: mem::take(&mut self.current),
            needs_stencil: self.needs_stencil,
            transforms: mem::replace(
                &mut self.transforms,
                Self::new_transforms(self.descriptors, self.dynamic_transforms),
            ),
            vertices: mem::replace(
                &mut self.vertices,
                Self::new_vertices(self.descriptors, self.dynamic_transforms),
            ),
        });
    }

    fn add_to_current(
        &mut self,
        matrix: Matrix,
        color_transform: ColorTransform,
        command_builder: impl FnOnce(wgpu::DynamicOffset) -> DrawCommand,
    ) {
        self.add_to_current_with_vertices(matrix, color_transform, None, |transform_buffer, _| {
            command_builder(transform_buffer)
        })
    }

    fn add_to_current_with_vertices(
        &mut self,
        matrix: Matrix,
        color_transform: ColorTransform,
        vertices: Option<&[PosUvVertex]>,
        command_builder: impl FnOnce(wgpu::DynamicOffset, Option<wgpu::BufferAddress>) -> DrawCommand,
    ) {
        // Commands are in the coordinates of the surface that started this
        // draw; this target may cover only part of it.
        let transform = Transforms {
            world_matrix: [
                [matrix.a, matrix.b, 0.0, 0.0],
                [matrix.c, matrix.d, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    matrix.tx.to_pixels() as f32 - self.rect.x as f32,
                    matrix.ty.to_pixels() as f32 - self.rect.y as f32,
                    0.0,
                    1.0,
                ],
            ],
            mult_color: color_transform.mult_rgba_normalized(),
            add_color: color_transform.add_rgba_normalized(),
        };
        if let (Ok(transform_range), Ok(vertices_range)) = (
            self.transforms.add(&[transform]),
            vertices.map(|v| self.vertices.add(v)).transpose(),
        ) {
            self.current.push(command_builder(
                transform_range.start as wgpu::DynamicOffset,
                vertices_range.map(|v| v.start),
            ));
        } else {
            // A group being drawn onto a page had room reserved for all of it
            // before the region was handed out, so this cannot be reached from
            // there - and must not be, because the chunk would be drawn onto
            // the surface rather than onto the page.
            debug_assert!(
                !self.paging,
                "a paged group overflowed the buffers reserved for it"
            );
            self.result.push(Chunk::Draw {
                chunk: mem::take(&mut self.current),
                needs_stencil: self.needs_stencil,
                transforms: mem::replace(
                    &mut self.transforms,
                    Self::new_transforms(self.descriptors, self.dynamic_transforms),
                ),
                vertices: mem::replace(
                    &mut self.vertices,
                    Self::new_vertices(self.descriptors, self.dynamic_transforms),
                ),
            });
            let transform_range = self
                .transforms
                .add(&[transform])
                .expect("Buffer must be able to fit a new thing, it was just emptied");
            let vertices_range = vertices.map(|v| {
                self.vertices
                    .add(v)
                    .expect("Buffer must be able to fit a new thing, it was just emptied")
            });
            self.current.push(command_builder(
                transform_range.start as wgpu::DynamicOffset,
                vertices_range.map(|v| v.start),
            ));
        }
    }
}

/// The format every blended group is rendered through, whatever the surface it
/// is composited onto is.
const BLEND_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The source rectangle of a group that has a whole texture to itself.
pub const WHOLE_TEXTURE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Whether a blended group can share a page with its siblings instead of taking
/// a render target of its own, and if not, why not.
///
/// A page draws every group on it in one render pass, scissored to each group's
/// own region, so a group can only join if its commands are a plain run of
/// draws: anything that would want a pass or a target *inside* the shared one -
/// a nested blend, an alpha mask, a stencil mask - keeps its own target, where
/// the existing code can give it one.
///
/// This is a wider door than [`trivial_fast_path`]: that one needs a group of
/// exactly one drawable under a blend state that cannot saturate, because it
/// draws the group's contents straight onto the destination. A page still
/// composites the group as a unit through a texture, so the group may have as
/// many children as it likes and any blend mode that is not arbitrary code.
fn page_eligible(
    commands: &CommandList,
    blend_type: &BlendType,
) -> Result<(), crate::PageFallback> {
    use crate::PageFallback;

    // A `PixelBender` blend is arbitrary code that may write anywhere its quad
    // covers, so it keeps the full-sized target it is given.
    if matches!(blend_type, BlendType::Shader(_)) {
        return Err(PageFallback::Shader);
    }

    for command in &commands.commands {
        match command {
            Command::RenderBitmap { .. }
            | Command::RenderShape { .. }
            | Command::DrawRect { .. }
            | Command::DrawLine { .. }
            | Command::DrawLineRect { .. } => {}
            Command::Blend(..) => return Err(PageFallback::NestedBlend),
            Command::RenderAlphaMask { .. } => return Err(PageFallback::AlphaMask),
            Command::PushMask
            | Command::ActivateMask
            | Command::DeactivateMask
            | Command::PopMask => return Err(PageFallback::Masked),
            Command::RenderStage3D { .. } => return Err(PageFallback::Stage3D),
        }
    }

    Ok(())
}

/// What a page-eligible group will need of a chunk's buffers, exactly.
///
/// Every draw takes one aligned slot in the uniform buffer, and the ones that
/// carry their own quad take four vertices. A group is only put on a page once
/// there is certainly room for all of it, because a run split between two
/// buffers would be split between two render passes and half of it would land
/// on the wrong page.
fn page_reserve(commands: &CommandList, emulate_lines: bool) -> (usize, usize) {
    let mut transforms = 0;
    let mut vertices = 0;
    for command in &commands.commands {
        match command {
            Command::RenderBitmap { .. } => {
                transforms += 1;
                vertices += 4;
            }
            // A rectangle of emulated lines becomes four rectangles.
            Command::DrawLineRect { .. } if emulate_lines => transforms += 4,
            _ => transforms += 1,
        }
    }
    (transforms, vertices)
}

/// Whether a blended group can be drawn straight onto its destination instead
/// of through a render target of its own, and if not, why not.
///
/// The intermediate target exists so that a group composites as a unit: its
/// children are drawn together and the blend is applied once to the result. A
/// group of *one* drawable has nothing to composite, so for a blend mode that
/// is just a GPU blend state, drawing that one thing with the blend state set
/// gives the same picture without the target, the render pass, the pool take,
/// the bind group or the second sampling step.
///
/// This is deliberately narrow. Anything that makes the group more than a
/// single unconditional draw - a second command, a mask around it, a nested
/// blend, a shape whose pipelines have no blend-state variants - keeps the
/// target.
fn trivial_fast_path(
    commands: &CommandList,
    blend_type: &BlendType,
) -> Result<(), crate::FallbackReason> {
    use crate::FallbackReason;

    // A complex blend reads the destination in a shader and genuinely needs the
    // group rendered out first; a PixelBender blend is arbitrary code.
    let trivial = match blend_type {
        BlendType::Trivial(trivial) => trivial,
        BlendType::Complex(_) | BlendType::Shader(_) => {
            return Err(FallbackReason::ComplexBlend);
        }
    };

    // Of the blend states, only `Normal` gives the same answer applied per
    // multisample as applied to the resolved group.
    //
    // Take `Add` on a rotated bitmap. Through a target, the group is resolved
    // first, so an edge covering half the samples contributes half its colour
    // and the sum is clamped once: `min(1, dst + c*src)`. Drawn directly, the
    // sum is clamped at each covered sample and the resolve averages the
    // clamped values: `c*min(1, dst + src) + (1-c)*dst`. Those differ wherever
    // the sum saturates, which on a light background is most of the edge - it
    // showed as a 43-level difference along the edges of a rotated bitmap.
    // `Normal` cannot saturate: a premultiplied source is at most its own
    // alpha, so `src + dst*(1-a)` is at most 1, and the two orders agree
    // exactly.
    //
    // This still covers `BlendMode::LAYER`, which is the mode that wraps a
    // group so it composites as a unit, and so the one a cached or filtered
    // display object arrives under.
    if !matches!(trivial, TrivialBlend::Normal) {
        return Err(FallbackReason::UnsupportedBlendMode);
    }

    let [command] = commands.commands.as_slice() else {
        return Err(FallbackReason::MultipleDraws);
    };

    match command {
        // A bitmap is the case that matters: a filtered or cached display
        // object reaches the renderer as exactly one `render_bitmap` of its
        // cache texture, which is what most of a crowded room's blended
        // objects are.
        Command::RenderBitmap { .. } => Ok(()),
        // Shapes are drawn through the colour, gradient and bitmap-fill
        // pipelines, which are only built with premultiplied-alpha blending;
        // there is no `Add` or `Screen` variant of them to select.
        Command::RenderShape { .. }
        | Command::DrawRect { .. }
        | Command::DrawLine { .. }
        | Command::DrawLineRect { .. }
        | Command::RenderStage3D { .. } => Err(FallbackReason::UnsupportedCommand),
        Command::Blend(..) => Err(FallbackReason::NestedBlend),
        Command::RenderAlphaMask { .. } => Err(FallbackReason::RequiresIntermediate),
        Command::PushMask | Command::ActivateMask | Command::DeactivateMask | Command::PopMask => {
            Err(FallbackReason::Masked)
        }
    }
}

impl CommandHandler for WgpuCommandHandler<'_, '_> {
    fn blend(&mut self, commands: CommandList, blend_mode: RenderBlendMode) {
        let target_layer = if let RenderBlendMode::Builtin(BlendMode::Layer) = &blend_mode {
            LayerRef::Current
        } else {
            self.nearest_layer
        };
        let blend_type = BlendType::from(blend_mode);

        // A group of one drawable does not need a target of its own.
        match trivial_fast_path(&commands, &blend_type) {
            Ok(()) => {
                crate::render_stats::record_fastpath(true, None);
                let BlendType::Trivial(blend_mode) = blend_type else {
                    unreachable!("the fast path only accepts trivial blends")
                };
                let Some(Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                    region,
                }) = commands.commands.into_iter().next()
                else {
                    unreachable!("the fast path only accepts a single bitmap")
                };
                self.render_bitmap_with_blend(
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                    region,
                    blend_mode,
                );
                return;
            }
            Err(reason) => crate::render_stats::record_fastpath(false, Some(reason)),
        }

        // A group that is a plain run of draws can share a page with its
        // siblings instead of taking a target, a render pass and a pool entry
        // of its own.
        let commands = match self.blend_through_page(commands, &blend_type) {
            Ok(()) => return,
            Err(commands) => commands,
        };

        // Every built-in blend leaves the destination untouched where the
        // blended group is transparent - the complex-blend shaders `discard`
        // there, and each trivial blend state is the identity on the
        // destination for a zero source - so a target that covers only what the
        // group draws composites to the same picture as a screen-sized one. A
        // PixelBender blend is arbitrary code that may write anywhere its quad
        // covers, so it keeps the full-sized target.
        let rect = match &blend_type {
            BlendType::Shader(_) => self.rect,
            _ => target_rect_for(content_bounds(&commands), self.rect),
        };

        let surface = Surface::for_rect(
            self.descriptors,
            self.quality,
            rect,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        crate::render_stats::blend_target_taken(rect.width as usize * rect.height as usize * 4);
        let clear_color = blend_type.default_color();
        let target = surface.draw_commands(
            RenderTargetMode::FreshWithColor(clear_color),
            self.descriptors,
            self.meshes,
            commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            target_layer,
            self.texture_pool,
        );
        target.ensure_cleared(self.draw_encoder);

        // We currently do not support shader blends in masks. In order not to
        // break other parts of the scene, we just fall back to a normal blend.
        //
        // TODO Add support for shader blends in masks.
        let is_shader_blend_in_mask =
            self.num_masks > 0 && matches!(blend_type, BlendType::Shader(_));
        let blend_type = if is_shader_blend_in_mask {
            BlendType::Trivial(TrivialBlend::Normal)
        } else {
            blend_type
        };

        match blend_type {
            BlendType::Trivial(blend_mode) => {
                let transform = Transform {
                    matrix: rect_matrix(rect),
                    color_transform: Default::default(),
                    perspective_projection: None,
                };
                let texture = target.take_color_texture();
                let bind_group = texture.bitmap_bind_group(self.descriptors).clone();
                self.add_to_current(
                    transform.matrix,
                    transform.color_transform,
                    |transform_buffer| DrawCommand::RenderTexture {
                        _texture: Some(texture),
                        binds: bind_group,
                        transform_buffer,
                        vertex_offset: None,
                        blend_mode,
                    },
                );
            }
            blend_type => {
                self.flush_current();
                let chunk_blend_mode = match blend_type {
                    BlendType::Complex(complex) => ChunkBlendMode::Complex(complex),
                    BlendType::Shader(shader) => ChunkBlendMode::Shader(shader),
                    _ => unreachable!(),
                };
                self.result.push(Chunk::Blend {
                    texture: Arc::new(target.take_color_texture()),
                    source_uv: WHOLE_TEXTURE,
                    blend_mode: chunk_blend_mode,
                    needs_stencil: self.num_masks > 0,
                    rect,
                });
                self.needs_stencil = self.num_masks > 0;
            }
        }
    }

    fn render_bitmap(
        &mut self,
        bitmap: BitmapHandle,
        transform: Transform,
        smoothing: bool,
        pixel_snapping: PixelSnapping,
        region: PixelRegion,
    ) {
        self.render_bitmap_with_blend(
            bitmap,
            transform,
            smoothing,
            pixel_snapping,
            region,
            TrivialBlend::Normal,
        )
    }

    fn render_stage3d(&mut self, bitmap: BitmapHandle, transform: Transform) {
        let mut matrix = transform.matrix;
        {
            let texture = as_texture(&bitmap);
            matrix *= Matrix::scale(
                texture.texture.width() as f32,
                texture.texture.height() as f32,
            );
        }
        self.add_to_current(matrix, transform.color_transform, |transform_buffer| {
            DrawCommand::RenderBitmap {
                bitmap,
                transform_buffer,
                vertex_offset: None,
                smoothing: false,
                blend_mode: TrivialBlend::Normal,
                render_stage3d: true,
            }
        });
    }

    fn render_shape(&mut self, shape: ShapeHandle, transform: Transform) {
        self.add_to_current(
            transform.matrix,
            transform.color_transform,
            |transform_buffer| DrawCommand::RenderShape {
                shape,
                transform_buffer,
            },
        );
    }

    fn draw_rect(&mut self, color: Color, matrix: Matrix) {
        self.add_to_current(
            matrix,
            ColorTransform::multiply_from(color),
            |transform_buffer| DrawCommand::DrawRect { transform_buffer },
        );
    }

    fn draw_line(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                ColorTransform::multiply_from(color),
                |transform_buffer| DrawCommand::DrawLine { transform_buffer },
            );
        }
    }

    fn draw_line_rect(&mut self, color: Color, mut matrix: Matrix) {
        if self.emulate_lines {
            let mut cl = CommandList::new();
            emulate_line_rect(&mut cl, color, matrix);
            cl.execute(self);
        } else {
            matrix.tx += Twips::HALF_PX;
            matrix.ty += Twips::HALF_PX;
            self.add_to_current(
                matrix,
                ColorTransform::multiply_from(color),
                |transform_buffer| DrawCommand::DrawLineRect { transform_buffer },
            );
        }
    }

    fn push_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks += 1;
        self.current.push(DrawCommand::PushMask);
    }

    fn activate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::ActivateMask);
    }

    fn deactivate_mask(&mut self) {
        self.needs_stencil = true;
        self.current.push(DrawCommand::DeactivateMask);
    }

    fn pop_mask(&mut self) {
        self.needs_stencil = true;
        self.num_masks -= 1;
        self.current.push(DrawCommand::PopMask);
    }

    fn render_alpha_mask(&mut self, maskee_commands: CommandList, mask_commands: CommandList) {
        // The result is the maskee scaled by the mask's alpha, so it is inside
        // both of them and transparent everywhere else.
        let bounds = content_bounds(&maskee_commands).intersect(content_bounds(&mask_commands));
        let rect = target_rect_for(bounds, self.rect);
        let surface = Surface::for_rect(
            self.descriptors,
            self.quality,
            rect,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let maskee = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            maskee_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        maskee.ensure_cleared(self.draw_encoder);
        let matrix = rect_matrix(rect);
        let maskee = maskee.take_color_texture();

        let mask = surface.draw_commands(
            RenderTargetMode::FreshWithColor(wgpu::Color::TRANSPARENT),
            self.descriptors,
            self.meshes,
            mask_commands,
            self.staging_belt,
            self.dynamic_transforms,
            self.draw_encoder,
            LayerRef::None,
            self.texture_pool,
        );
        mask.ensure_cleared(self.draw_encoder);
        let mask = mask.take_color_texture();

        let descriptors = self.descriptors;
        let binds = maskee.binds().paired(mask.binds().id(), || {
            descriptors
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &descriptors.bind_layouts.alpha_mask,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(maskee.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(mask.view()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                descriptors.bitmap_samplers.get_sampler(false, false),
                            ),
                        },
                    ],
                    label: create_debug_label!("Alpha mask").as_deref(),
                })
        });

        self.add_to_current(matrix, Default::default(), |transform_buffer| {
            DrawCommand::RenderAlphaMask {
                maskee,
                mask,
                binds,
                transform_buffer,
            }
        });
    }
}
