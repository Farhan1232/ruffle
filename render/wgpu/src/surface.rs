mod commands;
mod page;
pub mod target;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::bounds::TargetRect;
use crate::buffer_builder::BufferBuilder;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{Chunk, ChunkedCommands, CommandRenderer, chunk_blends};
use crate::surface::target::PoolOrArcTexture;
use crate::utils::run_copy_pipeline;
use crate::utils::supported_sample_count;
use crate::{Descriptors, MaskState, Pipelines};
use crate::{PosUvVertex, Transforms};
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::sync::Arc;
use target::CommandTarget;
use tracing::instrument;
use wgpu_profiler::Scope;

pub use crate::surface::commands::LayerRef;

use self::commands::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    /// Where this surface's target sits in the coordinate space of the commands
    /// drawn into it, and how big it is.
    ///
    /// The surface a frame is drawn to covers the whole viewport, but the
    /// temporary surfaces that blends and alpha masks are rendered through
    /// cover only the part of it their contents can reach, so their origin is
    /// not the origin of the commands they draw.
    rect: TargetRect,
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        Self::for_rect(
            descriptors,
            quality,
            TargetRect::from_size(width, height),
            frame_buffer_format,
        )
    }

    /// A surface covering only `rect` of the space its commands are drawn in.
    pub fn for_rect(
        descriptors: &Descriptors,
        quality: StageQuality,
        rect: TargetRect,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        let sample_count = supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format);
        Self {
            rect,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
        }
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'encoder, 'global: 'encoder>(
        &self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef<'encoder>,
        texture_pool: &'global mut TexturePool,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
        );

        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.whole_frame_bind_group(descriptors),
            target.globals(),
            1,
            draw_encoder,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'encoder, 'global: 'encoder>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'encoder Descriptors,
        meshes: &'encoder Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'encoder mut wgpu::util::StagingBelt,
        dynamic_transforms: &'encoder DynamicTransforms,
        draw_encoder: &'encoder mut Scope<'global, wgpu::CommandEncoder>,
        nearest_layer: LayerRef<'encoder>,
        texture_pool: &'encoder mut TexturePool,
    ) -> CommandTarget {
        let target = CommandTarget::new(
            descriptors,
            texture_pool,
            self.rect,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );

        let mut num_masks = 0;
        let mut mask_state = MaskState::NoMask;
        // `pages` holds the shared targets the chunks composite from. Their
        // passes are already encoded; keeping them here stops the pool handing
        // one of them out again - and clearing it - before the chunk that reads
        // it has been encoded.
        let ChunkedCommands {
            chunks,
            pages: _pages,
        } = chunk_blends(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            self.rect,
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
        );

        // Consecutive complex blends that cannot see each other's work are
        // composited together, in one pass off one snapshot of the destination.
        for pass in batch_passes(
            chunks,
            max_batch(descriptors, dynamic_transforms),
            nearest_layer,
        ) {
            match pass {
                Pass::Draw {
                    chunk,
                    needs_stencil,
                    transforms,
                    vertices,
                } => {
                    transforms.copy_to(staging_belt, draw_encoder, &dynamic_transforms.buffer);
                    vertices.copy_to(
                        staging_belt,
                        draw_encoder,
                        &dynamic_transforms.vertex_buffer,
                    );
                    crate::render_stats::record_render_pass();
                    let mut render_pass = draw_encoder.scoped_render_pass(
                        format!(
                            "Chunked draw calls {}",
                            if needs_stencil {
                                "(with stencil)"
                            } else {
                                "(Stencilless)"
                            }
                        ),
                        wgpu::RenderPassDescriptor {
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        },
                    );
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
                    let mut renderer = CommandRenderer::new(
                        &self.pipelines,
                        descriptors,
                        dynamic_transforms,
                        num_masks,
                        mask_state,
                        needs_stencil,
                    );

                    for command in &chunk {
                        renderer.execute(&mut render_pass.scope(command.name()), command);
                    }

                    num_masks = renderer.num_masks();
                    mask_state = renderer.mask_state();
                }
                Pass::Shader {
                    texture,
                    shader,
                    needs_stencil,
                    rect,
                } => {
                    assert!(!needs_stencil, "Shader blend mode not implemented in masks");
                    let parent_blend_buffer = target.update_blend_buffer(
                        descriptors,
                        texture_pool,
                        draw_encoder,
                        sample_region(rect, target.rect()),
                    );
                    run_pixelbender_shader_impl(
                        descriptors,
                        shader,
                        ShaderMode::Filter,
                        &[
                            PixelBenderShaderArgument::ImageInput {
                                index: 0,
                                channels: 0xFF,
                                name: "background".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(
                                    parent_blend_buffer.texture(),
                                )),
                            },
                            PixelBenderShaderArgument::ImageInput {
                                index: 1,
                                channels: 0xff,
                                name: "foreground".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                            },
                        ],
                        parent_blend_buffer.texture(),
                        draw_encoder,
                        target.color_attachments(),
                        target.sample_count(),
                        &FilterSource::for_entire_texture(texture.texture()),
                    )
                    .expect("Failed to run PixelBender blend mode");
                }
                Pass::Blends {
                    blends,
                    needs_stencil,
                    reads_layer,
                } => {
                    let parent = if reads_layer {
                        match nearest_layer {
                            // An Alpha or Erase with no Layer above it is
                            // dropped when the batches are built.
                            LayerRef::None => continue,
                            LayerRef::Current => &target,
                            LayerRef::Parent(layer) => layer,
                        }
                    } else {
                        &target
                    };

                    // Every blend in the batch reads a part of the destination
                    // that no other one in it writes, so one snapshot serves
                    // them all; each of them still only refreshes its own
                    // rectangle of it.
                    for blend in &blends {
                        parent.update_blend_buffer(
                            descriptors,
                            texture_pool,
                            draw_encoder,
                            sample_region(blend.rect, parent.rect()),
                        );
                    }
                    let parent_blend_buffer = parent
                        .blend_buffer()
                        .expect("the snapshots above created the buffer");

                    // The blend covers only the rectangle its group drew into.
                    // The quad's own coordinates are the blended texture's, and
                    // its `uv` attribute is where each corner lands in the
                    // destination it reads - which is `parent`, not necessarily
                    // the target being drawn into.
                    let mut offsets = Vec::with_capacity(blends.len());
                    {
                        let mut transforms = BufferBuilder::new_for_uniform(&descriptors.limits);
                        transforms.set_buffer_limit(dynamic_transforms.buffer.size());
                        let mut vertices = BufferBuilder::new_for_vertices(&descriptors.limits);
                        vertices.set_buffer_limit(dynamic_transforms.vertex_buffer.size());

                        let target_rect = target.rect();
                        for blend in &blends {
                            let rect = blend.rect;
                            let transform_range = transforms
                                .add(&[Transforms {
                                    world_matrix: [
                                        [rect.width as f32, 0.0, 0.0, 0.0],
                                        [0.0, rect.height as f32, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [
                                            (rect.x - target_rect.x) as f32,
                                            (rect.y - target_rect.y) as f32,
                                            0.0,
                                            1.0,
                                        ],
                                    ],
                                    // The blend shader has no colour transform
                                    // of its own to apply, so this carries
                                    // where the group's pixels are in the
                                    // texture it reads them from: the whole of
                                    // a target it had to itself, or one region
                                    // of a shared page.
                                    mult_color: blend.source_uv,
                                    add_color: [0.0, 0.0, 0.0, 0.0],
                                }])
                                .expect("a batch is sized to fit the buffers");
                            let vertex_range = vertices
                                .add(&blend_quad(rect, parent.rect()))
                                .expect("a batch is sized to fit the buffers");
                            offsets.push((
                                transform_range.start as wgpu::DynamicOffset,
                                vertex_range.start,
                            ));
                        }

                        transforms.copy_to(staging_belt, draw_encoder, &dynamic_transforms.buffer);
                        vertices.copy_to(
                            staging_belt,
                            draw_encoder,
                            &dynamic_transforms.vertex_buffer,
                        );
                    }

                    // The destination a blend reads is the same one frame
                    // after frame, so the bind group pairing it with this
                    // target is kept on the target rather than rebuilt. Every
                    // group sharing a page shares the bind group too, and
                    // picks its own region out of it with `source_uv`.
                    let binds: Vec<wgpu::BindGroup> = blends
                        .iter()
                        .map(|blend| {
                            blend
                                .texture
                                .binds()
                                .paired(parent_blend_buffer.binds_id(), || {
                                    descriptors.device.create_bind_group(
                                        &wgpu::BindGroupDescriptor {
                                            label: create_debug_label!("Complex blend binds")
                                                .as_deref(),
                                            layout: &descriptors.bind_layouts.blend,
                                            entries: &[
                                                wgpu::BindGroupEntry {
                                                    binding: 0,
                                                    resource: wgpu::BindingResource::TextureView(
                                                        parent_blend_buffer.view(),
                                                    ),
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding: 1,
                                                    resource: wgpu::BindingResource::TextureView(
                                                        blend.texture.view(),
                                                    ),
                                                },
                                                wgpu::BindGroupEntry {
                                                    binding: 2,
                                                    resource: wgpu::BindingResource::Sampler(
                                                        descriptors
                                                            .bitmap_samplers
                                                            .get_sampler(false, false),
                                                    ),
                                                },
                                            ],
                                        },
                                    )
                                })
                        })
                        .collect();

                    crate::render_stats::record_render_pass();
                    crate::render_stats::record_complex_batch(blends.len() as u64);
                    let mut render_pass =
                        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: create_debug_label!(
                                "Complex blends x{} {}",
                                blends.len(),
                                if needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            )
                            .as_deref(),
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        });
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);

                    if needs_stencil {
                        match mask_state {
                            MaskState::NoMask => {}
                            MaskState::DrawMaskStencil => {
                                render_pass.set_stencil_reference(num_masks - 1);
                            }
                            MaskState::DrawMaskedContent => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                            MaskState::ClearMaskStencil => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                        }
                    }
                    render_pass.set_index_buffer(
                        descriptors.quad.indices.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    for ((blend, (transform_offset, vertex_offset)), bind_group) in
                        blends.iter().zip(&offsets).zip(&binds)
                    {
                        if needs_stencil {
                            render_pass.set_pipeline(
                                self.pipelines.complex_blends[blend.blend_mode]
                                    .pipeline_for(mask_state),
                            );
                        } else {
                            render_pass.set_pipeline(
                                self.pipelines.complex_blends[blend.blend_mode]
                                    .stencilless_pipeline(),
                            );
                        }
                        render_pass.set_bind_group(
                            1,
                            &dynamic_transforms.bind_group,
                            &[*transform_offset],
                        );
                        render_pass.set_bind_group(2, bind_group, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            dynamic_transforms.vertex_buffer.slice(*vertex_offset..),
                        );
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                }
            }
        }

        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);

        target
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.rect.extent()
    }
}

/// One complex blend waiting to be composited.
struct BlendItem {
    /// The blended group's pixels: a target it had to itself, or the page it
    /// shares with its siblings.
    texture: Arc<PoolOrArcTexture>,
    /// `[u0, v0, du, dv]`: which part of `texture` is this group's.
    source_uv: [f32; 4],
    blend_mode: ComplexBlend,
    rect: TargetRect,
}

/// One render pass.
enum Pass {
    Draw {
        chunk: Vec<commands::DrawCommand>,
        needs_stencil: bool,
        transforms: BufferBuilder,
        vertices: BufferBuilder,
    },
    Shader {
        texture: Arc<PoolOrArcTexture>,
        shader: ruffle_render::pixel_bender::PixelBenderShaderHandle,
        needs_stencil: bool,
        rect: TargetRect,
    },
    /// Complex blends composited together.
    Blends {
        blends: Vec<BlendItem>,
        needs_stencil: bool,
        /// Whether these read the nearest layer above them - an `Alpha` or an
        /// `Erase` - rather than the target they are drawn onto.
        reads_layer: bool,
    },
}

/// Whether two blends can be composited in the same pass.
///
/// A complex blend reads the destination underneath itself and writes over it,
/// so two of them that cover the same pixels have to be ordered: the second has
/// to see what the first wrote. Two that cover no pixel in common cannot see
/// each other's work at all, so drawing them together off one snapshot of the
/// destination gives exactly the picture drawing them one at a time gives.
///
/// A pixel of slack on each side, because a blend samples the destination
/// through the same rectangle it covers and the fragments at its edge can round
/// into the neighbouring texel - which is why [`sample_region`] copies a pixel
/// more than the blend covers.
fn can_share_a_pass(a: TargetRect, b: TargetRect) -> bool {
    a.right() + 1 <= b.x - 1
        || b.right() + 1 <= a.x - 1
        || a.bottom() + 1 <= b.y - 1
        || b.bottom() + 1 <= a.y - 1
}

/// The most blends one pass can hold, which is what a chunk's buffers can
/// describe: an aligned slot each in the uniform buffer, and a quad each in the
/// vertex buffer.
fn max_batch(descriptors: &Descriptors, dynamic_transforms: &DynamicTransforms) -> usize {
    if !crate::tuning::blend_batching_enabled() {
        return 1;
    }
    let align = descriptors.limits.min_uniform_buffer_offset_alignment as u64;
    let transform = std::mem::size_of::<Transforms>() as u64;
    let stride = if align > 0 {
        transform.div_ceil(align) * align
    } else {
        transform
    };
    let by_transforms = dynamic_transforms.buffer.size() / stride.max(1);
    let by_vertices =
        dynamic_transforms.vertex_buffer.size() / (4 * std::mem::size_of::<PosUvVertex>() as u64);
    by_transforms.min(by_vertices).max(1) as usize
}

/// Groups the chunks into render passes, putting consecutive complex blends
/// that cannot see each other's work into one.
///
/// Order is never changed. A blend joins the batch being built only if it
/// shares no pixel with anything already in it, so within a batch no blend
/// depends on another, and batches run in the order their blends were given.
fn batch_passes(chunks: Vec<Chunk>, max_batch: usize, nearest_layer: LayerRef) -> Vec<Pass> {
    let mut passes: Vec<Pass> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match chunk {
            Chunk::Draw {
                chunk,
                needs_stencil,
                transforms,
                vertices,
            } => passes.push(Pass::Draw {
                chunk,
                needs_stencil,
                transforms,
                vertices,
            }),
            Chunk::Blend {
                texture,
                source_uv: _,
                blend_mode: ChunkBlendMode::Shader(shader),
                needs_stencil,
                rect,
            } => passes.push(Pass::Shader {
                texture,
                shader,
                needs_stencil,
                rect,
            }),
            Chunk::Blend {
                texture,
                source_uv,
                blend_mode: ChunkBlendMode::Complex(blend_mode),
                needs_stencil,
                rect,
            } => {
                let reads_layer = matches!(blend_mode, ComplexBlend::Alpha | ComplexBlend::Erase);
                // An Alpha or Erase with no Layer above it should be ignored.
                if reads_layer && matches!(nearest_layer, LayerRef::None) {
                    continue;
                }
                let item = BlendItem {
                    texture,
                    source_uv,
                    blend_mode,
                    rect,
                };
                if let Some(Pass::Blends {
                    blends,
                    needs_stencil: batch_stencil,
                    reads_layer: batch_layer,
                }) = passes.last_mut()
                    && *batch_stencil == needs_stencil
                    && *batch_layer == reads_layer
                    && blends.len() < max_batch
                    && blends
                        .iter()
                        .all(|other| can_share_a_pass(other.rect, item.rect))
                {
                    blends.push(item);
                    continue;
                }
                passes.push(Pass::Blends {
                    blends: vec![item],
                    needs_stencil,
                    reads_layer,
                });
            }
        }
    }
    passes
}

/// The four corners of the quad a complex blend draws.
///
/// A vertex's position is its place in the blended group's own target, which is
/// what the blend shader samples as its source, and its `uv` is where that
/// corner lands in the destination the blend reads underneath itself. The two
/// were the same when every target was the size of the whole surface; they are
/// not once a target covers only its own contents.
fn blend_quad(rect: TargetRect, parent: TargetRect) -> [PosUvVertex; 4] {
    let u = |x: i32| (x - parent.x) as f32 / parent.width.max(1) as f32;
    let v = |y: i32| (y - parent.y) as f32 / parent.height.max(1) as f32;
    let (u0, u1) = (u(rect.x), u(rect.right()));
    let (v0, v1) = (v(rect.y), v(rect.bottom()));
    [
        PosUvVertex::new(0.0, 0.0, u0, v0, 1.0),
        PosUvVertex::new(1.0, 0.0, u1, v0, 1.0),
        PosUvVertex::new(1.0, 1.0, u1, v1, 1.0),
        PosUvVertex::new(0.0, 1.0, u0, v1, 1.0),
    ]
}

/// Which of `parent`'s pixels a blend covering `rect` reads, in `parent`'s own
/// coordinates.
///
/// Grown by a pixel on every side: the fragments at the quad's edge sample the
/// destination at coordinates that floating-point rounding can put in the
/// neighbouring texel.
fn sample_region(rect: TargetRect, parent: TargetRect) -> TargetRect {
    TargetRect {
        x: rect.x - parent.x - 1,
        y: rect.y - parent.y - 1,
        width: rect.width + 2,
        height: rect.height + 2,
    }
}
