// Remove this when we decide on how to handle multithreaded rendering (especially on wasm)
#![allow(clippy::arc_with_non_send_sync)]

use crate::backend::ActiveFrame;
use crate::bitmaps::BitmapSamplers;
use crate::buffer_pool::{BufferPool, PoolEntry};
use crate::mesh::BitmapBinds;
use crate::pipelines::Pipelines;
use crate::target::{RenderTarget, SwapChainTarget};
use crate::utils::{
    BufferDimensions, capture_image, create_buffer_with_data, format_list, get_backend_names,
};
use bytemuck::{Pod, Zeroable};
use descriptors::Descriptors;
use enum_map::Enum;
use ruffle_render::backend::RawTexture;
use ruffle_render::bitmap::{BitmapHandle, BitmapHandleImpl, PixelRegion, SyncHandle};
use ruffle_render::shape_utils::GradientType;
use ruffle_render::tessellator::{Gradient as TessGradient, Vertex as TessVertex};
use std::any::Any;
use std::cell::{Cell, OnceCell};
use std::sync::Arc;
use swf::GradientSpread;
pub use wgpu;
pub use wgpu_profiler;

type Error = Box<dyn std::error::Error>;

#[macro_use]
pub mod utils;

mod bind_cache;
mod bitmaps;
mod bounds;
pub mod cache_pool;
mod context3d;
mod globals;
mod pipelines;
mod pixel_bender;
pub mod target;

pub mod backend;
mod blend;
mod buffer_builder;
pub mod buffer_pool;
#[cfg(feature = "clap")]
pub mod clap;
pub mod descriptors;
mod dynamic_transforms;
mod filters;
mod layouts;
mod mesh;
mod shaders;
mod surface;

impl BitmapHandleImpl for Texture {}

pub fn as_texture(handle: &BitmapHandle) -> &Texture {
    <dyn Any>::downcast_ref(&*handle.0).unwrap()
}

pub fn raw_texture_as_texture(handle: &dyn RawTexture) -> &wgpu::Texture {
    <dyn Any>::downcast_ref(handle).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum MaskState {
    NoMask,
    DrawMaskStencil,
    DrawMaskedContent,
    ClearMaskStencil,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Transforms {
    world_matrix: [[f32; 4]; 4],
    mult_color: [f32; 4],
    add_color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosVertex {
    position: [f32; 2],
}

impl From<TessVertex> for PosVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosUvVertex {
    position: [f32; 2],
    uv: [f32; 3],
}

impl PosUvVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, t: f32) -> Self {
        let position = [x, y];
        let uv = [u, v, t];
        Self { position, uv }
    }

    pub fn from_tessellator(vertex: TessVertex, texture_matrix: &[[f32; 3]; 3]) -> Self {
        let position = [vertex.x, vertex.y];
        let uv = Self::transform_uv(texture_matrix, vertex.x, vertex.y);
        Self { position, uv }
    }

    fn transform_uv(matrix: &[[f32; 3]; 3], x: f32, y: f32) -> [f32; 3] {
        [
            matrix[0][0] * x + matrix[1][0] * y + matrix[2][0],
            matrix[0][1] * x + matrix[1][1] * y + matrix[2][1],
            1.0,
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PosColorVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl From<TessVertex> for PosColorVertex {
    fn from(vertex: TessVertex) -> Self {
        Self {
            position: [vertex.x, vertex.y],
            color: [
                f32::from(vertex.color.r) / 255.0,
                f32::from(vertex.color.g) / 255.0,
                f32::from(vertex.color.b) / 255.0,
                f32::from(vertex.color.a) / 255.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GradientUniforms {
    focal_point: f32,
    interpolation: i32,
    shape: i32,
    repeat: i32,
}

impl From<TessGradient> for GradientUniforms {
    fn from(gradient: TessGradient) -> Self {
        Self {
            focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
            interpolation: (gradient.interpolation == swf::GradientInterpolation::LinearRgb) as i32,
            shape: match gradient.gradient_type {
                GradientType::Linear => 1,
                GradientType::Radial => 2,
                GradientType::Focal => 3,
            },
            repeat: match gradient.repeat_mode {
                GradientSpread::Pad => 1,
                GradientSpread::Reflect => 2,
                GradientSpread::Repeat => 3,
            },
        }
    }
}

#[derive(Debug)]
pub enum QueueSyncHandle {
    AlreadyCopied {
        index: Option<wgpu::SubmissionIndex>,
        buffer: PoolEntry<wgpu::Buffer, BufferDimensions>,
        copy_dimensions: BufferDimensions,
        descriptors: Arc<Descriptors>,
    },
    NotCopied {
        handle: BitmapHandle,
        copy_area: PixelRegion,
        descriptors: Arc<Descriptors>,
        pool: Arc<BufferPool<wgpu::Buffer, BufferDimensions>>,
    },
}

impl SyncHandle for QueueSyncHandle {}

impl QueueSyncHandle {
    pub fn capture<R, F: FnOnce(&[u8], u32) -> R>(
        self,
        with_rgba: F,
        frame: &mut ActiveFrame,
    ) -> R {
        match self {
            QueueSyncHandle::AlreadyCopied {
                index,
                buffer,
                copy_dimensions,
                descriptors,
            } => capture_image(
                &descriptors.device,
                &buffer,
                &copy_dimensions,
                index,
                with_rgba,
            ),
            QueueSyncHandle::NotCopied {
                handle,
                copy_area,
                descriptors,
                pool,
            } => {
                let texture = as_texture(&handle);

                let buffer_dimensions = BufferDimensions::new(
                    copy_area.width() as usize,
                    copy_area.height() as usize,
                    texture.texture.format(),
                );

                let buffer = pool.take(&descriptors, buffer_dimensions.clone());
                frame.command_encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: copy_area.x_min,
                            y: copy_area.y_min,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(buffer_dimensions.padded_bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: copy_area.width(),
                        height: copy_area.height(),
                        depth_or_array_layers: 1,
                    },
                );
                let index = frame.submit_direct(&descriptors);

                let image = capture_image(
                    &descriptors.device,
                    &buffer,
                    &buffer_dimensions,
                    Some(index),
                    with_rgba,
                );

                // After we've read pixels from a texture enough times, we'll store this buffer so that
                // future reads will be faster (it'll copy as part of the draw process instead)
                texture
                    .copy_count
                    .set(texture.copy_count.get().saturating_add(1));

                image
            }
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    pub(crate) texture: wgpu::Texture,
    /// What this texture is for, for the memory report only.
    pub(crate) kind: TextureKind,
    pub(crate) repeating_linear: OnceCell<BitmapBinds>,
    pub(crate) repeating_nearest: OnceCell<BitmapBinds>,
    pub(crate) clamped_linear: OnceCell<BitmapBinds>,
    pub(crate) clamped_nearest: OnceCell<BitmapBinds>,
    copy_count: Cell<u8>,
    /// The pool to give this texture back to when its owner lets it go.
    ///
    /// Only `cacheAsBitmap` textures have one: they are renderer-owned,
    /// disposable, and fully cleared before every redraw. A bitmap registered
    /// from pixels is not recycled, because its content is its identity.
    pub(crate) cache_pool: Option<Arc<std::sync::Mutex<crate::cache_pool::CacheTexturePool>>>,
}

/// Measurement-only accounting of every texture Ruffle asks the GPU for.
///
/// The point of splitting this by [`TextureKind`] is that the kinds have very
/// different lifetimes, and the question this build exists to answer is which
/// of them the process is still holding when the working set stays high:
///
/// * `Bitmap` and `CacheAsBitmap` are owned by content, and fall when the
///   content does. Memory retained here would be a live leak.
/// * `PoolMain` and `PoolOffscreen` are scratch render targets. Memory
///   retained here is the renderer's own pooling, not content.
/// * `Temporary` is a one-off render output that should never accumulate.
///
/// None of these counters change what is allocated, when it is freed, or how
/// long anything lives; they only observe.
mod texture_stats {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    /// What a tracked texture is for.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum TextureKind {
        /// A decoded image registered by the player: owned by a movie library.
        Bitmap = 0,
        /// The backing store of a `cacheAsBitmap` or filtered display object.
        CacheAsBitmap = 1,
        /// A one-off render output, such as a Pixel Bender result.
        Temporary = 2,
        /// A render target from the surface pool, which lives across frames.
        PoolMain = 3,
        /// A render target from the offscreen pool, which the renderer
        /// replaces every frame.
        PoolOffscreen = 4,
    }

    pub(crate) const KINDS: usize = 5;

    pub(crate) const KIND_NAMES: [&str; KINDS] = [
        "bitmap",
        "cache_as_bitmap",
        "temporary",
        "pool_main",
        "pool_offscreen",
    ];

    static LIVE_COUNT: [AtomicUsize; KINDS] = [const { AtomicUsize::new(0) }; KINDS];
    static LIVE_BYTES: [AtomicUsize; KINDS] = [const { AtomicUsize::new(0) }; KINDS];
    static CREATED_COUNT: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    static CREATED_BYTES: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    static DROPPED_COUNT: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    static DROPPED_BYTES: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

    /// The most texture memory Ruffle has held at once, over the whole run.
    /// Compared against the process' working set, this is what separates a
    /// high-water mark from memory that is still live.
    static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

    /// Times a pool handed back an existing target instead of building one.
    static POOL_REUSES: AtomicU64 = AtomicU64::new(0);
    /// Times a pool had nothing to hand back and constructed a new target.
    static POOL_MISSES: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn record_created(kind: TextureKind, bytes: usize) {
        let i = kind as usize;
        LIVE_BYTES[i].fetch_add(bytes, Ordering::Relaxed);
        LIVE_COUNT[i].fetch_add(1, Ordering::Relaxed);
        CREATED_BYTES[i].fetch_add(bytes as u64, Ordering::Relaxed);
        CREATED_COUNT[i].fetch_add(1, Ordering::Relaxed);

        let live: usize = LIVE_BYTES.iter().map(|b| b.load(Ordering::Relaxed)).sum();
        PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
    }

    pub(crate) fn record_dropped(kind: TextureKind, bytes: usize) {
        let i = kind as usize;
        LIVE_BYTES[i].fetch_sub(bytes, Ordering::Relaxed);
        LIVE_COUNT[i].fetch_sub(1, Ordering::Relaxed);
        DROPPED_BYTES[i].fetch_add(bytes as u64, Ordering::Relaxed);
        DROPPED_COUNT[i].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_pool_reuse() {
        POOL_REUSES.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_pool_miss() {
        POOL_MISSES.fetch_add(1, Ordering::Relaxed);
    }

    /// A snapshot of the texture counters, per kind and in total.
    #[derive(Clone, Debug, Default)]
    pub struct TextureStats {
        pub live_count: [usize; KINDS],
        pub live_bytes: [usize; KINDS],
        pub created_count: [u64; KINDS],
        pub created_bytes: [u64; KINDS],
        pub dropped_count: [u64; KINDS],
        pub dropped_bytes: [u64; KINDS],
        pub peak_live_bytes: usize,
        pub pool_reuses: u64,
        pub pool_misses: u64,
    }

    impl TextureStats {
        pub fn total_live_bytes(&self) -> usize {
            self.live_bytes.iter().sum()
        }

        pub fn total_live_count(&self) -> usize {
            self.live_count.iter().sum()
        }
    }

    pub fn texture_stats() -> TextureStats {
        let load_usize =
            |a: &[AtomicUsize; KINDS]| std::array::from_fn(|i| a[i].load(Ordering::Relaxed));
        let load_u64 =
            |a: &[AtomicU64; KINDS]| std::array::from_fn(|i| a[i].load(Ordering::Relaxed));
        TextureStats {
            live_count: load_usize(&LIVE_COUNT),
            live_bytes: load_usize(&LIVE_BYTES),
            created_count: load_u64(&CREATED_COUNT),
            created_bytes: load_u64(&CREATED_BYTES),
            dropped_count: load_u64(&DROPPED_COUNT),
            dropped_bytes: load_u64(&DROPPED_BYTES),
            peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
            pool_reuses: POOL_REUSES.load(Ordering::Relaxed),
            pool_misses: POOL_MISSES.load(Ordering::Relaxed),
        }
    }
}

pub(crate) use texture_stats::{KIND_NAMES, TextureKind};
pub use texture_stats::{TextureStats, texture_stats};

/// What a frame costs the renderer in work rather than in memory.
///
/// The blend-target sizing fix removed the bandwidth half of a crowded room's
/// cost; what is left is a fixed price per target - a render pass, a bind
/// group, a pool take and return - which is proportional to how *many* targets
/// a frame wants, not how big they are. These counters are what let that price
/// be measured and attributed.
pub mod render_stats {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    static RENDER_PASSES: AtomicU64 = AtomicU64::new(0);
    static BIND_GROUPS_CREATED: AtomicU64 = AtomicU64::new(0);
    static BIND_GROUP_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
    static BIND_GROUP_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
    static BLEND_TARGETS_LIVE: AtomicUsize = AtomicUsize::new(0);
    static BLEND_TARGET_BYTES: AtomicUsize = AtomicUsize::new(0);
    static PEAK_BLEND_TARGETS: AtomicUsize = AtomicUsize::new(0);
    static PEAK_BLEND_TARGET_BYTES: AtomicUsize = AtomicUsize::new(0);
    static FASTPATH_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
    static FASTPATH_USED: AtomicU64 = AtomicU64::new(0);
    static MULTIPLY_ON_DRAW_USED: AtomicU64 = AtomicU64::new(0);
    static MULTIPLY_ON_DRAW_SHAPE: AtomicU64 = AtomicU64::new(0);
    static MULTIPLY_ON_DRAW_TRANSPARENT: AtomicU64 = AtomicU64::new(0);

    /// Why a blend could not take the direct path. Kept as counters rather
    /// than log lines: a crowded frame asks the question hundreds of times.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum FallbackReason {
        MultipleDraws,
        Filtered,
        Masked,
        NestedBlend,
        ComplexBlend,
        /// A trivial blend whose state does not survive being applied per
        /// multisample rather than to the resolved group.
        UnsupportedBlendMode,
        UnsupportedCommand,
        RequiresIntermediate,
        Other,
    }

    pub const FALLBACK_NAMES: &[&str] = &[
        "multiple_draws",
        "filtered",
        "masked",
        "nested_blend",
        "complex_blend",
        "unsupported_blend_mode",
        "unsupported_command",
        "requires_intermediate",
        "other",
    ];

    static FALLBACKS: [AtomicU64; 9] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    pub(crate) fn record_render_pass() {
        RENDER_PASSES.fetch_add(1, Ordering::Relaxed);
        FRAME_RENDER_PASSES.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_bind_group_created() {
        BIND_GROUPS_CREATED.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_bind_group_cache(hit: bool) {
        if hit {
            BIND_GROUP_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        } else {
            BIND_GROUP_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
        }
    }

    static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    static CACHE_ENTRIES_NS: AtomicU64 = AtomicU64::new(0);
    static FRAME_COMMANDS_NS: AtomicU64 = AtomicU64::new(0);
    static QUEUE_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    static SLOW_FRAMES: AtomicU64 = AtomicU64::new(0);
    static VERY_SLOW_FRAMES: AtomicU64 = AtomicU64::new(0);
    static SLOW_CACHE_ENTRIES_NS: AtomicU64 = AtomicU64::new(0);
    static SLOW_FRAME_COMMANDS_NS: AtomicU64 = AtomicU64::new(0);
    static SLOW_QUEUE_SUBMIT_NS: AtomicU64 = AtomicU64::new(0);

    /// One frame at AdventureQuest Worlds' 24 frames a second.
    const FRAME_BUDGET_NS: u64 = 41_670_000;
    const VERY_SLOW_NS: u64 = 100_000_000;

    /// Where the renderer's share of a frame went.
    ///
    /// The renderer can only account for its own half; the rest of a frame is
    /// ActionScript, garbage collection and walking the display list. Reading
    /// `render_ns_total` against the frame time the frontend measures is what
    /// splits the two, and the `slow_` totals say where the renderer's time
    /// went in the frames that missed the budget - which are the only ones
    /// worth optimising.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct FrameTiming {
        pub total_ns: u64,
        pub cache_entries_ns: u64,
        pub frame_commands_ns: u64,
        pub queue_submit_ns: u64,
        pub slow_frames: u64,
        pub very_slow_frames: u64,
        pub slow_cache_entries_ns: u64,
        pub slow_frame_commands_ns: u64,
        pub slow_queue_submit_ns: u64,
    }

    /// Records what one frame's phases cost. Called once per frame from the
    /// backend, so four clock reads a frame.
    pub(crate) fn record_frame_timing(cache_entries: u64, frame_commands: u64, queue_submit: u64) {
        let total = cache_entries + frame_commands + queue_submit;
        SUBMIT_NS.fetch_add(total, Ordering::Relaxed);
        CACHE_ENTRIES_NS.fetch_add(cache_entries, Ordering::Relaxed);
        FRAME_COMMANDS_NS.fetch_add(frame_commands, Ordering::Relaxed);
        QUEUE_SUBMIT_NS.fetch_add(queue_submit, Ordering::Relaxed);
        if total > FRAME_BUDGET_NS {
            SLOW_FRAMES.fetch_add(1, Ordering::Relaxed);
            SLOW_CACHE_ENTRIES_NS.fetch_add(cache_entries, Ordering::Relaxed);
            SLOW_FRAME_COMMANDS_NS.fetch_add(frame_commands, Ordering::Relaxed);
            SLOW_QUEUE_SUBMIT_NS.fetch_add(queue_submit, Ordering::Relaxed);
        }
        if total > VERY_SLOW_NS {
            VERY_SLOW_FRAMES.fetch_add(1, Ordering::Relaxed);
        }
    }

    static FRAME_BLEND_TARGETS: AtomicUsize = AtomicUsize::new(0);
    static FRAME_BLEND_TARGET_BYTES: AtomicUsize = AtomicUsize::new(0);
    static FRAME_RENDER_PASSES: AtomicU64 = AtomicU64::new(0);
    static LAST_FRAME_RENDER_PASSES: AtomicU64 = AtomicU64::new(0);

    /// A blend has taken a render target of `bytes`.
    ///
    /// Counted per frame rather than as a live gauge: every target a frame
    /// takes is held until the chunk that composites it is encoded, so the
    /// frame's total *is* what was live at once.
    pub(crate) fn blend_target_taken(bytes: usize) {
        FRAME_BLEND_TARGETS.fetch_add(1, Ordering::Relaxed);
        FRAME_BLEND_TARGET_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Closes a frame: this frame's counts become the reported ones.
    pub(crate) fn end_frame() {
        let targets = FRAME_BLEND_TARGETS.swap(0, Ordering::Relaxed);
        let bytes = FRAME_BLEND_TARGET_BYTES.swap(0, Ordering::Relaxed);
        let passes = FRAME_RENDER_PASSES.swap(0, Ordering::Relaxed);
        BLEND_TARGETS_LIVE.store(targets, Ordering::Relaxed);
        BLEND_TARGET_BYTES.store(bytes, Ordering::Relaxed);
        LAST_FRAME_RENDER_PASSES.store(passes, Ordering::Relaxed);
        PEAK_BLEND_TARGETS.fetch_max(targets, Ordering::Relaxed);
        PEAK_BLEND_TARGET_BYTES.fetch_max(bytes, Ordering::Relaxed);

        let pages = FRAME_PAGES.swap(0, Ordering::Relaxed);
        let page_bytes = FRAME_PAGE_BYTES.swap(0, Ordering::Relaxed);
        PAGES_LAST_FRAME.store(pages, Ordering::Relaxed);
        PAGE_BYTES_LAST_FRAME.store(page_bytes, Ordering::Relaxed);
        PEAK_PAGE_BYTES.fetch_max(page_bytes, Ordering::Relaxed);
        DESTINATION_COPIES_LAST_FRAME.store(
            FRAME_DESTINATION_COPIES.swap(0, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        DESTINATION_COPY_PIXELS_LAST_FRAME.store(
            FRAME_DESTINATION_COPY_PIXELS.swap(0, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// Why a blended group could not share a page with its siblings.
    ///
    /// Separate from [`FallbackReason`], which is about the direct path: a
    /// group that cannot be drawn straight onto its destination may still be
    /// perfectly able to share a page, and these say which of the two it
    /// missed.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum PageFallback {
        /// A `PixelBender` blend, which is arbitrary code over its whole quad.
        Shader,
        /// The group draws another blended group of its own.
        NestedBlend,
        /// The group contains an alpha mask, which needs targets of its own.
        AlphaMask,
        /// The group pushes a stencil mask, which a shared pass has no
        /// per-region state for.
        Masked,
        /// Stage3D, which is drawn through its own pipelines.
        Stage3D,
        /// Bigger than a page region is allowed to be.
        Size,
        /// The group's draws do not fit one chunk's uniform or vertex buffer.
        Capacity,
        /// No page could be opened for it.
        NoPage,
    }

    pub const PAGE_FALLBACK_NAMES: &[&str] = &[
        "shader",
        "nested_blend",
        "alpha_mask",
        "masked",
        "stage3d",
        "size",
        "capacity",
        "no_page",
    ];

    static PAGE_FALLBACKS: [AtomicU64; 8] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    static BATCH_ELIGIBLE: AtomicU64 = AtomicU64::new(0);
    static BATCH_USED: AtomicU64 = AtomicU64::new(0);
    static FRAME_PAGES: AtomicUsize = AtomicUsize::new(0);
    static FRAME_PAGE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static PAGES_LAST_FRAME: AtomicUsize = AtomicUsize::new(0);
    static PAGE_BYTES_LAST_FRAME: AtomicUsize = AtomicUsize::new(0);
    static PEAK_PAGE_BYTES: AtomicUsize = AtomicUsize::new(0);

    /// A blended group was offered a page, and either took a region on one or
    /// did not.
    pub(crate) fn record_batch(used: bool, reason: Option<PageFallback>) {
        BATCH_ELIGIBLE.fetch_add(1, Ordering::Relaxed);
        if used {
            BATCH_USED.fetch_add(1, Ordering::Relaxed);
        } else if let Some(reason) = reason {
            PAGE_FALLBACKS[reason as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A page of `bytes` has been taken for this frame.
    pub(crate) fn page_taken(bytes: usize) {
        FRAME_PAGES.fetch_add(1, Ordering::Relaxed);
        FRAME_PAGE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    static DESTINATION_COPIES: AtomicU64 = AtomicU64::new(0);
    static DESTINATION_COPY_PIXELS: AtomicU64 = AtomicU64::new(0);
    static FRAME_DESTINATION_COPIES: AtomicU64 = AtomicU64::new(0);
    static FRAME_DESTINATION_COPY_PIXELS: AtomicU64 = AtomicU64::new(0);
    static DESTINATION_COPIES_LAST_FRAME: AtomicU64 = AtomicU64::new(0);
    static DESTINATION_COPY_PIXELS_LAST_FRAME: AtomicU64 = AtomicU64::new(0);
    static COMPLEX_BLENDS: AtomicU64 = AtomicU64::new(0);
    static COMPLEX_BLEND_PASSES: AtomicU64 = AtomicU64::new(0);

    /// A complex blend has taken a snapshot of the destination it reads.
    pub(crate) fn record_destination_copy(pixels: u64) {
        DESTINATION_COPIES.fetch_add(1, Ordering::Relaxed);
        DESTINATION_COPY_PIXELS.fetch_add(pixels, Ordering::Relaxed);
        FRAME_DESTINATION_COPIES.fetch_add(1, Ordering::Relaxed);
        FRAME_DESTINATION_COPY_PIXELS.fetch_add(pixels, Ordering::Relaxed);
    }

    /// `blends` complex blends were composited in one render pass.
    pub(crate) fn record_complex_batch(blends: u64) {
        COMPLEX_BLENDS.fetch_add(blends, Ordering::Relaxed);
        COMPLEX_BLEND_PASSES.fetch_add(1, Ordering::Relaxed);
    }

    /// What became of a multiply that could have been carried by its own draw.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum MultiplyOnDraw {
        /// Carried: no target, no pass, no composite.
        Used,
        /// The destination was opaque and the group was a single draw, but that
        /// draw was a shape.
        ///
        /// Two things stand in the way, and this counter is what says whether
        /// clearing them is worth it. The shape pipelines are only built with
        /// premultiplied-alpha blending, so there is no multiply variant to
        /// select; and a shape is a mesh of several draws, which a target
        /// composites into one picture before the blend applies. Carrying the
        /// blend on each draw instead would multiply the destination once per
        /// draw wherever two of them overlap, so a mesh may only take this path
        /// if it is a single draw.
        SoleShape,
        /// The destination was not known to be opaque, so the algebra does not
        /// hold and the shader has to run.
        TransparentDestination,
    }

    pub(crate) fn record_multiply_on_draw(outcome: MultiplyOnDraw) {
        match outcome {
            MultiplyOnDraw::Used => &MULTIPLY_ON_DRAW_USED,
            MultiplyOnDraw::SoleShape => &MULTIPLY_ON_DRAW_SHAPE,
            MultiplyOnDraw::TransparentDestination => &MULTIPLY_ON_DRAW_TRANSPARENT,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_fastpath(used: bool, reason: Option<FallbackReason>) {
        FASTPATH_ELIGIBLE.fetch_add(1, Ordering::Relaxed);
        if used {
            FASTPATH_USED.fetch_add(1, Ordering::Relaxed);
        } else if let Some(reason) = reason {
            FALLBACKS[reason as usize].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// What the renderer has done so far. Differences between two readings
    /// give the cost of the span between them.
    #[derive(Clone, Debug, Default)]
    pub struct RenderStats {
        pub render_passes: u64,
        pub bind_groups_created: u64,
        pub bind_group_cache_hits: u64,
        pub bind_group_cache_misses: u64,
        pub blend_targets_live: usize,
        pub blend_target_bytes: usize,
        pub peak_blend_targets: usize,
        pub peak_blend_target_bytes: usize,
        pub fastpath_eligible: u64,
        pub fastpath_used: u64,
        pub fallbacks: Vec<u64>,
        /// Multiplies carried by their own draw, and the two reasons the rest
        /// were not. `used` is a subset of `fastpath_used`.
        pub multiply_on_draw_used: u64,
        pub multiply_on_draw_shape: u64,
        pub multiply_on_draw_transparent: u64,
        /// Render passes encoded for the most recent frame.
        pub render_passes_last_frame: u64,
        /// Blended groups offered a shared page, and the ones that took a
        /// region on one.
        pub batch_eligible: u64,
        pub batch_used: u64,
        /// Why the rest did not, indexed by [`PageFallback`].
        pub page_fallbacks: Vec<u64>,
        /// Pages taken for the most recent frame, and what they cost.
        pub pages_last_frame: usize,
        pub page_bytes_last_frame: usize,
        pub peak_page_bytes: usize,
        /// Snapshots complex blends took of the destination they read.
        pub destination_copies: u64,
        pub destination_copy_pixels: u64,
        pub destination_copies_last_frame: u64,
        pub destination_copy_pixels_last_frame: u64,
        /// Complex blends composited, and the render passes that took them.
        pub complex_blends: u64,
        pub complex_blend_passes: u64,
        /// Where the renderer's share of the frames went.
        pub timing: FrameTiming,
    }

    pub fn render_stats() -> RenderStats {
        RenderStats {
            render_passes: RENDER_PASSES.load(Ordering::Relaxed),
            bind_groups_created: BIND_GROUPS_CREATED.load(Ordering::Relaxed),
            bind_group_cache_hits: BIND_GROUP_CACHE_HITS.load(Ordering::Relaxed),
            bind_group_cache_misses: BIND_GROUP_CACHE_MISSES.load(Ordering::Relaxed),
            blend_targets_live: BLEND_TARGETS_LIVE.load(Ordering::Relaxed),
            blend_target_bytes: BLEND_TARGET_BYTES.load(Ordering::Relaxed),
            peak_blend_targets: PEAK_BLEND_TARGETS.load(Ordering::Relaxed),
            peak_blend_target_bytes: PEAK_BLEND_TARGET_BYTES.load(Ordering::Relaxed),
            fastpath_eligible: FASTPATH_ELIGIBLE.load(Ordering::Relaxed),
            fastpath_used: FASTPATH_USED.load(Ordering::Relaxed),
            fallbacks: FALLBACKS
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            multiply_on_draw_used: MULTIPLY_ON_DRAW_USED.load(Ordering::Relaxed),
            multiply_on_draw_shape: MULTIPLY_ON_DRAW_SHAPE.load(Ordering::Relaxed),
            multiply_on_draw_transparent: MULTIPLY_ON_DRAW_TRANSPARENT.load(Ordering::Relaxed),
            render_passes_last_frame: LAST_FRAME_RENDER_PASSES.load(Ordering::Relaxed),
            batch_eligible: BATCH_ELIGIBLE.load(Ordering::Relaxed),
            batch_used: BATCH_USED.load(Ordering::Relaxed),
            page_fallbacks: PAGE_FALLBACKS
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            pages_last_frame: PAGES_LAST_FRAME.load(Ordering::Relaxed),
            page_bytes_last_frame: PAGE_BYTES_LAST_FRAME.load(Ordering::Relaxed),
            peak_page_bytes: PEAK_PAGE_BYTES.load(Ordering::Relaxed),
            destination_copies: DESTINATION_COPIES.load(Ordering::Relaxed),
            destination_copy_pixels: DESTINATION_COPY_PIXELS.load(Ordering::Relaxed),
            destination_copies_last_frame: DESTINATION_COPIES_LAST_FRAME.load(Ordering::Relaxed),
            destination_copy_pixels_last_frame: DESTINATION_COPY_PIXELS_LAST_FRAME
                .load(Ordering::Relaxed),
            complex_blends: COMPLEX_BLENDS.load(Ordering::Relaxed),
            complex_blend_passes: COMPLEX_BLEND_PASSES.load(Ordering::Relaxed),
            timing: FrameTiming {
                total_ns: SUBMIT_NS.load(Ordering::Relaxed),
                cache_entries_ns: CACHE_ENTRIES_NS.load(Ordering::Relaxed),
                frame_commands_ns: FRAME_COMMANDS_NS.load(Ordering::Relaxed),
                queue_submit_ns: QUEUE_SUBMIT_NS.load(Ordering::Relaxed),
                slow_frames: SLOW_FRAMES.load(Ordering::Relaxed),
                very_slow_frames: VERY_SLOW_FRAMES.load(Ordering::Relaxed),
                slow_cache_entries_ns: SLOW_CACHE_ENTRIES_NS.load(Ordering::Relaxed),
                slow_frame_commands_ns: SLOW_FRAME_COMMANDS_NS.load(Ordering::Relaxed),
                slow_queue_submit_ns: SLOW_QUEUE_SUBMIT_NS.load(Ordering::Relaxed),
            },
        }
    }
}

pub use render_stats::{FallbackReason, FrameTiming, PageFallback, RenderStats, render_stats};

/// Switches for the two ways a frame's blended groups are batched.
///
/// Both are on. They are here so that the same scene can be rendered with and
/// without them and the two pictures compared pixel for pixel, which is how the
/// batching is tested; and so that, if a driver in the field ever disagrees,
/// there is a way to render the old way without a new build.
pub mod tuning {
    use std::sync::atomic::{AtomicBool, Ordering};

    static BLEND_PAGES: AtomicBool = AtomicBool::new(true);
    static BLEND_BATCHING: AtomicBool = AtomicBool::new(true);
    static CACHE_POOL: AtomicBool = AtomicBool::new(true);
    static MULTIPLY_ON_DRAW: AtomicBool = AtomicBool::new(true);
    static FRUGAL_DEVICE_MEMORY: AtomicBool = AtomicBool::new(true);

    /// Whether blended groups share pages instead of each taking a target.
    pub fn blend_pages_enabled() -> bool {
        BLEND_PAGES.load(Ordering::Relaxed)
    }

    pub fn set_blend_pages_enabled(enabled: bool) {
        BLEND_PAGES.store(enabled, Ordering::Relaxed);
    }

    /// Whether complex blends that cannot see each other's work are composited
    /// in one render pass.
    pub fn blend_batching_enabled() -> bool {
        BLEND_BATCHING.load(Ordering::Relaxed)
    }

    pub fn set_blend_batching_enabled(enabled: bool) {
        BLEND_BATCHING.store(enabled, Ordering::Relaxed);
    }

    /// Whether a multiply over an opaque destination is carried by the draw
    /// that produced it instead of taking a target, a render pass and a
    /// composite of its own. Off, every multiply goes through the shader, which
    /// is exactly the phase 2 behaviour.
    pub fn multiply_on_draw_enabled() -> bool {
        MULTIPLY_ON_DRAW.load(Ordering::Relaxed)
    }

    pub fn set_multiply_on_draw_enabled(enabled: bool) {
        MULTIPLY_ON_DRAW.store(enabled, Ordering::Relaxed);
    }

    /// Whether the graphics allocator is asked for small memory blocks rather
    /// than the large ones wgpu defaults to.
    ///
    /// `gpu_allocator` destroys a block only when the *whole* block is empty
    /// (`MemoryType::free`), so one surviving allocation pins the block it sits
    /// in. wgpu's default `MemoryHints::Performance` asks for device blocks of
    /// 128 to 256 MB and host blocks of 64 to 128 MB, which makes that unit of
    /// waste very large: the client's 40-minute session ended holding 1,472 MB
    /// of reserve across 7 blocks for 319 MB of live allocations.
    ///
    /// `MemoryHints::MemoryUsage` asks for 8-64 MB device blocks and 4-32 MB
    /// host blocks instead, so the same live set pins a quarter as much. Read
    /// once, when the device is created; `RUFFLE_DEVICE_MEMORY` overrides it.
    pub fn frugal_device_memory_enabled() -> bool {
        FRUGAL_DEVICE_MEMORY.load(Ordering::Relaxed)
    }

    pub fn set_frugal_device_memory_enabled(enabled: bool) {
        FRUGAL_DEVICE_MEMORY.store(enabled, Ordering::Relaxed);
    }

    /// Whether released `cacheAsBitmap` textures are recycled instead of
    /// destroyed. Off, this is exactly the phase 1 behaviour, which is how the
    /// two are measured against each other.
    pub fn cache_pool_enabled() -> bool {
        CACHE_POOL.load(Ordering::Relaxed)
    }

    pub fn set_cache_pool_enabled(enabled: bool) {
        CACHE_POOL.store(enabled, Ordering::Relaxed);
    }
}

pub(crate) fn texture_stats_record_pool_reuse() {
    texture_stats::record_pool_reuse();
}

pub(crate) fn texture_stats_record_pool_miss() {
    texture_stats::record_pool_miss();
}

/// Approximate memory of a texture: its pixels at the format's block size.
pub(crate) fn texture_bytes(texture: &wgpu::Texture) -> usize {
    let size = texture.size();
    let bytes_per_block = texture.format().block_copy_size(None).unwrap_or(4) as usize;
    size.width as usize
        * size.height as usize
        * size.depth_or_array_layers as usize
        * bytes_per_block
}

pub(crate) fn track_texture_created(texture: &wgpu::Texture, kind: TextureKind) {
    texture_stats::record_created(kind, texture_bytes(texture));
}

pub(crate) fn track_texture_dropped(texture: &wgpu::Texture, kind: TextureKind) {
    texture_stats::record_dropped(kind, texture_bytes(texture));
}

/// `(textures alive, their bytes)` as tracked by Ruffle.
pub fn tracked_texture_totals() -> (usize, usize) {
    let stats = texture_stats();
    (stats.total_live_count(), stats.total_live_bytes())
}

/// `(created, dropped, bytes created)` across every kind, since the process
/// started.
///
/// Cumulative allocation traffic, not memory in use: subtract two readings to
/// get the churn of the span between them. The client's session held about
/// 112 MB of texture at the end and had allocated 207 GiB to get there, and it
/// is the second number that a driver's allocator, its fragmentation and its
/// deferred frees actually see.
pub fn texture_churn() -> (u64, u64, u64) {
    let stats = texture_stats();
    (
        stats.created_count.iter().sum(),
        stats.dropped_count.iter().sum(),
        stats.created_bytes.iter().sum(),
    )
}

impl Drop for Texture {
    fn drop(&mut self) {
        // A pooled cache texture is still allocated after this, so it stays in
        // the live accounting; the pool subtracts it when it really destroys
        // it.
        if crate::cache_pool::release(self) {
            return;
        }
        track_texture_dropped(&self.texture, self.kind);
    }
}

impl Texture {
    pub(crate) fn new(texture: wgpu::Texture, kind: TextureKind) -> Self {
        track_texture_created(&texture, kind);
        Self {
            texture,
            kind,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
            cache_pool: None,
        }
    }

    /// A cache texture that goes back to `pool` when its owner releases it.
    ///
    /// `recycled` is a texture the pool already had, with the bind groups that
    /// name it; they are still valid, since it is the same texture.
    pub(crate) fn recycled(
        recycled: crate::cache_pool::Recycled,
        pool: Arc<std::sync::Mutex<crate::cache_pool::CacheTexturePool>>,
    ) -> Self {
        Self {
            texture: recycled.texture,
            kind: TextureKind::CacheAsBitmap,
            repeating_linear: recycled.repeating_linear,
            repeating_nearest: recycled.repeating_nearest,
            clamped_linear: recycled.clamped_linear,
            clamped_nearest: recycled.clamped_nearest,
            copy_count: Cell::new(0),
            cache_pool: Some(pool),
        }
    }

    /// A freshly allocated cache texture, which will join `pool` when released.
    pub(crate) fn new_pooled(
        texture: wgpu::Texture,
        pool: Arc<std::sync::Mutex<crate::cache_pool::CacheTexturePool>>,
    ) -> Self {
        track_texture_created(&texture, TextureKind::CacheAsBitmap);
        Self {
            texture,
            kind: TextureKind::CacheAsBitmap,
            repeating_linear: Default::default(),
            repeating_nearest: Default::default(),
            clamped_linear: Default::default(),
            clamped_nearest: Default::default(),
            copy_count: Cell::new(0),
            cache_pool: Some(pool),
        }
    }

    pub fn bind_group(
        &self,
        repeating: bool,
        smoothed: bool,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        handle: BitmapHandle,
        samplers: &BitmapSamplers,
    ) -> &BitmapBinds {
        let bind = match (repeating, smoothed) {
            (true, true) => &self.repeating_linear,
            (true, false) => &self.repeating_nearest,
            (false, true) => &self.clamped_linear,
            (false, false) => &self.clamped_nearest,
        };
        bind.get_or_init(|| {
            BitmapBinds::new(
                device,
                layout,
                samplers.get_sampler(repeating, smoothed),
                self.texture.create_view(&Default::default()),
                create_debug_label!("Bitmap {:?} bind group (smoothed: {})", handle.0, smoothed),
            )
        })
    }
}
