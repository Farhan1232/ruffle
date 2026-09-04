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
    kind: TextureKind,
    repeating_linear: OnceCell<BitmapBinds>,
    repeating_nearest: OnceCell<BitmapBinds>,
    clamped_linear: OnceCell<BitmapBinds>,
    clamped_nearest: OnceCell<BitmapBinds>,
    copy_count: Cell<u8>,
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
        /// Render passes encoded for the most recent frame.
        pub render_passes_last_frame: u64,
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
            render_passes_last_frame: LAST_FRAME_RENDER_PASSES.load(Ordering::Relaxed),
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

pub use render_stats::{FallbackReason, FrameTiming, RenderStats, render_stats};

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

impl Drop for Texture {
    fn drop(&mut self) {
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
