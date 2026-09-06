pub mod null;

use crate::bitmap::{Bitmap, BitmapHandle, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle};
use crate::commands::CommandList;
use crate::error::Error;
use crate::filters::Filter;
use crate::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use crate::pixel_bender_support::PixelBenderShaderArgument;
use crate::quality::StageQuality;
use crate::shape_utils::DistilledShape;
use ruffle_wstr::{FromWStr, WStr};
use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;
use swf::{Color, Rectangle, Twips};

/// One render-target pool key, with what it holds and how much of it was ever
/// wanted at once. `peak_borrowed` is the ceiling a pool can grow to, because
/// a pool only builds a new entry when its free list is empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolKeyReport {
    pub pool: &'static str,
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub format: String,
    pub usage: String,
    pub idle_entries: usize,
    pub idle_bytes: usize,
    pub borrowed: usize,
    pub peak_borrowed: usize,
    pub recent_peak_borrowed: usize,
    pub reuses: u64,
    pub misses_pool_empty: u64,
    pub misses_new_key: u64,
    /// What the pool decided to keep for this key at its last trim.
    pub retained_target: usize,
}

pub struct BitmapCacheEntry {
    pub handle: BitmapHandle,
    pub commands: CommandList,
    pub clear: Color,
    pub filters: Vec<Filter>,
    /// The size of the picture inside the texture.
    ///
    /// The texture may be *larger* than this: a cache keeps its texture while
    /// the picture inside it only changes by a few pixels, which is what stops
    /// an animating avatar reallocating one every frame. See
    /// [`crate::cache_capacity`]. Everything that redraws or filters a cache
    /// takes its extent from here and never from the texture, which is the
    /// whole difference between this and the padding experiment that broke
    /// `displacement_map`.
    pub logical_width: u32,
    pub logical_height: u32,
}

/// GPU memory held by a render backend at one moment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderMemoryUsage {
    /// Number of textures alive in the backend.
    pub textures: usize,
    /// Bytes of texture memory those textures occupy.
    pub texture_bytes: usize,
    /// Number of buffers alive in the backend.
    pub buffers: usize,
    /// Bytes of buffer memory those buffers occupy.
    pub buffer_bytes: usize,
    /// Number of tessellated shape meshes alive.
    pub meshes: usize,
    /// Bytes of vertex and index data those meshes hold.
    pub mesh_bytes: usize,
    /// Textures Ruffle itself created and still holds (bitmaps, cached
    /// display objects, pooled render targets), counted by Ruffle so that the
    /// figure exists on every backend.
    pub tracked_textures: usize,
    /// Approximate bytes of those textures' pixels.
    pub tracked_texture_bytes: usize,
    /// Live textures and their bytes, split by what the texture is for:
    /// decoded bitmaps, `cacheAsBitmap` backing stores, one-off render
    /// outputs, and render targets from each of the two pools. Separating
    /// these is what tells memory owned by live content apart from memory the
    /// renderer is holding as reusable scratch.
    pub texture_kind_names: &'static [&'static str],
    pub texture_kind_live_counts: Vec<usize>,
    pub texture_kind_live_bytes: Vec<usize>,
    pub texture_kind_created: Vec<u64>,
    pub texture_kind_created_bytes: Vec<u64>,
    pub texture_kind_dropped: Vec<u64>,
    pub texture_kind_dropped_bytes: Vec<u64>,
    /// The most texture memory held at once over the whole run. Against the
    /// process' working set, this separates a high-water mark from live use.
    pub peak_texture_bytes: usize,
    /// Render targets handed back from a pool's free list, versus those the
    /// pool had to build. A high miss rate means the pools are churning.
    pub pool_reuses: u64,
    pub pool_misses: u64,
    /// Render targets idle in the surface pool (kept across frames) and in
    /// the offscreen pool (replaced every frame), and how many distinct sizes
    /// each is keyed on.
    pub main_pool_idle_textures: usize,
    pub main_pool_idle_bytes: usize,
    pub main_pool_size_classes: usize,
    pub offscreen_pool_idle_textures: usize,
    pub offscreen_pool_idle_bytes: usize,
    pub offscreen_pool_size_classes: usize,
    /// Readback/upload buffers idle in the renderer's buffer pool.
    pub buffer_pool_idle_entries: usize,
    pub buffer_pool_idle_bytes: usize,
    /// The heaviest pool keys, with the whole key and its demand figures.
    pub pool_keys: Vec<PoolKeyReport>,
    /// Textures created and dropped since the process started; the difference
    /// between two samples is the churn over that span.
    pub textures_created: u64,
    pub texture_bytes_created: u64,
    pub textures_dropped: u64,
    pub texture_bytes_dropped: u64,
    /// Objects the graphics backend itself is holding, as it counts them.
    ///
    /// Ruffle's own figures say what it asked for; these say what is still
    /// alive underneath, which is the only way to tell a resource Ruffle has
    /// dropped from one the backend has actually released. A backend that
    /// cannot report a figure leaves it zero.
    pub hal: HalResourceUsage,
    /// What the backend's memory allocator has taken from the driver, when it
    /// can say. `reserved` counts whole blocks including their unused parts,
    /// so `reserved - allocated` is memory the allocator owns and is not
    /// using.
    pub allocator: Option<AllocatorUsage>,
    /// What a frame costs in work rather than in memory.
    pub work: RenderWorkUsage,
}

/// Live graphics-backend objects, as the backend counts them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HalResourceUsage {
    pub textures: usize,
    pub texture_views: usize,
    pub buffers: usize,
    pub bind_groups: usize,
    pub bind_group_layouts: usize,
    pub render_pipelines: usize,
    pub compute_pipelines: usize,
    pub pipeline_layouts: usize,
    pub samplers: usize,
    pub command_encoders: usize,
    pub shader_modules: usize,
    pub query_sets: usize,
    pub fences: usize,
    pub texture_memory: usize,
    pub buffer_memory: usize,
    /// Separate allocations the backend has made from the driver.
    pub memory_allocations: usize,
}

/// The graphics allocator's own account of what it holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocatorUsage {
    /// Bytes in live sub-allocations.
    pub allocated_bytes: u64,
    /// Bytes in the blocks those sub-allocations sit in, unused parts
    /// included.
    pub reserved_bytes: u64,
    /// How many blocks the allocator holds.
    pub blocks: usize,
}

/// The reasons a blended group can fail to qualify for direct drawing, in the
/// order they appear in [`RenderWorkUsage::fallbacks`]. Named here so that the
/// report's columns exist even on a backend that never fills them.
/// Column names for the page-fallback and pool-miss counters, so a CSV writer
/// does not have to depend on the wgpu backend to name its own columns.
pub const PAGE_FALLBACK_COLUMN_NAMES: &[&str] = &[
    "shader",
    "nested_blend",
    "alpha_mask",
    "masked",
    "stage3d",
    "size",
    "capacity",
    "no_page",
];

pub const POOL_MISS_COLUMN_NAMES: &[&str] = &[
    "new_size_class",
    "format_mismatch",
    "sample_count_mismatch",
    "usage_mismatch",
    "evicted_by_budget",
    "free_list_empty",
];

pub const FALLBACK_COLUMN_NAMES: &[&str] = &[
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

/// What the renderer did, as opposed to what it holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderWorkUsage {
    /// Render passes encoded for the most recent frame.
    pub render_passes: u64,
    /// Render targets blended groups took during the most recent frame, and
    /// their bytes; every one is held until its chunk is encoded, so the
    /// frame's total is what was live at once.
    pub blend_targets: usize,
    pub blend_target_bytes: usize,
    pub peak_blend_targets: usize,
    pub peak_blend_target_bytes: usize,
    /// Bind groups built, against those served from the cache kept with the
    /// texture they describe.
    pub bind_groups_created: u64,
    pub bind_group_cache_hits: u64,
    pub bind_group_cache_misses: u64,
    /// Blended groups considered for direct drawing, and those that took it.
    pub fastpath_eligible: u64,
    pub fastpath_used: u64,
    /// Why the rest did not, in the order of `fallback_names`.
    pub fallback_names: &'static [&'static str],
    pub fallbacks: Vec<u64>,
    /// Multiplies carried by their own draw, which is a subset of
    /// `fastpath_used`, and the two reasons the rest were not: the destination
    /// was not known opaque, or the group's single draw was a shape.
    pub multiply_on_draw_used: u64,
    pub multiply_on_draw_shape: u64,
    pub multiply_on_draw_transparent: u64,
    /// Where the renderer's share of the frames went, in nanoseconds. The rest
    /// of a frame is ActionScript, collection and the display list, which the
    /// renderer cannot see; the frontend's frame time minus `render_ns_total`
    /// is that share.
    pub render_ns_total: u64,
    pub render_ns_cache_entries: u64,
    pub render_ns_frame_commands: u64,
    /// Includes waiting for the display to accept the frame, so a large share
    /// here means the GPU or the presentation queue could not keep up, not
    /// that the CPU was busy. `render_ns_frame_commands` is the CPU encode.
    pub render_ns_queue_submit: u64,
    /// Frames whose *rendering* alone missed the 41.67 ms budget, and those
    /// that took over 100 ms, with where their time went.
    pub slow_frames: u64,
    pub very_slow_frames: u64,
    pub slow_ns_cache_entries: u64,
    pub slow_ns_frame_commands: u64,
    pub slow_ns_queue_submit: u64,

    // --- phase 1: blend pages and batched complex blends ------------------
    /// Blended groups offered a shared page, and those that took a region on
    /// one; then why the rest did not, in the order of `page_fallback_names`.
    pub batch_eligible: u64,
    pub batch_used: u64,
    pub page_fallback_names: &'static [&'static str],
    pub page_fallbacks: Vec<u64>,
    /// Pages the most recent frame took, what they cost, and the most any
    /// frame has cost.
    pub pages_last_frame: usize,
    pub page_bytes_last_frame: usize,
    pub peak_page_bytes: usize,
    /// Snapshots complex blends took of the destination underneath them.
    /// These are the term phase 1 measured and did not attack.
    pub destination_copies: u64,
    pub destination_copy_pixels: u64,
    pub destination_copies_last_frame: u64,
    pub destination_copy_pixels_last_frame: u64,
    /// Complex blends composited, and the passes that took them. The ratio is
    /// how many share a pass.
    pub complex_blends: u64,
    pub complex_blend_passes: u64,

    // --- phase 2: cache and offscreen allocation --------------------------
    /// Caches asked to redraw, and those that kept the texture they had
    /// because the picture still fitted its capacity.
    pub cache_redraws: u64,
    pub cache_texture_kept: u64,
    /// Why a cache decided its picture was out of date, in the order of
    /// `cache_invalidation_names`.
    pub cache_invalidation_names: &'static [&'static str],
    pub cache_invalidations: Vec<u64>,
    /// Why redrawing it also needed a new texture, in the order of
    /// `cache_allocation_names`. `shrank` is the thrashing category.
    pub cache_allocation_names: &'static [&'static str],
    pub cache_allocations: Vec<u64>,
    pub cache_allocated_pixels: u64,
    /// The pool of released cache textures: asked, recycled, really built.
    pub cache_pool_takes: u64,
    pub cache_pool_hits: u64,
    pub cache_pool_builds: u64,
    pub cache_pool_idle_textures: usize,
    pub cache_pool_idle_bytes: usize,
    /// Why the offscreen pool could not serve a request from a free list, in
    /// the order of `pool_miss_names`. This is the term that dominates once
    /// the cache textures stop churning.
    pub pool_miss_names: &'static [&'static str],
    pub offscreen_pool_hits: u64,
    pub offscreen_pool_misses: Vec<u64>,
    pub offscreen_pool_miss_bytes: Vec<u64>,
    pub offscreen_pool_evictions: u64,
    pub offscreen_pool_evicted_bytes: u64,
    pub offscreen_pool_size_classes_seen: usize,
    pub main_pool_hits: u64,
    pub main_pool_misses: Vec<u64>,
}

pub trait RenderBackend: Any {
    fn viewport_dimensions(&self) -> ViewportDimensions;
    // Do not call this method directly - use `player.set_viewport_dimensions`,
    // which will ensure that the stage is properly updated as well.
    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions);
    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle;

    fn register_shape_with_scale(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
        _scale: f32,
    ) -> ShapeHandle {
        // Default implementation ignores scale
        self.register_shape(shape, bitmap_source)
    }

    fn render_offscreen(
        &mut self,
        handle: BitmapHandle,
        commands: CommandList,
        quality: StageQuality,
        bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>>;

    /// Applies the given filter with a `BitmapHandle` source onto a destination `BitmapHandle`.
    /// The `destination_rect` must be calculated by the caller and is assumed to be correct.
    /// Both `source_rect` and `destination_rect` must be valid (`BoundingBox::valid`).
    /// `source` may equal `destination`, in which case a temporary buffer is used internally.
    ///
    /// Returns None if the backend does not support this filter.
    fn apply_filter(
        &mut self,
        _source: BitmapHandle,
        _source_point: (u32, u32),
        _source_size: (u32, u32),
        _destination: BitmapHandle,
        _dest_point: (i32, i32),
        _filter: Filter,
    ) -> Option<Box<dyn SyncHandle>> {
        None
    }

    fn is_filter_supported(&self, _filter: &Filter) -> bool {
        false
    }

    fn is_offscreen_supported(&self) -> bool {
        false
    }

    fn submit_frame(
        &mut self,
        clear: swf::Color,
        commands: CommandList,
        cache_entries: Vec<BitmapCacheEntry>,
    );

    fn create_empty_texture(
        &mut self,
        width: NonZeroU32,
        height: NonZeroU32,
    ) -> Result<BitmapHandle, Error>;

    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, Error>;
    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        region: PixelRegion,
    ) -> Result<(), Error>;

    fn create_context3d(&mut self, profile: Context3DProfile) -> Result<Box<dyn Context3D>, Error>;

    fn debug_info(&self) -> Cow<'static, str>;

    /// How much memory this backend currently holds in GPU resources, if it
    /// can tell. Used by memory diagnostics; `None` means "not available".
    fn memory_usage(&mut self) -> Option<RenderMemoryUsage> {
        None
    }

    /// An internal name that is used to identify the render-backend.
    fn name(&self) -> &'static str;

    fn set_quality(&mut self, quality: StageQuality);

    fn compile_pixelbender_shader(
        &mut self,
        shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, Error>;

    fn run_pixelbender_shader(
        &mut self,
        handle: PixelBenderShaderHandle,
        arguments: &[PixelBenderShaderArgument],
        target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, Error>;

    fn resolve_sync_handle(
        &mut self,
        handle: Box<dyn SyncHandle>,
        with_rgba: RgbaBufRead,
    ) -> Result<(), Error>;
}

pub enum PixelBenderTarget {
    // The shader will write to the provided bitmap texture,
    // producing a `PixelBenderOutput::Bitmap` with the corresponding
    // `SyncHandle`
    Bitmap(BitmapHandle),
    // The shader will write to a temporary texture, which will then
    // be immediately read back as bytes (in `PixelBenderOutput::Bytes`)
    Bytes { width: u32, height: u32 },
}

pub enum PixelBenderOutput {
    Bitmap(Box<dyn SyncHandle>),
    Bytes(Vec<u8>),
}

pub trait IndexBuffer: Any {}
pub trait VertexBuffer: Any {}

pub trait ShaderModule: Any {}

pub trait Texture: Any + Debug {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

pub trait RawTexture: Any + Debug {
    fn equals(&self, other: &dyn RawTexture) -> bool;
}

#[cfg(feature = "wgpu")]
impl RawTexture for wgpu::Texture {
    fn equals(&self, other: &dyn RawTexture) -> bool {
        if let Some(other_texture) = (other as &dyn Any).downcast_ref::<wgpu::Texture>() {
            std::ptr::eq(self, other_texture)
        } else {
            false
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Context3DTextureFormat {
    Bgra,
    BgraPacked,
    BgrPacked,
    Compressed,
    CompressedAlpha,
    RgbaHalfFloat,
}

impl FromWStr for Context3DTextureFormat {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"bgra" {
            Ok(Context3DTextureFormat::Bgra)
        } else if s == b"bgraPacked4444" {
            Ok(Context3DTextureFormat::BgraPacked)
        } else if s == b"bgrPacked565" {
            Ok(Context3DTextureFormat::BgrPacked)
        } else if s == b"compressed" {
            Ok(Context3DTextureFormat::Compressed)
        } else if s == b"compressedAlpha" {
            Ok(Context3DTextureFormat::CompressedAlpha)
        } else if s == b"rgbaHalfFloat" {
            Ok(Context3DTextureFormat::RgbaHalfFloat)
        } else {
            Err(())
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Context3DBlendFactor {
    DestinationAlpha,
    DestinationColor,
    One,
    OneMinusDestinationAlpha,
    OneMinusDestinationColor,
    OneMinusSourceAlpha,
    OneMinusSourceColor,
    SourceAlpha,
    SourceColor,
    Zero,
}

impl FromWStr for Context3DBlendFactor {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"destinationAlpha" {
            Ok(Context3DBlendFactor::DestinationAlpha)
        } else if s == b"destinationColor" {
            Ok(Context3DBlendFactor::DestinationColor)
        } else if s == b"one" {
            Ok(Context3DBlendFactor::One)
        } else if s == b"oneMinusDestinationAlpha" {
            Ok(Context3DBlendFactor::OneMinusDestinationAlpha)
        } else if s == b"oneMinusDestinationColor" {
            Ok(Context3DBlendFactor::OneMinusDestinationColor)
        } else if s == b"oneMinusSourceAlpha" {
            Ok(Context3DBlendFactor::OneMinusSourceAlpha)
        } else if s == b"oneMinusSourceColor" {
            Ok(Context3DBlendFactor::OneMinusSourceColor)
        } else if s == b"sourceAlpha" {
            Ok(Context3DBlendFactor::SourceAlpha)
        } else if s == b"sourceColor" {
            Ok(Context3DBlendFactor::SourceColor)
        } else if s == b"zero" {
            Ok(Context3DBlendFactor::Zero)
        } else {
            Err(())
        }
    }
}

pub enum BufferUsage {
    DynamicDraw,
    StaticDraw,
}

pub enum ProgramType {
    Vertex,
    Fragment,
}

impl FromWStr for ProgramType {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"vertex" {
            Ok(ProgramType::Vertex)
        } else if s == b"fragment" {
            Ok(ProgramType::Fragment)
        } else {
            Err(())
        }
    }
}

pub trait Context3D: Any {
    fn profile(&self) -> Context3DProfile;
    // The BitmapHandle for the texture we're rendering to
    fn bitmap_handle(&self) -> BitmapHandle;
    // Whether or not we should actually render the texture
    // as part of stage rendering
    fn should_render(&self) -> bool;

    // Get a 'disposed' handle - this is what we store in all IndexBuffer3D
    // objects after dispose() has been called.
    fn disposed_index_buffer_handle(&self) -> Rc<dyn IndexBuffer>;

    // Get a 'disposed' handle - this is what we store in all VertexBuffer3D
    // objects after dispose() has been called.
    fn disposed_vertex_buffer_handle(&self) -> Rc<dyn VertexBuffer>;

    fn create_index_buffer(&mut self, usage: BufferUsage, num_indices: u32)
    -> Box<dyn IndexBuffer>;
    fn create_vertex_buffer(
        &mut self,
        usage: BufferUsage,
        num_vertices: u32,
        data_32_per_vertex: u8,
    ) -> Rc<dyn VertexBuffer>;

    fn create_texture(
        &mut self,
        width: u32,
        height: u32,
        format: Context3DTextureFormat,
        optimize_for_render_to_texture: bool,
        streaming_levels: u32,
    ) -> Result<Rc<dyn Texture>, Error>;
    fn create_cube_texture(
        &mut self,
        size: u32,
        format: Context3DTextureFormat,
        optimize_for_render_to_texture: bool,
        streaming_levels: u32,
    ) -> Result<Rc<dyn Texture>, Error>;

    fn upload_shaders(
        &mut self,
        module: &RefCell<Option<Rc<dyn ShaderModule>>>,
        vertex_shader_agal: Vec<u8>,
        fragment_shader_agal: Vec<u8>,
    ) -> Result<(), naga_agal::AgalError>;

    fn process_command(&mut self, command: Context3DCommand<'_>);

    fn present(&mut self);
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DVertexBufferFormat {
    Float1,
    Float2,
    Float3,
    Float4,
    Bytes4,
}

impl FromWStr for Context3DVertexBufferFormat {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"float1" {
            Ok(Context3DVertexBufferFormat::Float1)
        } else if s == b"float2" {
            Ok(Context3DVertexBufferFormat::Float2)
        } else if s == b"float3" {
            Ok(Context3DVertexBufferFormat::Float3)
        } else if s == b"float4" {
            Ok(Context3DVertexBufferFormat::Float4)
        } else if s == b"bytes4" {
            Ok(Context3DVertexBufferFormat::Bytes4)
        } else {
            Err(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DTriangleFace {
    None,
    Back,
    Front,
    FrontAndBack,
}

impl FromWStr for Context3DTriangleFace {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"none" {
            Ok(Context3DTriangleFace::None)
        } else if s == b"back" {
            Ok(Context3DTriangleFace::Back)
        } else if s == b"front" {
            Ok(Context3DTriangleFace::Front)
        } else if s == b"frontAndBack" {
            Ok(Context3DTriangleFace::FrontAndBack)
        } else {
            Err(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DProfile {
    Baseline,
    BaselineConstrained,
    BaselineExtended,
    Standard,
    StandardConstrained,
    StandardExtended,
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DCompareMode {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl FromWStr for Context3DCompareMode {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"never" {
            Ok(Context3DCompareMode::Never)
        } else if s == b"less" {
            Ok(Context3DCompareMode::Less)
        } else if s == b"equal" {
            Ok(Context3DCompareMode::Equal)
        } else if s == b"lessEqual" {
            Ok(Context3DCompareMode::LessEqual)
        } else if s == b"greater" {
            Ok(Context3DCompareMode::Greater)
        } else if s == b"notEqual" {
            Ok(Context3DCompareMode::NotEqual)
        } else if s == b"greaterEqual" {
            Ok(Context3DCompareMode::GreaterEqual)
        } else if s == b"always" {
            Ok(Context3DCompareMode::Always)
        } else {
            Err(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DStencilAction {
    DecrementSaturate,
    DecrementWrap,
    IncrementSaturate,
    IncrementWrap,
    Invert,
    Keep,
    Set,
    Zero,
}

impl FromWStr for Context3DStencilAction {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"decrementSaturate" {
            Ok(Context3DStencilAction::DecrementSaturate)
        } else if s == b"decrementWrap" {
            Ok(Context3DStencilAction::DecrementWrap)
        } else if s == b"incrementSaturate" {
            Ok(Context3DStencilAction::IncrementSaturate)
        } else if s == b"incrementWrap" {
            Ok(Context3DStencilAction::IncrementWrap)
        } else if s == b"invert" {
            Ok(Context3DStencilAction::Invert)
        } else if s == b"keep" {
            Ok(Context3DStencilAction::Keep)
        } else if s == b"set" {
            Ok(Context3DStencilAction::Set)
        } else if s == b"zero" {
            Ok(Context3DStencilAction::Zero)
        } else {
            Err(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DWrapMode {
    Clamp,
    ClampURepeatV,
    Repeat,
    RepeatUClampV,
}

impl FromWStr for Context3DWrapMode {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"clamp" {
            Ok(Context3DWrapMode::Clamp)
        } else if s == b"clamp_u_repeat_v" {
            Ok(Context3DWrapMode::ClampURepeatV)
        } else if s == b"repeat" {
            Ok(Context3DWrapMode::Repeat)
        } else if s == b"repeat_u_clamp_v" {
            Ok(Context3DWrapMode::RepeatUClampV)
        } else {
            Err(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Context3DTextureFilter {
    Anisotropic16X,
    Anisotropic2X,
    Anisotropic4X,
    Anisotropic8X,
    Linear,
    Nearest,
}

impl FromWStr for Context3DTextureFilter {
    type Err = ();

    fn from_wstr(s: &WStr) -> Result<Self, Self::Err> {
        if s == b"anisotropic16x" {
            Ok(Context3DTextureFilter::Anisotropic16X)
        } else if s == b"anisotropic2x" {
            Ok(Context3DTextureFilter::Anisotropic2X)
        } else if s == b"anisotropic4x" {
            Ok(Context3DTextureFilter::Anisotropic4X)
        } else if s == b"anisotropic8x" {
            Ok(Context3DTextureFilter::Anisotropic8X)
        } else if s == b"linear" {
            Ok(Context3DTextureFilter::Linear)
        } else if s == b"nearest" {
            Ok(Context3DTextureFilter::Nearest)
        } else {
            Err(())
        }
    }
}
pub enum Context3DCommand<'a> {
    Clear {
        red: f64,
        green: f64,
        blue: f64,
        alpha: f64,
        depth: f64,
        stencil: u32,
        mask: u32,
    },
    ConfigureBackBuffer {
        width: u32,
        height: u32,
        anti_alias: u32,
        depth_and_stencil: bool,
        wants_best_resolution: bool,
        wants_best_resolution_on_browser_zoom: bool,
    },
    SetRenderToTexture {
        texture: Rc<dyn Texture>,
        enable_depth_and_stencil: bool,
        anti_alias: u32,
        surface_selector: u32,
    },
    SetRenderToBackBuffer,

    UploadToIndexBuffer {
        buffer: &'a mut dyn IndexBuffer,
        start_offset: usize,
        data: &'a [u8],
    },

    UploadToVertexBuffer {
        buffer: Rc<dyn VertexBuffer>,
        start_vertex: usize,
        data32_per_vertex: u8,
        data: &'a [u8],
    },

    DrawTriangles {
        index_buffer: &'a dyn IndexBuffer,
        first_index: usize,
        num_triangles: isize,
    },

    SetVertexBufferAt {
        index: u32,
        buffer: Option<(Rc<dyn VertexBuffer>, Context3DVertexBufferFormat)>,
        buffer_offset: u32,
    },

    SetShaders {
        module: Option<Rc<dyn ShaderModule>>,
    },
    SetProgramConstants {
        program_type: ProgramType,
        first_register: u32,
        matrix_raw_data_column_major: &'a [[u8; 4]],
    },
    SetCulling {
        face: Context3DTriangleFace,
    },
    CopyBitmapToTexture {
        source: &'a [u8],
        source_width: u32,
        source_height: u32,
        dest: Rc<dyn Texture>,
        layer: u32,
    },
    SetTextureAt {
        sampler: u32,
        texture: Option<Rc<dyn Texture>>,
        cube: bool,
    },
    SetColorMask {
        red: bool,
        green: bool,
        blue: bool,
        alpha: bool,
    },
    SetDepthTest {
        depth_mask: bool,
        pass_compare_mode: Context3DCompareMode,
    },
    SetBlendFactors {
        source_factor: Context3DBlendFactor,
        destination_factor: Context3DBlendFactor,
    },
    SetSamplerStateAt {
        sampler: u32,
        wrap: Context3DWrapMode,
        filter: Context3DTextureFilter,
    },
    SetScissorRectangle {
        rect: Option<Rectangle<Twips>>,
    },
    SetStencilActions {
        triangle_face: Context3DTriangleFace,
        compare_mode: Context3DCompareMode,
        on_both_pass: Context3DStencilAction,
        on_depth_fail: Context3DStencilAction,
        on_depth_pass_stencil_fail: Context3DStencilAction,
    },
    SetStencilReferenceValue {
        reference_value: u32,
        read_mask: u32,
        write_mask: u32,
    },
}

#[derive(Clone, Debug)]
pub struct ShapeHandle(pub Arc<dyn ShapeHandleImpl>);

pub trait ShapeHandleImpl: Any + Debug {}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct ViewportDimensions {
    /// The dimensions of the stage's containing viewport.
    pub width: u32,
    pub height: u32,

    /// The scale factor of the containing viewport from standard-size pixels
    /// to device-scale pixels.
    pub scale_factor: f64,
}
