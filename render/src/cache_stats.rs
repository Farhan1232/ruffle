//! Why `cacheAsBitmap` textures are built, and how often.
//!
//! A cached display object keeps a texture of itself and redraws it when
//! something about it changes. An authenticated 43-minute AdventureQuest Worlds
//! session built 621,413 of those textures, so the question phase 2 has to
//! answer is not how many but *why*: a cache that is rebuilt because its object
//! genuinely changed shape is the cache working, and one rebuilt because its
//! object moved by a pixel is not.
//!
//! These counters separate the two. An **invalidation** is the cache deciding
//! its picture is out of date and redrawing it; an **allocation** is the cache
//! also having to build a new texture to redraw into. The first costs a render
//! pass, the second costs a driver allocation, and a session that does far more
//! of the second than of the first is thrashing.

use std::sync::atomic::{AtomicU64, Ordering};

/// Why a cached display object decided its picture was out of date.
///
/// The brief's categories, mapped onto what this cache actually stores: it
/// keeps the four scale/skew terms of the matrix it was drawn with and the
/// unfiltered size it was drawn at, and compares them. It has no separate
/// notion of a dirty filter, a device scale or a texture format, so those
/// categories would be counted as zero forever and are not offered.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheInvalidation {
    /// The object has no texture yet: the first draw, or the first after the
    /// cache was cleared.
    FirstAllocation,
    /// The scale or skew of the matrix changed, so the picture is drawn at a
    /// different size or angle. Pure translation does *not* land here - the
    /// cache deliberately ignores `tx`/`ty`.
    TransformChange,
    /// The object's own bounds changed: it animated into a different shape.
    SourceSizeChange,
    /// Something the cache cannot see for itself changed - a child moved, the
    /// filter list changed, the blend mode changed - and `make_dirty` was
    /// called.
    ContentDirty,
}

/// Why redrawing the cache also needed a new texture.
///
/// The cache keeps the texture it has whenever the size it needs is unchanged,
/// so everything here is a size that moved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheAllocation {
    /// There was no texture to keep.
    FirstAllocation,
    /// Wider than the texture it had.
    WidthExceeded,
    /// Taller than the texture it had.
    HeightExceeded,
    /// Fits inside the texture it had, and the texture was rebuilt anyway.
    /// This is the thrashing category: an avatar whose bounds breathe between
    /// 147x196 and 151x198 lands here on every frame that shrinks it.
    Shrank,
    /// The object is too large to cache, or the renderer refused the texture.
    Refused,
}

pub const CACHE_INVALIDATION_NAMES: &[&str] = &[
    "first_allocation",
    "transform_change",
    "source_size_change",
    "content_dirty",
];

pub const CACHE_ALLOCATION_NAMES: &[&str] = &[
    "first_allocation",
    "width_exceeded",
    "height_exceeded",
    "shrank",
    "refused",
];

static INVALIDATIONS: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

static ALLOCATIONS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Caches asked to redraw, and the ones that kept the texture they had.
static REDRAWS: AtomicU64 = AtomicU64::new(0);
static TEXTURE_KEPT: AtomicU64 = AtomicU64::new(0);
/// Pixels of cache texture built, so the bytes can be compared with the
/// renderer's own texture accounting.
static ALLOCATED_PIXELS: AtomicU64 = AtomicU64::new(0);
/// Pixels of physical capacity kept that a smaller logical size did not use.
static SPARE_CAPACITY_PIXELS: AtomicU64 = AtomicU64::new(0);

pub fn record_invalidation(reason: CacheInvalidation) {
    REDRAWS.fetch_add(1, Ordering::Relaxed);
    INVALIDATIONS[reason as usize].fetch_add(1, Ordering::Relaxed);
}

pub fn record_allocation(reason: CacheAllocation, width: u32, height: u32) {
    ALLOCATIONS[reason as usize].fetch_add(1, Ordering::Relaxed);
    ALLOCATED_PIXELS.fetch_add(u64::from(width) * u64::from(height), Ordering::Relaxed);
}

/// A redraw that kept the texture it had, with the capacity the smaller logical
/// size left unused.
pub fn record_texture_kept(spare_pixels: u64) {
    TEXTURE_KEPT.fetch_add(1, Ordering::Relaxed);
    SPARE_CAPACITY_PIXELS.fetch_add(spare_pixels, Ordering::Relaxed);
}

/// What the `cacheAsBitmap` caches have done so far. Subtract two readings to
/// measure a stretch of play.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub redraws: u64,
    pub texture_kept: u64,
    pub allocated_pixels: u64,
    pub spare_capacity_pixels: u64,
    /// Indexed by [`CacheInvalidation`].
    pub invalidations: Vec<u64>,
    /// Indexed by [`CacheAllocation`].
    pub allocations: Vec<u64>,
}

impl CacheStats {
    /// Textures built, whatever the reason.
    pub fn allocations_total(&self) -> u64 {
        self.allocations.iter().sum()
    }
}

pub fn cache_stats() -> CacheStats {
    CacheStats {
        redraws: REDRAWS.load(Ordering::Relaxed),
        texture_kept: TEXTURE_KEPT.load(Ordering::Relaxed),
        allocated_pixels: ALLOCATED_PIXELS.load(Ordering::Relaxed),
        spare_capacity_pixels: SPARE_CAPACITY_PIXELS.load(Ordering::Relaxed),
        invalidations: INVALIDATIONS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect(),
        allocations: ALLOCATIONS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect(),
    }
}
