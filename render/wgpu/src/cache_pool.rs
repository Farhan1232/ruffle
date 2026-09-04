//! Recycling for `cacheAsBitmap` textures.
//!
//! A cached display object keeps a texture of itself and rebuilds it whenever
//! the picture inside it changes size. An AdventureQuest Worlds avatar's bounds
//! breathe by a pixel or two every frame as its animation plays, so "changes
//! size" happens constantly: the client's 43-minute session built 621,413 of
//! these, each one a driver allocation, while never holding more than about a
//! hundred megabytes of them at a time.
//!
//! Nothing about that traffic is necessary. The sizes repeat - an avatar
//! oscillating between 147x196 and 151x198 asks for the same handful of sizes
//! over and over - so a texture that has just been given up is very often
//! exactly the one the next request wants.
//!
//! ## Why recycling one of these is safe
//!
//! These textures are renderer-owned and disposable, and they are the *only*
//! thing [`RenderBackend::create_empty_texture`] is used for: `BitmapData` goes
//! through `register_bitmap`, which uploads pixels and is untouched by any of
//! this. A cache texture has exactly one owner, the `BitmapCache` that asked
//! for it, and it is dropped when that cache replaces or releases it.
//!
//! Stale pixels cannot survive the reuse. Every redraw of a cache goes through
//! `Surface::draw_commands` with `RenderTargetMode::ExistingWithColor`, whose
//! first render pass loads the attachment with `LoadOp::Clear` over the whole
//! texture, and which ends with `ensure_cleared` so that a cache that draws
//! nothing at all is still cleared rather than left holding whatever was there.
//! A recycled texture is therefore fully overwritten before anything can sample
//! it.
//!
//! ## What bounds it
//!
//! Idle textures are held against a byte budget, and when it is exceeded the
//! sizes least recently asked for are dropped for real. Sizes nothing has
//! wanted for a while are forgotten entirely, so a long session that wanders
//! through thousands of sizes does not keep a few of each.

use crate::{Texture, texture_bytes};
use fnv::FnvHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// What the cache pool has done. Cumulative; subtract two readings for a span.
static TAKES: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static BUILDS: AtomicU64 = AtomicU64::new(0);
static RETURNS: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);
static EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachePoolStats {
    /// Requests for a cache texture.
    pub takes: u64,
    /// Requests served by recycling one.
    pub hits: u64,
    /// Requests that had to allocate.
    pub builds: u64,
    /// Textures given back when a cache released them.
    pub returns: u64,
    /// Textures really destroyed, because the pool was over its budget or had
    /// stopped being asked for them.
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// What the pool is holding idle right now, and how many sizes it knows.
    pub idle_textures: usize,
    pub idle_bytes: usize,
    pub size_classes: usize,
}

impl CachePoolStats {
    pub fn hit_rate(&self) -> f64 {
        if self.takes == 0 {
            return 0.0;
        }
        self.hits as f64 / self.takes as f64
    }
}

/// The parts of a [`Texture`] worth keeping when its owner lets it go.
///
/// The bind groups come back with it: they name this texture and nothing else,
/// so they are still valid, and rebuilding them per reuse would trade a texture
/// allocation for four bind-group allocations.
pub(crate) struct Recycled {
    pub texture: wgpu::Texture,
    pub repeating_linear: std::cell::OnceCell<crate::mesh::BitmapBinds>,
    pub repeating_nearest: std::cell::OnceCell<crate::mesh::BitmapBinds>,
    pub clamped_linear: std::cell::OnceCell<crate::mesh::BitmapBinds>,
    pub clamped_nearest: std::cell::OnceCell<crate::mesh::BitmapBinds>,
}

#[derive(Default)]
struct SizeClass {
    idle: Vec<Recycled>,
    /// When this size was last asked for, so the sizes a scene has moved on
    /// from are the ones given up first.
    last_used: u64,
}

pub struct CacheTexturePool {
    classes: FnvHashMap<(u32, u32), SizeClass>,
    idle_bytes: usize,
    budget: usize,
    clock: u64,
}

impl std::fmt::Debug for CacheTexturePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheTexturePool")
            .field("size_classes", &self.classes.len())
            .field("idle_bytes", &self.idle_bytes)
            .finish()
    }
}

/// What the pool will hold in textures nothing is using.
///
/// The client's session held about 112 MB of tracked texture in total, most of
/// it these, so a 48 MB idle budget is enough to serve a room coming back to
/// sizes it has just used without the pool becoming a second copy of the scene.
const DEFAULT_BUDGET: usize = 48 * 1024 * 1024;

impl CacheTexturePool {
    pub fn new() -> Arc<Mutex<Self>> {
        Self::with_budget(DEFAULT_BUDGET)
    }

    pub fn with_budget(budget: usize) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            classes: Default::default(),
            idle_bytes: 0,
            budget,
            clock: 0,
        }))
    }

    /// A texture of exactly this size, recycled if one is idle.
    pub(crate) fn take(&mut self, width: u32, height: u32) -> Option<Recycled> {
        self.clock += 1;
        let clock = self.clock;
        TAKES.fetch_add(1, Ordering::Relaxed);
        let class = self.classes.entry((width, height)).or_default();
        class.last_used = clock;
        match class.idle.pop() {
            Some(recycled) => {
                self.idle_bytes = self
                    .idle_bytes
                    .saturating_sub(texture_bytes(&recycled.texture));
                HITS.fetch_add(1, Ordering::Relaxed);
                Some(recycled)
            }
            None => {
                BUILDS.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Takes a released cache texture back.
    pub(crate) fn give_back(&mut self, recycled: Recycled) {
        let size = recycled.texture.size();
        let bytes = texture_bytes(&recycled.texture);
        RETURNS.fetch_add(1, Ordering::Relaxed);
        let clock = self.clock;
        let class = self.classes.entry((size.width, size.height)).or_default();
        class.last_used = class.last_used.max(clock);
        class.idle.push(recycled);
        self.idle_bytes += bytes;
        self.enforce_budget();
    }

    /// Drops the sizes least recently asked for until the idle set fits the
    /// budget. Whatever a frame is using is untouchable - it is not in here.
    fn enforce_budget(&mut self) {
        if self.idle_bytes <= self.budget {
            return;
        }
        let mut order: Vec<((u32, u32), u64)> = self
            .classes
            .iter()
            .filter(|(_, class)| !class.idle.is_empty())
            .map(|(key, class)| (*key, class.last_used))
            .collect();
        order.sort_unstable_by_key(|(_, last_used)| *last_used);

        for (key, _) in order {
            if self.idle_bytes <= self.budget {
                break;
            }
            if let Some(class) = self.classes.get_mut(&key) {
                for recycled in class.idle.drain(..) {
                    let bytes = texture_bytes(&recycled.texture);
                    self.idle_bytes = self.idle_bytes.saturating_sub(bytes);
                    EVICTIONS.fetch_add(1, Ordering::Relaxed);
                    EVICTED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
                    crate::track_texture_dropped(&recycled.texture);
                }
                self.classes.remove(&key);
            }
        }
    }

    /// Forgets sizes nothing has asked for in a while, so a long session does
    /// not keep a key for every size it has ever met.
    ///
    /// Called on the same schedule as the render-target pools' trimming.
    pub fn trim_idle(&mut self) {
        let cutoff = self.clock.saturating_sub(FORGET_AFTER);
        let mut give_up = Vec::new();
        for (key, class) in &self.classes {
            if class.last_used < cutoff {
                give_up.push(*key);
            }
        }
        for key in give_up {
            if let Some(class) = self.classes.remove(&key) {
                for recycled in class.idle {
                    let bytes = texture_bytes(&recycled.texture);
                    self.idle_bytes = self.idle_bytes.saturating_sub(bytes);
                    EVICTIONS.fetch_add(1, Ordering::Relaxed);
                    EVICTED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
                    crate::track_texture_dropped(&recycled.texture);
                }
            }
        }
    }

    fn snapshot(&self) -> (usize, usize, usize) {
        (
            self.classes.values().map(|c| c.idle.len()).sum(),
            self.idle_bytes,
            self.classes.len(),
        )
    }
}

/// Requests after which a size nothing has asked for is forgotten. A crowded
/// room asks for a few hundred a frame, so this is a few seconds of play.
const FORGET_AFTER: u64 = 4096;

/// What the cache pool has done so far, plus what it is holding now.
pub fn cache_pool_stats(pool: &Mutex<CacheTexturePool>) -> CachePoolStats {
    let (idle_textures, idle_bytes, size_classes) =
        pool.lock().map(|pool| pool.snapshot()).unwrap_or((0, 0, 0));
    CachePoolStats {
        takes: TAKES.load(Ordering::Relaxed),
        hits: HITS.load(Ordering::Relaxed),
        builds: BUILDS.load(Ordering::Relaxed),
        returns: RETURNS.load(Ordering::Relaxed),
        evictions: EVICTIONS.load(Ordering::Relaxed),
        evicted_bytes: EVICTED_BYTES.load(Ordering::Relaxed),
        idle_textures,
        idle_bytes,
        size_classes,
    }
}

/// The counters alone, for callers that have no handle on the pool.
pub fn cache_pool_counters() -> CachePoolStats {
    CachePoolStats {
        takes: TAKES.load(Ordering::Relaxed),
        hits: HITS.load(Ordering::Relaxed),
        builds: BUILDS.load(Ordering::Relaxed),
        returns: RETURNS.load(Ordering::Relaxed),
        evictions: EVICTIONS.load(Ordering::Relaxed),
        evicted_bytes: EVICTED_BYTES.load(Ordering::Relaxed),
        idle_textures: 0,
        idle_bytes: 0,
        size_classes: 0,
    }
}

/// Hands a released cache texture back to its pool, if it came from one.
pub(crate) fn release(texture: &mut Texture) -> bool {
    let Some(pool) = texture.cache_pool.take() else {
        return false;
    };
    let Ok(mut pool) = pool.lock() else {
        return false;
    };
    pool.give_back(Recycled {
        texture: texture.texture.clone(),
        repeating_linear: std::mem::take(&mut texture.repeating_linear),
        repeating_nearest: std::mem::take(&mut texture.repeating_nearest),
        clamped_linear: std::mem::take(&mut texture.clamped_linear),
        clamped_nearest: std::mem::take(&mut texture.clamped_nearest),
    });
    true
}
