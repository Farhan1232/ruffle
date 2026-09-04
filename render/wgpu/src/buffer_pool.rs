use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::{FnvHashMap, FnvHashSet};
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// What the render-target pool has been asked for since the process started.
///
/// The pixels are the headline number: a frame's temporary targets are
/// allocated, cleared, drawn into, sampled and composited, so what they cost in
/// bandwidth is proportional to how many pixels of them a frame asks for. The
/// builds are the other half - a target that has to be created is a driver
/// allocation, which the pool exists to avoid.
static POOL_TAKES: AtomicU64 = AtomicU64::new(0);
static POOL_BUILDS: AtomicU64 = AtomicU64::new(0);
static POOL_PIXELS: AtomicU64 = AtomicU64::new(0);

/// Counts of what the render-target pool has handed out.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolUsage {
    /// Targets handed out, whether re-used or freshly built.
    pub takes: u64,
    /// Targets that had to be built because no idle one was available.
    pub builds: u64,
    /// Pixels of all the targets handed out.
    pub pixels: u64,
}

impl std::ops::Sub for PoolUsage {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            takes: self.takes - rhs.takes,
            builds: self.builds - rhs.builds,
            pixels: self.pixels - rhs.pixels,
        }
    }
}

/// What the render-target pool has handed out so far. Subtract two readings to
/// measure a stretch of rendering.
pub fn pool_usage() -> PoolUsage {
    PoolUsage {
        takes: POOL_TAKES.load(Ordering::Relaxed),
        builds: POOL_BUILDS.load(Ordering::Relaxed),
        pixels: POOL_PIXELS.load(Ordering::Relaxed),
    }
}

/// Which of the two texture pools a measurement belongs to.
///
/// They behave completely differently and have to be read apart: the main pool
/// serves a handful of size classes and reuses nearly everything, while the
/// offscreen pool's sizes follow the content and it is the one that built
/// 758,880 textures in the client's session.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PoolKind {
    /// Targets a frame is composed from.
    Main,
    /// Scratch space for filters and offscreen draws.
    Offscreen,
}

/// Why a request could not be served from the free list.
///
/// The first three are about the key: this pool is a map from an exact
/// `(size, usage, format, sample_count)` to a free list, so a request whose key
/// is not in the map has to build, and *which part* of the key was new is the
/// whole question. The rest are about a key that exists.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PoolMiss {
    /// A size this pool has never been asked for.
    NewSizeClass,
    /// A size it has been asked for, at a format it has not.
    FormatMismatch,
    /// A size and format it has been asked for, at another sample count.
    SampleCountMismatch,
    /// A size, format and sample count it has been asked for, with other usage
    /// flags.
    UsageMismatch,
    /// A size it held and gave up to stay inside its idle budget, asked for
    /// again. This is the category that says the budget is too small rather
    /// than that the content is too varied.
    EvictedByBudget,
    /// The key was registered and every texture under it was already lent out.
    /// This one is not waste - the frame really is using them all at once.
    FreeListEmpty,
}

pub const POOL_MISS_NAMES: &[&str] = &[
    "new_size_class",
    "format_mismatch",
    "sample_count_mismatch",
    "usage_mismatch",
    "evicted_by_budget",
    "free_list_empty",
];

const MISS_REASONS: usize = 6;
const POOL_KINDS: usize = 2;

#[expect(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static POOL_HITS: [AtomicU64; POOL_KINDS] = [ZERO; POOL_KINDS];
#[expect(clippy::declare_interior_mutable_const)]
const ZERO_ROW: [AtomicU64; MISS_REASONS] = [ZERO; MISS_REASONS];
static POOL_MISSES: [[AtomicU64; MISS_REASONS]; POOL_KINDS] = [ZERO_ROW; POOL_KINDS];
static POOL_MISS_BYTES: [[AtomicU64; MISS_REASONS]; POOL_KINDS] = [ZERO_ROW; POOL_KINDS];
static POOL_EVICTIONS: [AtomicU64; POOL_KINDS] = [ZERO; POOL_KINDS];
static POOL_EVICTED_BYTES: [AtomicU64; POOL_KINDS] = [ZERO; POOL_KINDS];

/// What one pool has done so far. Subtract two readings to measure a stretch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolTelemetry {
    /// Requests served from a free list.
    pub hits: u64,
    /// Requests that had to build, indexed by [`PoolMiss`].
    pub misses: Vec<u64>,
    /// Bytes those builds came to, indexed the same way.
    pub miss_bytes: Vec<u64>,
    /// Idle textures given up to stay inside the budget, and their bytes.
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// The sizes asked for most often, biggest first: `(width, height,
    /// requests, builds)`. Published when the pool is trimmed.
    pub top_sizes: Vec<(u32, u32, u64, u64)>,
    /// How many distinct sizes the pool is holding keys for.
    pub live_size_classes: usize,
    /// Distinct sizes met since the process started.
    pub size_classes_seen: usize,
}

impl PoolTelemetry {
    pub fn misses_total(&self) -> u64 {
        self.misses.iter().sum()
    }
}

/// The published size histograms, one per pool kind. Written when a pool is
/// trimmed rather than on every request, so that taking a texture stays a
/// couple of atomic adds.
static SIZE_REPORTS: Mutex<Option<[PoolSizeReport; POOL_KINDS]>> = Mutex::new(None);

#[derive(Clone, Debug, Default)]
struct PoolSizeReport {
    top_sizes: Vec<(u32, u32, u64, u64)>,
    live_size_classes: usize,
    size_classes_seen: usize,
}

/// What one of the texture pools has done so far.
pub fn pool_telemetry(kind: PoolKind) -> PoolTelemetry {
    let index = kind as usize;
    let report = SIZE_REPORTS
        .lock()
        .ok()
        .and_then(|reports| reports.as_ref().map(|reports| reports[index].clone()))
        .unwrap_or_default();
    PoolTelemetry {
        hits: POOL_HITS[index].load(Ordering::Relaxed),
        misses: POOL_MISSES[index]
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect(),
        miss_bytes: POOL_MISS_BYTES[index]
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect(),
        evictions: POOL_EVICTIONS[index].load(Ordering::Relaxed),
        evicted_bytes: POOL_EVICTED_BYTES[index].load(Ordering::Relaxed),
        top_sizes: report.top_sizes,
        live_size_classes: report.live_size_classes,
        size_classes_seen: report.size_classes_seen,
    }
}

/// What the offscreen pool keeps in idle scratch space.
///
/// Measured rather than guessed. A room of a hundred filtered, animating
/// objects asks this pool for a few hundred targets a frame across a couple of
/// hundred live sizes, and at 64 MiB it spent between a quarter and a half of
/// its misses on sizes it had held and given up moments earlier - it was
/// evicting what the next frame wanted. The sweep in
/// `render/wgpu/tests/cache_churn.rs` is what settled the figure;
/// `RUFFLE_OFFSCREEN_POOL_MB` overrides it so the same build can be measured
/// both ways.
fn offscreen_idle_budget() -> usize {
    const DEFAULT_MB: usize = 192;
    std::env::var("RUFFLE_OFFSCREEN_POOL_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|mb| *mb > 0)
        .unwrap_or(DEFAULT_MB)
        * 1024
        * 1024
}

type PoolInner<T> = Mutex<PoolState<T>>;

/// How many trim intervals of demand a pool remembers. A pool keeps enough
/// entries for the busiest interval in this window, so demand has to have been
/// low for the whole window before anything is released.
const DEMAND_HISTORY: usize = 4;

/// Entries kept beyond observed demand, so a scene that varies a little from
/// interval to interval does not have to rebuild targets.
const DEMAND_HEADROOM: usize = 2;

/// A pool is never trimmed below this, which leaves small pools alone.
const MIN_RETAINED: usize = 4;

/// Excess below this is not worth releasing.
const TRIM_THRESHOLD: usize = 8;

/// A pool's free list, plus what it needs to size itself.
///
/// `borrowed` is the pair to `available`: a pool only ever builds a new entry
/// when its free list is empty, so the number of entries it holds can never
/// exceed the most it has had lent out at once. A pool sitting on hundreds of
/// idle entries is therefore not failing to re-use them - hundreds really were
/// in use at the same time, once.
#[derive(Debug)]
struct PoolState<T> {
    available: Vec<T>,
    borrowed: usize,
    /// Peak borrows during the current trim interval, and the peaks of the
    /// last few intervals, which is what the retained set is sized from.
    interval_peak: usize,
    demand_history: [usize; DEMAND_HISTORY],
}

impl<T> Default for PoolState<T> {
    fn default() -> Self {
        Self {
            available: Vec::new(),
            borrowed: 0,
            interval_peak: 0,
            demand_history: [0; DEMAND_HISTORY],
        }
    }
}

impl<T> PoolState<T> {
    /// Records that an entry has been lent out.
    fn borrow(&mut self) {
        self.borrowed += 1;
        self.interval_peak = self.interval_peak.max(self.borrowed);
    }

    /// Records that a lent entry has come back.
    fn restore(&mut self, entry: T) {
        self.borrowed = self.borrowed.saturating_sub(1);
        self.available.push(entry);
    }

    /// Closes the current demand interval and releases entries the pool has
    /// stopped needing, returning how many were released.
    ///
    /// Sizing on the busiest of the last [`DEMAND_HISTORY`] intervals, and
    /// releasing only half of whatever is above that, means a scene that keeps
    /// needing its entries keeps them, and one whose demand has really gone
    /// gives the memory back over the following intervals rather than in one
    /// step the next frame would have to undo.
    fn trim(&mut self) -> usize {
        let interval_peak = self.interval_peak;
        self.demand_history.rotate_right(1);
        self.demand_history[0] = interval_peak;
        self.interval_peak = self.borrowed;

        let demand = self.demand_history.iter().copied().max().unwrap_or(0);
        let retained = demand.saturating_add(DEMAND_HEADROOM).max(MIN_RETAINED);
        let held = self.available.len() + self.borrowed;
        if held <= retained || held - retained < TRIM_THRESHOLD {
            return 0;
        }

        let release = ((held - retained) / 2).min(self.available.len());
        self.available.truncate(self.available.len() - release);
        release
    }

    /// Whether nothing has asked for this size for a whole demand window.
    ///
    /// Sizes that keep coming back are worth keeping registered, but a size
    /// that has not been wanted for the whole window is not: the offscreen
    /// pool's sizes follow the content, so a long session meets thousands of
    /// them and would otherwise keep [`MIN_RETAINED`] targets for every one it
    /// had ever seen.
    fn is_dormant(&self) -> bool {
        self.borrowed == 0
            && self.interval_peak == 0
            && self.demand_history.iter().all(|&peak| peak == 0)
    }
}
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

/// A pooled render target. Accounted for in the memory report like any other
/// texture Ruffle creates; see `tracked_texture_totals`.
#[derive(Debug)]
pub struct PooledTexture(
    pub wgpu::Texture,
    pub wgpu::TextureView,
    pub(crate) crate::bind_cache::BindGroupCache,
);

impl PooledTexture {
    fn new(texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        crate::track_texture_created(&texture);
        Self(texture, view, crate::bind_cache::BindGroupCache::default())
    }
}

impl Drop for PooledTexture {
    fn drop(&mut self) {
        crate::track_texture_dropped(&self.0);
    }
}

impl TexturePool {
    /// Releases render targets this pool has stopped needing.
    ///
    /// A pool grows to the busiest frame it has ever drawn and, without this,
    /// stays there for the rest of the session: one crowded scene full of
    /// blended objects, each of which is rendered through its own screen-sized
    /// target, leaves hundreds of those targets behind long after the scene is
    /// gone. They are still perfectly re-usable, so the aim is not to stop
    /// pooling but to stop the pool being sized by a burst that ended minutes
    /// ago.
    ///
    /// A size whose free list empties keeps its key, so a size that comes back
    /// does not have to be registered again - unless nothing has wanted it for
    /// a whole demand window, in which case the key goes too. That matters for
    /// the offscreen pool, whose sizes follow the content rather than a handful
    /// of size classes.
    /// Whatever the demand history says, a pool never keeps more than
    /// `idle_budget` in targets nothing is using: the sizes least recently
    /// asked for are given up until it is under it again.
    pub fn trim_idle(&mut self) {
        self.pools
            .retain(|_, size_pool| !size_pool.pool.trim_idle());
        self.enforce_idle_budget();
        self.trim_globals();
        self.publish_size_report();
    }

    /// Gives up the sizes least recently asked for until what is held idle fits
    /// the budget.
    ///
    /// Called whenever a size is met for the first time, not only when the pool
    /// is trimmed: a scene of animating filtered objects can meet two thousand
    /// sizes between one trim and the next, and a bound that applies only every
    /// couple of seconds is not a bound on the peak. Only idle targets are
    /// counted and released - what a frame is using is untouchable.
    fn enforce_idle_budget(&mut self) {
        let idle: Vec<(u64, TextureKey, usize)> = self
            .pools
            .iter()
            .filter(|(_, size_pool)| size_pool.pool.idle_len() > 0)
            .map(|(key, size_pool)| {
                (
                    size_pool.last_used,
                    *key,
                    size_pool.pool.idle_len() * key.bytes(),
                )
            })
            .collect();
        let index = self.kind as usize;
        for key in sizes_over_budget(idle, self.idle_budget) {
            if let Some(size_pool) = self.pools.get_mut(&key) {
                let given_up = size_pool.pool.idle_len();
                size_pool.pool.release_idle();
                POOL_EVICTIONS[index].fetch_add(given_up as u64, Ordering::Relaxed);
                POOL_EVICTED_BYTES[index]
                    .fetch_add((given_up * key.bytes()) as u64, Ordering::Relaxed);
                if !size_pool.pool.is_borrowed() {
                    self.pools.remove(&key);
                    // Remembered so that asking for this size again is
                    // reported as the budget biting rather than as a size the
                    // content had never used.
                    if self.evicted.len() >= 16_384 {
                        self.evicted.clear();
                    }
                    self.evicted.insert(key);
                }
            }
        }
    }
}

/// Which sizes a pool should give up so that what it keeps idle fits `budget`.
///
/// Sizes are given as `(when it was last asked for, the size, what it is
/// holding idle)`. The ones least recently wanted go first, so the sizes the
/// last few frames used are the ones that survive.
fn sizes_over_budget<Key: Copy>(mut sizes: Vec<(u64, Key, usize)>, budget: usize) -> Vec<Key> {
    let mut held: usize = sizes.iter().map(|(_, _, bytes)| bytes).sum();
    if held <= budget {
        return Vec::new();
    }
    sizes.sort_unstable_by_key(|(last_used, _, _)| *last_used);
    let mut give_up = Vec::new();
    for (_, key, bytes) in sizes {
        if held <= budget {
            break;
        }
        held -= bytes;
        give_up.push(key);
    }
    give_up
}

/// One size's free list, and when that size was last asked for.
#[derive(Debug)]
struct SizePool {
    pool: BufferPool<PooledTexture, AlwaysCompatible>,
    last_used: u64,
}

#[derive(Debug)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, SizePool>,
    /// A projection per target size, with whether anything has asked for it
    /// since the last trim.
    globals_cache: FnvHashMap<GlobalsKey, (Arc<Globals>, bool)>,
    /// Ticks on every request, so sizes can be ordered by how recently they
    /// were wanted.
    clock: u64,
    /// The most this pool will keep in idle targets.
    idle_budget: usize,
    /// Which pool this is, so the two can be measured apart.
    kind: PoolKind,
    /// Every size this pool has been asked for, with how many requests and how
    /// many builds it took. This is what says whether the content asks for a
    /// handful of sizes over and over or thousands of nearly identical ones.
    size_history: FnvHashMap<(u32, u32), (u64, u64)>,
    /// Keys this pool gave up to stay inside its budget. A request for one of
    /// these is the budget's fault rather than the content's, and the two want
    /// opposite fixes.
    evicted: FnvHashSet<TextureKey>,
}

impl TexturePool {
    /// A pool for the targets a frame is composed from.
    ///
    /// Those come in size classes, so there are tens of sizes rather than
    /// thousands and demand-aware trimming is enough on its own; the budget is
    /// a backstop, set well above the couple of hundred megabytes a crowded
    /// room's targets come to.
    pub fn new() -> Self {
        Self::with_idle_budget(256 * 1024 * 1024)
    }

    /// A pool for the scratch space filters and offscreen draws run through.
    ///
    /// These sizes follow the content - a filter's targets are exactly its
    /// source's size, which changes whenever a cached object's bounds do - so a
    /// scene of animating filtered objects meets thousands of them. Keeping a
    /// few of every size ever seen is how this pool reached 3,673 sizes and 2.5
    /// GiB in a harness of forty rotating filtered avatars. What is worth
    /// keeping is the handful of sizes the last few frames used, a few
    /// megabytes; the budget is set an order of magnitude above that, and the
    /// sizes given up when it bites are the ones least recently wanted.
    pub fn new_offscreen() -> Self {
        Self::with_kind(offscreen_idle_budget(), PoolKind::Offscreen)
    }

    fn with_idle_budget(idle_budget: usize) -> Self {
        Self::with_kind(idle_budget, PoolKind::Main)
    }

    fn with_kind(idle_budget: usize, kind: PoolKind) -> Self {
        Self {
            pools: Default::default(),
            globals_cache: Default::default(),
            clock: 0,
            idle_budget,
            kind,
            size_history: Default::default(),
            evicted: Default::default(),
        }
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<PooledTexture, AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        self.clock += 1;
        let clock = self.clock;
        // Classified before the entry is made, because the answer depends on
        // what the map held when the request arrived.
        let miss_reason = self.classify_miss(&key);
        let mut fresh_key = false;
        let size_pool = self.pools.entry(key).or_insert_with(|| {
            fresh_key = true;
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            let pool = BufferPool::new(Box::new(move |descriptors, _description| {
                let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: label.as_deref(),
                    size,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage,
                });
                let view = texture.create_view(&Default::default());
                PooledTexture::new(texture, view)
            }));
            SizePool {
                pool,
                last_used: clock,
            }
        });
        size_pool.last_used = clock;
        let served_from_free_list = size_pool.pool.idle_len() > 0;
        let entry = size_pool.pool.take(descriptors, AlwaysCompatible);
        self.record_request(&key, miss_reason, served_from_free_list);
        if fresh_key {
            self.enforce_idle_budget();
        }
        POOL_TAKES.fetch_add(1, Ordering::Relaxed);
        POOL_PIXELS.fetch_add(
            u64::from(size.width) * u64::from(size.height),
            Ordering::Relaxed,
        );
        entry
    }

    /// Why a request for `key` cannot be served from a free list, if it
    /// cannot. `None` means the key is registered and might have one.
    fn classify_miss(&self, key: &TextureKey) -> Option<PoolMiss> {
        if self.pools.contains_key(key) {
            return None;
        }
        if self.evicted.contains(key) {
            return Some(PoolMiss::EvictedByBudget);
        }
        // Which part of the key is the new one. Checked from the most specific
        // outwards, so a request that differs only in usage is not reported as
        // a whole new size.
        let mut size_seen = false;
        let mut format_seen = false;
        let mut samples_seen = false;
        for other in self.pools.keys() {
            if other.size != key.size {
                continue;
            }
            size_seen = true;
            if other.format != key.format {
                continue;
            }
            format_seen = true;
            if other.sample_count != key.sample_count {
                continue;
            }
            samples_seen = true;
        }
        Some(if samples_seen {
            PoolMiss::UsageMismatch
        } else if format_seen {
            PoolMiss::SampleCountMismatch
        } else if size_seen {
            PoolMiss::FormatMismatch
        } else {
            PoolMiss::NewSizeClass
        })
    }

    /// Records one request against this pool's counters and size history.
    fn record_request(
        &mut self,
        key: &TextureKey,
        miss_reason: Option<PoolMiss>,
        served_from_free_list: bool,
    ) {
        let index = self.kind as usize;
        let built = !served_from_free_list;
        let entry = self
            .size_history
            .entry((key.size.width, key.size.height))
            .or_insert((0, 0));
        entry.0 += 1;
        if built {
            entry.1 += 1;
        }
        if !built {
            POOL_HITS[index].fetch_add(1, Ordering::Relaxed);
            return;
        }
        // A registered key whose free list happened to be empty is a different
        // thing from a key that was never there.
        let reason = miss_reason.unwrap_or(PoolMiss::FreeListEmpty);
        POOL_MISSES[index][reason as usize].fetch_add(1, Ordering::Relaxed);
        POOL_MISS_BYTES[index][reason as usize].fetch_add(key.bytes() as u64, Ordering::Relaxed);
    }

    /// Publishes the size histogram so it can be read from outside without
    /// locking anything on the path a frame takes.
    fn publish_size_report(&self) {
        let mut sizes: Vec<(u32, u32, u64, u64)> = self
            .size_history
            .iter()
            .map(|((width, height), (requests, builds))| (*width, *height, *requests, *builds))
            .collect();
        sizes.sort_unstable_by_key(|(_, _, requests, _)| std::cmp::Reverse(*requests));
        sizes.truncate(24);
        let report = PoolSizeReport {
            top_sizes: sizes,
            live_size_classes: self.pools.len(),
            size_classes_seen: self.size_history.len(),
        };
        if let Ok(mut reports) = SIZE_REPORTS.lock() {
            let reports = reports.get_or_insert_with(Default::default);
            reports[self.kind as usize] = report;
        }
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        let entry = self
            .globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                (
                    Arc::new(Globals::new(
                        &descriptors.device,
                        &descriptors.bind_layouts.globals,
                        viewport_width,
                        viewport_height,
                    )),
                    true,
                )
            });
        entry.1 = true;
        entry.0.clone()
    }

    /// Forgets projections for sizes nothing has drawn since the last trim.
    ///
    /// A projection is a 64-byte buffer and a bind group, so a handful of them
    /// costs nothing - but the offscreen pool's sizes follow the content, and
    /// this cache used to be thrown away with the pool every frame. Now that
    /// the pool survives, this has to be bounded the same way the free lists
    /// are, or a long session accumulates one for every size it has ever drawn.
    fn trim_globals(&mut self) {
        self.globals_cache.retain(|_, (globals, used_since_trim)| {
            // Still lent to a live target: keep it whatever the flag says.
            let keep = *used_since_trim || Arc::strong_count(globals) > 1;
            *used_since_trim = false;
            keep
        });
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

impl TextureKey {
    /// What one texture of this size costs, near enough to budget with.
    fn bytes(&self) -> usize {
        let block = self.format.block_copy_size(None).unwrap_or(4) as usize;
        self.size.width as usize
            * self.size.height as usize
            * self.size.depth_or_array_layers as usize
            * block
            * self.sample_count as usize
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;

    /// The lowest cost [`cost_to_use`](Self::cost_to_use) can return.
    ///
    /// An entry that costs this much cannot be beaten, so the search for one
    /// stops as soon as it finds it. Without that, taking an entry scans the
    /// whole free list, and a frame that takes hundreds of render targets
    /// spends its time walking a list that shrinks by one each time.
    fn best_possible_cost() -> Option<Self::Cost> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }

    fn best_possible_cost() -> Option<()> {
        // Every entry in one of these pools is an equally good answer.
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(PoolState::default())),
            constructor,
        }
    }

    /// Releases entries this pool has not needed for a while.
    ///
    /// Sizing on the busiest of the last [`DEMAND_HISTORY`] intervals, and
    /// releasing only half of whatever is above that, means a scene that keeps
    /// needing its entries keeps them, and one whose demand has really gone
    /// gives the memory back over the following intervals rather than in one
    /// step that the next frame would have to undo.
    ///
    /// Returns whether the pool has gone dormant and is worth forgetting
    /// entirely.
    fn trim_idle(&mut self) -> bool {
        let mut state = self.lock();
        state.trim();
        state.is_dormant()
    }

    /// How many entries are sitting unused.
    fn idle_len(&self) -> usize {
        self.lock().available.len()
    }

    /// Whether anything is currently using an entry from this pool.
    fn is_borrowed(&self) -> bool {
        self.lock().borrowed > 0
    }

    /// Gives up every idle entry, for when the pool is over its budget.
    fn release_idle(&mut self) {
        self.lock().available.clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState<(Type, Description)>> {
        self.available
            .lock()
            .expect("Should not be able to lock recursively")
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self.lock();
        guard.borrow();
        let unbeatable = Description::best_possible_cost();
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.available.len() {
            if let Some(cost) = description.cost_to_use(&guard.available[i].1) {
                let unbeatable = Some(&cost) == unbeatable.as_ref();
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else {
                    best = Some((cost, i));
                }
                if unbeatable {
                    break;
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.available.swap_remove(best)
        } else {
            POOL_BUILDS.fetch_add(1, Ordering::Relaxed);
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            pool.lock()
                .expect("Should not be able to lock recursively")
                .restore((item, self.description.clone()));
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool's state with `count` entries lent out and none returned.
    fn with_burst(count: usize) -> PoolState<u32> {
        let mut state = PoolState::default();
        for _ in 0..count {
            state.borrow();
        }
        for i in 0..count {
            state.restore(i as u32);
        }
        state
    }

    /// One interval of `demand` entries being taken and returned.
    fn interval(state: &mut PoolState<u32>, demand: usize) {
        let mut taken = Vec::new();
        for _ in 0..demand {
            state.borrow();
            taken.push(state.available.pop().unwrap_or(0));
        }
        for entry in taken {
            state.restore(entry);
        }
    }

    /// The bug this guards against: a scene that briefly needs many targets at
    /// once leaves the pool holding every one of them for the rest of the
    /// session, however small its demand becomes afterwards.
    #[test]
    fn a_burst_does_not_size_the_pool_forever() {
        let mut state = with_burst(120);
        assert_eq!(state.available.len(), 120, "the burst is served");

        for _ in 0..12 {
            interval(&mut state, 4);
            state.trim();
        }

        let retained = state.available.len();
        assert!(
            retained < 20,
            "pool still holds {retained} targets for an ongoing demand of 4"
        );
        assert!(
            retained >= 4,
            "pool trimmed to {retained}, below the demand it is still serving"
        );
    }

    /// The other half: a scene whose demand has not gone away keeps what it is
    /// using, so trimming never costs a reallocation.
    #[test]
    fn steady_demand_is_never_trimmed() {
        let mut state = with_burst(40);
        for _ in 0..12 {
            interval(&mut state, 40);
            state.trim();
        }
        assert_eq!(
            state.available.len(),
            40,
            "a pool in constant use should keep exactly what it is using"
        );
    }

    /// Releasing is gradual, so a burst that returns shortly afterwards is
    /// still served from the pool instead of rebuilding everything.
    #[test]
    fn trimming_is_gradual() {
        let mut state = with_burst(120);
        for _ in 0..DEMAND_HISTORY {
            interval(&mut state, 2);
            state.trim();
        }
        let retained = state.available.len();
        assert!(
            retained > 20,
            "the first trims dropped {} of 120 at once",
            120 - retained
        );
    }

    /// Demand is remembered for a whole window, so a burst that recurs every
    /// few intervals never has its targets taken away between appearances.
    #[test]
    fn recurring_bursts_keep_their_targets() {
        let mut state = with_burst(60);
        for _ in 0..9 {
            interval(&mut state, 3);
            state.trim();
            interval(&mut state, 60);
            state.trim();
        }
        assert_eq!(
            state.available.len(),
            60,
            "a burst that keeps recurring should keep its targets"
        );
    }

    /// A size the content has stopped using is forgotten, so that a long
    /// session does not keep a few targets for every size it has ever met.
    #[test]
    fn a_size_that_stops_being_used_is_forgotten() {
        let mut state = with_burst(6);
        for _ in 0..DEMAND_HISTORY {
            assert!(!state.is_dormant(), "forgotten while still in demand");
            state.trim();
        }
        state.trim();
        assert!(
            state.is_dormant(),
            "a size unused for a whole window is still registered"
        );
    }

    /// A size still being drawn every interval is never forgotten, however
    /// small its pool.
    #[test]
    fn a_size_in_use_is_never_forgotten() {
        let mut state = with_burst(6);
        for _ in 0..20 {
            interval(&mut state, 1);
            state.trim();
            assert!(!state.is_dormant(), "forgot a size that is still in use");
        }
    }

    /// The budget is the backstop against a pool whose sizes follow the
    /// content: however many of them there are, and whatever the demand history
    /// says about each one on its own, the idle set stays bounded. This is the
    /// case the offscreen pool met - 3,673 sizes holding 2.5 GiB between them.
    #[test]
    fn the_idle_budget_bounds_a_pool_of_many_sizes() {
        const TARGET: usize = 1024 * 1024;
        let sizes: Vec<(u64, u32, usize)> = (0..3673u32)
            .map(|i| (u64::from(i), i, 4 * TARGET))
            .collect();
        let budget = 64 * TARGET;
        let given_up = sizes_over_budget(sizes.clone(), budget);

        let kept: usize = sizes
            .iter()
            .filter(|(_, key, _)| !given_up.contains(key))
            .map(|(_, _, bytes)| bytes)
            .sum();
        assert!(
            kept <= budget,
            "kept {kept} bytes against a {budget} byte budget"
        );
        // What survives is what was wanted most recently.
        assert!(
            !given_up.contains(&3672),
            "gave up the size the last frame used"
        );
        assert!(
            given_up.contains(&0),
            "kept the size nothing has used since"
        );
    }

    /// A pool inside its budget is left entirely alone, so the sizes a scene is
    /// cycling through are never taken away from it.
    #[test]
    fn a_pool_inside_its_budget_gives_up_nothing() {
        let sizes: Vec<(u64, u32, usize)> = (0..20u32).map(|i| (u64::from(i), i, 1024)).collect();
        assert!(sizes_over_budget(sizes, 64 * 1024).is_empty());
    }

    /// Small pools are left alone entirely.
    #[test]
    fn small_pools_are_left_alone() {
        let mut state = with_burst(6);
        for _ in 0..12 {
            interval(&mut state, 1);
            assert_eq!(state.trim(), 0, "a six-entry pool should not be trimmed");
        }
        assert_eq!(state.available.len(), 6);
    }
}
