use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
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
pub struct PooledTexture(pub wgpu::Texture, pub wgpu::TextureView);

impl PooledTexture {
    fn new(texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        crate::track_texture_created(&texture);
        Self(texture, view)
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
    pub fn trim_idle(&mut self) {
        self.pools.retain(|_, pool| !pool.trim_idle());
        self.trim_globals();
    }
}

#[derive(Debug, Default)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, BufferPool<PooledTexture, AlwaysCompatible>>,
    /// A projection per target size, with whether anything has asked for it
    /// since the last trim.
    globals_cache: FnvHashMap<GlobalsKey, (Arc<Globals>, bool)>,
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
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
        let pool = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            BufferPool::new(Box::new(move |descriptors, _description| {
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
            }))
        });
        POOL_TAKES.fetch_add(1, Ordering::Relaxed);
        POOL_PIXELS.fetch_add(
            u64::from(size.width) * u64::from(size.height),
            Ordering::Relaxed,
        );
        pool.take(descriptors, AlwaysCompatible)
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
