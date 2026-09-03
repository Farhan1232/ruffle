use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<PoolState<T>>;

/// A pool's free list plus the demand figures the memory report needs.
///
/// `borrowed` and `peak_borrowed` are the important pair. A pool only ever
/// builds a new entry when its free list is empty, so the number of entries it
/// holds can never exceed the most it has ever had lent out at once - which
/// means a pool sitting on hundreds of idle entries is evidence that hundreds
/// really were in use simultaneously, not that look-up failed to re-use them.
/// How many trim intervals of demand a pool remembers. A pool keeps enough
/// entries for the busiest interval in this window, so demand has to stay low
/// for the whole window before anything is released.
const DEMAND_HISTORY: usize = 4;

/// Entries kept beyond observed demand, so that a scene which varies a little
/// from interval to interval does not have to rebuild targets.
const DEMAND_HEADROOM: usize = 2;

/// A pool never trims below this, so small pools are left alone entirely.
const MIN_RETAINED: usize = 4;

/// Excess below this is not worth releasing.
const TRIM_THRESHOLD: usize = 8;

#[derive(Debug)]
pub(crate) struct PoolState<T> {
    pub(crate) available: Vec<T>,
    borrowed: usize,
    peak_borrowed: usize,
    /// Peak borrows during the current trim interval, and the peaks of the
    /// last few intervals. The retained working set is sized from these, so it
    /// follows real demand instead of the highest burst of the session.
    interval_peak: usize,
    demand_history: [usize; DEMAND_HISTORY],
    /// Peak since the last report, so a policy can see present demand rather
    /// than the highest burst of the whole session.
    recent_peak_borrowed: usize,
    reuses: u64,
    misses_pool_empty: u64,
    misses_new_key: u64,
    /// What the last trim decided to keep, so the report can show the target
    /// beside the count it actually holds.
    retained_target: usize,
}

impl<T> Default for PoolState<T> {
    fn default() -> Self {
        Self {
            available: Vec::new(),
            borrowed: 0,
            peak_borrowed: 0,
            interval_peak: 0,
            demand_history: [0; DEMAND_HISTORY],
            recent_peak_borrowed: 0,
            reuses: 0,
            misses_pool_empty: 0,
            misses_new_key: 0,
            retained_target: 0,
        }
    }
}

impl<T> PoolState<T> {
    /// Records that an entry has been lent out.
    fn borrow(&mut self) {
        self.borrowed += 1;
        self.peak_borrowed = self.peak_borrowed.max(self.borrowed);
        self.recent_peak_borrowed = self.recent_peak_borrowed.max(self.borrowed);
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
        self.retained_target = retained;
        let held = self.available.len() + self.borrowed;
        if held <= retained || held - retained < TRIM_THRESHOLD {
            return 0;
        }

        let release = ((held - retained) / 2).min(self.available.len());
        self.available.truncate(self.available.len() - release);
        release
    }
}
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

/// A pooled render target. Accounted for in the memory report like any other
/// texture Ruffle creates; see `tracked_texture_totals`.
#[derive(Debug)]
pub struct PooledTexture(
    pub wgpu::Texture,
    pub wgpu::TextureView,
    pub(crate) crate::TextureKind,
);

impl PooledTexture {
    fn new(texture: wgpu::Texture, view: wgpu::TextureView, kind: crate::TextureKind) -> Self {
        crate::track_texture_created(&texture, kind);
        Self(texture, view, kind)
    }
}

impl Drop for PooledTexture {
    fn drop(&mut self) {
        crate::track_texture_dropped(&self.0, self.2);
    }
}

#[derive(Debug)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, BufferPool<PooledTexture, AlwaysCompatible>>,
    globals_cache: FnvHashMap<GlobalsKey, Arc<Globals>>,
    /// Which pool this is, so the memory report can tell the surface pool
    /// (which lives across frames) from the offscreen one (which the renderer
    /// replaces every frame). Reporting only; it changes nothing.
    kind: crate::TextureKind,
}

/// One pool key held by a [`TexturePool`], for the memory report.
///
/// This reports the *whole* key the pool is looked up by, not just the
/// dimensions: two keys with identical width, height and sample count are
/// still separate pools if their usage flags or format differ, and reporting
/// only the size makes those look like one pool that is failing to re-use
/// itself.
#[derive(Clone, Copy, Debug)]
pub struct PoolSizeClass {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    /// Entries sitting unused in this key's free list.
    pub idle_entries: usize,
    pub idle_bytes: usize,
    /// Entries currently lent out for this key.
    pub borrowed: usize,
    /// The most that were lent out at once for this key, ever. A pool can
    /// only ever grow to this, so it is also the total the key holds.
    pub peak_borrowed: usize,
    /// The most lent out at once since the previous report, which is what a
    /// retention policy would have to respect.
    pub recent_peak_borrowed: usize,
    /// Requests served from the free list, and requests that had to build a
    /// new texture because the free list was empty.
    pub reuses: u64,
    pub misses_pool_empty: u64,
    /// Requests that created this key's pool in the first place.
    pub misses_new_key: u64,
    /// What the last trim decided this key should keep. Comparing it with
    /// `idle_entries` is what shows a crowded scene's surplus being released.
    pub retained_target: usize,
}

impl TexturePool {
    /// Releases render targets this pool has stopped needing, returning the
    /// bytes released. Keys are kept even when their free list empties, so a
    /// size that comes back does not have to be re-registered.
    pub fn trim_idle(&mut self) -> usize {
        self.pools
            .values_mut()
            .map(|pool| pool.trim_idle(|texture| crate::texture_bytes(&texture.0)))
            .sum()
    }

    pub fn new(kind: crate::TextureKind) -> Self {
        Self {
            pools: Default::default(),
            globals_cache: Default::default(),
            kind,
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
        let kind = self.kind;
        let mut fresh_key = false;
        let pool = self.pools.entry(key).or_insert_with(|| {
            fresh_key = true;
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
                PooledTexture::new(texture, view, kind)
            }))
        });
        if fresh_key {
            pool.note_new_key();
        }
        pool.take(descriptors, AlwaysCompatible)
    }

    /// `(distinct sizes pooled, textures idle in free lists, their bytes)`.
    ///
    /// A texture is idle when nothing is currently borrowing it: it has been
    /// returned to its pool and is being kept for reuse. This is what the
    /// pool retains between frames, as opposed to what a frame is using, and
    /// so is the number that says whether a high working set is the
    /// renderer's pooling rather than live content.
    pub fn idle_totals(&self) -> (usize, usize, usize) {
        let mut textures = 0;
        let mut bytes = 0;
        for pool in self.pools.values() {
            for (texture, _) in pool.available().iter() {
                textures += 1;
                bytes += crate::texture_bytes(&texture.0);
            }
        }
        (self.pools.len(), textures, bytes)
    }

    /// Every size class this pool holds, heaviest first. Lets the report name
    /// the sizes that are actually retaining memory rather than only totalling
    /// them.
    pub fn size_classes(&mut self) -> Vec<PoolSizeClass> {
        let mut classes: Vec<_> = self
            .pools
            .iter_mut()
            .map(|(key, pool)| {
                let stats = pool.stats();
                let available = pool.available();
                let idle_bytes = available
                    .iter()
                    .map(|(texture, _)| crate::texture_bytes(&texture.0))
                    .sum();
                PoolSizeClass {
                    width: key.size.width,
                    height: key.size.height,
                    sample_count: key.sample_count,
                    format: key.format,
                    usage: key.usage,
                    idle_entries: available.len(),
                    idle_bytes,
                    borrowed: stats.0,
                    peak_borrowed: stats.1,
                    recent_peak_borrowed: stats.2,
                    reuses: stats.3,
                    misses_pool_empty: stats.4,
                    misses_new_key: stats.5,
                    retained_target: stats.6,
                }
            })
            .collect();
        classes.sort_by(|a, b| b.idle_bytes.cmp(&a.idle_bytes));
        classes
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        self.globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                Arc::new(Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport_width,
                    viewport_height,
                ))
            })
            .clone()
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
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
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

    /// Records that this pool was created to serve a request, so the report
    /// can tell a brand new key from one whose free list simply ran dry.
    pub(crate) fn note_new_key(&self) {
        self.lock().misses_new_key += 1;
    }

    /// Releases idle entries a pool has not needed for a while, and returns
    /// the bytes released.
    ///
    /// A pool grows to the busiest moment it has ever seen and, with nothing
    /// like this, stays there for the rest of the session: one crowded room
    /// full of blended avatars leaves hundreds of screen-sized targets behind
    /// long after the room empties. The entries are still perfectly reusable,
    /// so the aim is not to stop pooling but to stop the pool being sized by
    /// a burst that has been over for minutes.
    ///
    /// Sizing on the busiest of the last [`DEMAND_HISTORY`] intervals, and
    /// releasing only half of whatever is above that, means a scene which
    /// keeps needing the entries keeps them, and one whose demand has really
    /// gone gives the memory back over the following intervals rather than
    /// all at once.
    /// Releases entries this pool has not needed for a while, returning the
    /// bytes released so the report can show what a trim recovered.
    pub(crate) fn trim_idle(&mut self, size_of: impl Fn(&Type) -> usize) -> usize {
        let mut state = self.lock();
        let before: usize = state.available.iter().map(|(item, _)| size_of(item)).sum();
        let released = state.trim();
        if released == 0 {
            return 0;
        }
        let after: usize = state.available.iter().map(|(item, _)| size_of(item)).sum();
        before.saturating_sub(after)
    }

    /// `(borrowed, peak, recent peak, reuses, empty-misses, new-key misses)`,
    /// resetting the recent peak so the next report covers the next interval.
    pub(crate) fn stats(&mut self) -> (usize, usize, usize, u64, u64, u64, usize) {
        let mut state = self.lock();
        let recent = state.recent_peak_borrowed;
        state.recent_peak_borrowed = state.borrowed;
        (
            state.borrowed,
            state.peak_borrowed,
            recent,
            state.reuses,
            state.misses_pool_empty,
            state.misses_new_key,
            state.retained_target,
        )
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState<(Type, Description)>> {
        self.available
            .lock()
            .expect("Should not be able to lock recursively")
    }

    /// The entries sitting unused in this pool, kept for reuse.
    pub(crate) fn available(&self) -> AvailableGuard<'_, Type, Description> {
        AvailableGuard(self.lock())
    }

    /// How many entries are idle in this pool, and the bytes they describe.
    /// `size_of` maps a description to its byte size, since only the caller
    /// knows how its descriptions are measured.
    pub fn idle_totals(&self, size_of: impl Fn(&Description) -> usize) -> (usize, usize) {
        let guard = self.lock();
        (
            guard.available.len(),
            guard.available.iter().map(|(_, d)| size_of(d)).sum(),
        )
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self.lock();
        guard.borrow();
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.available.len() {
            if let Some(cost) = description.cost_to_use(&guard.available[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            crate::texture_stats_record_pool_reuse();
            guard.reuses += 1;
            guard.available.swap_remove(best)
        } else {
            // The free list was empty. With `AlwaysCompatible` every entry in
            // it matches, so this is the only way a request can miss: nothing
            // is ever rejected for being busy, fenced or in the wrong state.
            crate::texture_stats_record_pool_miss();
            guard.misses_pool_empty += 1;
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

/// Read-only view of a pool's free list, so the memory report can walk it
/// without seeing the rest of the pool's state.
pub(crate) struct AvailableGuard<'a, Type, Description: BufferDescription>(
    std::sync::MutexGuard<'a, PoolState<(Type, Description)>>,
);

impl<Type, Description: BufferDescription> Deref for AvailableGuard<'_, Type, Description> {
    type Target = Vec<(Type, Description)>;

    fn deref(&self) -> &Self::Target {
        &self.0.available
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool's state with `count` entries lent out and then returned, which
    /// is the shape of a frame that renders `count` blended objects.
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
