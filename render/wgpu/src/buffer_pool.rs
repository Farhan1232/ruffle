use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<Vec<T>>;
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

/// One size class held by a [`TexturePool`], for the memory report.
#[derive(Clone, Copy, Debug)]
pub struct PoolSizeClass {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    /// Entries sitting unused in this size's free list.
    pub idle_entries: usize,
    pub idle_bytes: usize,
}

impl TexturePool {
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
                PooledTexture::new(texture, view, kind)
            }))
        });
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
    pub fn size_classes(&self) -> Vec<PoolSizeClass> {
        let mut classes: Vec<_> = self
            .pools
            .iter()
            .map(|(key, pool)| {
                let available = pool.available();
                let idle_bytes = available
                    .iter()
                    .map(|(texture, _)| crate::texture_bytes(&texture.0))
                    .sum();
                PoolSizeClass {
                    width: key.size.width,
                    height: key.size.height,
                    sample_count: key.sample_count,
                    idle_entries: available.len(),
                    idle_bytes,
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
            available: Arc::new(Mutex::new(vec![])),
            constructor,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(Type, Description)>> {
        self.available
            .lock()
            .expect("Should not be able to lock recursively")
    }

    /// The entries sitting unused in this pool, kept for reuse.
    pub(crate) fn available(&self) -> std::sync::MutexGuard<'_, Vec<(Type, Description)>> {
        self.lock()
    }

    /// How many entries are idle in this pool, and the bytes they describe.
    /// `size_of` maps a description to its byte size, since only the caller
    /// knows how its descriptions are measured.
    pub fn idle_totals(&self, size_of: impl Fn(&Description) -> usize) -> (usize, usize) {
        let guard = self.lock();
        (guard.len(), guard.iter().map(|(_, d)| size_of(d)).sum())
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self.lock();
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
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
            guard.swap_remove(best)
        } else {
            crate::texture_stats_record_pool_miss();
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
                .push((item, self.description.clone()))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}
