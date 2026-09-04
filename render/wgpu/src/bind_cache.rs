//! Bind groups kept with the texture they describe.
//!
//! A blend composites its render target back through a bind group, and a
//! complex blend needs a second one pairing that target with the destination it
//! reads. Both were built fresh every frame, so a crowded room built one or two
//! per blended object per frame - hundreds of `create_bind_group` calls a frame,
//! which is part of the fixed price per target that content-bounded targets
//! could not touch.
//!
//! The targets themselves come from a pool and outlive the frame, so the bind
//! groups can too. Keeping them *on* the pooled texture is what makes that safe:
//! a bind group cannot outlive the view it names, because it is dropped with it.
//!
//! Bind groups that name two textures - a blend reading its destination, an
//! alpha mask reading its mask - are keyed on the other texture's id. Ids are
//! unique for the life of the process and are never reissued, so a hit means
//! the partner really is the same texture, not a different one that happens to
//! sit at the same address.

use std::cell::{OnceCell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies one texture for as long as the process runs.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_texture_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The bind groups a texture can be asked for, built on demand and kept.
#[derive(Debug)]
pub struct BindGroupCache {
    id: u64,
    /// Samples this whole texture: the layout and sampler never vary, so one
    /// bind group serves for the texture's whole life.
    whole: OnceCell<wgpu::BindGroup>,
    /// Pairs this texture with another. Only the most recent partner is kept -
    /// a target is composited onto one destination at a time, and the same one
    /// frame after frame.
    paired: RefCell<Option<(u64, wgpu::BindGroup)>>,
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self {
            id: next_texture_id(),
            whole: OnceCell::new(),
            paired: RefCell::new(None),
        }
    }
}

impl BindGroupCache {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The bind group that samples the whole texture, built once.
    pub fn whole(&self, build: impl FnOnce() -> wgpu::BindGroup) -> &wgpu::BindGroup {
        let mut hit = true;
        let group = self.whole.get_or_init(|| {
            hit = false;
            crate::render_stats::record_bind_group_created();
            build()
        });
        crate::render_stats::record_bind_group_cache(hit);
        group
    }

    /// The bind group pairing this texture with `partner`, rebuilt only when
    /// the partner changes.
    ///
    /// Returns a clone rather than a reference because the entry can be
    /// replaced; `wgpu::BindGroup` is a cheap handle, so this costs a refcount
    /// rather than a rebuild.
    pub fn paired(&self, partner: u64, build: impl FnOnce() -> wgpu::BindGroup) -> wgpu::BindGroup {
        let mut cache = self.paired.borrow_mut();
        if let Some((cached_partner, group)) = cache.as_ref()
            && *cached_partner == partner
        {
            crate::render_stats::record_bind_group_cache(true);
            return group.clone();
        }
        crate::render_stats::record_bind_group_cache(false);
        crate::render_stats::record_bind_group_created();
        let group = build();
        *cache = Some((partner, group.clone()));
        group
    }
}
