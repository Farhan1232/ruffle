//! Physical capacity for renderer-owned cache textures, kept separate from the
//! logical picture inside them.
//!
//! An AdventureQuest Worlds avatar does not hold still. Its bounds breathe by a
//! pixel or two every frame as the animation plays - 147x196, 151x198, 149x197 -
//! and a cache that keys its texture on the exact size rebuilds it every time.
//! That is what 621,413 `cacheAsBitmap` textures in 43 minutes are made of: not
//! an object changing shape, but an object being drawn.
//!
//! The fix is to let the texture be a little bigger than the picture. The
//! texture becomes a **capacity**, rounded up to a granularity coarse enough to
//! absorb the breathing; the picture stays exactly the size it was, drawn into
//! the top-left corner, and everything downstream is told the logical size
//! rather than the texture's.
//!
//! ## This is not the padding design that broke `displacement_map`
//!
//! An earlier experiment made the texture bigger and let the rest of the
//! renderer keep asking the texture how big it was. Filters then read the
//! padding, and a displacement map - which samples by coordinate rather than by
//! neighbourhood - read it hardest. The rule that avoids it is that **no
//! consumer may take its extent from the texture**: the logical size travels
//! with the handle, the surface that redraws the cache is built at the logical
//! size, and the filter chain is given a `FilterSource` of the logical rectangle
//! rather than of the whole texture. Unused physical pixels are never sampled
//! and never contribute to the output.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Whether a cache texture may be kept when the picture inside it changes size.
///
/// On. It is here so the same scene can be rendered with it and without it and
/// the two frames compared pixel for pixel, which is how it is tested, and so
/// that a build in the field can be told to behave exactly as the one before it
/// did without being rebuilt: `RUFFLE_CACHE_CAPACITY=0`.
static CAPACITY_REUSE: LazyLock<AtomicBool> =
    LazyLock::new(|| AtomicBool::new(env_flag("RUFFLE_CACHE_CAPACITY", true)));

/// The rounding, in pixels. Sizes are rounded up to a multiple of this.
/// `RUFFLE_CACHE_GRANULARITY` overrides it.
static GRANULARITY: LazyLock<AtomicU32> = LazyLock::new(|| {
    AtomicU32::new(
        std::env::var("RUFFLE_CACHE_GRANULARITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|granularity| *granularity > 0)
            .unwrap_or(DEFAULT_GRANULARITY),
    )
});

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(value.trim(), "0" | "false" | "off" | "no"),
        Err(_) => default,
    }
}

/// 32 pixels, chosen by measurement rather than by taste.
///
/// `the_granularity_is_chosen_by_measurement` in
/// `render/wgpu/tests/cache_churn.rs` sweeps it over a hundred cached objects
/// whose bounds wander - the worst case, harder than the looping animation a
/// real avatar plays. Rebuilds per object per frame:
///
/// | rounding | rebuilds | live texture |
/// |---|---|---|
/// | exact | 1.00 | 260.6 MB |
/// | 8 | 0.83 | 260.6 MB |
/// | 16 | 0.45 | 261.7 MB |
/// | **32** | **0.03** | **265.1 MB** |
/// | 64 | 0.01 | 268.9 MB |
/// | 128 | 0.00 | 264.5 MB |
///
/// 32 is the knee. Below it most of the rebuilds survive - a few pixels of
/// breathing still crosses a bucket boundary - and above it there is almost
/// nothing left to remove, so the extra texture buys nothing.
pub const DEFAULT_GRANULARITY: u32 = 32;

pub fn capacity_reuse_enabled() -> bool {
    CAPACITY_REUSE.load(Ordering::Relaxed)
}

pub fn set_capacity_reuse_enabled(enabled: bool) {
    CAPACITY_REUSE.store(enabled, Ordering::Relaxed);
}

pub fn granularity() -> u32 {
    GRANULARITY.load(Ordering::Relaxed).max(1)
}

pub fn set_granularity(granularity: u32) {
    GRANULARITY.store(granularity.max(1), Ordering::Relaxed);
}

/// Rounds one dimension up to the next multiple of the granularity.
fn round_up(value: u32, granularity: u32) -> u32 {
    // Saturating, so a dimension near `u32::MAX` cannot wrap to something
    // small. The caller has already refused sizes a texture could not be.
    match value.checked_next_multiple_of(granularity) {
        Some(rounded) => rounded.max(granularity),
        None => value,
    }
}

/// The texture to build for a picture of this size.
///
/// With reuse off this is the picture's own size, which is what makes the
/// switch a true A/B: the old behaviour is the new one with the rounding
/// removed.
pub fn capacity_for(width: u32, height: u32) -> (u32, u32) {
    if !capacity_reuse_enabled() {
        return (width, height);
    }
    let granularity = granularity();
    (round_up(width, granularity), round_up(height, granularity))
}

/// Whether a texture of `physical` size should be kept for a picture of
/// `logical` size.
///
/// Two things have to hold. The picture must fit - otherwise there is nothing
/// to draw it into. And the texture must not be extravagantly larger than the
/// picture needs, or an object that was briefly huge would pin that memory for
/// the rest of the session. The slack of one granularity step is the hysteresis:
/// it means an object breathing across a bucket boundary keeps its texture
/// instead of rebuilding on alternate frames, which is the exact failure the
/// rounding is here to prevent.
pub fn capacity_fits(physical: (u32, u32), logical: (u32, u32)) -> bool {
    if !capacity_reuse_enabled() {
        return physical == logical;
    }
    if logical.0 > physical.0 || logical.1 > physical.1 {
        return false;
    }
    let granularity = granularity();
    let wanted = capacity_for(logical.0, logical.1);
    physical.0 <= wanted.0.saturating_add(granularity)
        && physical.1 <= wanted.1.saturating_add(granularity)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Restores the switches whatever a test does to them, since they are
    /// process-wide.
    struct Restore(bool, u32);

    impl Restore {
        fn new() -> Self {
            Self(capacity_reuse_enabled(), granularity())
        }
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            set_capacity_reuse_enabled(self.0);
            set_granularity(self.1);
        }
    }

    /// The case the whole thing exists for: an avatar whose bounds breathe by a
    /// few pixels a frame is drawn into one texture, not five.
    #[test]
    fn a_breathing_avatar_keeps_one_texture() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(true);
        set_granularity(DEFAULT_GRANULARITY);

        let sizes = [(147, 196), (151, 198), (149, 197), (153, 199), (147, 196)];
        let physical = capacity_for(sizes[0].0, sizes[0].1);
        for size in sizes {
            assert!(
                capacity_fits(physical, size),
                "{size:?} did not fit the {physical:?} texture the first frame built"
            );
        }
    }

    /// The picture always fits inside the texture, so nothing is ever clipped.
    #[test]
    fn a_capacity_always_holds_its_content() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(true);
        for granularity in [1, 4, 16, 32, 64] {
            set_granularity(granularity);
            for width in [1u32, 7, 63, 147, 256, 1023, 4096] {
                for height in [1u32, 3, 100, 196, 512, 2047] {
                    let (pw, ph) = capacity_for(width, height);
                    assert!(pw >= width && ph >= height, "{width}x{height} did not fit");
                    assert!(
                        pw < width + granularity && ph < height + granularity,
                        "{width}x{height} rounded to {pw}x{ph}, more than one step"
                    );
                }
            }
        }
    }

    /// An object that grows past its texture gets a new one, so growth is never
    /// silently clipped.
    #[test]
    fn growing_past_the_capacity_rebuilds() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(true);
        set_granularity(32);
        let physical = capacity_for(147, 196); // 160x224
        assert!(
            !capacity_fits(physical, (161, 196)),
            "wider than the texture"
        );
        assert!(
            !capacity_fits(physical, (147, 225)),
            "taller than the texture"
        );
    }

    /// And one that shrinks a long way gives the memory back rather than
    /// sitting on a texture sized by a burst that is over.
    #[test]
    fn shrinking_a_long_way_gives_the_memory_back() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(true);
        set_granularity(32);
        let physical = capacity_for(600, 600);
        assert!(
            !capacity_fits(physical, (40, 40)),
            "a 608x608 texture was kept for a 40x40 picture"
        );
    }

    /// The hysteresis: a size sitting exactly on a bucket boundary must not
    /// rebuild on alternate frames, which would be the churn all over again.
    #[test]
    fn a_size_on_a_bucket_boundary_does_not_alternate() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(true);
        set_granularity(32);
        // 160 rounds to itself; 161 rounds to 192. Without the slack, the
        // 192-wide texture would be judged too big the moment the object
        // stepped back to 160.
        let physical = capacity_for(161, 100);
        assert_eq!(physical.0, 192);
        assert!(
            capacity_fits(physical, (160, 100)),
            "stepping back over a bucket boundary rebuilt the texture"
        );
    }

    /// With the switch off, the policy is exactly the old one, which is what
    /// makes the A/B comparison meaningful.
    #[test]
    fn the_switch_restores_the_old_behaviour() {
        let _restore = Restore::new();
        set_capacity_reuse_enabled(false);
        assert_eq!(capacity_for(147, 196), (147, 196));
        assert!(capacity_fits((147, 196), (147, 196)));
        assert!(!capacity_fits((160, 224), (147, 196)));
    }
}
