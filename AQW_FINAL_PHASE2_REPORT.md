# AQW renderer, phase 2: what a room of animating avatars allocates

Phase 1 took the fixed per-blend cost out of a crowded room. Phase 2 was the
other half of the client's problem: 1.4 million textures and 207 GiB of
allocation traffic in a 43-minute session that never held more than about
112 MB at once. This is what was found, what was built, what it measured, and
what is left.

## Where to start

| | |
|---|---|
| Phase 1 SHA used | `10519f326682f9ca22df7e8e84b37aad28060b2d` - verified against the handoff before anything was changed |
| Production branch | `fix/aqw-render-final-performance` |
| Production SHA | `4174140feef907c11d0b23cd3c4d22b8fca19ce4` |
| Diagnostic branch | `diagnostic/aqw-final-windows` |
| Instrumentation ID | `aqw-final-diag-4` |
| Local GPU | Intel HD Graphics 4400, Mesa 25.2, Vulkan. Weak and thermally noisy, so the numbers to trust are the counters; frame times are only quoted from runs that measure both builds a few seconds apart. |

## Phase 1 was verified first, and is preserved

`git rev-parse HEAD` was `10519f326…` before any change, the tree was clean, and
the branch matched `fork/`. The handoff was read in full. Everything it lists as
preserved still is, and the benchmark reproduces its tables to the digit:

| scene, 800 objects | passes before | after | phase 1 recorded |
|---|---|---|---|
| `Layer` groups | 804 | 13 | 61.8x |
| `Multiply` groups | 1600 | 75 | 21.3x |
| room 60% cached / 10% complex | 488 | 23 | 21.2x |
| room 0% / 20% | 968 | 60 | 16.1x |
| room 0% / 100% | 1600 | 185 | 8.6x |
| direct fast path | 4 passes, 7.2 MB | unchanged | unchanged |

`cargo test -p ruffle_render_wgpu --lib` is 20 passed, `--test blend_pages` is 10
passed, including `a_texel_boundary_picks_a_neighbouring_texel_at_worst`.

## 1. The cacheAsBitmap root cause

`BitmapCache::update` kept the texture it had only when the new size was
**exactly** the old one:

```rust
if current.info.width == actual_width && current.info.height == actual_height {
    return; // No need to resize it
}
```

An AdventureQuest Worlds avatar does not hold still. Its bounds move by a pixel
or two every frame as the animation plays - 147x196, 151x198, 149x197 - so that
test failed constantly and the cache built a new texture. Measured on a room of
a hundred animating cached objects: **1.00 texture per object per frame**, 2,400
a second, 275.9 MB a second of allocation. That is the whole of the client's
621,413.

`create_empty_texture` has exactly one production caller - this one. `BitmapData`
goes through `register_bitmap`, which uploads pixels. Nothing else was at risk.

### Cache rebuild reasons

Counted per reason in `ruffle_render::cache_stats`, split into why a cache
decided it was out of date and why redrawing it also needed a new texture. The
brief's `DEVICE_SCALE_CHANGE`, `FILTER_DIRTY` and `FORMAT_CHANGE` are not offered:
this cache stores four matrix terms and a source size and compares those, so
those categories would read zero forever and a column that can only be zero is
worse than no column.

| invalidation | what it is | CSV column |
|---|---|---|
| `first_allocation` | no texture yet | `cache_dirty_first_allocation` |
| `transform_change` | scale or skew moved; pure translation is deliberately ignored | `cache_dirty_transform_change` |
| `source_size_change` | the object's own bounds changed | `cache_dirty_source_size_change` |
| `content_dirty` | a child moved, or the filter list changed | `cache_dirty_content_dirty` |

| allocation | what it is | CSV column |
|---|---|---|
| `first_allocation` | there was no texture to keep | `cache_alloc_first_allocation` |
| `width_exceeded` | wider than the texture it had | `cache_alloc_width_exceeded` |
| `height_exceeded` | taller | `cache_alloc_height_exceeded` |
| **`shrank`** | **fits inside the texture it had, and it was rebuilt anyway** | `cache_alloc_shrank` |
| `refused` | too large to cache, or the renderer said no | `cache_alloc_refused` |

`shrank` is the thrashing category. Before this work an avatar breathing between
147x196 and 151x198 landed there on every frame that made it smaller.

## 2. The fix: physical capacity, logical picture

A cache texture is now a **capacity**. The picture keeps its exact size; the
texture is rounded up to a multiple of 32 pixels and kept while the picture still
fits and has not shrunk more than one step below it - the step is the hysteresis,
so a size sitting on a bucket boundary does not rebuild on alternate frames.

`render/src/cache_capacity.rs` holds the policy, `capacity_for` and
`capacity_fits`, with a switch (`RUFFLE_CACHE_CAPACITY=0`) so one build can be
run both ways.

### The rounding was measured, not chosen

A hundred cached objects whose bounds *wander and never repeat* - harder than the
looping animation a real avatar plays:

| rounding | rebuilds per object per frame | live texture |
|---|---|---|
| exact | 1.00 | 260.6 MB |
| 8 | 0.83 | 260.6 MB |
| 16 | 0.45 | 261.7 MB |
| **32** | **0.03** | **265.1 MB** |
| 64 | 0.01 | 268.9 MB |
| 128 | 0.00 | 264.5 MB |

32 is the knee. Below it most of the rebuilds survive; above it there is almost
nothing left to remove.

### Why this is not the padding design that broke `displacement_map`

The earlier experiment made the texture bigger and let the rest of the renderer
keep asking the texture how big it was. The rule that avoids it is that **no
consumer takes its extent from the texture**:

* the picture's size travels with the cache entry (`BitmapCacheEntry::logical_width`);
* the surface that redraws it is built at that size, and its passes are held to
  the picture's rectangle by a viewport and a scissor (`CommandTarget::apply_viewport`);
* the stencil attachment follows the *texture*, not the picture, because a render
  pass requires every attachment to be the same size. This was a real defect in
  the first version of the design, caught by reading the code rather than by a
  failure: a stencil sized on the picture would have failed wgpu validation the
  first time a cached object with a mask inside it was drawn, which in AQW is
  routine. `a_padded_cache_with_a_mask_inside_it` and the 29 mask image tests
  cover it now;
* the composite samples the picture's rectangle through the `PixelRegion` the
  renderer already had, so that path needed no change at all;
* **filters are never handed a rectangle of a larger texture.**

That last point is the one that matters, and it was not a theory. The first
attempt did pass filters a sub-rectangle, and `visual/filters/any_blur_scales_with_screen`
failed. The cause: `glow.rs` binds the source *and* the blurred copy of it to one
pass and samples both with the same coordinates. The blurred copy is a target of
its own, exactly the size of the picture; the source was now larger, so the two
sets of coordinates were on different scales. `displacement_map` would have been
worse - it takes `source_width` and `source_height` straight from the texture.

So a filtered cache is drawn into a target of exactly its own size and filtered
from there. It costs nothing: the filtered result was always going to be copied
back into the cache texture, and that is the same copy.

### Stale pixels

A recycled or kept texture cannot show what was in it before. Every redraw goes
through `Surface::draw_commands` with `RenderTargetMode::ExistingWithColor`,
whose first pass loads the attachment with `LoadOp::Clear` over the whole
texture, and which ends with `ensure_cleared` so that a cache drawing nothing is
still cleared. `a_kept_texture_does_not_show_the_picture_before_it` renders a
large bright picture, then a smaller one into the same kept texture, and compares
it against the same small picture in a texture that has never held anything -
for all eleven filter configurations, masked and unmasked.

### BitmapData safety

Not touched. `create_empty_texture` is the only entry point that was changed and
`BitmapData` does not use it. Only renderer-owned, disposable cache textures are
recycled, and they have exactly one owner.

## 3. Cache churn, before and after

A hundred cached objects, 48 frames, measured with the policy off and on a few
seconds apart in the same process.

| | phase 1 | recycled only | recycled + capacity |
|---|---|---|---|
| **looping animation** | | | |
| cache rebuilds | 4,800 | 4,800 | **50** |
| textures really allocated | 4,800 (551.9 MB) | 4,459 | **3 (0.3 MB)** |
| **wandering bounds** | | | |
| cache rebuilds | 4,793 | 4,793 | **132** |
| per object per frame | 1.00 | 1.00 | **0.03** |

Recycling released cache textures **on its own does not scale**: 98.2% recycled
at 25 objects, 7.1% at 100, because a pool keyed on the exact size meets more
sizes than it can hold. With the rounding the sizes collapse onto a handful and
it recovers 94%. Both are kept; the rounding is what does the work.

## 4. The offscreen pool

### Root cause

Filter scratch is sized on the exact content it filters, so its sizes follow the
content the same way the cache textures did. A hundred objects with wandering
bounds met **3,300 distinct scratch sizes in 48 frames**.

### Miss reasons

Counted in `buffer_pool.rs`, per pool:

| reason | what it is |
|---|---|
| `new_size_class` | a size this pool has never been asked for |
| `format_mismatch` / `sample_count_mismatch` / `usage_mismatch` | a size it has met, at another format, sample count or usage |
| `evicted_by_budget` | a size it held, gave up to stay inside its budget, and was asked for again |
| `free_list_empty` | the key was there and every texture under it was lent out - not waste; the frame really needs them at once |

### The budget was measured

At 64 MiB a quarter to a half of the misses were `evicted_by_budget`: the pool
was evicting what the next frame wanted. Sweeping it, on a hundred filtered
objects with looping animation, counting **all** textures the run allocated:

| offscreen idle budget | textures allocated |
|---|---|
| 64 MB | 6,123 - *worse than phase 1's 4,800* |
| **192 MB** | **3** |
| 384 MB | 3 |

192 MB is the knee and 384 MB buys nothing, so 192 MB is the default
(`RUFFLE_OFFSCREEN_POOL_MB` overrides it). At 64 MB the exactly-sized filter
target this work introduced pushed the pool over its budget and it thrashed;
that is why the figure had to move with the fix rather than after it.

### What is left, and why

Under bounds that **wander and never repeat**, the same sweep gives 10,583 at
64 MB and 10,001 at 192 MB - a tenth, not an order of magnitude. No retention
policy fixes that: the sizes are genuinely new, and the only remedy is to
quantise scratch sizes the way cache textures are now quantised.

That is **not** done, deliberately, and the two things blocking it are named
rather than guessed at:

1. `glow.rs`, `bevel.rs` and `drop_shadow.rs` bind the source and its blurred
   copy to one pass and sample both with one set of coordinates. Two textures of
   different physical sizes need two.
2. `displacement_map.rs` takes `source_width` and `source_height` from the
   texture, so a padded source moves every sample.

Both are fixable, and both change filter shaders that 32 image tests hold to the
pixel. It is the right next piece of work; it was not worth risking a filter
regression to reach for it in the same session as the cache fix.

## 5. The allocator, and the unaccounted Windows memory

**No claim is made that the ~2.2 GB is explained.** What can be said is that a
mechanism which produces exactly that shape was reproduced locally and measured.

`generate_allocator_report()` gives what the graphics allocator has taken from
the driver:

* **allocated** - what is really in use;
* **reserved** - whole blocks the allocator owns, including their unused parts.

Over a 30-minute soak, allocated stayed flat at 109.8 MB while reserved
oscillated between 192.0 MB (2 blocks) and 320.0 MB (3 blocks) as demand rose and
fell. The gap - up to 210 MB here - is resident, is charged to the process, and
appears in **no count of live textures**. On a machine doing hundreds of times
more allocation, this is the shape of memory that no Ruffle-side counter
explains.

That is why the churn work is the right attack on the memory question and not
only on stutter: 207 GiB of allocation traffic is what drives a suballocator's
high-water mark and its fragmentation.

The diagnostic build reports `allocator_allocated_bytes`,
`allocator_reserved_bytes` and `allocator_blocks` every interval, so the client's
run can be read the same way.

## 6. wgpu / HAL findings

The counters phase 1 was asked to preserve still work, and were confirmed against
a real run of the diagnostic build:

`hal_texture_views` 19, `hal_buffers` 38, `hal_bind_groups` 9, `hal_samplers` 26,
`hal_shader_modules` 21, `hal_texture_memory` 16.5 MB, `hal_buffer_memory`
760 KB - all live and moving.

Two read zero on this Mesa/Vulkan stack: **`hal_textures`** and
**`hal_memory_allocations`**. That is wgpu's own accounting on this backend, not
a break introduced here - `hal_texture_memory` is populated from the same
structure. They are left in the schema rather than removed, because they may
populate on the client's NVIDIA driver, but nothing in this report is read from
them, and `allocator_*` is the figure that carries the argument instead.

## 7. Deferred destruction

Compared tracked textures against HAL resources across the soak; they track each
other and nothing accumulates between them. **No `Device::poll(Wait)` was added**
- the brief's warning is right, and there was no measurement suggesting it would
help: the reserve oscillates rather than climbing, which is what release
happening on time looks like.

## 8. Loading-phase memory

**Not measured, and this is the one piece of the brief that is not delivered.**
It needs the real client loading real assets, which this machine cannot do - the
content is behind a login and the local GPU is not the target. The diagnostic
build carries every counter the timeline needs (`swf_bytes`, `characters`,
`bitmap_source_bytes`, `bitmap_decoded_bytes`, `pending_loaders`,
`gc_external_bytes`, `tex_*_created`, `allocator_*`, `rss_bytes`) sampled every
interval from process start, so the 0-30 / 30-60 / 60-120 second breakdown falls
straight out of the first rows of the CSV from the Windows run. It is analysis
waiting on data, not work waiting to be done.

## 9. Phase 1's remaining renderer issues

Investigated as asked.

1. **Destination copies.** Unchanged and confirmed: 800 copies and **96.90 MB**
   per frame at 800 `Multiply` objects, exactly what phase 1 recorded. Not
   attacked. The two candidate fixes want opposite things - one copy of the whole
   region versus fewer, larger copies - and choosing between them needs the
   client's card, where the bandwidth and the per-command barrier cost are
   different from this laptop's. `destination_copies` and
   `destination_copy_pixels` are in the CSV so that run can decide it.
2. **Mixed-room alternation.** Reproduced and quantified: a room of 0% cached /
   20% complex blends batches **4.1 complex blends per pass**, against 10.0 for
   the 60% / 10% mixture - phase 1's estimate was about four against twelve.
   Not fixed. Closing it means letting a complex blend move earlier past drawing
   commands it shares no pixel with, which is a reordering, and `Pass::Draw`
   carries no per-draw coverage to test against. It is a genuine piece of work,
   not a small one, and it is the largest pass reduction still available.
3. **Alpha-mask batching.** Still incomplete, still counted as
   `page_fallback_alpha_mask`. Untouched.
4. **The half-pixel nearest-sampling edge case.** Verified rather than assumed:
   `a_texel_boundary_picks_a_neighbouring_texel_at_worst` still passes, and the
   full image suite passes at unchanged tolerances. It cannot produce a
   meaningful visual regression - a differing pixel holds what the pixel beside
   it holds.

The phase 1 blend-page architecture was not undone. `blend_pages_enabled` and
`blend_batching_enabled` are still on, still switchable, and all ten page tests
pass.

## 10. Tests

| suite | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --all-targets` | clean |
| `cargo test -p ruffle_render --lib` | 58 passed |
| `cargo test -p ruffle_render_wgpu --lib` | 20 passed |
| `cargo test -p ruffle_render_wgpu --test blend_pages` | 10 passed |
| `cargo test -p ruffle_render_wgpu --test cache_capacity` | **5 passed** (new) |
| `cargo test -p ruffle_render_wgpu --test blend_render_targets` | 5 passed |
| `cargo test --release --package tests --features imgtests` | **4,539 passed, 0 failed, 37 ignored** |

Exactly phase 1's count. No threshold was weakened and no test was disabled.
Named groups, all passing: 32 filter tests including `displacement_map`,
`displacement_map_through_filters`, `displacement_map_through_applyFilter` and
`displacement_map_scales_with_screen`; 30 `cache_as_bitmap`; 29 mask; 36 blend;
17 `unload*`; 53 loader.

### New tests

`render/wgpu/tests/cache_capacity.rs` - every one renders the same content twice,
once with the rounding on and once off, and compares the two frames **pixel for
pixel**:

| test | what it holds |
|---|---|
| `a_padded_cache_draws_the_same_picture` | 11 filter configurations x 4 picture sizes x 2 qualities, padded against exactly sized |
| `a_kept_texture_does_not_show_the_picture_before_it` | a large bright picture, then a smaller one in the same kept texture, masked and unmasked, for every filter |
| `a_padded_cache_with_a_mask_inside_it` | the stencil case, which needs the stencil to follow the texture |
| `growing_past_the_capacity_is_never_clipped` | a picture that outgrows its texture gets a new one |
| `a_padded_cache_survives_awkward_placement` | fractional, negative and edge-of-viewport placement |

`render/src/cache_capacity.rs` unit tests: a breathing avatar keeps one texture,
a capacity always holds its content, growth rebuilds, a long shrink gives the
memory back, a bucket boundary does not alternate, and the switch restores the
old behaviour exactly.

`core/src/memory_report.rs` unit tests: the CSV header and its rows have the same
number of fields, and every column the verifier prints exists.

## 11. Soak

30 minutes, **51 cycles, 8,568 frames**, walking crowd → quiet → crowd → complex
blends → masks → filtered cacheAsBitmap objects whose bounds breathe → a worst
case of 700 groups.

```text
 cycle  frames   pages   page MB    passes    pool MB  bg built   textures   slowest
     1     168       7      88.0       574      113.3       190        458    473.2ms
     2     168       7      88.0       574      101.5       176        185    416.1ms
     3     168       7      88.0       574      101.7       176        187    463.4ms
   ...
    51     168       7      88.0       574      101.7       176        187    785.8ms

settled at cycle 2, finished at cycle 51: 7 -> 7 pages,
101.5 -> 101.7 MB of texture, 176 -> 176 bind groups built
```

* **cache rebuilds per cycle: 70 → 21 → 10 → 0 from cycle 4 onward.** The
  capacity holds completely once the room has warmed up; over the whole run,
  31 in the first half and **0** in the second.
* pages, page bytes, render passes, tracked textures and pool bytes: constant.
* textures allocated: **25,768 over the first 25 settled cycles and 25,756 over
  the last 25** - flat to a tenth of a percent, and almost all of it the
  offscreen scratch discussed in §4.
* bind groups: 4,453 in each half, identical.
* allocator allocated: flat at 109.8 MB. Reserve: 192-320 MB over the first half,
  320-320 MB over the second - it found its high-water mark and stayed there.

The soak's first version asserted that the reserve must not be larger at the end
than at the beginning, and on a 42-cycle run it **failed**. That assertion was wrong, not the
renderer: a suballocator takes and releases whole blocks, so a scene whose demand
rises and falls oscillates between two values and which one a run ends on says
nothing. It now asserts what actually matters - that the reserve's *ceiling* in
the second half of a long run is no higher than in the first, and that what is
allocated inside it is flat. The failure is written down here because it was a
real one and the fix was to test the right property, not to loosen the threshold.

## 12. Files changed

| file | what |
|---|---|
| `render/src/cache_capacity.rs` | new: the capacity policy, its switch, and its tests |
| `render/src/cache_stats.rs` | new: cache invalidation and allocation reason counters |
| `render/wgpu/src/cache_pool.rs` | new: recycling for released cache textures, with a budget |
| `core/src/display_object.rs` | `BitmapCache` keeps a capacity; `is_dirty` returns *why*; the size limit is judged on the texture that would really be built, with an exact-size fallback |
| `render/src/backend.rs` | `BitmapCacheEntry` carries the picture's size |
| `render/wgpu/src/backend.rs` | a filtered cache with spare capacity is drawn into an exactly sized target; the copy back is held to the picture's rectangle |
| `render/wgpu/src/surface/target.rs` | `attachment_size`, `apply_viewport`, and the stencil sized on the attachment |
| `render/wgpu/src/surface.rs` | the viewport applied at both pass sites |
| `render/wgpu/src/utils.rs` | `run_copy_pipeline` takes a viewport |
| `render/wgpu/src/buffer_pool.rs` | pool miss reasons, size histogram, eviction counters, the measured offscreen budget |
| `render/wgpu/src/lib.rs` | cumulative texture churn, the cache-pool switch |
| `render/wgpu/tests/cache_churn.rs` | new: the AQW churn reproduction and the two sweeps |
| `render/wgpu/tests/cache_capacity.rs` | new: the correctness suite |
| `render/wgpu/tests/blend_page_soak.rs` | breathing caches, phase 2 counters, allocator assertions |

## 13. Limitations

1. **Loading-phase memory is not profiled.** Needs the Windows run.
2. **The unaccounted ~2.2 GB is not explained**, only given a measured candidate
   mechanism and the counters to test it with.
3. **Offscreen scratch still churns under non-repeating sizes** - about a tenth
   better, not an order of magnitude. The two blockers are named in §4.
4. **Destination copies and the mixed-room alternation are unchanged.** Both are
   measured; neither is fixed.
5. **Alpha masks still do not take page regions.**
6. Frame times on the local GPU are not trustworthy in absolute terms. Every
   performance claim here is a counter, or a before/after measured seconds apart
   in one process.
7. One accounting nuance worth knowing before two counters are compared: a cache
   texture released into the recycling pool is marked as freed against the
   collector's external-memory pacing at that moment, because the display object
   really has let go of it, while the texture itself stays allocated until the
   pool gives it up. So `gc_external_bytes` can read up to the pool's 48 MB
   budget lower than the texture memory actually resident. It is bounded by that
   budget and it is visible in `cache_pool_idle_bytes`.
8. **No client testing was requested or performed**, as instructed.

## 14. Windows runs

Both runs use the diagnostic branch, never the production branch: the production
build does not log any of this, deliberately.

### The local laptop run (first)

An 8 GB machine with an integrated GPU is fine for this. It is not authoritative
for frame rate; it is there to say that the build is correct, that memory does
not climb, that the counters are populated and that the schema is right.

```powershell
cd $HOME\ruffle
git fetch --all
git checkout diagnostic/aqw-final-windows
git pull --ff-only
git rev-parse HEAD          # must be the diagnostic SHA in this report
git status                  # must be clean, or build-info records it as modified

cargo build --release --package ruffle_desktop
.\Make-BuildInfo.ps1        # writes $HOME\Desktop\aqw-final\build-info.txt
```

`build-info.txt` must say `instrument : aqw-final-diag-4` and `worktree : clean`
before the run counts.

In a second PowerShell window, left running for the whole session:

```powershell
cd $HOME\ruffle
.\Log-WindowsRam.ps1
```

Then the run itself:

```powershell
cd $HOME\ruffle
.\target\release\ruffle_desktop.exe `
    --memory-report "$HOME\Desktop\aqw-final\aqw-memory.csv" `
    --memory-report-interval 5 `
    https://game.aq.com/game/
```

15-30 minutes: log in, let the character and equipment load, go to a busy area,
change map several times, change equipment, use skills in combat, find a crowded
room if one is available, then sit in a quiet room for the last few minutes.
Crowd → quiet → crowd matters more than total time: it is what shows whether
memory comes back.

Then:

```powershell
Stop-Process -Name ruffle_desktop -ErrorAction SilentlyContinue
# Ctrl+C the RAM logger window
.\Verify-AqwLog.ps1 -Dir "$HOME\Desktop\aqw-final"
```

`Verify-AqwLog.ps1` prints `RUNTIME: NEW BUILD CONFIRMED (aqw-final-diag-4)` only
when the build, the clean checkout and every required column check out. If it
prints `RUNTIME: CHECKS FAILED`, the figures above it are not this build's and
the run has to be repeated.

There is no PowerShell on the machine this was written on, so the script could
not be *executed* here. What was done instead: the diagnostic build was run and
made to write a real CSV, and every column the script requires (81 of them) and
every column it prints was checked against that file programmatically. Two unit
tests keep it that way - the header and the rows must have the same number of
fields, and the columns the verifier reads must exist. The first time it is run
on Windows, read its output before trusting the run.

Files to send back, all from `$HOME\Desktop\aqw-final`:

* `build-info.txt`
* `aqw-memory.csv`
* `windows-ram.csv`
* the `Verify-AqwLog.ps1` output

Not the console log: it has carried login traffic before. `aqw-final/` is in
`.gitignore` so none of it can be committed by accident.

### The client's RTX run (only after the local run is analysed)

Windows 11, RTX 5060 Ti. Same commands. 40-60 minutes: startup, equipment load,
a busy room, several map swaps, equipment changes, skills and combat, a very
crowded room, a quiet room, then another crowded room. The last three are the
important part.

This is prepared, not sent. The client has not been asked to test.

## Verdict

Local engineering pass. Windows acceptance pending: the local laptop run first,
then the RTX 5060 Ti run, before any of this is called accepted.
