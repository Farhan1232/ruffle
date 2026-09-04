# AQW renderer, phase 1: what a crowded room costs

Phase 1 was one job: take the fixed per-blend cost out of a populated
AdventureQuest Worlds room. This is what was found, what was built, what it
measured, and what is left for phase 2.

## Where to start

| | |
|---|---|
| Base SHA | `436f7d71bc3070758d99ee8de9472b8bc796c5dc` (`fix/aqw-blend-render-performance`) |
| Phase 1 branch | `fix/aqw-render-final-performance` |
| Renderer work | `6cc98dcb9` - one commit, the whole of it |
| Phase 1 final SHA | the tip of `fix/aqw-render-final-performance`, which is this file's own commit on top of `6cc98dcb9` |
| Local GPU | Intel HD Graphics 4400, Mesa 25.2, Vulkan. Weak, and thermally noisy: absolute frame times drift by tens of per cent over a run, so the numbers to trust here are the counters, and frame times are only quoted from runs that measure both builds a few frames apart. |

## The bottleneck

The blend-target *sizing* work was already done and had already paid: peak
texture 5,281 MiB → 329 MiB, peak blend pool 3,884 MiB → 106 MiB. Fitting the
client's 43-minute run gave 106 µs per live target before it and 57 µs after -
the bandwidth half had gone and a **fixed price per target** was left, one that
scales with how *many* targets a frame wants rather than how big they are.

At the base commit a frame with 800 blended objects encoded:

* 800 render passes to draw the groups' contents, one per group;
* 800 pool takes and returns, one `CommandTarget`, one globals lookup and one
  bind group each;
* for a complex blend, a second render pass and a destination snapshot;
* which, when complex blends split the surface's own draw stream, added a third.

Measured at the base commit, per frame, 800 objects:

| scene | render passes | targets | target MB |
|---|---|---|---|
| `Multiply` groups | 1600 | 800 | 164.4 |
| AQW room, 60% cached / 10% complex | 488 | 400 | 103.4 |
| AQW room, 0% cached / 20% complex | 968 | 800 | 199.3 |
| AQW room, 0% cached / 100% complex | 1600 | 800 | 199.3 |

Nine hundred `Multiply` objects took 1800 passes and 2,128 ms a frame locally.

## The architecture

Two changes, each with an off switch (`ruffle_render_wgpu::tuning`), which is
how they are tested: the same scene is rendered with them on and off and the two
frames compared pixel for pixel.

### 1. Blend pages (`render/wgpu/src/surface/page.rs`)

Nothing about a blended group needs a render target all to itself. What it needs
is a rectangle of transparent pixels that only its own commands draw into, and a
way to sample exactly that rectangle afterwards.

A page is a pooled texture that many groups share. Each group is given a region
of it; every group on the page is drawn in **one render pass**, scissored to its
own region; and each is composited back by the same quad it would have used for
its own target, carrying the region's texture coordinates.

* **Physical page** and **logical region** are separate types. A `PageRegion`
  carries its own `x`, `y`, `width`, `height` *and* the page's `page_width` and
  `page_height`; nothing assumes the origin is `(0, 0)` or that the page is the
  size of the group.
* **Packing** is a shelf: a row grows to the right, a new row starts above the
  tallest region of the last. O(1), allocation-free, and 98% efficient on a room
  of one-size objects - which is what a room of avatars is. `regions_never_overlap`
  and `a_room_of_one_size_fills_its_page` are the tests.
* **Page sizes** are `[1024, 1024, 2048]`, the last repeating, and `[1024]`
  throughout when the surface is multisampled. The first two are small so a
  quiet scene with a dozen blended objects does not take sixteen megabytes to
  hold them; a scene that has filled two of those has shown it can fill a large
  one. A 2048 page holds about ninety of the client's avatar-with-equipment
  groups. A multisampled page carries `sample_count` copies of every pixel on
  top of its resolve target, so it stays small - a 2048 page at four samples is
  eighty megabytes.
* **Regions are the content's own size**, not a size class: a page region is not
  a pool key, and an avatar wants 150x200 where a size-classed target gives it
  192x256. A third less area, for free.
* **Gutter** of one pixel between regions and around the page edge.
* **Lifetime**: a page's pass is encoded before any pass that composites from
  it, because every page is drawn before `chunk_blends` returns and the chunks
  are executed after. The pooled textures are held in `ChunkedCommands::pages`
  for as long as the surface is drawing, so the pool cannot hand a page out
  again - and clear it - while a composite that reads it is still to be encoded.
  There is no page generation counter and no double buffering, because neither
  is what makes this safe: a region is never reissued within a frame, and once
  the surface has finished with a page and the pool takes it back, anything that
  clears it is encoded *after* everything that read it, in the same encoder.
  wgpu executes a submission's commands in order, so a later frame's clear
  cannot overtake an earlier frame's read. The `Arc` is what stops the reuse
  happening too early; command order is what makes reuse afterwards correct.

**Why the picture does not change.** Pages are a power of two on a side and a
region is composited through a quad exactly as many pixels wide as the region is,
so the fragment at the region's `i`-th pixel samples the page at
`(x + i + 0.5) / page` - the centre of page texel `x + i`, exactly, because the
division is by a power of two. That is the same texel the group's own target
would have handed back. `a_region_samples_its_own_texels_and_no_others` checks
this for the corners of the space. Writes are held inside the region by the
scissor rectangle, whatever rasterisation or multisampling would otherwise do.
The page is cleared to transparent by the load operation of its one pass, so a
region is transparent everywhere its group did not draw, exactly as a freshly
taken target is; and regions are handed out in one direction and never reissued
within a frame. Sub-pixel phase is kept because a region starts on a whole pixel
and the group's world matrices have the region's origin put back on.

### 2. Complex blends composited together (`render/wgpu/src/surface.rs`)

A complex blend reads the destination underneath itself and writes over it, so
two of them that cover the same pixels have to be ordered. Two that cover no
pixel in common cannot see each other's work at all.

`batch_passes` walks the chunks and merges a run of consecutive complex blends
into one `Pass::Blends` while each new one shares no pixel - grown by one, which
is what the destination snapshot copies - with anything already in the batch, and
while the stencil state and the destination they read are the same. The batch
then refreshes each member's own rectangle of the snapshot and composites all of
them in **one render pass**, each with its own pipeline, transform and quad.

Order is never changed: two blends that overlap always land in different batches,
and batches run in the order their blends arrived. Twelve blends a pass in a
crowd; `overlapping_complex_blends_keep_their_order` proves the overlapping case
still takes one pass each, and both cases are compared pixel for pixel against
the unbatched render.

### 3. A chunk's buffers were sized for 200 vertices, not 200 objects

`DynamicTransforms` sized its vertex buffer at one `PosUvVertex` per
`ESTIMATED_OBJECTS_PER_CHUNK` rather than the four a quad needs, which capped a
chunk at fifty draws and cost a crowded room a render pass per fifty objects.
Both buffers are now sized for the same number of draws - 256, what the uniform
buffer's 64 KiB binding limit allows at the usual alignment. 800 fast-path
objects went from 16 passes to 4.

## Direct fast path: unchanged

`trivial_fast_path` is exactly as it was: a single `Command::RenderBitmap` under
`TrivialBlend::Normal`, drawn straight onto its destination with the blend state
set. It is tried **first**, before pages, so an object that can skip the
intermediate entirely still does. It was not widened: `Add`, `Subtract` and
`Screen` do not survive being applied per multisample rather than to the resolved
group, which the existing `visual/blend_direct` test caught when the path
accepted them.

The order is: direct draw, then a page region, then a target of the group's own.

## Page eligibility and fallbacks

A group takes a region when its commands are a plain run of draws:
`RenderBitmap`, `RenderShape`, `DrawRect`, `DrawLine`, `DrawLineRect`. Anything
that would want a pass or a target *inside* the shared one keeps its own target,
where the existing code already knows what to do with it. Counted, by reason,
in `RenderStats::page_fallbacks`:

| reason | what it is |
|---|---|
| `shader` | a `BlendMode.SHADER` PixelBender blend - arbitrary code that may write anywhere its quad covers, so it keeps its full-sized target |
| `nested_blend` | the group draws a blended group of its own |
| `alpha_mask` | the group contains a `RenderAlphaMask`, which needs two targets and a paired bind group |
| `masked` | the group pushes a stencil mask; a shared pass has no per-region stencil state |
| `stage3d` | drawn through its own pipelines |
| `size` | larger than 510 pixels on a side; it would take most of a page and get none of the benefit |
| `capacity` | the group's draws do not fit one chunk's uniform or vertex buffer |
| `no_page` | no page could be opened |

Masks are neither disabled nor approximated - a masked group simply keeps the
target it had. Filters are untouched: a filtered or cached display object
reaches the renderer as one `render_bitmap` of its cache texture and usually
takes the *direct* path, and where it does not it is a plain draw and takes a
region. Every filter test in the suite passes, `displacement_map` included.

The reserve is exact rather than a guess: `page_reserve` counts one uniform slot
per draw and four vertices per quad-carrying draw (four slots for a rectangle of
emulated lines), and a group only joins a page once `BufferBuilder::has_room_for`
says all of it will fit. A run split between two buffers would be split between
two render passes and half of it would land on the wrong page, so this is
enforced rather than hoped for; `paging` carries a debug assertion for the
overflow path that must therefore never be reached.

## Destination copies

Unchanged in *count* and *bytes*: each complex blend still refreshes its own
rectangle of the snapshot, grown by a pixel. What changed is that a batch takes
its copies together and then composites in one pass, so the copies are no longer
separated by a render pass each.

At 800 `Multiply` objects that is 800 copies and 96.9 MB a frame - about 123 KB
each. On the client's card that is roughly a millisecond of bandwidth; what it
is not free of is 800 copy commands with the barriers a driver puts around them.
`destination_copies` and `destination_copy_pixels` are in `RenderStats` so phase
2 can decide whether that term is worth attacking. It is not the term that was
attacked here, and the measurements say the pass count was the right one: see
below.

## Numbers

### Interleaved before/after, 50 to 900 objects

Both builds measured a few frames apart on the same machine in the same process,
because frame times on this laptop drift with its temperature over a run.
"before" is this branch with both switches off, so it isolates the two batching
changes.

**Groups of one bitmap each under `Layer`** - the case where every group wants a
target and none can take the direct path:

| objects | passes before | after | ratio | target MB before | after | mean ms before | after |
|---|---|---|---|---|---|---|---|
| 50 | 51 | 4 | 12.8x | 16.6 | 31.2 | 76.1 | 66.5 |
| 100 | 101 | 4 | 25.2x | 26.0 | 31.2 | 117.9 | 94.1 |
| 250 | 251 | 5 | 50.2x | 54.1 | 47.2 | 305.1 | 235.9 |
| 500 | 502 | 9 | 55.8x | 101.0 | 95.2 | 422.3 | 368.6 |
| 800 | 804 | 13 | 61.8x | 157.2 | 127.2 | 1154.6 | 662.9 |
| 900 | 904 | 14 | 64.6x | 176.0 | 143.2 | 749.6 | 627.4 |

**The same under `Multiply`** - every group also reads the destination:

| objects | passes before | after | ratio | copies | target MB before | after | mean ms before | after |
|---|---|---|---|---|---|---|---|---|
| 50 | 100 | 7 | 14.3x | 50 | 23.8 | 38.4 | 88.5 | 69.8 |
| 100 | 200 | 12 | 16.7x | 100 | 33.2 | 38.4 | 144.1 | 104.3 |
| 250 | 500 | 25 | 20.0x | 250 | 61.3 | 54.4 | 314.6 | 209.8 |
| 500 | 1000 | 48 | 20.8x | 500 | 108.2 | 102.4 | 644.1 | 384.8 |
| 800 | 1600 | 75 | 21.3x | 800 | 164.4 | 134.4 | 1391.5 | 663.2 |
| 900 | 1800 | 84 | 21.4x | 900 | 183.2 | 150.4 | 1679.2 | 695.3 |

**AQW-shaped rooms** - mixtures of cached singles (which take the direct path)
and multi-child containers (which do not):

| mixture | objects | passes before | after | ratio | mean ms before | after |
|---|---|---|---|---|---|---|
| 60% cached, 10% complex | 800 | 488 | 23 | 21.2x | 505.2 | 399.9 |
| 60% cached, 10% complex | 900 | 549 | 26 | 21.1x | 568.5 | 581.8 |
| 0% cached, 20% complex | 800 | 968 | 60 | 16.1x | 984.0 | 734.2 |
| 0% cached, 20% complex | 900 | 1089 | 67 | 16.3x | 1339.5 | 925.0 |
| 0% cached, 100% complex | 800 | 1600 | 185 | 8.6x | 1233.2 | 899.5 |
| 0% cached, 100% complex | 900 | 1800 | 207 | 8.7x | 1365.1 | 974.9 |

**Objects that take the direct path** are untouched, as intended: 1 to 4 passes
and 7.2 MB at every size, before and after.

### Against the base commit

Full-run numbers, base commit versus this branch, 800 objects:

| scene | passes | | targets/pages | | target MB | | CPU encode ms | |
|---|---|---|---|---|---|---|---|---|
| | base | now | base | now | base | now | base | now |
| `Multiply` singles | 1600 | 74 | 800 targets | 8 pages | 164.4 | 118.4 | 26.6 | 6.1 |
| `Multiply`, 900 objects | 1800 | 83 | 900 targets | 9 pages | 183.2 | 134.4 | 46.5 | 7.0 |
| room 60/10 | 488 | 23 | 400 | 7 | 103.4 | 102.4 | - | - |
| room 0/20 | 968 | 60 | 800 | 13 | 199.3 | 198.4 | - | - |
| room 0/100 | 1600 | 185 | 800 | 13 | 199.3 | 198.4 | - | - |
| `Layer` fast path | 16 | 4 | 0 | 0 | 7.2 | 7.2 | 2.0 | 2.8 |

CPU encode time for 900 `Multiply` objects: **46.5 ms → 7.0 ms**. That is the
number that matters most for the client's stutter, because it is the main
thread's own cost and it is what a `p99` of 727 ms is made of.

### What this predicts on the client's card

Their previous run fitted at 57 µs of fixed cost per live blend target after the
sizing fix. What a page removes from that per group is the render pass, the pool
take and return, the `CommandTarget` and its globals, and - for a complex blend -
one of the two passes. What is left per group is a draw call inside a shared
pass, plus a destination copy for a complex blend.

At 800 targets, 57 µs each is about 46 ms of a frame. The pass count for the
room shapes measured here falls by 8.6x to 21x, and the CPU encode time by 6.6x.
If the fixed cost falls with it, a crowded 800-object frame should lose the large
majority of that 46 ms, which is most of the distance between the 100-165 ms
means the client reported and the 41.67 ms budget. It cannot be claimed to land
under the budget without their machine: what is left in those frames is
ActionScript, garbage collection, the display-list walk, the destination copies
and the fill, none of which this touched. The renderer's own fixed share is what
was removed.

## Correctness

Every batching test renders the same scene twice - with the batching on and with
it off - and compares the two frames **pixel for pixel**. That is stronger than a
stored reference: it holds for every blend mode, transform and arrangement the
test cares to build, and it fails on the difference rather than on a tolerance.

`render/wgpu/tests/blend_pages.rs`, all passing:

| test | what it holds |
|---|---|
| `every_blend_mode_survives_sharing_a_page` | all fourteen blend modes x six placements (whole pixels, fractional translation, rotation, skew, negative scale, fractional scale) x two qualities (no multisampling and 4x), each wrapped in a `Layer` so `Alpha` and `Erase` have one - 168 exact comparisons |
| `groups_over_the_edge_survive_sharing_a_page` | groups clipped by the viewport's edges |
| `a_neighbour_on_a_page_does_not_bleed_through` | red, empty, green and blue regions side by side on a page plus a rotated one over them; the empty group's area is still the background afterwards |
| `a_page_region_does_not_show_the_last_frame` | a frame of white through the pages, then a frame of transparent groups in the same places: no white survives anywhere |
| `overlapping_complex_blends_keep_their_order` | four overlapping destination-reading groups, each of which changes what the next reads: one pass each, and the same picture |
| `separated_complex_blends_share_a_pass` | eighteen separated complex blends in under four passes, same picture |
| `a_crowd_on_pages_looks_like_a_crowd_on_targets` | 220 mixed-mode groups: identical, on 2 pages, in 179 passes against 397 with a target each |
| `groups_that_cannot_share_a_page_say_why` | stencil masks, alpha masks, nested blends and an oversized group all fall back, each counted under its own reason, and the picture is unchanged |
| `groups_under_a_mask_survive_sharing_a_page` | blended groups drawn while a stencil mask is up, straddling the mask's edge, under `Layer`, `Add`, `Multiply` and `Alpha`: the composite is stencil-tested and the batch does not reach across the mask |
| `a_texel_boundary_picks_a_neighbouring_texel_at_worst` | the one thing a page does change - see below |

Unit tests in `render/wgpu/src/surface/page.rs` and `bounds.rs`:
`regions_never_overlap`, `regions_keep_a_gutter_between_them`,
`a_room_of_one_size_fills_its_page`,
`a_region_samples_its_own_texels_and_no_others`,
`a_region_is_the_content_and_nothing_more`, `the_region_covers_the_content`.

### The one thing that does change

A bitmap placed exactly half a pixel from a whole one puts every one of its texel
boundaries on a sample point. Which of the two texels a *nearest* sampler picks
there is arbitrary, and it is settled by the last bits of the interpolated
coordinate - which a page moves, because the group is drawn at its region's place
on the page rather than at the corner of a target of its own.

So the tie can land the other way. `a_texel_boundary_picks_a_neighbouring_texel_at_worst`
builds forty groups at exactly that offset and holds the effect to what it is: a
pixel that differs holds what the pixel beside it holds - the neighbouring texel,
never anything new - and it is a seam along the boundaries inside the sprites,
1.1% of the frame, not a shifted image. Everything Flash draws is on a twentieth
of a pixel, so this is a one-in-twenty placement per axis; the effect is one
texel on a hairline, of the same size and kind as the differences the project's
own image tests already tolerate between drivers. It is called out here so
phase 2 knows it is understood rather than undiscovered.

## Full test results

| suite | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --all-targets` | clean, no warnings |
| `cargo test -p ruffle_render_wgpu --lib` | 20 passed, 0 failed |
| `cargo test -p ruffle_render_wgpu --test blend_pages` | 10 passed, 0 failed |
| `cargo test -p ruffle_render_wgpu --test blend_render_targets` | 5 passed, 0 failed |
| `cargo test --release --package tests --features imgtests` | **4539 passed, 0 failed**, 37 ignored |

Named in the brief and confirmed individually: all 36 blend image tests; 29 mask
tests; 32 filter tests including `displacement_map`,
`displacement_map_through_filters`, `displacement_map_through_applyFilter` and
`displacement_map_scales_with_screen`; 30 `cache_as_bitmap` tests; 53 loader
tests; 17 `unload*` tests; `weak_dictionary_releases_unreferenced_keys`;
`application_domain`; and the `MovieLibrary` lifetime tests
(`loader_unload_releases_library`, `retained_class_keeps_library`,
`released_class_frees_library`). No threshold was weakened and no test was
disabled.

## Soak

`render/wgpu/tests/blend_page_soak.rs`, ignored by default:

```text
RUFFLE_SOAK_SECONDS=1200 cargo test --release -p ruffle_render_wgpu \
    --test blend_page_soak -- --ignored --nocapture
```

It walks a cycle - crowd, quiet, crowd again, complex blends, masks, filtered
`cacheAsBitmap` objects, and a worst case of 700 groups with masks and complex
blends among them - twenty-four frames at a time, with everything drifting so no
frame repeats the last. Each cycle it records the busiest frame's pages, page
bytes and render passes, the textures the renderer owns, and the bind groups
built, and it insists the cycles after the first look like each other.

**Result of a 20-minute run: 33 cycles, 5,544 frames, passing and flat.**

```text
 cycle  frames   pages   page MB    passes    pool MB  bg built   textures   slowest
     1     168       7      88.0       574       98.4       190        159    504.8ms
     2     168       7      88.0       574       98.4       176        159    816.9ms
   ...
    20     168       7      88.0       574       98.4       176        159    687.8ms
   ...
    33     168       7      88.0       574       98.4       176        159    735.5ms

settled at cycle 2, finished at cycle 33: 7 -> 7 pages,
98.4 -> 98.4 MB of texture, 176 -> 176 bind groups built
```

Every column is constant: the busiest frame wants the same seven pages and the
same 88.0 MB of them, encodes the same 574 render passes, and the renderer holds
the same 98.4 MB of texture. The pool oscillates between holding 159 textures at
98.4 MB and 255 at 101.9 as it trims the sizes a scene has stopped using and
takes them back when it returns - which is what it is for, and it is bounded on
both sides. Nothing accumulates over 5,544 frames.

The one number that does not go to zero is bind groups built, and it took two
wrong assertions to say what it actually is.

Measured per phase, **almost all of them are built by two phase changes**: 99
when the masked phase comes back and 77 when the worst case does, out of about
176 a cycle. Those two phases are the ones whose groups keep targets of their
own, and between visits the pool gives up the sizes they were using -
demand-aware trimming, working as designed. A size that comes back comes back as
new textures, and a bind group lives on the texture it names, which is what makes
keeping it safe. So that cost is the price of handing the memory back.

The rest is the drift: the soak moves every object a little every frame, so now
and then a group's bounds cross a size class and the pool genuinely has to build
a texture. Across the whole 33-cycle run that accounted for **25** bind groups,
in one cycle, out of 5,692. That is why the soak does not assert zero: a drifting
scene meeting a new size is the pool doing its job, not the cache missing.

What it asserts instead is the difference between the two. A cache that was
missing would build a bind group per blend per frame - about 49,700 a cycle here.
The soak requires fewer than one per fifty blends, a fifty-fold margin over what
is actually built, and it requires the second half of a long run not to build
appreciably more than the first: **2,873 over the first sixteen settled cycles,
2,819 over the last sixteen.** All of this is pre-existing behaviour of the pool
and the bind-group cache from the base commit; pages did not add to it.

## Preserved

None of these were touched, and all their tests pass: the `MovieLibrary` and
unloaded-SWF lifetime fix, weak `flash.utils.Dictionary` keys, the avatar and
player retention fixes, completed image-loader cleanup, the `ApplicationDomain`
and class-lifetime corrections, `unloadAndStop` and the external-memory work, the
render-surface pool's demand trimming and idle budget, content-bounded blend
targets, the removal of full-viewport targets, target-origin and UV handling, the
bounded offscreen pool, composite bind-group caching, and the restricted direct
fast path.

## Files changed

| file | what |
|---|---|
| `render/wgpu/src/surface/page.rs` | new: pages, regions, the shelf packer, and their tests |
| `render/wgpu/src/surface/commands.rs` | page eligibility and reserve, the paged blend path, region-mapped drawing, `Chunk::Blend` naming a shared source, `ChunkedCommands` |
| `render/wgpu/src/surface.rs` | passes rather than chunks: complex blends batched into one pass, the source rectangle passed to the blend shaders |
| `render/wgpu/src/surface/target.rs` | `blend_buffer()` for a batch to read once, destination-copy counter |
| `render/wgpu/src/bounds.rs` | `region_rect_for`, and its tests |
| `render/wgpu/src/buffer_builder.rs` | `has_room_for`, so a group can ask before it is placed |
| `render/wgpu/src/dynamic_transforms.rs` | a chunk's buffers sized for draws rather than vertices |
| `render/wgpu/src/lib.rs` | page, destination-copy and batch counters; `PageFallback`; the `tuning` switches |
| `render/wgpu/shaders/blend/*.wgsl` (9) | the source rectangle: `current_uv` comes from `transforms.mult_color`, which the blend shaders had no other use for. `[0, 0, 1, 1]` is the whole texture and gives back exactly what the shader computed before |
| `render/wgpu/tests/blend_pages.rs` | new: the correctness suite |
| `render/wgpu/tests/blend_page_soak.rs` | new: the soak |
| `render/wgpu/tests/blend_render_targets.rs` | the new counters, and the interleaved before/after benchmark |

## Memory

Bounded and, at the sizes that matter, lower. 800 `Layer` groups: 157.2 MB of
render target a frame → 127.2 MB. 800 `Multiply`: 164.4 → 134.4. The AQW room
shapes are level (199.3 → 198.4) because their groups are taller and pack fewer
to a page. Small scenes pay a little more - 50 groups go from 16.6 MB to 31.2 -
because a page is a page whether it is full or not; the pool's idle budget still
bounds what is kept, and 31 MB is not the order of the problem phase 2 is chasing.

No new allocation is unbounded: pages come from the same pool as the targets they
replace, are keyed on two sizes rather than twenty, and are released with the
surface that drew them. Nothing about the pool's trimming, budgets or counters
was changed.

## Left for phase 2

1. **The unexplained ~2.2 GB Windows working set.** Untouched here, as
   instructed. Nothing found in this work explains it; the counters added
   (`pages_last_frame`, `page_bytes_last_frame`, `peak_page_bytes`) give the new
   architecture's share of it directly.
2. **Destination copies.** Still one per complex blend: 800 copies and 96.9 MB a
   frame at 800 objects. Now measured (`destination_copies`,
   `destination_copy_pixels`) rather than assumed. Worth deciding whether the
   cost is the bandwidth or the 800 copy commands and their barriers - the two
   want different fixes.
3. **Scene-change bind groups.** About 176 `create_bind_group` calls per trip
   through the soak's seven scenes: 176 of them at two phase changes, where the
   pool gives memory back and takes it again, and a handful more when a drifting
   object's bounds cross a size class. Negligible in itself, and one blend in
   fifty thousand; listed because phase 2 will see it in any instrumentation it
   adds and should know it is understood rather than a cache miss.
4. **The mixed room's alternation.** A trivial group's composite goes into the
   surface's own chunk and a complex blend closes that chunk, so a room that
   alternates between them gets `Draw, Blend, Draw, Blend` and complex blends
   batch only about four to a pass instead of twelve. Keeping the chunk open
   across a complex blend that shares no pixel with anything in it would close
   that gap - it needs a coverage test per pending draw, and it is the single
   largest remaining pass reduction (about 60 passes to about 20 at 800 objects).
5. **Alpha masks do not take page regions.** They need two regions and a paired
   bind group naming both, and the alpha-mask shader takes no texture
   coordinates of its own. Rarer than blends, so it was left; the fallback is
   counted as `alpha_mask`.
6. **PixelBender blends** keep their full-viewport target. Deliberate: arbitrary
   code may write anywhere its quad covers.
7. **Filter and cache allocation churn**, which phase 1 was told not to chase.
   The offscreen pool's budget still bounds it; the client session's 1.86 million
   created-and-destroyed offscreen targets and 621,413 `cacheAsBitmap` textures
   are phase 2's.
8. **Windows validation.** No client build was prepared and the client was not
   asked to test, as instructed.

Nothing suspicious about memory was found while doing this work. The one thing
worth knowing is item 4's shape: the surface's chunk boundaries, not the pages,
are now what decides how many passes a mixed room encodes.

## Final SHA

The renderer work and its tests are `6cc98dcb9`, one commit. This file is the
commit on top of it, so that that SHA could be written down. Both are on
`fix/aqw-render-final-performance`; its tip is the phase 1 final SHA, and
`git log --oneline -2` on the branch shows both.

Start phase 2 from the branch tip.
