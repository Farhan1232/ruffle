# What the Windows run and the reference client say the next phase is

Written 2026-09-06 from three things that arrived together: the client's 40-minute
Windows run of the phase 2 diagnostic build (`aqw-memory.csv`, `windows-ram.csv`,
`verify.txt`, `build-info.txt`), a 28-second screen capture taken 2.5 minutes into
that same run, and the published source of another AQW client built on Ruffle
(`github.com/Uzipuzy123/AQW-Ruffle-v0.7.8-Source`, the "Aether" client, MIT/Apache-2.0
like Ruffle itself).

## 1. Phase 2 held on Windows

RTX 5060 Ti, Windows 11, 15.6 GB, 40 minutes, commit `2d9bb770b`, instrumentation
`aqw-final-diag-4`.

| | phase 1 run | this run |
|---|---|---|
| peak texture bytes | 5,537 MB | **436 MB** |
| cache textures built | 621,413 | **37,929** |
| offscreen textures built | 758,880 | **192,635** |
| cache redraws keeping their texture | - | **85.7%** |
| offscreen pool hit rate | - | **84.5%** |
| bind group cache hit rate | - | **99.8%** |

The 32-pixel capacity for `cacheAsBitmap` textures is doing exactly what the local
measurement predicted. Nothing regressed.

## 2. The memory climb is not in anything we account for

Working set 307 MB -> 3,948 MB peak, ending 3,747 MB. Private bytes 501 MB -> 6,163 MB
peak, ending 5,706 MB. It climbs for the first 28 minutes and is flat for the last 13.

Everything we count is flat across those 40 minutes:

| | start | end | trend |
|---|---|---|---|
| Rust live heap | 32 MB | 370 MB | oscillates 300-670, no trend |
| GPU allocator reserve | 192 MB | 1,472 MB | flat from minute 4 |
| movies resident | 1 | 121 | oscillates 64-348 |
| SWF bytes | 0 | 21 MB | flat |
| GC arena | 3 MB | 53 MB | oscillates |
| live GPU textures | 15 MB | 298 MB | oscillates |

Private bytes minus (Rust live heap + GPU allocator reserve) goes from **161 MB to
3,864 MB**. That is the whole climb, and it is outside both instruments.

It ratchets rather than leaks: +5,017 MB across 241 sample windows, -1,468 MB across 55.
The largest single steps (+540, +482, +363, +250 MB in five seconds) all land in windows
where the frame rate is 11-17 fps, `movies` is spiking, and `render_passes` is about a
thousand - room loads and crowded rooms. It steps up there and never comes back.

**Hypothesis for phase 3, with the evidence for it:** it is the graphics driver's own
host memory, held per render pass and released only at submit, ratcheting to the
high-water mark of a crowded frame.

- We record ~1,050 render passes in a crowded frame and submit all of them in one
  encoder - there is no submission splitting in our tree.
- 81% of the renderer's own time is `queue_submit`.
- The other client found the same shape from the other end: "Every drawing step in a
  frame takes a small allocation from the driver that is only released once the batch is
  handed over, and a crowded town is seven hundred to a thousand steps." They shipped
  `AETHER_MAX_PASSES_PER_SUBMISSION` as a knob for it (notes/0.6.44.md).

Two measurements settle it, and both are cheap:
1. Split the frame's submission every N passes and A/B it inside one session.
2. Add a heap census that separates *retained* from *live* - mimalloc's `mi_process_info`,
   or a `HeapWalk` sum of committed-vs-busy - so heap retention can be ruled in or out
   without another guess.

## 3. The stutter and the memory climb have the same cause

The capture was taken during the run, so it can be read against the counters.

Video, by frame differencing: 645 distinct frames in 27.8 s = **23.2 updates a second**,
144 intervals where the picture is held for two capture frames (15 fps stretches), and
**19 hitches of 100-167 ms** in 28 seconds.

The counters for that wall clock (t = 150-185 s):

- frames fall to 56-70 per 5 s: **11-14 fps**
- `frame_ms_max` 527, 593, 961 ms
- `render_passes` per frame: 5 before, **~1,050 during**
- 6,460 destination copies a second

Over the whole run: 8,417,623 complex blends, 8,417,623 destination copies, **84,572 MB
of pixels copied back and forth**, and `complex_blends / passes = 1.0` - the blends share
no passes at all. Phase 1's pages carry 99.7% of blended groups, but a complex blend still
takes its own pass because it reads the destination in a shader.

So a crowded room costs one render pass and one destination read-back **per multiply
layer**. That is the stutter, and it is very likely also the ratchet in section 2.

## 4. The fix the other client already measured

`render/wgpu/src/blend.rs` in that source carries `TrivialBlend::MultiplyOpaque`.
Premultiplied Flash multiply is `src*(1-dst.a) + dst*(1-src.a) + src*dst`, which no blend
state expresses - but when `dst.a == 1` the first term vanishes and what is left is
exactly:

```
src_factor: Dst, dst_factor: OneMinusSrcAlpha, operation: Add    (colour and alpha both)
```

The alpha channel comes out `dst.a`, so an opaque target stays opaque and the condition
holds for the next layer. They apply it when the target was cleared with an opaque colour
and the blend's whole content is a single shape or bitmap draw, and the sub-target, its
pass and its composite all disappear.

Their measurements (A/B alternating every 45 s inside one session): **64% of complex
blends qualify**, about 188 of 280 blend passes a frame; **+50.8% frame rate** weighted
across scene-size bins, growing with crowding (+9.6% at 150-200 complex blends a frame,
+60.1% at 300-350); -16 to -18 ms a frame; render passes per complex blend -50.6%;
multisample resolve traffic -61.1%.

It is also **safe under multisampling**, which is what blocks `Add`/`Screen`/`Subtract` in
our fast path. Multiply over an opaque destination is linear in the source and cannot
saturate (`dst*src + dst*(1-a) <= dst` for premultiplied source), so applying it per
sample and resolving gives the same answer as resolving and then applying it - the
algebra that fails for `Add` works here.

What it costs us: our `trivial_fast_path` rejects `RenderShape` because the shape
pipelines are only built with premultiplied blending, so shape pipelines per blend state
are part of the job. The bitmap case is free - and `fastpath_fallback_multiple_draws` is
**128 for the whole 40-minute run**, so essentially every blended group we see is already
a single draw. The only gate is the opaque backdrop.

Second, smaller: let complex blends that do not overlap share one pass by setting the
pipeline per draw rather than per pass (their notes/0.6.37.md; they went from 457 blended
layers in 455 passes to grouping properly). Ours is at 1.0 blends per pass today.

Third: we still multisample the small offscreen targets - 7,841 sample-count pool misses
this run. They measured ~135 such targets a frame at ~2 megapixels each.

## 4a. What was ported, and what it measured

`MultiplyOpaque` is in, on `fix/aqw-render-final-performance`
(`render/wgpu/src/blend.rs`, `surface/commands.rs`, `surface.rs`), with the
counters carried into the diagnostic build as `aqw-final-diag-5`.

A multiply whose group is a single `RenderBitmap` over a destination known
opaque is now drawn straight onto that destination with the blend state above.
Opacity comes from the target's clear colour, asked before the mode is consumed,
and is only ever allowed to under-report: a target cleared from an existing
texture counts as transparent, a group being drawn into a page is over the
page's transparent clear rather than over the surface, and `Alpha`, `Erase` or a
PixelBender blend ends the claim for the rest of the surface, since those are
the three that write alpha back out of it.

Measured on 220 single-bitmap multiplies over an opaque stage
(`tests/multiply_on_draw.rs`):

| | render passes | destination copies |
|---|---|---|
| a target per group | 441 | 220 |
| pages and batching (what ships today) | 20 | 220 |
| **carried on the draw** | **1** | **0** |

The two paths are not byte-identical and the tests say so rather than tolerating
it quietly. The shader un-premultiplies both colours, multiplies, and
premultiplies the result back; the blend unit multiplies directly. That round
trip is worth up to one level per channel, and a pixel drawn through two
overlapping multiplies can carry the last bit of each. The unit tests assert the
algebra exactly in f64; the render tests allow one level per multiply a pixel
passes through and fail on anything more. The existing suites - `blend_pages`,
`blend_render_targets`, `cache_capacity` - all still pass.

One thing found while writing those tests, worth keeping: in a scene of 220
overlapping sprites placed on exact texel boundaries, the three paths disagree
at 687 pixels by up to 175 levels. Off the boundaries they agree exactly. That
is the nearest sampler breaking a tie, which each path is entitled to break
differently because each places the group on the pixel grid its own way - the
same reason `blend_pages.rs` has `off_the_half_pixel`. It is not a bug and it is
not new, but a scene that lands on ties will show it.

Shapes stay on the old path and are counted rather than guessed at
(`multiply_on_draw_shape`). Two things are in the way, not one: the colour and
gradient pipelines are only built with premultiplied blending, and a shape is a
mesh of several draws that a target composites into one picture before the blend
applies - carrying the blend on each draw would multiply the destination once
per draw wherever two of them overlap. So the shape case needs a per-mesh check
as well as new pipelines, and the counter is what says whether that is worth
building.

## 5. Where their memory problem differs from ours

Their own census (`aether/src/memory_census.rs`) documents the climb the client is
describing: five hours idle in Yulgar took `movies` from 81 to 5,479 without a single
release and RSS from 1.2 GB to 31.7 GB, 5.7 MB per SWF, perfectly linear. They could not
release them - AQW loads into a shared application domain, and dropping definitions on
unload "stripped every avatar in the room to a black silhouette" - so they de-duplicate by
URL instead (4,693 movies for 1,935 distinct URLs) and accept one permanent copy of every
distinct file.

Our run does not have that: `movies` rises and falls between 64 and 348 and `swf_bytes`
holds at 21 MB over 40 minutes, which is the phase 1 ephemeron library fix doing its job.
Their remaining climb and ours are not the same bug.

## 6. One thing to fix in our own reporting

`verify.txt` says "frames drawn 11,086,877, mean frame time 0.2 ms". Two 5-second windows
(t = 1687 and 1807) contain 369,414 samples between them, because the player keeps ticking
while the window is occluded and does not render. Those two windows drag the aggregate to
nonsense. The per-window rows are fine; the summary needs to drop windows whose tick rate
says the window was not on screen.
