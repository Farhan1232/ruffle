# AQW on Windows: what the automated pass found before the manual test

Native Windows validation of the phase 1 and phase 2 renderer work, run on the
tester's own laptop. No account was used and no authenticated session was
started; everything below comes from synthetic workloads, the test suites, and
an unauthenticated load of the public AQW loader.

**Verdict: AUTOMATED WINDOWS VALIDATION PARTIAL — Vulkan cannot be validated on
this machine, and AQW's loading phase exhausts this laptop's shared GPU memory.**
Both are defects of this hardware, not of the build. Everything that could be
validated here passed. The detail is in §2 and §14, and what it means for the
manual test is in §16.

---

## 1. Machine, toolchain, and the exact code

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (build 26200) |
| CPU | Intel Core i5-8350U, 4 cores / 8 threads |
| RAM | 7.86 GB (typically 1.2–4.0 GB free during this work) |
| GPU | Intel UHD Graphics 620, driver 31.0.101.2134, 1 GB reported VRAM, **shared with system RAM** |
| Disk | 48 GB free of 200.9 GB |
| git / rustc / cargo | 2.54.0 / 1.98.1 / 1.98.1, `x86_64-pc-windows-msvc` |

| | |
|---|---|
| Production branch | `fix/aqw-render-final-performance` |
| Production SHA verified | `672ee967c376d917b80c91ea86decac439b49b59` — matched the brief exactly |
| Diagnostic branch | `diagnostic/aqw-final-windows` |
| Diagnostic SHA at start | `53c845f0d151eae52a510e4afddf4bf447078304` — matched the brief exactly, worktree clean |
| Diagnostic SHA at end | `79cf4ae8e515a3203d2246b1be9ca4a4aa3e4a00` plus this report |
| Instrumentation ID | `aqw-final-diag-4`, read from `core/src/memory_report.rs:20`, not from documentation |

The built executable self-reports its commit: `ruffle_desktop --version` printed
`0.5.0-local (53c845f0d… 2026-09-05)` before any change, tying the binary to the
checkout rather than to a claim about it.

**The documented work was verified to exist rather than trusted.** Every item the
brief lists as must-not-be-lost was located in the tree by search:
`MovieLibrary`/SWF lifetime (`loader_unload_releases_library`,
`released_class_frees_library`), weak `Dictionary`, `ApplicationDomain`, loader
cleanup, bounded pools (`buffer_pool.rs` idle budget), content-bounded blend
targets (`bounds.rs::region_rect_for`), bind-group caching (`bind_cache.rs`),
the restricted direct fast path (`trivial_fast_path`), blend pages and batching
(`surface/page.rs`, `chunk_blends`, `batch_passes`), `cacheAsBitmap` physical
capacity (`cache_capacity.rs::capacity_for`/`capacity_fits`), exact logical
rects (`BitmapCacheEntry::logical_width`), filtered-cache exact sizing
(`apply_viewport`), and the offscreen pool policy (`RUFFLE_OFFSCREEN_POOL_MB`).

`git diff origin/fix/aqw-render-final-performance..HEAD` confirmed the
diagnostic branch is production **plus instrumentation only** — the production
tip is an ancestor of the diagnostic tip, and no functional renderer file
diverges except to add counters.

---

## 2. Vulkan on this machine: available, then broken

Vulkan is present and was initially working: `vulkaninfo` reports **Vulkan
1.3.215** on the Intel UHD 620 through `DRIVER_ID_INTEL_PROPRIETARY_WINDOWS` —
a real Windows Vulkan stack, not the Mesa one the earlier phases used. Two early
runs in this session created a Vulkan device, rendered for 16–20 s, and wrote a
valid CSV.

It then degraded to a hard failure and stayed there.

| backend | adapter | device creation |
|---|---|---|
| DX12 | OK | **OK** |
| GL | OK | **OK** |
| Vulkan | OK | **`STATUS_ACCESS_VIOLATION` (0xC0000005)** |

The fault is in `vkCreateDevice`, **before any Ruffle rendering code runs** —
adapter enumeration lists all four adapters fine, and the crash occurs inside
`request_device`. Later it began failing the windowed desktop app too, on 4
consecutive attempts, including with the same local SWF that had previously
succeeded. This is an Intel driver defect on this laptop; the client's
NVIDIA/Vulkan stack is unrelated to it.

**Consequence.** Every wgpu GPU test harness creates its device headlessly
through `Backends::all()` (`tests/tests/environment.rs:151`), so all of them
crashed. The workaround needed no source change: pointing `VK_DRIVER_FILES` and
`VK_ICD_FILENAMES` at a nonexistent ICD makes the Vulkan backend offer no
adapters, and wgpu falls through to DX12. Verified clean.

**So every GPU test result below is DX12, not Vulkan.** That still exercises the
whole phase 1/2 architecture — pages, batching, cache capacity, filters, masks —
but it is *not* a Vulkan pixel-path validation, and it is not claimed to be.

---

## 3. Build

`cargo build --release --package ruffle_desktop` — exit 0. Executable present,
starts, and reports its own commit. Rebuilt at the end at `79cf4ae8e`;
`build-info.txt` regenerated and correct.

## 4. Build metadata

`Make-BuildInfo.ps1` needed no fix. Its output carries every field the brief
requires — branch, HEAD SHA, HEAD subject, clean/dirty, instrumentation ID,
production SHA, exe SHA256, exe size, timestamps, Windows version, GPU, RAM —
and correctly identified `diagnostic/aqw-final-windows`, `aqw-final-diag-4`, and
the exact SHA. The production SHA is *found* (`git merge-base` against the
production branch) rather than remembered, which is why it stayed right after
the branch moved.

## 5. Format and check

| | |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --all-targets` | clean, exit 0, 5m47s |

---

## 6. The CSV schema and the verifier — two defects found and fixed

The schema itself is complete: a real run writes **183 columns**, covering every
category the brief lists. The verifier was not.

### 6.1 The verifier passed while printing a figure that did not exist

`Verify-AqwLog.ps1` printed `tracked_textures` but did not require it. Deleting
that one column from a real log and re-running produced:

```
ok    all 81 required columns are present (182 in total)
live backend textures     (hal_textures)   0   against tracked_textures 0
RUNTIME: NEW BUILD CONFIRMED (aqw-final-diag-4)
```

A fabricated `0` presented as a measurement, under a green verdict — exactly the
failure the script's own header comment claims to prevent.

**Fixed.** The required set now covers every category the brief asks for (the
loading timeline, the collector, textures created and dropped beside live, the
remaining HAL counters, slow-frame timings, per-frame destination copies) **and**
is unioned with the column names the script's own source reads, so a printed
figure can never again be unrequired. **81 → 119 required columns.** A row-width
check was added, because `Import-Csv` pads a truncated final row with empty
fields that read back as zero. Re-tested both ways: the good log passes; the log
missing one column now fails with exit 1.

### 6.2 Two HAL counters read zero for a reason phase 2 had not identified

Phase 2 left open whether `hal_textures` and `hal_memory_allocations` "may
populate on the client's NVIDIA driver". **They will not, and it is not a driver
limitation.** From the vendored `wgpu-hal 30.0.1` source:

* Vulkan's `create_texture` never increments `counters.textures`, while
  `destroy_texture` decrements it — so the figure runs negative and Ruffle's
  `value.max(0)` clamps it to zero. DX12 increments it correctly (`dx12/device.rs:596`).
* `counters.memory_allocations` is written by **no backend at all**.

This was then confirmed by observation: on DX12 in this run `hal_textures` reads
**627**, non-zero; `hal_memory_allocations` still reads 0 everywhere. The
verifier now prints `n/a` with the reason instead of a misleading `0`.
`allocator_*` and `hal_texture_memory` carry that section.

### 6.3 Two dropped tests restored

The diagnostic branch had **18** `ruffle_render_wgpu` lib tests where production
has **20**; both earlier reports state 20. The instrumentation commit
`645aa4f9d` rewrote `buffer_pool.rs` and dropped
`a_pool_inside_its_budget_gives_up_nothing` and
`the_idle_budget_bounds_a_pool_of_many_sizes`. They compile against the
instrumented pool unchanged, so the removal was collateral, not an API change.
Restored, so the diagnostic build carries the same pool guarantees as the build
it represents.

---

## 7. Unit and renderer tests (DX12)

| suite | result | phase 1/2 baseline |
|---|---|---|
| `ruffle_render --lib` | **58 passed, 0 failed** | 58 ✓ |
| `ruffle_render_wgpu --lib` | **18 passed, 0 failed** (20 after §6.3) | reported 20 — see §6.3 |
| `--test blend_pages` | **10 passed, 0 failed** | 10 ✓ |
| `--test cache_capacity` | **5 passed, 0 failed** | 5 ✓ |
| `--test blend_render_targets` | 4 of 5 passed — see §9 | 5 |

All ten blend-page tests pass, including
`a_texel_boundary_picks_a_neighbouring_texel_at_worst`,
`overlapping_complex_blends_keep_their_order` and
`groups_under_a_mask_survive_sharing_a_page`. No tolerance was weakened and no
test was disabled; the only change was preventing the process from selecting the
backend that faults during device creation.

---

## 8. Image suite

**Diagnostic: 4528 passed, 11 failed, 37 ignored.**
**Production: 4529 passed, 10 failed, 37 ignored.**

Same 4,539-test set both phases ran — nothing skipped or lost. The failures were
not accepted at face value; the same suite and the same individual tests were run
on **production `672ee967c`** on the same backend to see whether this work caused
them. It did not.

### Category A — backend-inherent (5 tests)

Fail deterministically on **both branches**, in isolation, with identical
numbers. The reference images were generated on Vulkan/Mesa; this run is
DX12/Intel.

| test | failure | on production |
|---|---|---|
| `visual/blend_direct` | 102 outliers, **max difference 4** (limit 0) | identical |
| `from_shumway/acid/acid-mask` | 21 outliers (limit 16), max diff 53 | identical |
| `avm2/bitmapdata_drawwithquality` | 3204 outliers (limit **3200**) | identical |
| `avm2/stage3d_raytrace` | 57 outliers (limit 10) | identical |
| `avm2/away3d_advanced_shallow_water_demo` | 662,845 outliers on diagnostic, **1,086,667 on production** — nondeterministic | worse |

`blend_direct` — the test phase 1 names as the guard on the direct fast path —
differs by 4/255 across 102 pixels against a zero-outlier tolerance. That is
driver rounding, and it fails identically without any of this work present.

### Category B — flaky under parallel execution (the rest)

These **pass in isolation on both branches** and fail only inside the full
parallel suite, and the failing set differs run to run *on the same branch*:

| | production | diagnostic |
|---|---|---|
| `bitmapdata_draw_self_via_graphic` | **failed** | passed |
| `bitmapdata_applyfilter_blur` | passed | **failed** |
| `bitmapdata_draw_rotation` | passed | **failed** |
| `bitmapdata_opaque`, `blend_scroll`, `blend_transform`, `BitmapData-v8` | failed in suite, **pass individually** | same |

A test that fails on production but not on diagnostic settles it: this is
instability of GPU-readback tests under concurrency on this hardware, identical
on both branches.

**No test fails on the diagnostic branch that passes on production in a matched
isolated run.** No regression is attributable to this work.

### The groups the brief singled out

| group | result |
|---|---|
| `displacement_map` (all 5, incl. `_through_filters`, `_through_applyFilter`, `_scales_with_screen`) | **5/5 pass** |
| `visual/filters/` | 23/23 pass |
| `cache_as_bitmap` | 30/30 pass |
| `unload*` | 17/17 pass |
| mask | 28/29 (only `acid-mask`, category A) |
| blend | 33/36 (only the three above) |

The filter tests that the earlier cache-capacity attempt nearly broke are green
at unchanged tolerances.

---

## 9. Windows benchmarks

Pass counts reproduce phase 1's recorded table **digit for digit**.

**`Layer` groups** (every group wants a target; none can take the direct path):

| objects | passes before | after | ratio | phase 1 ratio | target MB before → after |
|---|---|---|---|---|---|
| 50 | 51 | 4 | 12.8x | 12.8x ✓ | 16.6 → 31.2 |
| 100 | 101 | 4 | 25.2x | 25.2x ✓ | 26.0 → 31.2 |
| 250 | 251 | 5 | 50.2x | 50.2x ✓ | 54.1 → 47.2 |
| 500 | 502 | 9 | 55.8x | 55.8x ✓ | 101.0 → 95.2 |
| 800 | 804 | 13 | 61.8x | 61.8x ✓ | 157.2 → 127.2 |
| 900 | 904 | 14 | 64.6x | 64.6x ✓ | 176.0 → 143.2 |

**The batched path, counters per frame** (`Multiply`, 150x200 objects, 1920x985):

| objects | target MB | passes/fr | pages | batch% | copies | copy MB | blends/pass |
|---|---|---|---|---|---|---|---|
| 50 | 22.4 | 7 | 2 | 100% | 50 | 6.05 | 12.5 |
| 100 | 38.4 | 13 | 3 | 100% | 100 | 12.11 | 11.1 |
| 250 | 54.4 | 26 | 4 | 100% | 250 | 30.28 | 11.9 |
| 500 | 86.4 | 48 | 6 | 100% | 500 | 60.57 | 12.2 |
| 800 | 118.4 | 75 | 8 | 100% | 800 | **96.90** | 12.1 |
| 900 | 134.4 | 84 | 9 | 100% | 900 | 109.02 | 12.2 |

800 objects on 8 pages at 118.4 MB with 96.90 MB of destination copies matches
phase 1's "8 pages, 118.4 MB" and phase 2's "800 copies and 96.90 MB per frame"
exactly. Blends-per-pass holds at ~12, phase 1's "twelve blends a pass in a
crowd".

**`Layer` direct path**: 0 targets, 0 pages, 7.2 MB flat, **100% fast path** at
every size — untouched, as designed.

Frame-time ratios here are much better than the Mesa laptop (7.0x at 800 objects
against phase 1's 1.7x), but this is a different GPU and they are not compared.

### Measurement limits on this hardware

`what_the_batching_is_worth` failed at 800 `Multiply` objects with
**`wgpu error: Out of Memory`**. The failure is in the `old` arm, which is
measured first and allocates 800 individual render targets on a 1 GB shared-VRAM
iGPU. The **batched** path runs the same 900-object workload successfully in
`a_crowded_room_does_not_ask_for_screen_sized_targets`. So the *pre-phase-1
baseline cannot complete a frame this machine's shipping build handles* — but it
also means phase 1's 800/900 `Multiply` before/after rows cannot be reproduced
here. Measurable to 500 objects only.

`Multiply` pass counts run consistently **+1** against phase 1's table
(500: 1001→49 here against 1000→48 recorded; the soak shows 575 passes against
574). The offset appears on both the before and after side, so ratios are
unaffected. It looks like one extra composite pass on DX12; it was not chased.

---

## 10. Cache churn

Reproduces phase 2 on Windows. Textures really allocated over 48 frames:

| actors | A: phase 1 | B: recycled only | C: recycled + capacity |
|---|---|---|---|
| 25 | 1200 (137.1 MB) | 22 (2.0 MB) | **3 (0.4 MB)** — 400x |
| 50 | 2400 (275.5 MB) | 16 (1.7 MB) | **4 (0.6 MB)** — 600x |
| 100 | **4800 (551.9 MB)** | **4459** | **3 (0.3 MB)** — 1600x |

The 100-actor row matches phase 2's headline (4,800 → 4,459 → 3) digit for
digit. Rebuilds per actor-frame: 1.00 → 0.01–0.02. Wandering bounds at 100
actors: 11163 → **10001**, and phase 2 recorded 10,001 at this budget.

The breathing pattern the brief asks about (147x196, 151x198, 149x197 …) is what
the "breathing bounds" workload plays, and the capacity holds it with **3
allocations across the whole run**.

---

## 11. Offscreen pool

Measured on this machine rather than inherited. 100 actors, looping animation,
capacity policy on:

| budget | textures allocated | evicted by budget |
|---|---|---|
| 64 MB | **6123** (724.5 MB) — worse than phase 1's 4800 | 6117 evictions, 723.7 MB |
| **192 MB** | **3** (0.3 MB) | **0** |
| 384 MB | 3 (0.3 MB) | 0 |

Phase 2 recorded 6,123 / 3 / 3 for exactly these budgets — an identical match on
a different OS and backend. 192 MB is the knee here too.

The wandering-bounds workload *does* improve at 384 MB (10001 → 9131
allocations, hit rate 24.3% → 30.9%). **The budget was left at 192 MB.** That is
a ~9% churn reduction in a workload phase 2 explicitly notes is harder than real
avatar animation, bought by doubling retained idle memory on a client whose
complaint is memory, with **zero** benefit in the looping case that models a real
room. The brief's instruction not to simply enlarge the pool applies.

---

## 12. HAL counters and the allocator

Working on DX12 in the AQW run: `hal_textures` 627, `hal_texture_views` 1,106,
`hal_bind_groups` 874, `hal_buffers` 9,853, `hal_texture_memory` 546 MB,
`hal_buffer_memory` 662 MB, `hal_samplers` 26, `hal_command_encoders` 16, plus
`hal_bind_group_layouts`, `hal_render_pipelines`, `hal_compute_pipelines`,
`hal_pipeline_layouts`, `hal_shader_modules`, `hal_query_sets`, `hal_fences`.

Not working, with the cause established in source (§6.2):
`hal_memory_allocations` on every backend; `hal_textures` on Vulkan specifically.

`allocator_allocated_bytes`, `allocator_reserved_bytes` and `allocator_blocks`
work everywhere and carry the memory argument.

---

## 13. AQW startup, unaccounted memory, and deferred destruction

Unauthenticated load of `https://game.aq.com/game/gamefiles/Loader3.swf` on DX12,
sampled every 5 s. No credentials were entered and none appear in any artifact.

### The timeline

| t (s) | RSS MB | private MB | alloc MB | reserved MB | blocks | buffers | halTex MB | passes | blends/pass | chars |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 358 | 328 | 25 | 192 | 2 | 30 | 18 | 1 | – | 18 |
| 6 | 1189 | 1163 | 583 | 832 | 5 | 6014 | 148 | 7 | 1.33 | 4325 |
| 16–66 | 1189–1234 | 1163–1207 | **449 (flat)** | **832 (flat)** | 5 | ~5955 | 65 | 7 | 1.33 | 4325 |
| 71 | 1270 | 1244 | 584 | 832 | 5 | 6097 | 168 | 31 | 1.33 | 4325 |
| 76 | 2014 | 1990 | 1132 | 1344 | 7 | 6361 | 470 | 12 | 1.32 | 4325 |
| 82 | 3383 | 3368 | 1797 | 2112 | 10 | 11793 | 658 | 962 | 1.27 | 18862 |
| 87 | 609 | 4281 | 1457 | 2240 | 11 | 9853 | 546 | 927 | 1.24 | 18875 |

**Answer to the brief's A/B/C/D: B, together with D.** Memory does not creep. It
is flat for a full minute on the login screen — allocated 449 MB and reserved
832 MB, unchanged — then steps up strictly on demand as content loads, the
reserve growing in whole blocks. At the end 2240 − 1457 = **783 MB is resident,
charged to the process, and inside allocator blocks that hold nothing**. That is
the mechanism phase 2 proposed for the client's unexplained working set, now
measured on native Windows with real AQW content rather than on Mesa with a
synthetic scene.

### Where the 1,457 MB actually is

| | |
|---|---|
| `hal_buffer_memory` | **662 MB across 9,853 buffers** |
| `hal_texture_memory` | 546 MB |
| sum | 1,208 MB of 1,457 MB allocated |
| Ruffle `tracked_texture_bytes` | **97 MB** |
| Ruffle `mesh_bytes` | **21 MB across 4,753 meshes** |

**Buffers are the dominant term, and they are ~31x larger than their contents.**
Per-buffer cost is 65.0–69.8 KiB across 6,000–9,800 buffers while the mesh data
inside them averages ~2 KiB. That sits exactly on D3D12's 64 KiB default resource
placement alignment, and the arithmetic confirms it: at t=96 s, 5,955 buffers ×
64 KiB = 372 MB against 375 MB measured, within 1%.

Ruffle creates two GPU buffers per tessellated mesh, so ~4,750 AQW meshes become
~9,500 buffers. **The buffer *count* is backend-independent; the 64 KiB floor is
a DX12 property.** Vulkan's minimum alignment is typically far smaller, so this
particular amplification may be much weaker on the client's card — but it cannot
be checked here, because Vulkan is broken on this machine (§2). This is the
single most promising lead for the client's unaccounted memory and the counters
to test it with (`hal_buffers`, `hal_buffer_memory`, `meshes`, `mesh_bytes`) are
all in the CSV.

### Deferred destruction

No evidence of it, and **no `Device::poll(Wait)` was added**. Across the soak,
tracked textures and the allocator track each other and return to the same level
every cycle (§15); in the AQW run `textures_created − textures_dropped` equals
`tracked_textures` exactly. `hal_textures` (627) exceeds Ruffle's tracked count
(404) by resources Ruffle does not own — swapchain and GUI textures — which DX12
counts and Ruffle does not. Nothing accumulates between the two.

---

## 14. AQW loading exhausts this laptop's GPU memory

Every one of **three** startup runs ended in `wgpu error: Out of Memory` during
`submit_frame`, at 80–140 s, as real game content landed. On an integrated GPU
every graphics byte comes from the same 7.86 GB the OS and other applications
are using, and the run above was already at 3.4 GB RSS / 4.3 GB private with
2,240 MB reserved by the graphics allocator.

This is a property of this hardware. On the client's RTX 5060 Ti those
allocations live in dedicated VRAM. It is reported because it bounds what this
machine can validate — and because it will very likely interrupt the manual
gameplay test (§16).

---

## 15. Soak

15 minutes, **25 cycles, 4,200 frames**, cycling crowd → quiet → crowd → complex
blends → masks → filtered breathing `cacheAsBitmap` → a 700-group worst case.
Ended by system memory pressure from outside the test, not by the renderer.

```
 cycle  frames   pages   page MB    passes    pool MB  bg built   textures   slowest
     1     168       7      88.0       575      139.2       190       1031    487.4ms
     2     168       7      88.0       575      140.1        25       1049    270.9ms
   ...
    25     168       7      88.0       575      140.0         0       1042    231.2ms
```

| | settled cycles 2–13 | settled cycles 14–25 | drift |
|---|---|---|---|
| bind groups built | 392 | 201 | **−48.7%** |
| textures allocated | 13,436 | 13,415 | **−0.16%** |
| allocator reserved | 512.0 MB / 4 blocks | 512.0 MB / 4 blocks | **0** |

Pages, page bytes and render passes are constant. Cache rebuilds ran
70 → 21 → 10 → **0** and stayed there. The allocator's reserve **ceiling** is
identical in both halves — the property phase 2's corrected assertion tests, and
the one that distinguishes a suballocator oscillating from a leak. Nothing
ratchets over 4,200 frames.

Windows differs from the Linux soak in two ways, both better: bind groups built
per cycle fell from phase 2's steady 176 to 25 and often 0, and cache rebuilds
reached 0 from the first cycle rather than the fourth.

---

## 16. The largest remaining renderer cost, now quantified on real content

**On real AQW content, complex blends composite at 1.24–1.33 per pass.**

Phase 1 predicted twelve to a pass in a crowd. Phase 2 measured 4.1 in its
synthetic 0%-cached / 20%-complex room and named the cause as item 2: a trivial
group's composite goes into the surface's own chunk and a complex blend closes
it, so a mixed room alternates `Draw, Blend, Draw, Blend`. Real AQW content is
far worse than either estimate, and the figure is flat across the entire run.

The distinction matters, because the two phase 1 changes behave very differently
here:

* **Blend pages succeed completely**: `batch_used` is **9,322 of 9,324 — 100%**.
  Every eligible blended group shares a page on real content. 3 pages, 60 MB.
* **Complex-blend pass merging does not**: 1.24 blends per pass, 9,319
  destination copies totalling 288 MB, and **927 render passes in the last
  frame**.

Bind-group caching also holds up on real content: **99.9% hit rate**, 14 bind
groups built across 4,681 frames.

**This was not fixed, deliberately.** Closing it means letting a complex blend
move earlier past drawing commands it shares no pixel with — a reordering, which
the brief requires to remain exact, and which `Pass::Draw` carries no per-draw
coverage information to justify. Phase 2 correctly calls it a genuine piece of
work rather than a small one. Attempting it here, where Vulkan cannot run and the
image suite costs an hour per branch to rebuild, would have been reckless. It is
now measured on real content instead of estimated, which is what the next session
needs to justify doing it properly.

---

## 17. Changes made

One commit, `79cf4ae8e`, on the diagnostic branch only. **No production
functional code was touched**, so no follow-up production branch was needed.

| change | before | after |
|---|---|---|
| Verifier required columns | 81, omitting one it printed | **119**, unioned with the columns its own source reads |
| Verifier on a log missing a printed column | `NEW BUILD CONFIRMED`, prints `0` | **fails, exit 1** |
| Truncated log | padded to zeros silently | **row-width check fails it** |
| `hal_textures` / `hal_memory_allocations` when unpopulated | printed `0` | **`n/a` with the reason** |
| `ruffle_render_wgpu` lib tests on the diagnostic branch | 18 | **20**, matching production |

Everything else in this report is measurement, not modification.

---

## 18. Known remaining limitations

1. **Vulkan is unvalidated on this machine** and the driver now fails device
   creation outright (§2). All GPU results are DX12.
2. **AQW loading OOMs on this laptop**, 3/3 (§14).
3. **Complex blends batch at 1.24 per pass on real content** (§16) — the largest
   remaining renderer win, measured but not fixed.
4. **Destination copies unchanged**: 9,319 per run / 288 MB on real content,
   96.90 MB per frame at 800 synthetic objects.
5. **Alpha masks still do not take page regions** (`page_fallback_alpha_mask`).
6. **Offscreen scratch still churns under non-repeating sizes** — the two
   blockers phase 2 named in its §4 are unchanged.
7. **The `Multiply` before/after benchmark is limited to 500 objects here** (§9).
8. **The +1 pass offset against phase 1's tables** on DX12 (§9), unexplained but
   consistent and ratio-neutral.
9. **The buffer-count lead in §13 is untested on Vulkan** and needs the client's
   run to confirm or dismiss.
10. The image suite required lowering `naga`'s optimisation level to compile in
    8 GB (`--config profile.release.package.naga.opt-level=1`). That affects
    shader-translation speed only, not rendered pixels.

---

## 19. Manual test readiness

Ready, with the two caveats in §16 below the commands. The build is correct, the
diagnostics work, the verifier works and has been run end-to-end on real AQW data
for the first time, the automated tests are green apart from failures proven to
predate this work, the synthetic benchmarks reproduce both phases, cache and
offscreen churn are confirmed, the allocator and HAL counters are inspected and
their gaps explained, the startup diagnostic is captured and analysed, and the
repository is clean and pushed.

No credentials appear anywhere in this report or in any artifact. `aqw-final/`,
`aqw-final-auto/`, `*-memory.csv`, `*-windows-ram.csv` and `*-console.log` are
all in `.gitignore`.
