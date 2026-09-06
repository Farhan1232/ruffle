# The memory investigation: what was ruled out, what was fixed, what is still open

Written 2026-09-06 against the client's 40-minute Windows run of the phase 2
diagnostic build (RTX 5060 Ti, Windows 11 Home, 15.6 GB, `aqw-memory.csv`,
`windows-ram.csv`, `verify.txt`).

**The leak is not fixed.** Two real defects were found and fixed, one of them
worth roughly a gigabyte of committed memory, and one of them a counter that has
been pointing the investigation at the wrong thing. Neither of them is the climb.
What the climb *is* has been narrowed to two candidates that no instrument in the
process could tell apart, and the instrument that can has been built and is in
the next diagnostic build. Nothing here should be read as "the leak is solved".

## 1. What the run actually shows

| | start | end | shape |
|---|---|---|---|
| working set | 307 MB | 3,747 MB (peak 3,948) | climbs for 28 min, flat for 13 |
| private bytes | 501 MB | 5,706 MB (peak 6,163) | same |
| Rust heap, live | 32 MB | 370 MB | oscillates 300-670, **no trend** |
| graphics allocator, reserved | 192 MB | 1,472 MB | **flat from minute 4** |
| graphics allocator, in use | 18 MB | 319 MB | oscillates 250-620 |
| movies resident | 1 | 121 | oscillates 64-348 |
| SWF bytes | 0 | 21 MB | flat |

Private bytes minus the Rust heap's live bytes minus the entire graphics
allocator reserve: **161 MB at the start, 3,864 MB at the end**. That difference
is the whole problem, and every counter we own is flat across it.

It ratchets rather than leaks smoothly: +5,017 MB across 241 five-second
windows, -1,468 MB across 55. The largest single steps (+540, +482, +363 MB in
five seconds) all land in windows at 11-17 fps with `movies` spiking and about a
thousand render passes a frame - room loads and crowded rooms. It steps up there
and does not come back.

## 2. What has been ruled out, and on what evidence

**Textures and buffers are not leaking.** 209,996 textures created against
209,704 dropped over the 40 minutes; 47,683 MB created against 47,460 MB
dropped. The 292 that survive are 223 MB and every one is accounted for by kind:
275 `cacheAsBitmap` surfaces (161 MB), 12 pooled targets (55 MB), 5 bitmaps
(6.7 MB), 0 temporaries, 0 offscreen. Every kind balances individually.

**Nothing resident in the Rust heap is leaking.** The heap counter is a global
allocator that counts every allocation and free, so an `Arc` cycle, a cache
without eviction, a retained display object or a growing map would all show
there. It oscillates between 300 and 670 MB with no trend across 40 minutes.
This rules out the whole class in one measurement.

**The SWF library is not accumulating**, which is worth saying because it is the
leak the other AQW client documents and cannot fix: five hours idle in Yulgar
took their resident movies from 81 to 5,479 and their RSS from 1.2 to 31.7 GB.
Ours rises and falls between 64 and 348 with SWF bytes flat at 21 MB. The
phase 1 ephemeron fix is doing its job.

**The pools are bounded.** Both have an idle budget and a demand-aware trim. The
offscreen pool held 40 MB idle on average - against a 192 MB budget - and never
exceeded 117 MB.

**wgpu is not deferring destruction for want of a poll.** This was the strongest
a-priori suspect, because Ruffle's render loop never calls `Device::poll`. It
does not need to: `Queue::submit` calls `device.maintain(PollType::Poll)` itself
(`wgpu-core-30.0.1/src/device/queue.rs:1541`), which triages completed
submissions and runs the map callbacks the staging belt's `recall()` depends on.
The belt is also correctly `finish()`ed and `recall()`ed around every submit
(`render/wgpu/src/backend.rs:1563-1591`).

## 3. Fixed: the graphics allocator holds four times what it uses

`gpu_allocator` destroys a memory block only when the *whole* block is empty
(`gpu-allocator-0.28.0/src/vulkan/mod.rs:658-686`), so the block size is also
the granularity at which one surviving allocation pins memory. wgpu's default
`MemoryHints::Performance` asks for device blocks of 128-256 MB and host blocks
of 64-128 MB (`wgpu-hal-30.0.1/src/lib.rs:439-456`), and Ruffle has always
passed `memory_hints: Default::default()`.

The log: **1,472 MB of reserve across 7 blocks against 319 MB of live
allocations**, flat for the whole session. Roughly 1.15 GB of committed memory
that nothing is using, and on Windows a process's committed memory is charged to
it whether it is touched or not.

`MemoryHints::MemoryUsage` asks the same allocator for 8-64 MB device blocks and
4-32 MB host blocks - a quarter of the waste per pinned block. It costs more
`vkAllocateMemory` calls, which this content makes few of (the offscreen pool
serves 84.5% of its requests from idle textures), so it is on by default, with
`RUFFLE_DEVICE_MEMORY=performance|memory|<min>:<max>` to A/B without a rebuild.

Measured here, on a 150-second soak of the same scene both ways:

| | reserved | blocks | in use |
|---|---|---|---|
| `performance` (the default) | 192.0 MB | 2 | 109.8 MB |
| `memory` | 156.0 MB | 7 | 109.8 MB |

**And then the longer runs said not to ship it as the default.** Three
15-minute soaks, same scene, live set flat at 110 MB throughout all of them:

| setting | reserve, first half | reserve, second half | verdict |
|---|---|---|---|
| `performance` | 192-320 MB | 192-192 MB | ceiling fell; **passes** |
| `memory`, run 1 | 156 MB | 188 MB | ceiling rose 32 MB; fails |
| `memory`, run 2 | 156-188 MB | 188-220 MB | ceiling rose 32 MB; fails |

That is repeatable, not noise, and the mechanism is the same one the change is
about read backwards: a block is only freed when it is *wholly* empty, so
cutting the block size does not remove the fragmentation, it moves it from
inside the blocks to between them. Nine small blocks each holding a survivor
free nothing, where two large ones holding the same 110 MB were stable.

So the default is unchanged - `Performance`, exactly as before - and the frugal
setting is a switch. On the client's machine the ratio is nothing like this
one's: 319 MB live in 1,472 MB reserved, against 110 MB in 192 MB here, and a
4.6x waste has far more room to win than a 1.75x one. **It is worth one run of
`RUFFLE_DEVICE_MEMORY=memory` on his machine, and that run decides it.** What I
will not do is ship a default on the strength of a prediction that my own
measurement argues against.

## 4. Fixed: a counter that has been blaming the wrong thing

`offscreen_miss_evicted_by_budget` held **87,341 of the 192,635 offscreen
texture builds** in the client's log, which reads as "the idle budget is too
small and is throwing away textures the next frame wants". The same log shows
the pool holding 40 MB against a 192 MB budget it never came close to filling.
Both cannot be true.

The counter was wrong. A key entered the evicted set the first time the budget
gave it up and **was never taken out again**, so every later drop of that size -
including the ordinary dormancy drop, which is a different problem with the
opposite fix - was reported as the budget biting. A key now leaves the list as
soon as it is registered again, and dormancy drops are counted as `dormant`.

This matters because it decides the next fix. The offscreen pool built 192,635
textures worth 35.7 GB over the session across **6,678 distinct size classes** -
each blended group's target is sized to its exact content bounds, so nearly
every size is its own pool with too little traffic to be worth retaining. If the
next run says `dormant`, the answer is to quantise those sizes the way phase 2
quantised `cacheAsBitmap` textures (621,413 rebuilds became 37,929). If it says
`evicted_by_budget`, the answer is a larger budget. Until now the log could not
distinguish them.

## 5. Built: the measurement that can settle the climb

Committed memory is either private to the process or a mapping of something
else. A heap that has stopped giving pages back is private commit - and the Rust
heap is the candidate, because the counter we have reports *live* bytes and says
nothing about what the allocator holds underneath. The graphics driver maps what
it allocates rather than committing it privately, so its memory lands in the
other column.

`VirtualQuery` on Windows, `/proc/self/maps` on Linux, walking the region table
rather than the pages, so it costs microseconds. Five new columns:
`committed_private_bytes`, `committed_mapped_bytes`, `committed_image_bytes`,
`committed_private_regions`, `largest_private_region_bytes`. The verifier prints
them beside the Rust heap and the allocator reserve and subtracts, so one line
says how much is still unexplained and which side of the process it is on.

Private climbing with the Rust heap flat means the heap is retaining, and the
fix is an allocator that returns pages (mimalloc, or `HeapSetInformation` with
low-fragmentation tuning). Mapped climbing means the driver, and no amount of
work inside the client will touch it. **These want opposite fixes, which is why
guessing was not worth doing.**

## 6. Files changed

* `render/wgpu/src/backend.rs` - `device_memory_hints()`, and the device asks
  for it instead of taking the default.
* `render/wgpu/src/lib.rs` - `tuning::frugal_device_memory_enabled`, off by
  default and documented with the measurement that keeps it off.
* `render/wgpu/src/buffer_pool.rs` - `PoolMiss::Dormant`, the `dormant` key set,
  `re_admit`, and a unit test that the budget and dormancy are told apart.
* `desktop/src/memory_reporter.rs` - the address-space census, the header as one
  function, two tests.
* `desktop/Cargo.toml` - `Win32_System_Memory`, `Win32_System_SystemInformation`.
* `Verify-AqwLog.ps1` - the new columns, a "where the private bytes are"
  section, and the occluded-window fix from earlier in the phase.

## 7. Risks

* The Windows half of the census cannot be compiled here. It is typechecked
  against `windows-sys` 0.61.2 for `x86_64-pc-windows-gnu`, which catches API
  misuse but not linking. If the client's build fails, that is where to look.
* `MemoryHints::MemoryUsage` trades allocation count for reserve. Nothing here
  measured a frame-time cost, but this laptop is not the client's machine and
  the scene is not the game.
* The frugal allocator setting fails the soak's ceiling assertion, repeatably,
  which is why it is off. If it is ever turned on by default, that assertion has
  to be understood first rather than relaxed.
* The `size_history` map in the texture pool has no eviction and grew to 6,678
  entries (~160 KB) over the session. Left alone deliberately: it is the input
  to the trim policy that was tuned by measurement, and 160 KB is not the
  problem being chased.
