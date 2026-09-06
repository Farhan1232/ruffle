//! Periodic memory sampling, used to measure SWF retention over a play session.
//!
//! Enabled with `--memory-report <FILE>`. Every interval it writes one CSV row
//! combining the process' resident set size with Ruffle's own accounting of
//! what each still-loaded movie is keeping alive, so that growth in RSS can be
//! attributed to (or ruled out as) movies that were supposed to be unloaded.

use ruffle_core::Player;
use ruffle_core::memory_report::MemoryReport;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The system allocator with a running total of live bytes, so that the
/// memory report can say how much of the process is Rust heap as opposed to
/// graphics driver and other native memory.
pub struct CountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            LIVE_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

/// Bytes currently allocated through the Rust allocator.
pub fn rust_heap_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Frame intervals seen since the last sample, in milliseconds, so that a
/// run can be judged on smoothness as well as on footprint.
#[derive(Default)]
struct FrameTimes {
    samples: Vec<f64>,
    last_frame: Option<Instant>,
}

impl FrameTimes {
    /// Records the gap since the previous frame. Called once per tick.
    fn record(&mut self, now: Instant) {
        if let Some(last) = self.last_frame {
            self.samples
                .push(now.duration_since(last).as_secs_f64() * 1000.0);
        }
        self.last_frame = Some(now);
    }

    /// `(count, mean, p50, p95, p99, max, frames over 33 ms, over 100 ms)`,
    /// draining what has been recorded so far.
    fn drain_stats(&mut self) -> (usize, f64, f64, f64, f64, f64, usize, usize) {
        if self.samples.is_empty() {
            return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);
        }
        let long = self.samples.iter().filter(|ms| **ms > 33.0).count();
        let very_long = self.samples.iter().filter(|ms| **ms > 100.0).count();
        let sum: f64 = self.samples.iter().sum();
        let count = self.samples.len();
        self.samples
            .sort_by(|a, b| a.partial_cmp(b).expect("Frame times are finite"));
        let at = |q: f64| {
            let i = ((count as f64 - 1.0) * q).round() as usize;
            self.samples[i]
        };
        let stats = (
            count,
            sum / count as f64,
            at(0.50),
            at(0.95),
            at(0.99),
            self.samples[count - 1],
            long,
            very_long,
        );
        self.samples.clear();
        stats
    }
}

/// The whole header, core's columns wrapped in the ones the frontend adds.
///
/// One function rather than a literal at the write site, so that the test which
/// checks the verifier can find its columns is checking the same string the log
/// is written with.
fn csv_header(core_columns: String) -> String {
    // Built from the same array the buckets are filled from, so a bucket
    // cannot be added without its column appearing beside it.
    let buckets: String = SIZE_BUCKET_NAMES
        .iter()
        .map(|name| format!(",private_{name}"))
        .collect();
    format!(
        "rss_bytes,peak_rss_bytes,private_bytes,peak_private_bytes,{core_columns},\
         peak_gpu_texture_bytes_sampled,rust_heap_bytes,\
         frames,frame_ms_mean,frame_ms_p50,frame_ms_p95,frame_ms_p99,frame_ms_max,\
         frames_over_33ms,frames_over_100ms,\
         committed_private_bytes,committed_mapped_bytes,committed_image_bytes,\
         committed_private_regions,largest_private_region_bytes{buckets}"
    )
}

pub struct MemoryReporter {
    output: BufWriter<File>,
    interval: Duration,
    started: Instant,
    last_sample: Option<Instant>,
    /// Retained bytes at the first sample, so the log states growth directly.
    baseline_retained: Option<usize>,
    frame_times: FrameTimes,
    /// Cumulative created-texture bytes at the previous sample, for the churn
    /// rate between the two.
    last_created_bytes: Option<u64>,
    last_elapsed: f64,
    /// The header names one group of columns per texture kind, and only the
    /// renderer knows those names, so it is written with the first sample.
    header_written: bool,
    /// Every sample so far, so the classification can be made from the whole
    /// run at every sample rather than only when the process exits - which is
    /// not how these runs end.
    history: Vec<crate::memory_classify::Sample>,
    /// Where the summary is written, beside the log.
    summary_path: std::path::PathBuf,
    /// Running maxima across samples, so a row states the high-water mark
    /// beside the current value.
    peak_gpu_texture_bytes: usize,
    peak_rss: usize,
    peak_private: usize,
}

impl MemoryReporter {
    pub fn new(path: &Path, interval: Duration) -> Result<Self, std::io::Error> {
        let output = BufWriter::new(File::create(path)?);
        tracing::info!(
            "Memory instrumentation {} active",
            ruffle_core::memory_report::INSTRUMENTATION_VERSION
        );

        Ok(Self {
            output,
            interval,
            started: Instant::now(),
            last_sample: None,
            baseline_retained: None,
            frame_times: FrameTimes::default(),
            last_created_bytes: None,
            last_elapsed: 0.0,
            header_written: false,
            history: Vec::new(),
            summary_path: path.with_extension("summary.txt"),
            peak_gpu_texture_bytes: 0,
            peak_rss: 0,
            peak_private: 0,
        })
    }

    /// Rewrites the summary beside the log.
    ///
    /// Every sample, and overwriting rather than appending, because a
    /// diagnostic session ends by being closed or killed rather than by
    /// returning, and a summary that only exists on a clean exit is a summary
    /// nobody ever reads.
    fn write_summary(&self) {
        let classification = crate::memory_classify::classify(&self.history);
        let text = crate::memory_classify::summary(&self.history, &classification);
        if let Err(e) = std::fs::write(&self.summary_path, text) {
            tracing::error!("Could not write memory summary: {e}");
        }
    }

    /// Takes a sample if the interval has elapsed. Cheap to call every frame.
    pub fn maybe_sample(&mut self, player: &mut Player) {
        let now = Instant::now();
        self.frame_times.record(now);
        if let Some(last) = self.last_sample
            && now.duration_since(last) < self.interval
        {
            return;
        }
        self.last_sample = Some(now);

        let report = player.mutate_with_update_context(MemoryReport::capture);
        let elapsed = now.duration_since(self.started).as_secs_f64();
        let rss = resident_set_size().unwrap_or(0);
        let private = private_bytes().unwrap_or(0);
        self.peak_rss = self.peak_rss.max(rss);
        self.peak_private = self.peak_private.max(private);
        self.peak_gpu_texture_bytes = self.peak_gpu_texture_bytes.max(report.gpu_texture_bytes);

        if !self.header_written {
            self.header_written = true;
            if let Err(e) = writeln!(
                self.output,
                "{}",
                csv_header(MemoryReport::csv_header_for(report.texture_kind_names()))
            ) {
                tracing::error!("Could not write memory report header: {e}");
            }
        }

        let rust_heap = rust_heap_bytes();
        let space = address_space();
        let allocator = report.allocator.unwrap_or_default();
        self.history.push(crate::memory_classify::Sample {
            elapsed_s: elapsed,
            working_set: rss as u64,
            private_bytes: private as u64,
            rust_heap: rust_heap as u64,
            allocator_allocated: allocator.allocated_bytes,
            allocator_reserved: allocator.reserved_bytes,
            committed_private: space.private,
            committed_mapped: space.mapped,
            committed_image: space.image,
            private_regions: space.private_regions,
            largest_private_region: space.largest_private_region,
            private_by_size: space.private_by_size,
            render_passes_last_frame: report.work.render_passes,
            complex_blends: report.work.complex_blends,
            destination_copies: report.work.destination_copies,
            offscreen_builds: report.work.offscreen_pool_misses.iter().sum(),
        });
        self.write_summary();
        let (frames, mean, p50, p95, p99, max, long, very_long) = self.frame_times.drain_stats();
        if let Err(e) = writeln!(
            self.output,
            "{},{},{},{},{},{},{},{},{mean:.2},{p50:.2},{p95:.2},{p99:.2},{max:.2},{long},{very_long},\
             {},{},{},{},{},{},{},{},{},{}",
            rss,
            self.peak_rss,
            private,
            self.peak_private,
            report.to_csv_row(elapsed),
            self.peak_gpu_texture_bytes,
            rust_heap,
            frames,
            space.private,
            space.mapped,
            space.image,
            space.private_regions,
            space.largest_private_region,
            space.private_by_size[0],
            space.private_by_size[1],
            space.private_by_size[2],
            space.private_by_size[3],
            space.private_by_size[4],
        )
        .and_then(|_| self.output.flush())
        {
            tracing::error!("Could not write memory report: {e}");
        }

        let retained = report.swf_bytes + report.bitmap_decoded_bytes;
        let baseline = *self.baseline_retained.get_or_insert(retained);
        let mut kind_breakdown = String::new();
        for (i, name) in report.texture_kind_names().iter().enumerate() {
            let bytes = report
                .texture_kind_live_bytes
                .get(i)
                .copied()
                .unwrap_or_default();
            let count = report
                .texture_kind_live_counts
                .get(i)
                .copied()
                .unwrap_or_default();
            let _ = write!(
                kind_breakdown,
                " [{name} {} MiB/{count}]",
                bytes / (1024 * 1024)
            );
        }
        let churn_mib_per_s = self
            .last_created_bytes
            .replace(report.texture_bytes_created)
            .map(|previous| {
                let span = elapsed - self.last_elapsed;
                if span > 0.0 {
                    (report.texture_bytes_created.saturating_sub(previous) as f64)
                        / (1024.0 * 1024.0)
                        / span
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        self.last_elapsed = elapsed;

        tracing::info!(
            "memory @{elapsed:.0}s: rss {} MiB, private {} MiB (peak {} MiB), \
             rust heap {} MiB, {} movies retaining {} MiB \
             (+{} MiB since first sample), {} pending loaders, {} class aliases, \
             gc {} MiB / {} objects (+{} MiB external), gpu {} textures {} MiB + buffers {} MiB, \
             {} meshes {} MiB, textures live {} MiB (peak {} MiB){}, \
             work: {} passes, {} blend targets, {} bind groups built ({:.0}% cached), \
             {:.0}% of blends drawn without a target, \
             driver {} allocations {} MiB live of {} MiB reserved, \
             pools idle: main {} MiB / {} sizes, offscreen {} MiB / {} sizes, buffers {} MiB, \
             pool reuse {}/{}, churn {} MiB/s, \
             frame ms mean {:.1} p95 {:.1} p99 {:.1} max {:.1} ({} over 33 ms){}{}",
            rss / (1024 * 1024),
            private / (1024 * 1024),
            self.peak_private / (1024 * 1024),
            rust_heap / (1024 * 1024),
            report.movies.len(),
            retained / (1024 * 1024),
            retained.saturating_sub(baseline) / (1024 * 1024),
            report.pending_loaders,
            report.class_aliases,
            report.gc_allocation / (1024 * 1024),
            report.gc_objects,
            report.gc_external_bytes / (1024 * 1024),
            report.gpu_textures,
            report.gpu_texture_bytes / (1024 * 1024),
            report.gpu_buffer_bytes / (1024 * 1024),
            report.meshes,
            report.mesh_bytes / (1024 * 1024),
            report.tracked_texture_bytes / (1024 * 1024),
            report.peak_texture_bytes / (1024 * 1024),
            kind_breakdown,
            report.work.render_passes,
            report.work.blend_targets,
            report.work.bind_groups_created,
            {
                let total = report.work.bind_group_cache_hits + report.work.bind_group_cache_misses;
                if total > 0 {
                    100.0 * report.work.bind_group_cache_hits as f64 / total as f64
                } else {
                    0.0
                }
            },
            if report.work.fastpath_eligible > 0 {
                100.0 * report.work.fastpath_used as f64 / report.work.fastpath_eligible as f64
            } else {
                0.0
            },
            report.hal.memory_allocations,
            report.allocator.map(|a| a.allocated_bytes).unwrap_or(0) / (1024 * 1024),
            report.allocator.map(|a| a.reserved_bytes).unwrap_or(0) / (1024 * 1024),
            report.main_pool_idle_bytes / (1024 * 1024),
            report.main_pool_size_classes,
            report.offscreen_pool_idle_bytes / (1024 * 1024),
            report.offscreen_pool_size_classes,
            report.buffer_pool_idle_bytes / (1024 * 1024),
            report.pool_reuses,
            report.pool_misses,
            churn_mib_per_s,
            mean,
            p95,
            p99,
            max,
            long,
            report.top_pool_keys(),
            report.top_movies(5),
        );
    }
}

/// Resident set size of this process, in bytes.
#[cfg(target_os = "linux")]
fn resident_set_size() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Memory this process has committed that is not shared with any other, in
/// bytes: "Commit Size" in Task Manager.
///
/// The pair to the working set, and the one that answers whether memory has
/// really been given back. A Windows process can hold committed private pages
/// it is no longer touching - the graphics driver keeps a system-memory
/// commitment for the video memory a process has allocated, and does not
/// release it when the peak passes - so the working set can settle at a couple
/// of gigabytes while private bytes stay at the highest the session ever
/// reached. Reading both, beside the renderer's own peak texture bytes, is what
/// distinguishes "the pages are merely not resident" from "the memory is still
/// owned".
#[cfg(target_os = "linux")]
fn private_bytes() -> Option<usize> {
    let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    let field = |name: &str| -> Option<usize> {
        let line = rollup.lines().find(|line| line.starts_with(name))?;
        Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
    };
    Some(field("Private_Clean:")? + field("Private_Dirty:")?)
}

/// Working set size of this process, in bytes.
#[cfg(target_os = "windows")]
fn resident_set_size() -> Option<usize> {
    windows_memory_counters().map(|counters| counters.WorkingSetSize)
}

#[cfg(target_os = "windows")]
fn private_bytes() -> Option<usize> {
    windows_memory_counters().map(|counters| counters.PrivateUsage)
}

#[cfg(target_os = "windows")]
fn windows_memory_counters()
-> Option<windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS_EX> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            // `GetProcessMemoryInfo` takes the shorter struct; passing the
            // longer one with its own size is how the documentation says to ask
            // for `PrivateUsage`.
            (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    };
    (ok != 0).then_some(counters)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn resident_set_size() -> Option<usize> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn private_bytes() -> Option<usize> {
    None
}

/// What the process's committed address space is made of.
///
/// The reason this exists: over the client's 40-minute session, private bytes
/// minus the Rust heap's live bytes minus the graphics allocator's whole
/// reserve went from 161 MB to 3,864 MB, and every counter the client and the
/// renderer keep was flat across the same run. The growth is real and it is
/// outside both instruments, so the next question is which side of the process
/// it is on, and no counter we own can answer it.
///
/// The operating system can. Committed memory is either private to the process
/// - which is where a heap that has stopped giving pages back would show, the
/// Rust heap included - or a mapping of something else, which is where the
/// graphics driver's own memory shows, since it maps what it allocates rather
/// than committing it privately. One number each, sampled beside the rest.
///
/// It is a walk of the region table rather than of the pages, so it costs
/// microseconds and is safe to take every five seconds.
#[derive(Clone, Copy, Default)]
pub struct AddressSpace {
    pub private: u64,
    pub mapped: u64,
    pub image: u64,
    pub private_regions: u64,
    pub largest_private_region: u64,
    /// Committed private bytes by region size, in the buckets named by
    /// [`SIZE_BUCKET_NAMES`].
    ///
    /// This is what separates the two things private commit can be. A heap
    /// grows by taking segments, so a heap that is retaining shows as a great
    /// many regions in the middle buckets; a driver's arenas are few and large.
    /// Without it, "private is growing" leaves our allocator and the display
    /// driver's own heaps indistinguishable.
    pub private_by_size: [u64; 5],
}

/// The edges of [`AddressSpace::private_by_size`], in bytes.
const SIZE_BUCKET_EDGES: [u64; 4] = [64 * 1024, 1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024];

pub const SIZE_BUCKET_NAMES: [&str; 5] = [
    "under_64kb",
    "64kb_to_1mb",
    "1mb_to_16mb",
    "16mb_to_256mb",
    "over_256mb",
];

fn size_bucket(size: u64) -> usize {
    SIZE_BUCKET_EDGES
        .iter()
        .position(|edge| size < *edge)
        .unwrap_or(SIZE_BUCKET_EDGES.len())
}

#[cfg(windows)]
pub fn address_space() -> AddressSpace {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

    let mut info: SYSTEM_INFO = unsafe { zeroed() };
    // SAFETY: `info` is a valid, correctly sized `SYSTEM_INFO` to write into.
    unsafe { GetSystemInfo(&mut info) };
    let mut address = info.lpMinimumApplicationAddress as usize;
    let maximum = info.lpMaximumApplicationAddress as usize;

    let mut census = AddressSpace::default();
    while address < maximum {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { zeroed() };
        // SAFETY: `mbi` is a valid, correctly sized structure to write into,
        // and `address` is only ever a region boundary this walk was given.
        let written = unsafe {
            VirtualQuery(
                address as *const _,
                &mut mbi,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if written == 0 {
            break;
        }
        let size = mbi.RegionSize as u64;
        if mbi.State == MEM_COMMIT {
            match mbi.Type {
                MEM_PRIVATE => {
                    census.private += size;
                    census.private_regions += 1;
                    census.largest_private_region = census.largest_private_region.max(size);
                    census.private_by_size[size_bucket(size)] += size;
                }
                MEM_MAPPED => census.mapped += size,
                MEM_IMAGE => census.image += size,
                _ => {}
            }
        }
        // A region that does not advance would spin the walk forever.
        let next = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize);
        if next <= address {
            break;
        }
        address = next;
    }
    census
}

/// The same census from `/proc/self/maps`, so the columns mean something on the
/// machine the work is done on as well as on the one it is measured on.
///
/// An anonymous mapping is the counterpart of Windows' private commit, and a
/// file-backed one of its mapped and image regions. The split is coarser -
/// Linux does not distinguish an image from any other file mapping here, so
/// executables and libraries are counted as images by their permissions - but
/// the question it answers is the same one.
#[cfg(target_os = "linux")]
pub fn address_space() -> AddressSpace {
    let mut census = AddressSpace::default();
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return census;
    };
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let Some(range) = fields.next() else { continue };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (u64::from_str_radix(start, 16), u64::from_str_radix(end, 16))
        else {
            continue;
        };
        let size = end.saturating_sub(start);
        let permissions = fields.next().unwrap_or("");
        // offset, device, inode, then the path if there is one.
        let path = fields.nth(3).unwrap_or("");
        if path.is_empty() {
            census.private += size;
            census.private_regions += 1;
            census.largest_private_region = census.largest_private_region.max(size);
            census.private_by_size[size_bucket(size)] += size;
        } else if permissions.contains('x') {
            census.image += size;
        } else {
            census.mapped += size;
        }
    }
    census
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn address_space() -> AddressSpace {
    AddressSpace::default()
}

#[cfg(test)]
mod address_space_tests {
    use super::address_space;

    /// The verifier reads these by name, and a column it cannot find is a
    /// figure nobody sees. The core half of the header has its own test; this
    /// is the half the desktop frontend adds.
    #[test]
    fn the_header_carries_the_census_columns() {
        let header = super::csv_header(String::from("elapsed_s"));
        for wanted in [
            "committed_private_bytes",
            "committed_mapped_bytes",
            "committed_image_bytes",
            "committed_private_regions",
            "largest_private_region_bytes",
            "rust_heap_bytes",
            "private_under_64kb",
            "private_over_256mb",
        ] {
            assert!(
                header.split(',').any(|column| column.trim() == wanted),
                "the memory report header has no `{wanted}` column"
            );
        }

        // The row writer emits one field per bucket; a header that has grown a
        // bucket the row has not would silently shift every column after it.
        let bucket_columns = super::SIZE_BUCKET_NAMES
            .iter()
            .filter(|name| {
                let wanted = format!("private_{name}");
                header.split(',').any(|column| column.trim() == wanted)
            })
            .count();
        assert_eq!(
            bucket_columns,
            super::SIZE_BUCKET_NAMES.len(),
            "the header carries {bucket_columns} of the {} bucket columns",
            super::SIZE_BUCKET_NAMES.len()
        );
    }

    /// A smoke test, because a census that silently reports nothing would look
    /// exactly like a process that is holding nothing - and the whole point of
    /// these columns is to be believed when they are large.
    #[test]
    fn the_census_finds_this_process() {
        let space = address_space();
        assert!(
            space.private > 1024 * 1024,
            "a running test process has more than a megabyte of private commit; \
             the census reported {} bytes over {} regions",
            space.private,
            space.private_regions
        );
        assert!(
            space.private_regions > 0 && space.largest_private_region > 0,
            "regions {} largest {}",
            space.private_regions,
            space.largest_private_region
        );
        println!(
            "private {:.1} MB over {} regions (largest {:.1} MB), mapped {:.1} MB, image {:.1} MB",
            space.private as f64 / (1024.0 * 1024.0),
            space.private_regions,
            space.largest_private_region as f64 / (1024.0 * 1024.0),
            space.mapped as f64 / (1024.0 * 1024.0),
            space.image as f64 / (1024.0 * 1024.0),
        );
    }
}
