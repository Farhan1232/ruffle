//! Which part of the process the unexplained memory is in.
//!
//! Four rounds have now established what the climb is *not*: not textures
//! (209,996 created against 209,704 dropped over the client's 40 minutes, and
//! every kind balances), not anything resident in the Rust heap (a counting
//! allocator's live bytes with no trend across the same run), not the SWF
//! library, not a pool without a bound, and not wgpu holding destroyed
//! resources for want of a poll. What is left is memory the process owns that
//! none of our own counters can see: private bytes minus the Rust heap's live
//! bytes minus the whole graphics allocator reserve went from 161 MB to
//! 3,864 MB while every one of those counters stayed flat.
//!
//! The operating system can say where it is, and this is the arithmetic that
//! reads its answer. Committed memory is either private to the process or a
//! mapping of something else, so the growth lands in one column or the other
//! and the two want opposite fixes: a heap that has stopped giving pages back
//! is fixed inside this process, and the graphics driver's own memory is not.
//!
//! **One honest limit, stated here because it decides what the next step is.**
//! Private commit is not the same thing as "the Rust heap". The display
//! driver's user-mode half keeps its own private heaps, and those are private
//! commit too. So a `PrivateHeap` verdict narrows the search to the process's
//! own allocators - ours and the driver's - and the region-size histogram is
//! what separates those two, because a heap grows in segments and a driver's
//! arenas do not look like segments. A `MappedDriver` verdict has no such
//! ambiguity.

use std::fmt::Write as _;

const MB: f64 = 1024.0 * 1024.0;

/// The sampled quantities the classification is made from.
///
/// Every one of these is already a column in the memory log; this is the
/// subset the question needs, kept in memory so the summary can be rewritten
/// from the whole run at every sample rather than only at the end. A run that
/// is killed - which is how most of them end - still leaves a complete answer
/// on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    pub elapsed_s: f64,
    pub working_set: u64,
    pub private_bytes: u64,
    pub rust_heap: u64,
    pub allocator_allocated: u64,
    pub allocator_reserved: u64,
    pub committed_private: u64,
    pub committed_mapped: u64,
    pub committed_image: u64,
    pub private_regions: u64,
    pub largest_private_region: u64,
    /// Committed private bytes by region size, in the buckets of
    /// [`AddressSpace`](crate::memory_reporter::AddressSpace).
    pub private_by_size: [u64; 5],
    pub render_passes_last_frame: u64,
    pub complex_blends: u64,
    pub destination_copies: u64,
    pub offscreen_builds: u64,
}

impl Sample {
    /// Private bytes that neither the Rust heap nor the graphics allocator can
    /// account for. The quantity the whole investigation is about.
    pub fn unexplained(&self) -> f64 {
        self.private_bytes as f64 - self.rust_heap as f64 - self.allocator_reserved as f64
    }
}

/// Which side of the process the growth is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    /// Committed private memory is what is growing. The process's own
    /// allocators are holding it - ours or the driver's, which the region
    /// histogram then separates.
    PrivateHeap,
    /// Mapped memory is what is growing, which is how a display driver's
    /// memory appears. Nothing inside the client will move it.
    MappedDriver,
    /// Both are growing enough to matter, so neither explains it alone.
    Both,
    /// Nothing is growing over the measured window.
    Flat,
    /// The log has no address-space census, so the question cannot be asked of
    /// it. Older builds, before `aqw-final-diag-5`.
    NoCensus,
    /// There is growth, and the census does not account for it. Worth knowing
    /// rather than rounding away: it would mean the two are being measured at
    /// different moments, or that the growth is in something neither column
    /// covers.
    Unaccounted,
}

impl Category {
    pub fn verdict(self) -> &'static str {
        match self {
            Category::PrivateHeap => "PRIVATE/HEAP is growing",
            Category::MappedDriver => "MAPPED/DRIVER is growing",
            Category::Both => "BOTH are growing",
            Category::Flat => "nothing is growing after warm-up",
            Category::NoCensus => "UNDECIDED - this log has no address-space census",
            Category::Unaccounted => "UNDECIDED - the census does not account for the growth",
        }
    }
}

/// Growth of one series over the classified window, as a rate and a total.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Growth {
    /// Least-squares slope, in MB per minute.
    pub mb_per_min: f64,
    /// Last sample minus first, in MB.
    pub delta_mb: f64,
    /// The same slope over each half of the window, so a series that has
    /// levelled off can be told from one that has not. This is the difference
    /// between a lower peak and a fixed leak.
    pub first_half_mb_per_min: f64,
    pub second_half_mb_per_min: f64,
}

/// What the run says.
#[derive(Clone, Debug)]
pub struct Classification {
    pub samples: usize,
    pub warmup_s: f64,
    /// The window the rates are measured over, in seconds since the start.
    pub window: (f64, f64),
    pub working_set: Growth,
    pub private_bytes: Growth,
    pub rust_heap: Growth,
    pub allocator_reserved: Growth,
    pub committed_private: Growth,
    pub committed_mapped: Growth,
    pub unexplained: Growth,
    pub category: Category,
}

/// Warm-up before the rates are measured. A session spends its first minutes
/// loading the game and the first room it lands in, which is real memory that
/// is not the leak, and including it makes every rate look like a climb.
const DEFAULT_WARMUP_S: f64 = 300.0;

/// Growth below this is noise at the scale being chased.
///
/// Two rather than one, and measured rather than picked: a session that
/// oscillates by 80 MB either side of a settled figure - which is what a room
/// filling and emptying looks like - regresses to 1.18 MB/min purely from
/// where its window happens to start and stop. One would call that a leak. The
/// thing actually being chased ran at 33 MB/min for half an hour, so there is
/// no risk of hiding it at this threshold, and the rate is printed either way
/// so the reader is never left with only the verdict.
const NOISE_MB_PER_MIN: f64 = 2.0;

/// The share of the growth one column has to hold to be named as the cause.
const DOMINANT_SHARE: f64 = 0.65;

fn slope_mb_per_min(samples: &[Sample], value: impl Fn(&Sample) -> f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean_t = samples.iter().map(|s| s.elapsed_s).sum::<f64>() / n;
    let mean_v = samples.iter().map(&value).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance = 0.0;
    for sample in samples {
        let dt = sample.elapsed_s - mean_t;
        covariance += dt * (value(sample) - mean_v);
        variance += dt * dt;
    }
    if variance == 0.0 {
        return 0.0;
    }
    // Bytes per second becomes megabytes per minute.
    covariance / variance * 60.0 / MB
}

fn growth(samples: &[Sample], value: impl Fn(&Sample) -> f64 + Copy) -> Growth {
    let half = samples.len() / 2;
    Growth {
        mb_per_min: slope_mb_per_min(samples, value),
        delta_mb: match (samples.first(), samples.last()) {
            (Some(first), Some(last)) => (value(last) - value(first)) / MB,
            _ => 0.0,
        },
        first_half_mb_per_min: slope_mb_per_min(&samples[..half], value),
        second_half_mb_per_min: slope_mb_per_min(&samples[half..], value),
    }
}

/// Reads the run.
///
/// `warmup_s` is how much of the start to leave out; a run too short to spare
/// that much leaves out a fifth of itself instead, so a quick check still says
/// something rather than nothing.
pub fn classify(samples: &[Sample]) -> Classification {
    classify_with_warmup(samples, DEFAULT_WARMUP_S)
}

pub fn classify_with_warmup(samples: &[Sample], warmup_s: f64) -> Classification {
    let run_length = samples.last().map(|s| s.elapsed_s).unwrap_or(0.0);
    let warmup = if run_length > warmup_s * 2.0 {
        warmup_s
    } else {
        run_length / 5.0
    };
    let window: Vec<Sample> = samples
        .iter()
        .copied()
        .filter(|s| s.elapsed_s >= warmup)
        .collect();

    let working_set = growth(&window, |s| s.working_set as f64);
    let private_bytes = growth(&window, |s| s.private_bytes as f64);
    let rust_heap = growth(&window, |s| s.rust_heap as f64);
    let allocator_reserved = growth(&window, |s| s.allocator_reserved as f64);
    let committed_private = growth(&window, |s| s.committed_private as f64);
    let committed_mapped = growth(&window, |s| s.committed_mapped as f64);
    let unexplained = growth(&window, |s| s.unexplained());

    let has_census = window
        .iter()
        .any(|s| s.committed_private > 0 || s.committed_mapped > 0);

    let category = if !has_census {
        Category::NoCensus
    } else if private_bytes.mb_per_min.abs() < NOISE_MB_PER_MIN
        && unexplained.mb_per_min.abs() < NOISE_MB_PER_MIN
    {
        Category::Flat
    } else {
        let private_growth = committed_private.mb_per_min.max(0.0);
        let mapped_growth = committed_mapped.mb_per_min.max(0.0);
        let total = private_growth + mapped_growth;
        if total < NOISE_MB_PER_MIN {
            Category::Unaccounted
        } else if private_growth / total >= DOMINANT_SHARE {
            Category::PrivateHeap
        } else if mapped_growth / total >= DOMINANT_SHARE {
            Category::MappedDriver
        } else {
            Category::Both
        }
    };

    Classification {
        samples: samples.len(),
        warmup_s: warmup,
        window: (
            window.first().map(|s| s.elapsed_s).unwrap_or(0.0),
            window.last().map(|s| s.elapsed_s).unwrap_or(0.0),
        ),
        working_set,
        private_bytes,
        rust_heap,
        allocator_reserved,
        committed_private,
        committed_mapped,
        unexplained,
        category,
    }
}

/// The whole answer as a page of text, written beside the log so that reading
/// the run does not require the log to be opened at all.
pub fn summary(samples: &[Sample], classification: &Classification) -> String {
    let mut out = String::new();
    let row = |out: &mut String, name: &str, growth: &Growth| {
        let _ = writeln!(
            out,
            "  {name:<26} {:>9.1} MB {:>9.2} MB/min   halves {:>7.2} -> {:>7.2}",
            growth.delta_mb,
            growth.mb_per_min,
            growth.first_half_mb_per_min,
            growth.second_half_mb_per_min,
        );
    };

    let _ = writeln!(out, "=== WHERE THE MEMORY IS ============================");
    let _ = writeln!(
        out,
        "{} samples; rates measured from {:.0}s to {:.0}s, leaving out the first {:.0}s as warm-up",
        classification.samples,
        classification.window.0,
        classification.window.1,
        classification.warmup_s,
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {:<26} {:>12} {:>13}   {}",
        "series", "change", "rate", "first half -> second half"
    );
    row(&mut out, "working set", &classification.working_set);
    row(&mut out, "private bytes", &classification.private_bytes);
    row(&mut out, "Rust heap (live)", &classification.rust_heap);
    row(
        &mut out,
        "GPU allocator reserved",
        &classification.allocator_reserved,
    );
    row(
        &mut out,
        "committed private",
        &classification.committed_private,
    );
    row(
        &mut out,
        "committed mapped",
        &classification.committed_mapped,
    );
    let _ = writeln!(out);
    row(&mut out, "UNEXPLAINED", &classification.unexplained);
    let _ = writeln!(
        out,
        "  (unexplained = private bytes - Rust heap live - GPU allocator reserve)"
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "  VERDICT: {}", classification.category.verdict());
    match classification.category {
        Category::PrivateHeap => {
            let _ = writeln!(
                out,
                "  Committed private is where it is going. That is this process's own\n  \
                 allocators - ours and the display driver's user-mode half, which keeps\n  \
                 private heaps of its own. The region histogram below separates them: a\n  \
                 heap grows in segments, a driver's arenas do not."
            );
        }
        Category::MappedDriver => {
            let _ = writeln!(
                out,
                "  Mapped memory is where it is going, which is how a display driver's\n  \
                 memory appears. Nothing inside the client will move it; the next step is\n  \
                 the rendering workload - backend, render passes per submission, and which\n  \
                 subsystem's absence flattens the slope."
            );
        }
        Category::Flat => {
            let _ = writeln!(
                out,
                "  Memory settled after warm-up. That is the success criterion, not a\n  \
                 lower peak - check that the session really was crowded for its length."
            );
        }
        Category::NoCensus => {
            let _ = writeln!(
                out,
                "  This log predates the census. Re-run with a build reporting\n  \
                 committed_private_bytes and committed_mapped_bytes."
            );
        }
        Category::Both | Category::Unaccounted => {
            let _ = writeln!(
                out,
                "  Neither column holds enough of the growth to be named on its own.\n  \
                 The per-sample rows are the thing to read."
            );
        }
    }

    if !samples.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "=== OVER TIME ======================================");
        let _ = writeln!(
            out,
            "  {:>6} {:>7} {:>8} {:>7} {:>8} {:>8} {:>8} {:>7} {:>6} {:>10} {:>10} {:>9}",
            "t",
            "workset",
            "private",
            "heap",
            "gpu.live",
            "gpu.rsv",
            "priv.cmt",
            "mapped",
            "passes",
            "cplx.blend",
            "dst.copies",
            "offscreen"
        );
        // A dozen rows however long the run is, so the shape is readable at a
        // glance and the CSV is only needed for the detail.
        let step = (samples.len() / 12).max(1);
        let last_is_shown = (samples.len() - 1) % step == 0;
        let tail = samples.last().filter(|_| !last_is_shown);
        for sample in samples.iter().step_by(step).chain(tail) {
            let _ = writeln!(
                out,
                "  {:>6.0} {:>7.0} {:>8.0} {:>7.0} {:>8.0} {:>8.0} {:>8.0} {:>7.0} {:>6} {:>10} {:>10} {:>9}",
                sample.elapsed_s,
                sample.working_set as f64 / MB,
                sample.private_bytes as f64 / MB,
                sample.rust_heap as f64 / MB,
                sample.allocator_allocated as f64 / MB,
                sample.allocator_reserved as f64 / MB,
                sample.committed_private as f64 / MB,
                sample.committed_mapped as f64 / MB,
                sample.render_passes_last_frame,
                sample.complex_blends,
                sample.destination_copies,
                sample.offscreen_builds,
            );
        }
        let _ = writeln!(
            out,
            "  (megabytes; the last three are totals for the session so far)"
        );
    }

    if let Some(last) = samples.last() {
        let _ = writeln!(out);
        let _ = writeln!(out, "=== AT THE END =====================================");
        let _ = writeln!(
            out,
            "  working set {:.0} MB, private {:.0} MB, Rust heap {:.0} MB,\n  \
             GPU allocator {:.0} MB live in {:.0} MB reserved, unexplained {:.0} MB",
            last.working_set as f64 / MB,
            last.private_bytes as f64 / MB,
            last.rust_heap as f64 / MB,
            last.allocator_allocated as f64 / MB,
            last.allocator_reserved as f64 / MB,
            last.unexplained() / MB,
        );
        let _ = writeln!(
            out,
            "  committed: {:.0} MB private over {} regions (largest {:.0} MB), \
             {:.0} MB mapped, {:.0} MB images",
            last.committed_private as f64 / MB,
            last.private_regions,
            last.largest_private_region as f64 / MB,
            last.committed_mapped as f64 / MB,
            last.committed_image as f64 / MB,
        );
        let names = ["<64 KB", "64 KB-1 MB", "1-16 MB", "16-256 MB", ">256 MB"];
        let _ = writeln!(out, "  private commit by region size:");
        for (name, bytes) in names.iter().zip(last.private_by_size.iter()) {
            let _ = writeln!(out, "    {name:<12} {:>9.1} MB", *bytes as f64 / MB);
        }
        let _ = writeln!(
            out,
            "  renderer, at the last sample: {} render passes in the frame, \
             {} complex blends and\n  {} destination copies over the session, \
             {} offscreen textures built",
            last.render_passes_last_frame,
            last.complex_blends,
            last.destination_copies,
            last.offscreen_builds,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(elapsed_s: f64) -> Sample {
        Sample {
            elapsed_s,
            ..Default::default()
        }
    }

    /// A series of `count` samples five seconds apart, with `value` bytes added
    /// to `field` every minute.
    fn ramp(count: usize, per_minute: f64, field: fn(&mut Sample, u64)) -> Vec<Sample> {
        (0..count)
            .map(|i| {
                let elapsed_s = i as f64 * 5.0;
                let mut sample = at(elapsed_s);
                field(&mut sample, (elapsed_s / 60.0 * per_minute) as u64);
                sample
            })
            .collect()
    }

    #[test]
    fn a_known_ramp_is_measured_exactly() {
        let samples = ramp(240, 10.0 * MB, |s, v| s.private_bytes = v);
        let growth = super::growth(&samples, |s| s.private_bytes as f64);
        assert!(
            (growth.mb_per_min - 10.0).abs() < 0.01,
            "a 10 MB/min ramp measured as {:.3}",
            growth.mb_per_min
        );
    }

    #[test]
    fn private_growth_is_named_as_the_heap() {
        let mut samples = ramp(240, 20.0 * MB, |s, v| s.private_bytes = v);
        for sample in &mut samples {
            sample.committed_private = sample.private_bytes;
            sample.committed_mapped = 64 * 1024 * 1024;
        }
        let classification = classify(&samples);
        assert_eq!(classification.category, Category::PrivateHeap);
    }

    #[test]
    fn mapped_growth_is_named_as_the_driver() {
        let mut samples = ramp(240, 20.0 * MB, |s, v| s.private_bytes = v);
        for sample in &mut samples {
            sample.committed_mapped = sample.private_bytes;
            sample.committed_private = 64 * 1024 * 1024;
        }
        let classification = classify(&samples);
        assert_eq!(classification.category, Category::MappedDriver);
    }

    #[test]
    fn a_settled_session_is_not_called_a_leak() {
        // The success criterion: oscillating, not climbing.
        let samples: Vec<Sample> = (0..240)
            .map(|i| {
                let elapsed_s = i as f64 * 5.0;
                let wobble = ((i % 20) as f64 - 10.0) * 8.0 * MB;
                Sample {
                    elapsed_s,
                    private_bytes: (3000.0 * MB + wobble) as u64,
                    committed_private: (3000.0 * MB + wobble) as u64,
                    committed_mapped: 64 * 1024 * 1024,
                    ..Default::default()
                }
            })
            .collect();
        let c = classify(&samples);
        assert_eq!(
            c.category,
            Category::Flat,
            "a session oscillating by 80 MB regressed to {:.2} MB/min and was \
             called a leak",
            c.private_bytes.mb_per_min
        );
    }

    #[test]
    fn a_log_without_the_census_says_so_rather_than_guessing() {
        let samples = ramp(240, 20.0 * MB, |s, v| s.private_bytes = v);
        assert_eq!(classify(&samples).category, Category::NoCensus);
    }

    /// The client's own 40-minute Windows run, sampled every two minutes.
    ///
    /// This is the series every figure in the investigation has been quoted
    /// from, so the arithmetic that will read the next run is checked against
    /// the last one: the Rust heap flat, the allocator reserve falling, and the
    /// unexplained remainder climbing at about 33 MB a minute for the whole
    /// half hour after warm-up. It has no census columns, so it also has to
    /// come back undecided rather than confident.
    #[test]
    fn the_clients_run_reads_the_way_it_was_reported() {
        const RUN: &[(f64, u64, u64, u64, u64, u64)] = &[
            (0.0, 208306176, 403337216, 33358671, 18566688, 201326592),
            (
                120.0, 2211004416, 3999178752, 405791187, 312268800, 1543503872,
            ),
            (
                241.0, 2694066176, 4572176384, 403659556, 645594256, 1543503872,
            ),
            (
                362.0, 2869264384, 5044633600, 343640370, 511481184, 1811939328,
            ),
            (
                482.0, 3102060544, 5302657024, 463168033, 444723392, 1811939328,
            ),
            (
                603.0, 3194359808, 5411115008, 382003344, 334153792, 1811939328,
            ),
            (
                723.0, 3210637312, 5439033344, 345263354, 490583344, 1811939328,
            ),
            (
                843.0, 3210670080, 5440229376, 350855058, 338301024, 1811939328,
            ),
            (
                964.0, 3501502464, 5766479872, 375124136, 315037696, 1811939328,
            ),
            (
                1084.0, 3529568256, 5795155968, 454623357, 489024800, 1811939328,
            ),
            (
                1205.0, 3607220224, 5877452800, 450909254, 507952032, 1811939328,
            ),
            (
                1325.0, 3844997120, 6166073344, 370430002, 511383552, 1811939328,
            ),
            (
                1445.0, 3920531456, 6241525760, 523269220, 621712096, 1811939328,
            ),
            (
                1566.0, 4137263104, 6458793984, 535147871, 386712160, 1811939328,
            ),
            (
                1686.0, 4134445056, 6187724800, 540426516, 264754384, 1543503872,
            ),
            (
                1806.0, 4134453248, 6187593728, 540693385, 264349792, 1543503872,
            ),
            (
                1926.0, 3937107968, 6017966080, 443762959, 584787968, 1543503872,
            ),
            (
                2047.0, 3957133312, 6037090304, 317834160, 498565584, 1543503872,
            ),
            (
                2167.0, 3956015104, 6037417984, 321037047, 311213248, 1543503872,
            ),
            (
                2287.0, 3928846336, 5982937088, 380542492, 335357712, 1543503872,
            ),
            (
                2407.0, 3928965120, 5983571968, 387539790, 334757072, 1543503872,
            ),
            (
                2412.0, 3928965120, 5983571968, 387891808, 334589072, 1543503872,
            ),
        ];
        let samples: Vec<Sample> = RUN
            .iter()
            .map(
                |&(elapsed_s, working_set, private_bytes, rust_heap, allocated, reserved)| Sample {
                    elapsed_s,
                    working_set,
                    private_bytes,
                    rust_heap,
                    allocator_allocated: allocated,
                    allocator_reserved: reserved,
                    ..Default::default()
                },
            )
            .collect();

        let c = classify(&samples);
        assert_eq!(
            c.category,
            Category::NoCensus,
            "this run predates the census and must not be classified from it"
        );
        assert!(
            (c.window.0 - 362.0).abs() < 1.0,
            "the warm-up should end at the first sample past five minutes, not {:.0}s",
            c.window.0
        );
        assert!(
            c.rust_heap.mb_per_min.abs() < 1.0,
            "the Rust heap is flat in this run; measured {:.2} MB/min",
            c.rust_heap.mb_per_min
        );
        assert!(
            c.allocator_reserved.mb_per_min < 0.0,
            "the allocator's reserve fell over this run; measured {:.2} MB/min",
            c.allocator_reserved.mb_per_min
        );
        assert!(
            (c.unexplained.mb_per_min - 33.3).abs() < 1.0,
            "the unexplained remainder climbed at about 33 MB/min; measured {:.2}",
            c.unexplained.mb_per_min
        );
        assert!(
            (c.unexplained.delta_mb - 1109.0).abs() < 5.0,
            "and by about 1,109 MB over the window; measured {:.1}",
            c.unexplained.delta_mb
        );

        // The page it writes has to name the numbers it was given.
        let text = summary(&samples, &c);
        assert!(text.contains("UNEXPLAINED"), "{text}");
        assert!(text.contains("no address-space census"), "{text}");
        println!("{text}");
    }

    #[test]
    fn a_short_run_still_leaves_out_a_warm_up() {
        let samples = ramp(24, 20.0 * MB, |s, v| s.private_bytes = v);
        let c = classify(&samples);
        assert!(
            c.warmup_s > 0.0 && c.warmup_s < 120.0,
            "a two-minute run should shorten its warm-up, not skip it: {:.0}s",
            c.warmup_s
        );
    }
}
