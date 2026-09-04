//! Live-memory accounting for loaded SWFs.
//!
//! Ruffle keeps one [`MovieLibrary`] per `SwfMovie`, and that map is *weakly*
//! keyed on the movie (see [`crate::library::Library`]). A movie's decoded
//! characters therefore stay resident for exactly as long as somebody still
//! holds a strong `Arc<SwfMovie>`. When content is unloaded but its memory is
//! never returned, the movie is still in this report and its strong count is
//! above the one reference the library itself holds.
//!
//! This module walks that state so a leak can be measured instead of guessed
//! at: run it once per interval while switching zones, and a movie that should
//! be gone shows up as a row that never disappears.

use std::fmt::Write as _;

use crate::context::UpdateContext;

/// Identifies this instrumentation, so a log can be tied to the build that
/// produced it. Bump it whenever the columns change.
pub const INSTRUMENTATION_VERSION: &str = "aqw-blend-overhead-diag-3";

/// What a single still-resident movie is keeping alive.
#[derive(Debug, Clone)]
pub struct MovieMemory {
    pub url: String,
    /// Strong `Arc<SwfMovie>` references, excluding the one this report holds.
    ///
    /// A movie that only its own library still references reads as 1. Anything
    /// higher is an outside reference keeping the movie (and its whole library)
    /// alive.
    pub strong_refs: usize,
    /// Size of the movie's decompressed SWF data.
    pub swf_bytes: usize,
    pub characters: usize,
    /// Strong `Arc<SwfMovie>` clones held from inside this movie's own library
    /// - its characters plus the library's own handle on the movie.
    ///
    /// When this equals `strong_refs`, nothing outside the library needs the
    /// movie any more and the only thing keeping it resident is the library
    /// that is keyed on it.
    pub self_refs: usize,
    pub bitmaps: usize,
    /// Bitmaps that have been uploaded to the render backend, and so are also
    /// holding GPU memory. Ruffle never releases these handles on its own.
    pub uploaded_bitmaps: usize,
    /// Bytes of still-compressed bitmap data held by the library.
    pub bitmap_source_bytes: usize,
    /// Bytes those bitmaps occupy once decoded to RGBA, whether or not they
    /// have been uploaded yet. This is the dominant cost for AQW-style content.
    pub bitmap_decoded_bytes: usize,
    pub sounds: usize,
    pub fonts: usize,
    /// Whether the movie's library still points at an AVM2 `ApplicationDomain`.
    pub has_domain: bool,
    /// Whether the display object this movie was loaded into is still
    /// reachable. Content that is gone but still listed here has not been
    /// swept yet.
    pub content_alive: bool,
}

/// A whole-player snapshot, cheap enough to take every frame.
#[derive(Debug, Clone, Default)]
pub struct MemoryReport {
    pub movies: Vec<MovieMemory>,
    pub swf_bytes: usize,
    pub bitmap_source_bytes: usize,
    pub bitmap_decoded_bytes: usize,
    pub characters: usize,
    /// Loaders still registered with the `LoadManager`. Should return to zero
    /// once every load settles; a number that only grows is itself a leak.
    pub pending_loaders: usize,
    /// `registerClassAlias` entries. These are strong roots with no eviction,
    /// so each one pins a class, its translation unit and its movie forever.
    pub class_aliases: usize,
    /// Bytes of live `Gc` objects.
    pub gc_allocation: usize,
    pub gc_objects: usize,
    /// Bytes reported to the collector as external allocations owned by
    /// `Gc` objects: movie libraries' SWF data and bitmap sources,
    /// `BitmapData` pixels, `cacheAsBitmap` textures.
    pub gc_external_bytes: usize,
    /// GPU memory the render backend reports holding, if it can tell: this
    /// is where decoded bitmaps, cached display objects and filter targets
    /// end up, and it is not part of any movie library's accounting.
    pub gpu_textures: usize,
    pub gpu_texture_bytes: usize,
    pub gpu_buffer_bytes: usize,
    /// Tessellated shape meshes alive in the renderer and their vertex and
    /// index bytes. Counted by Ruffle itself, so available on every backend.
    pub meshes: usize,
    pub mesh_bytes: usize,
    /// Textures Ruffle created and still holds, and their approximate pixel
    /// bytes; counted by Ruffle, so available on every backend.
    pub tracked_textures: usize,
    pub tracked_texture_bytes: usize,
    /// Live textures split by what they are for - decoded bitmaps,
    /// `cacheAsBitmap` backing stores, one-off render outputs, and render
    /// targets from each pool - so that memory owned by live content can be
    /// told apart from memory the renderer holds as reusable scratch.
    pub texture_kind_names: &'static [&'static str],
    pub texture_kind_live_counts: Vec<usize>,
    pub texture_kind_live_bytes: Vec<usize>,
    pub texture_kind_created: Vec<u64>,
    pub texture_kind_created_bytes: Vec<u64>,
    pub texture_kind_dropped: Vec<u64>,
    pub texture_kind_dropped_bytes: Vec<u64>,
    /// The most texture memory held at once over the whole run.
    pub peak_texture_bytes: usize,
    /// Pool free-list hits versus misses, which is how much the renderer is
    /// actually re-using its render targets.
    pub pool_reuses: u64,
    pub pool_misses: u64,
    /// Idle render targets in the surface pool (kept across frames) and the
    /// offscreen pool (replaced each frame), with their size-class counts.
    pub main_pool_idle_textures: usize,
    pub main_pool_idle_bytes: usize,
    pub main_pool_size_classes: usize,
    pub offscreen_pool_idle_textures: usize,
    pub offscreen_pool_idle_bytes: usize,
    pub offscreen_pool_size_classes: usize,
    /// Readback/upload buffers idle in the renderer's buffer pool.
    pub buffer_pool_idle_entries: usize,
    pub buffer_pool_idle_bytes: usize,
    /// The heaviest pool keys, with the whole key and its demand figures.
    pub pool_keys: Vec<ruffle_render::backend::PoolKeyReport>,
    /// Textures created and dropped since start; differences between samples
    /// give the renderer's texture allocation churn.
    pub textures_created: u64,
    pub texture_bytes_created: u64,
    pub textures_dropped: u64,
    pub texture_bytes_dropped: u64,
    /// What the graphics backend is still holding underneath Ruffle's own
    /// accounting, and what its allocator has taken from the driver.
    pub hal: ruffle_render::backend::HalResourceUsage,
    pub allocator: Option<ruffle_render::backend::AllocatorUsage>,
    /// What the last frame cost in render passes, targets and bind groups.
    pub work: ruffle_render::backend::RenderWorkUsage,
}

impl MemoryReport {
    pub fn capture(context: &mut UpdateContext<'_>) -> Self {
        let mut report = MemoryReport {
            pending_loaders: context.load_manager.len(),
            class_aliases: context.avm2.class_alias_count(),
            gc_allocation: context.gc_context.metrics().total_gc_allocation(),
            gc_objects: context.gc_context.metrics().total_gc_count(),
            gc_external_bytes: context.gc_context.metrics().total_external_allocation(),
            ..Default::default()
        };

        if let Some(gpu) = context.renderer.memory_usage() {
            report.gpu_textures = gpu.textures;
            report.gpu_texture_bytes = gpu.texture_bytes;
            report.gpu_buffer_bytes = gpu.buffer_bytes;
            report.meshes = gpu.meshes;
            report.mesh_bytes = gpu.mesh_bytes;
            report.tracked_textures = gpu.tracked_textures;
            report.tracked_texture_bytes = gpu.tracked_texture_bytes;
            report.texture_kind_names = gpu.texture_kind_names;
            report.texture_kind_live_counts = gpu.texture_kind_live_counts;
            report.texture_kind_live_bytes = gpu.texture_kind_live_bytes;
            report.texture_kind_created = gpu.texture_kind_created;
            report.texture_kind_created_bytes = gpu.texture_kind_created_bytes;
            report.texture_kind_dropped = gpu.texture_kind_dropped;
            report.texture_kind_dropped_bytes = gpu.texture_kind_dropped_bytes;
            report.peak_texture_bytes = gpu.peak_texture_bytes;
            report.pool_reuses = gpu.pool_reuses;
            report.hal = gpu.hal;
            report.allocator = gpu.allocator;
            report.work = gpu.work;
            report.pool_misses = gpu.pool_misses;
            report.main_pool_idle_textures = gpu.main_pool_idle_textures;
            report.main_pool_idle_bytes = gpu.main_pool_idle_bytes;
            report.main_pool_size_classes = gpu.main_pool_size_classes;
            report.offscreen_pool_idle_textures = gpu.offscreen_pool_idle_textures;
            report.offscreen_pool_idle_bytes = gpu.offscreen_pool_idle_bytes;
            report.offscreen_pool_size_classes = gpu.offscreen_pool_size_classes;
            report.buffer_pool_idle_entries = gpu.buffer_pool_idle_entries;
            report.buffer_pool_idle_bytes = gpu.buffer_pool_idle_bytes;
            report.pool_keys = gpu.pool_keys;
            report.textures_created = gpu.textures_created;
            report.texture_bytes_created = gpu.texture_bytes_created;
            report.textures_dropped = gpu.textures_dropped;
            report.texture_bytes_dropped = gpu.texture_bytes_dropped;
        }

        let movies: Vec<_> = context.library.known_movies().collect();
        for movie in movies {
            let Some(library) = context.library.library_for_movie(movie.clone()) else {
                continue;
            };
            let usage = library.memory_usage();
            let content_alive = library.has_live_content(context.gc());

            report.swf_bytes += movie.uncompressed_len().max(0) as usize;
            report.bitmap_source_bytes += usage.bitmap_source_bytes;
            report.bitmap_decoded_bytes += usage.bitmap_decoded_bytes;
            report.characters += usage.characters;

            report.movies.push(MovieMemory {
                url: movie.url().to_owned(),
                // Subtract the reference held by the `movies` vec above, so
                // that the number reads as "references other than ours".
                strong_refs: std::sync::Arc::strong_count(&movie) - 1,
                swf_bytes: movie.uncompressed_len().max(0) as usize,
                characters: usage.characters,
                self_refs: usage.self_refs + 1, // + the library's own `swf` field
                bitmaps: usage.bitmaps,
                uploaded_bitmaps: usage.uploaded_bitmaps,
                bitmap_source_bytes: usage.bitmap_source_bytes,
                bitmap_decoded_bytes: usage.bitmap_decoded_bytes,
                sounds: usage.sounds,
                fonts: usage.fonts,
                has_domain: usage.has_domain,
                content_alive,
            });
        }

        report
            .movies
            .sort_by(|a, b| b.bitmap_decoded_bytes.cmp(&a.bitmap_decoded_bytes));
        report
    }

    /// The texture kinds this build reports, in the order their columns
    /// appear. Taken from the renderer so the header cannot drift from the
    /// rows; empty until the first sample, when the backend names them.
    pub fn texture_kind_names(&self) -> &'static [&'static str] {
        self.texture_kind_names
    }

    /// The CSV header for a report, including one group of columns per
    /// texture kind. `kinds` must be the same names the rows will use.
    pub fn csv_header_for(kinds: &[&str]) -> String {
        let mut header = String::from(
            "elapsed_s,movies,characters,swf_bytes,bitmap_source_bytes,bitmap_decoded_bytes,\
             pending_loaders,class_aliases,gc_allocation,gc_objects,gc_external_bytes,\
             gpu_textures,gpu_texture_bytes,gpu_buffer_bytes,meshes,mesh_bytes,\
             tracked_textures,tracked_texture_bytes,peak_texture_bytes,pool_reuses,pool_misses,\
             main_pool_idle_textures,main_pool_idle_bytes,main_pool_size_classes,\
             offscreen_pool_idle_textures,offscreen_pool_idle_bytes,offscreen_pool_size_classes,\
             buffer_pool_idle_entries,buffer_pool_idle_bytes,\
             textures_created,texture_bytes_created,textures_dropped,texture_bytes_dropped,\
             hal_textures,hal_texture_views,hal_buffers,hal_bind_groups,hal_bind_group_layouts,\
             hal_render_pipelines,hal_compute_pipelines,hal_pipeline_layouts,hal_samplers,\
             hal_command_encoders,hal_shader_modules,hal_query_sets,hal_fences,\
             hal_texture_memory,hal_buffer_memory,hal_memory_allocations,\
             allocator_allocated_bytes,allocator_reserved_bytes,allocator_blocks,\
             render_passes,blend_targets_live,blend_target_bytes,\
             peak_blend_targets,peak_blend_target_bytes,\
             bind_groups_created,bind_group_cache_hits,bind_group_cache_misses,\
             trivial_blend_fastpath_eligible,trivial_blend_fastpath_used,\
             render_ns_total,render_ns_cache_entries,render_ns_frame_commands,\
             render_ns_queue_submit,render_slow_frames,render_very_slow_frames,\
             render_slow_ns_cache_entries,render_slow_ns_frame_commands,\
             render_slow_ns_queue_submit",
        );
        for name in ruffle_render::backend::FALLBACK_COLUMN_NAMES {
            let _ = write!(header, ",fastpath_fallback_{name}");
        }
        for kind in kinds {
            for suffix in [
                "live",
                "live_bytes",
                "created",
                "created_bytes",
                "dropped",
                "dropped_bytes",
            ] {
                let _ = write!(header, ",tex_{kind}_{suffix}");
            }
        }
        header
    }

    /// The header for a report whose renderer could not name its kinds.
    pub fn csv_header() -> String {
        Self::csv_header_for(&[])
    }

    /// One CSV row, for logging a time series across a zone-change run.
    pub fn to_csv_row(&self, elapsed_s: f64) -> String {
        let mut row = format!(
            "{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            elapsed_s,
            self.movies.len(),
            self.characters,
            self.swf_bytes,
            self.bitmap_source_bytes,
            self.bitmap_decoded_bytes,
            self.pending_loaders,
            self.class_aliases,
            self.gc_allocation,
            self.gc_objects,
            self.gc_external_bytes,
            self.gpu_textures,
            self.gpu_texture_bytes,
            self.gpu_buffer_bytes,
            self.meshes,
            self.mesh_bytes,
            self.tracked_textures,
            self.tracked_texture_bytes,
            self.peak_texture_bytes,
            self.pool_reuses,
            self.pool_misses,
            self.main_pool_idle_textures,
            self.main_pool_idle_bytes,
            self.main_pool_size_classes,
            self.offscreen_pool_idle_textures,
            self.offscreen_pool_idle_bytes,
            self.offscreen_pool_size_classes,
            self.buffer_pool_idle_entries,
            self.buffer_pool_idle_bytes,
            self.textures_created,
            self.texture_bytes_created,
            self.textures_dropped,
            self.texture_bytes_dropped,
        );
        let allocator = self.allocator.unwrap_or_default();
        let _ = write!(
            row,
            ",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.hal.textures,
            self.hal.texture_views,
            self.hal.buffers,
            self.hal.bind_groups,
            self.hal.bind_group_layouts,
            self.hal.render_pipelines,
            self.hal.compute_pipelines,
            self.hal.pipeline_layouts,
            self.hal.samplers,
            self.hal.command_encoders,
            self.hal.shader_modules,
            self.hal.query_sets,
            self.hal.fences,
            self.hal.texture_memory,
            self.hal.buffer_memory,
            self.hal.memory_allocations,
            allocator.allocated_bytes,
            allocator.reserved_bytes,
            allocator.blocks,
            self.work.render_passes,
            self.work.blend_targets,
            self.work.blend_target_bytes,
            self.work.peak_blend_targets,
            self.work.peak_blend_target_bytes,
            self.work.bind_groups_created,
            self.work.bind_group_cache_hits,
            self.work.bind_group_cache_misses,
            self.work.fastpath_eligible,
            self.work.fastpath_used,
        );
        let _ = write!(
            row,
            ",{},{},{},{},{},{},{},{},{}",
            self.work.render_ns_total,
            self.work.render_ns_cache_entries,
            self.work.render_ns_frame_commands,
            self.work.render_ns_queue_submit,
            self.work.slow_frames,
            self.work.very_slow_frames,
            self.work.slow_ns_cache_entries,
            self.work.slow_ns_frame_commands,
            self.work.slow_ns_queue_submit,
        );
        for i in 0..ruffle_render::backend::FALLBACK_COLUMN_NAMES.len() {
            let _ = write!(row, ",{}", self.work.fallbacks.get(i).copied().unwrap_or(0));
        }
        for i in 0..self.texture_kind_names.len() {
            let at = |v: &Vec<usize>| v.get(i).copied().unwrap_or(0);
            let at64 = |v: &Vec<u64>| v.get(i).copied().unwrap_or(0);
            let _ = write!(
                row,
                ",{},{},{},{},{},{}",
                at(&self.texture_kind_live_counts),
                at(&self.texture_kind_live_bytes),
                at64(&self.texture_kind_created),
                at64(&self.texture_kind_created_bytes),
                at64(&self.texture_kind_dropped),
                at64(&self.texture_kind_dropped_bytes),
            );
        }
        row
    }

    /// The retained pool keys, for a human reading the log. Prints the whole
    /// key and, next to the idle count, the most that key ever had lent out at
    /// once - which is the number that says whether idle entries are retention
    /// or genuine demand.
    pub fn top_pool_keys(&self) -> String {
        let mut out = String::new();
        for k in self.pool_keys.iter().take(8) {
            let _ = write!(
                out,
                "\n    [{}] {:>5}x{:<5} x{} {:<12} {:<38} {:>4} idle {:>4} busy  peak {:>4} recent {:>4} keep {:>4}  {:>8} KiB  reuse {} miss {}",
                k.pool,
                k.width,
                k.height,
                k.sample_count,
                k.format,
                k.usage,
                k.idle_entries,
                k.borrowed,
                k.peak_borrowed,
                k.recent_peak_borrowed,
                k.retained_target,
                k.idle_bytes / 1024,
                k.reuses,
                k.misses_pool_empty + k.misses_new_key,
            );
        }
        out
    }

    /// The heaviest movies still resident, for a human reading the log.
    pub fn top_movies(&self, count: usize) -> String {
        let mut out = String::new();
        for movie in self.movies.iter().take(count) {
            let _ = write!(
                out,
                "\n    {:>4} refs ({:>4} internal){}  {:>9} KiB decoded  {:>5} chars  {}",
                movie.strong_refs,
                movie.self_refs,
                if movie.content_alive {
                    "  live"
                } else {
                    "  dead"
                },
                movie.bitmap_decoded_bytes / 1024,
                movie.characters,
                movie.url,
            );
        }
        out
    }
}

/// Per-library totals, gathered inside `library.rs` where the fields live.
#[derive(Debug, Clone, Default)]
pub struct LibraryMemoryUsage {
    pub characters: usize,
    /// Strong `Arc<SwfMovie>` clones this library's own characters hold
    /// pointing back at the movie it is keyed on.
    pub self_refs: usize,
    pub bitmaps: usize,
    pub uploaded_bitmaps: usize,
    pub bitmap_source_bytes: usize,
    pub bitmap_decoded_bytes: usize,
    pub sounds: usize,
    pub fonts: usize,
    pub has_domain: bool,
}
