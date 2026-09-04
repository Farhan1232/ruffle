# Checks an AQW diagnostic log package and prints the result.
#
# Every figure it prints names the CSV column it came from, so that nothing in
# the output looks like a column that does not exist.
param(
    [string]$Dir = "$PWD",
    [string]$ExpectedInstrumentation = "aqw-blend-overhead-diag-3"
)

$ErrorActionPreference = "Stop"
$csvPath  = Join-Path $Dir "aqw-memory.csv"
$infoPath = Join-Path $Dir "build-info.txt"
$ramPath  = Join-Path $Dir "windows-ram.csv"

function Fail($message) { Write-Host "FAIL  $message" -ForegroundColor Red; $script:failed = $true }
function Pass($message) { Write-Host "ok    $message" -ForegroundColor Green }
$script:failed = $false

# --- the build ---------------------------------------------------------------
if (-not (Test-Path $infoPath)) {
    Fail "build-info.txt is missing - run .\Make-BuildInfo.ps1 before the test"
} else {
    $info = Get-Content $infoPath -Raw
    if ($info -match 'instrument\s*:\s*(\S+)') {
        if ($Matches[1] -eq $ExpectedInstrumentation) {
            Pass "instrumentation is $($Matches[1])"
        } else {
            Fail "instrumentation is $($Matches[1]), expected $ExpectedInstrumentation"
        }
    } else { Fail "build-info.txt does not name an instrumentation version" }
    if ($info -match 'worktree\s*:\s*clean') { Pass "worktree was clean" }
    else { Fail "the checkout was modified - the numbers cannot be tied to a commit" }
    if ($info -match 'commit\s*:\s*([0-9a-f]{40})') { Pass "commit $($Matches[1])" }
}

# --- the columns -------------------------------------------------------------
$required = @(
    # what the renderer is holding
    'peak_texture_bytes','tracked_texture_bytes','tex_pool_main_live','tex_pool_main_live_bytes',
    # what a frame costs in work
    'render_passes','blend_targets_live','blend_target_bytes',
    'peak_blend_targets','peak_blend_target_bytes',
    'bind_groups_created','bind_group_cache_hits','bind_group_cache_misses',
    'trivial_blend_fastpath_eligible','trivial_blend_fastpath_used',
    # where the renderer's time went
    'render_ns_total','render_ns_cache_entries','render_ns_frame_commands',
    'render_ns_queue_submit','render_slow_frames','render_very_slow_frames',
    # what the graphics backend owns underneath
    'hal_textures','hal_texture_views','hal_buffers','hal_bind_groups','hal_samplers',
    'hal_render_pipelines','hal_command_encoders','hal_shader_modules',
    'hal_texture_memory','hal_buffer_memory','hal_memory_allocations',
    'allocator_allocated_bytes','allocator_reserved_bytes','allocator_blocks',
    # the process
    'rss_bytes','peak_rss_bytes','private_bytes','peak_private_bytes','rust_heap_bytes',
    'frames','frame_ms_mean','frame_ms_p95','frame_ms_p99'
)

if (-not (Test-Path $csvPath)) {
    Fail "aqw-memory.csv is missing"
    exit 1
}
$rows = Import-Csv $csvPath
if ($rows.Count -lt 2) { Fail "aqw-memory.csv has $($rows.Count) rows - the run was too short" }
$columns = $rows[0].PSObject.Properties.Name
$missing = $required | Where-Object { $columns -notcontains $_ }
if ($missing) { Fail "aqw-memory.csv is missing columns: $($missing -join ', ')" }
else { Pass "all $($required.Count) required columns are present ($($columns.Count) in total)" }
if ($script:failed) { Write-Host "`nStop here and send me this output." -ForegroundColor Yellow; exit 1 }

# --- the result --------------------------------------------------------------
function Max($column) { ($rows | ForEach-Object { [double]$_.$column } | Measure-Object -Maximum).Maximum }
function Last($column) { [double]$rows[-1].$column }

$drawn = $rows | Where-Object { [int]$_.frames -gt 0 }
$frames = ($drawn | ForEach-Object { [int]$_.frames } | Measure-Object -Sum).Sum
$weighted = 0; $drawn | ForEach-Object { $weighted += [double]$_.frame_ms_mean * [int]$_.frames }
$meanFrame = if ($frames) { $weighted / $frames } else { 0 }
$overBudget = ($drawn | Where-Object { [double]$_.frame_ms_mean -gt 41.67 }).Count

Write-Host ""
Write-Host "=== MEMORY ===================================================="
"peak texture bytes        (peak_texture_bytes)       {0,8:N0} MB   was 5537 in the first test" -f ((Max 'peak_texture_bytes')/1MB)
"peak blend target bytes   (peak_blend_target_bytes)  {0,8:N0} MB" -f ((Max 'peak_blend_target_bytes')/1MB)
"peak blend targets        (peak_blend_targets)       {0,8:N0}" -f (Max 'peak_blend_targets')
"peak working set          (peak_rss_bytes)           {0,8:N0} MB   was 2481" -f ((Max 'peak_rss_bytes')/1MB)
"peak private bytes        (peak_private_bytes)       {0,8:N0} MB   was 9007" -f ((Max 'peak_private_bytes')/1MB)
"final Rust heap           (rust_heap_bytes)          {0,8:N0} MB" -f ((Last 'rust_heap_bytes')/1MB)
Write-Host ""
Write-Host "=== WHERE THE REST OF THE MEMORY IS ==========================="
"graphics driver allocations (hal_memory_allocations) {0,8:N0}" -f (Last 'hal_memory_allocations')
"  their live bytes    (allocator_allocated_bytes)    {0,8:N0} MB" -f ((Last 'allocator_allocated_bytes')/1MB)
"  their block bytes   (allocator_reserved_bytes)     {0,8:N0} MB   the gap is memory the allocator owns and is not using" -f ((Last 'allocator_reserved_bytes')/1MB)
"live backend textures     (hal_textures)             {0,8:N0}   against tracked_textures {1,0:N0}" -f (Last 'hal_textures'), (Last 'tracked_textures')
"live backend texture views(hal_texture_views)        {0,8:N0}" -f (Last 'hal_texture_views')
"live backend bind groups  (hal_bind_groups)          {0,8:N0}" -f (Last 'hal_bind_groups')
"live backend buffers      (hal_buffers)              {0,8:N0}" -f (Last 'hal_buffers')
Write-Host ""
Write-Host "=== WORK PER FRAME ============================================"
"render passes, last frame (render_passes)            {0,8:N0}" -f (Last 'render_passes')
"blend targets, last frame (blend_targets_live)       {0,8:N0}" -f (Last 'blend_targets_live')
$made = Last 'bind_groups_created'
$hits = Last 'bind_group_cache_hits'; $misses = Last 'bind_group_cache_misses'
$rate = if (($hits + $misses) -gt 0) { 100 * $hits / ($hits + $misses) } else { 0 }
"bind groups built, total  (bind_groups_created)      {0,8:N0}" -f $made
"bind group cache hit rate (bind_group_cache_hits)    {0,8:N1}%" -f $rate
$eligible = Last 'trivial_blend_fastpath_eligible'; $used = Last 'trivial_blend_fastpath_used'
$fast = if ($eligible -gt 0) { 100 * $used / $eligible } else { 0 }
"blends drawn without a target (trivial_blend_fastpath_used) {0,4:N1}% of {1,0:N0}" -f $fast, $eligible
foreach ($c in $columns | Where-Object { $_ -like 'fastpath_fallback_*' }) {
    $v = Last $c
    if ($v -gt 0) { "  {0,-44} {1,8:N0}" -f $c, $v }
}
Write-Host ""
Write-Host "=== FRAME TIME ================================================"
"frames drawn              (frames)                   {0,8:N0}" -f $frames
"mean frame time           (frame_ms_mean)            {0,8:N1} ms   target under 41.7" -f $meanFrame
"5-second windows over budget                         {0,8:N0} of {1,0:N0}" -f $overBudget, $drawn.Count
$total = Last 'render_ns_total'
if ($total -gt 0) {
    "renderer's own share of that time, split by phase:"
    "  cached objects redrawn  (render_ns_cache_entries)  {0,6:N1}%" -f (100 * (Last 'render_ns_cache_entries') / $total)
    "  the frame's own drawing (render_ns_frame_commands) {0,6:N1}%" -f (100 * (Last 'render_ns_frame_commands') / $total)
    "  handing work to the GPU (render_ns_queue_submit)   {0,6:N1}%" -f (100 * (Last 'render_ns_queue_submit') / $total)
    "frames whose rendering alone missed the budget (render_slow_frames)      {0,8:N0}" -f (Last 'render_slow_frames')
    "  of those, over 100 ms                       (render_very_slow_frames) {0,8:N0}" -f (Last 'render_very_slow_frames')
}

if (Test-Path $ramPath) {
    $ram = Import-Csv $ramPath
    if ($ram.Count -gt 1) {
        Write-Host ""
        Write-Host "=== TASK MANAGER'S VIEW ======================================="
        "peak working set  {0,8:N0} MB" -f (($ram | ForEach-Object { [double]$_.working_set_mb } | Measure-Object -Maximum).Maximum)
        "peak private      {0,8:N0} MB" -f (($ram | ForEach-Object { [double]$_.private_bytes_mb } | Measure-Object -Maximum).Maximum)
    }
} else {
    Write-Host "`nnote: windows-ram.csv is missing, so Task Manager's view is not shown."
}
Write-Host ""
