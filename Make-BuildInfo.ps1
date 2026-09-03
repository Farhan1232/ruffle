# Writes diagnostic-build-info.txt so the run can be tied to the exact binary.
# Run from the ruffle source folder, after building.
param(
    [string]$Exe = ".\target\release\ruffle_desktop.exe",
    [string]$Out = "$HOME\Desktop\aqw-diagnostic\diagnostic-build-info.txt"
)

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$branch = git rev-parse --abbrev-ref HEAD
$commit = git rev-parse HEAD
$dirty  = if ((git status --porcelain)) { "MODIFIED - not a clean checkout" } else { "clean" }
$item   = Get-Item $Exe
$sha    = (Get-FileHash $Exe -Algorithm SHA256).Hash

@"
=== AQW GPU MEMORY DIAGNOSTIC BUILD ===
generated    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
repo         : https://github.com/Farhan1232/ruffle.git
branch       : $branch
commit       : $commit
worktree     : $dirty
instrument   : aqw-gpu-diag-1
exe path     : $($item.FullName)
exe modified : $($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
exe size     : $($item.Length) bytes
exe sha256   : $sha
gpu          : $((Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name))
os           : $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)
ram          : $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)) GB
"@ | Set-Content -Encoding UTF8 $Out

Get-Content $Out
