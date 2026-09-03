# Samples the Ruffle process every 5 seconds: working set, private bytes, CPU.
# Leave this running in its own PowerShell window for the whole test.
param(
    [string]$Out = "$HOME\Desktop\aqw-diagnostic\diagnostic-windows-ram.csv",
    [int]$IntervalSeconds = 5
)

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
"timestamp,working_set_mb,working_set_gb,private_bytes_mb,cpu_seconds" | Set-Content -Encoding UTF8 $Out

Write-Host "Logging to $Out - press Ctrl+C to stop." -ForegroundColor Cyan
while ($true) {
    $p = Get-Process ruffle_desktop -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
    if ($p) {
        $ws  = [math]::Round($p.WorkingSet64 / 1MB, 1)
        $gb  = [math]::Round($p.WorkingSet64 / 1GB, 3)
        $pb  = [math]::Round($p.PrivateMemorySize64 / 1MB, 1)
        $cpu = [math]::Round($p.CPU, 1)
        "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$ws,$gb,$pb,$cpu" | Add-Content -Encoding UTF8 $Out
        Write-Host ("{0}  working set {1,8} MB   private {2,8} MB" -f (Get-Date -Format 'HH:mm:ss'), $ws, $pb)
    }
    Start-Sleep -Seconds $IntervalSeconds
}
