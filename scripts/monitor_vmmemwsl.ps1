<#
.SYNOPSIS
    Monitor VmmemWSL memory usage over time. Run in an Administrator PowerShell
    window on the Windows host WHILE running tests inside WSL2.

.DESCRIPTION
    Samples VmmemWSL working-set size every N seconds, logs to CSV, and prints
    a live summary. Press Ctrl+C to stop; the CSV is flushed after every sample.

.PARAMETER OutputPath
    Path to the CSV output file. Default: .\vmmemwsl_monitor.csv

.PARAMETER IntervalSeconds
    Sampling interval in seconds. Default: 5

.PARAMETER DurationMinutes
    Maximum monitoring duration in minutes. 0 = unlimited (run until Ctrl+C).
    Default: 0

.EXAMPLE
    # Monitor for 30 minutes, sample every 5 seconds:
    .\monitor_vmmemwsl.ps1 -DurationMinutes 30

    # Monitor indefinitely with 10-second intervals:
    .\monitor_vmmemwsl.ps1 -IntervalSeconds 10

    # Custom output path:
    .\monitor_vmmemwsl.ps1 -OutputPath C:\logs\vmmem_test.csv -DurationMinutes 60
#>

param(
    [string]$OutputPath = ".\vmmemwsl_monitor.csv",
    [int]$IntervalSeconds = 5,
    [int]$DurationMinutes = 0
)

# --- Helpers ---
function Get-VmmemWSLMemoryMB {
    $proc = Get-Process -Name "vmmemWSL" -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        $proc = Get-Process -Name "vmmem" -ErrorAction SilentlyContinue
    }
    if ($null -eq $proc) {
        return @{ WorkingSetMB = 0; PrivateMB = 0; Found = $false }
    }
    return @{
        WorkingSetMB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
        PrivateMB    = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
        Found        = $true
    }
}

function Get-SystemMemoryMB {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    return @{
        TotalMB     = [math]::Round($os.TotalVisibleMemorySize / 1KB, 1)
        FreeMB      = [math]::Round($os.FreePhysicalMemory / 1KB, 1)
        UsedPercent = [math]::Round((1 - $os.FreePhysicalMemory / $os.TotalVisibleMemorySize) * 100, 1)
    }
}

function Get-TopDiskWriters {
    <#
    .SYNOPSIS
        Returns the top N processes by cumulative disk write bytes using
        WMI performance counters (works without admin elevation).
        Note: IOWriteBytesPersec is a raw cumulative counter despite the name.
    #>
    param([int]$TopN = 5)

    try {
        # Query all process perf counters at once (fast, single WMI call)
        $allPerf = Get-CimInstance Win32_PerfRawData_PerfProc_Process -ErrorAction Stop |
            Where-Object { $_.Name -ne '_Total' -and $_.Name -ne 'Idle' -and $_.IOWriteBytesPersec -gt 0 }

        $procs = $allPerf | ForEach-Object {
            [PSCustomObject]@{
                Name         = $_.Name
                PID          = $_.IDProcess
                WriteBytesMB = [math]::Round([double]$_.IOWriteBytesPersec / 1MB, 2)
            }
        } | Sort-Object WriteBytesMB -Descending |
            Select-Object -First $TopN
    } catch {
        $procs = @()
    }

    return $procs
}

# Track previous write bytes per PID for rate calculation
$script:prevWriteBytes = @{}

# --- Header ---
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VmmemWSL Memory Monitor" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Output:   $OutputPath"
Write-Host "  Interval: ${IntervalSeconds}s"
if ($DurationMinutes -gt 0) {
    Write-Host "  Duration: ${DurationMinutes} min"
} else {
    Write-Host "  Duration: unlimited (Ctrl+C to stop)"
}
Write-Host ""

# --- CSV header ---
$csvHeader = "timestamp,elapsed_s,vmmem_ws_mb,vmmem_private_mb,system_used_pct,system_free_mb,system_total_mb,top1_name,top1_pid,top1_write_mb,top1_rate_mbs,top2_name,top2_pid,top2_write_mb,top2_rate_mbs,top3_name,top3_pid,top3_write_mb,top3_rate_mbs"
$csvHeader | Out-File -FilePath $OutputPath -Encoding UTF8

# --- Sampling loop ---
$startTime = Get-Date
$sampleCount = 0
$peakWS = 0
$maxSamples = if ($DurationMinutes -gt 0) {
    [math]::Ceiling($DurationMinutes * 60 / $IntervalSeconds)
} else { [int]::MaxValue }

Write-Host ("{0,-22} {1,10} {2,10} {3,8} {4,10} | {5,-16} {6,10}" -f "Timestamp", "VmmemMB", "PrivateMB", "SysUsed%", "SysFreeMB", "TopWriter", "Rate MB/s") -ForegroundColor Yellow
Write-Host ("{0,-22} {1,10} {2,10} {3,8} {4,10} | {5,-16} {6,10}" -f "-" * 22, "-" * 10, "-" * 10, "-" * 8, "-" * 10, "-" * 16, "-" * 10) -ForegroundColor DarkGray

try {
    while ($sampleCount -lt $maxSamples) {
        $now = Get-Date
        $elapsed = [math]::Round(($now - $startTime).TotalSeconds, 0)
        $ts = $now.ToString("yyyy-MM-dd HH:mm:ss")

        $vmmem = Get-VmmemWSLMemoryMB
        $sysmem = Get-SystemMemoryMB

        if ($vmmem.WorkingSetMB -gt $peakWS) { $peakWS = $vmmem.WorkingSetMB }

        # --- Top disk writers ---
        $topWriters = Get-TopDiskWriters -TopN 3
        $writerCsvParts = @()
        $topWriterDisplay = ""
        $topWriterRate = ""

        for ($w = 0; $w -lt 3; $w++) {
            if ($w -lt $topWriters.Count) {
                $tw = $topWriters[$w]
                $prevKey = "$($tw.PID)"
                $prevWB = $script:prevWriteBytes[$prevKey]
                $rateMBs = 0.0
                if ($null -ne $prevWB -and $tw.WriteBytesMB -ge $prevWB) {
                    $rateMBs = [math]::Round(($tw.WriteBytesMB - $prevWB) / $IntervalSeconds, 3)
                }
                $script:prevWriteBytes[$prevKey] = $tw.WriteBytesMB
                $writerCsvParts += "$($tw.Name),$($tw.PID),$($tw.WriteBytesMB),$rateMBs"

                if ($w -eq 0) {
                    $topWriterDisplay = $tw.Name
                    $topWriterRate = "{0:N3}" -f $rateMBs
                }
            } else {
                $writerCsvParts += ",,,"
            }
        }

        # Console line
        $status = if ($vmmem.Found) { "" } else { " [NOT FOUND]" }
        $color = if ($vmmem.WorkingSetMB -gt 15000) { "Red" }
                 elseif ($vmmem.WorkingSetMB -gt 8000) { "Yellow" }
                 else { "Green" }

        Write-Host ("{0,-22} {1,10:N0} {2,10:N0} {3,7:N1}% {4,10:N0}{5} | {6,-16} {7,10}" -f `
            $ts, $vmmem.WorkingSetMB, $vmmem.PrivateMB, `
            $sysmem.UsedPercent, $sysmem.FreeMB, $status, `
            $topWriterDisplay, $topWriterRate) -ForegroundColor $color

        # CSV line
        $csvLine = "$ts,$elapsed,$($vmmem.WorkingSetMB),$($vmmem.PrivateMB),$($sysmem.UsedPercent),$($sysmem.FreeMB),$($sysmem.TotalMB),$($writerCsvParts -join ',')"
        $csvLine | Out-File -FilePath $OutputPath -Encoding UTF8 -Append

        $sampleCount++
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    # --- Summary ---
    $endTime = Get-Date
    $totalElapsed = [math]::Round(($endTime - $startTime).TotalSeconds, 0)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Monitoring Summary" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Samples:     $sampleCount"
    Write-Host "  Duration:    ${totalElapsed}s"
    Write-Host "  Peak VmmemWSL: $($peakWS) MB"
    Write-Host "  Final VmmemWSL:"

    $finalVmmem = Get-VmmemWSLMemoryMB
    if ($finalVmmem.Found) {
        Write-Host "    Working Set: $($finalVmmem.WorkingSetMB) MB"
        Write-Host "    Private:     $($finalVmmem.PrivateMB) MB"
    } else {
        Write-Host "    VmmemWSL process not found"
    }

    Write-Host "  CSV saved to: $OutputPath"
    Write-Host ""

    # --- Quick reclaim timing analysis ---
    if (Test-Path $OutputPath) {
        $data = Import-Csv $OutputPath
        if ($data.Count -gt 2) {
            $peakRow = $data | Sort-Object { [double]$_.vmmem_ws_mb } -Descending | Select-Object -First 1
            $minAfterPeak = $data |
                Where-Object { [int]$_.elapsed_s -ge [int]$peakRow.elapsed_s } |
                Sort-Object { [double]$_.vmmem_ws_mb } |
                Select-Object -First 1

            if ([double]$peakRow.vmmem_ws_mb -gt [double]$minAfterPeak.vmmem_ws_mb + 1000) {
                $dropTime = [int]$minAfterPeak.elapsed_s - [int]$peakRow.elapsed_s
                Write-Host "  Memory drop detected:" -ForegroundColor Yellow
                Write-Host "    Peak:     $($peakRow.vmmem_ws_mb) MB at t=$($peakRow.elapsed_s)s"
                Write-Host "    Trough:   $($minAfterPeak.vmmem_ws_mb) MB at t=$($minAfterPeak.elapsed_s)s"
                Write-Host "    Reclaim:  ${dropTime}s after peak"
            } else {
                Write-Host "  No significant memory drop detected in recorded data." -ForegroundColor DarkGray
            }
        }
    }
}
