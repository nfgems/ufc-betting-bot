<#
.SYNOPSIS
    Supervises a `btc5m-opportunity` comparison run.

.DESCRIPTION
    Monitors the runner process, report.json, and runner logs; records status to
    watchdog_status.json and watchdog.log; creates periodic checkpoints; and stops
    itself when the run is complete.

    Cadence escalates as the run proves healthy:
      - Phase 1 (first $Phase1Minutes minutes):  every $Phase1IntervalSeconds s  (default 5 min)
      - Phase 2 (until $Phase2Minutes minutes):  every $Phase2IntervalSeconds s  (default 10 min)
      - Phase 3 (after that, until finished):    every $Phase3IntervalSeconds s  (default 15 min)
    If a check finds trouble (error markers, non-pending settlement errors, the
    runner not running before completion, or outside_trading_hours) and
    -TightenOnError is $true, the next interval drops back to Phase 1 until the
    run is clean again.

    This script DOES NOT patch code or restart the runner. When it flags trouble,
    a supervising operator/agent must carry out the remediation decision
    (continue / patch / resume / restore / restart) per NEXT_RUN_HANDOFF.md.

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\btc5m_opportunity_watchdog.ps1 `
        -RunRoot "C:\Users\Evan\betting-bot\ufc-betting-bot\data\logs\btc5m_opportunity\<run_id>"
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [int]$Phase1IntervalSeconds = 300,   # 5 minutes
    [int]$Phase2IntervalSeconds = 600,   # 10 minutes
    [int]$Phase3IntervalSeconds = 900,   # 15 minutes

    [int]$Phase1Minutes = 60,            # elapsed < 60 min  -> phase 1 (5 min)
    [int]$Phase2Minutes = 360,           # elapsed < 360 min -> phase 2 (10 min); else phase 3 (15 min)

    [int]$CheckpointMinutes = 45,
    [bool]$TightenOnError = $true
)

$ErrorActionPreference = "Continue"

$ReportPath = Join-Path $RunRoot "report.json"
$MarkdownPath = Join-Path $RunRoot "report.md"
$RunnerPidPath = Join-Path $RunRoot "runner.pid"
$StdoutLog = Join-Path $RunRoot "runner.out.log"
$StderrLog = Join-Path $RunRoot "runner.err.log"
$WatchdogLog = Join-Path $RunRoot "watchdog.log"
$WatchdogStatus = Join-Path $RunRoot "watchdog_status.json"
$LedgerDir = Join-Path $RunRoot "ledgers"
$CheckpointRoot = Join-Path $RunRoot "checkpoints"
$ManifestPath = Join-Path $RunRoot "RUN_PATHS.json"
$LastCheckpointPath = Join-Path $RunRoot ".watchdog_last_checkpoint"
$StartedPath = Join-Path $RunRoot ".watchdog_started"

$Patterns = @(
    "Traceback",
    "ERROR",
    "Exception",
    "snapshot_fetch_failed",
    "snapshot fetch failed",
    "exit_snapshot_fetch_failed",
    "exit snapshot fetch failed",
    "data_fetch_failed",
    "data fetch failed",
    "status=error",
    "settlement_errors",
    "ReadTimeout",
    "ConnectTimeout",
    "Max retries exceeded",
    "NameResolutionError",
    "451 Client Error",
    "invalid_trading_timezone",
    "outside_trading_hours"
)

function Now-Iso {
    return [DateTime]::UtcNow.ToString("o")
}

function Write-WatchdogLog {
    param([string]$Message)
    Add-Content -LiteralPath $WatchdogLog -Value "$(Now-Iso) $Message" -Encoding utf8
}

function Copy-ExistingFile {
    param([string]$Source, [string]$DestinationDirectory)
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $DestinationDirectory -Force
    }
}

function New-RunCheckpoint {
    param([string]$Name)
    try {
        New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null
        $dest = Join-Path $CheckpointRoot $Name
        if (Test-Path -LiteralPath $dest) {
            $suffix = 1
            do {
                $dest = Join-Path $CheckpointRoot "${Name}_$suffix"
                $suffix += 1
            } while (Test-Path -LiteralPath $dest)
        }
        New-Item -ItemType Directory -Path $dest | Out-Null
        Copy-ExistingFile -Source $ReportPath -DestinationDirectory $dest
        Copy-ExistingFile -Source $MarkdownPath -DestinationDirectory $dest
        Copy-ExistingFile -Source $ManifestPath -DestinationDirectory $dest
        Copy-ExistingFile -Source $RunnerPidPath -DestinationDirectory $dest
        Copy-ExistingFile -Source $StdoutLog -DestinationDirectory $dest
        Copy-ExistingFile -Source $StderrLog -DestinationDirectory $dest
        if (Test-Path -LiteralPath $LedgerDir) {
            $ledgerDest = Join-Path $dest "ledgers"
            Copy-Item -LiteralPath $LedgerDir -Destination $ledgerDest -Recurse -Force
        }
        Write-WatchdogLog "checkpoint_created name=$Name path=$dest"
        return $true
    } catch {
        Write-WatchdogLog "checkpoint_failed name=$Name error=$($_.Exception.Message)"
        return $false
    }
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ ok = $false; value = $null; error = "missing" }
    }
    try {
        $raw = Get-Content -Raw -LiteralPath $Path -Encoding utf8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return [pscustomobject]@{ ok = $false; value = $null; error = "empty" }
        }
        return [pscustomobject]@{ ok = $true; value = ($raw | ConvertFrom-Json); error = $null }
    } catch {
        return [pscustomobject]@{ ok = $false; value = $null; error = $_.Exception.Message }
    }
}

function Get-RunnerInfo {
    $pidValue = $null
    if (Test-Path -LiteralPath $RunnerPidPath) {
        $pidText = (Get-Content -Raw -LiteralPath $RunnerPidPath -ErrorAction SilentlyContinue).Trim()
        if ($pidText -match "^\d+$") {
            $pidValue = [int]$pidText
        }
    }

    $process = $null
    if ($null -ne $pidValue) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
    }

    $children = @()
    if ($null -ne $pidValue) {
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $pidValue" -ErrorAction SilentlyContinue |
            Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine)
    }

    return [pscustomobject]@{
        pid = $pidValue
        status = $(if ($null -eq $pidValue) { "missing_pid" } elseif ($null -eq $process) { "not_running" } else { "running" })
        executable_path = $(if ($null -eq $process) { $null } else { $process.ExecutablePath })
        command_line = $(if ($null -eq $process) { $null } else { $process.CommandLine })
        child_processes = $children
    }
}

function Get-LogMarkers {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return @()
    }
    $lines = @(Get-Content -LiteralPath $Path -Tail 250 -ErrorAction SilentlyContinue)
    $markers = @()
    foreach ($line in $lines) {
        foreach ($pattern in $Patterns) {
            if ($line -like "*$pattern*") {
                $markers += [pscustomobject]@{ pattern = $pattern; line = $line }
                break
            }
        }
    }
    return @($markers | Select-Object -Last 40)
}

function Add-ProfileSums {
    param([object]$Report)

    $settlementErrors = 0
    $pendingSettlementErrors = 0
    $nonPendingSettlementErrors = 0
    $settlementErrorSamples = @()
    $outsideTradingHours = 0
    $openHoldBets = 0

    if ($null -ne $Report -and $null -ne $Report.profiles) {
        foreach ($modeProp in $Report.profiles.PSObject.Properties) {
            foreach ($profileProp in $modeProp.Value.PSObject.Properties) {
                $payload = $profileProp.Value
                foreach ($settlementError in @($payload.settlement_errors)) {
                    $settlementErrors += 1
                    $errorText = [string]$settlementError
                    if ($settlementErrorSamples.Count -lt 10 -and $errorText) {
                        $settlementErrorSamples += $errorText
                    }
                    if ($errorText -like "*official Polymarket resolution not available yet*") {
                        $pendingSettlementErrors += 1
                    } else {
                        $nonPendingSettlementErrors += 1
                    }
                }
                $outsideTradingHours += [int]($payload.outside_trading_hours ?? 0)
                $openHoldBets += [int]($payload.open_bets ?? 0)
            }
        }
    }

    $openTakeProfit = 0
    if ($null -ne $Report -and $null -ne $Report.take_profit_positions) {
        foreach ($modeProp in $Report.take_profit_positions.PSObject.Properties) {
            foreach ($profileProp in $modeProp.Value.PSObject.Properties) {
                foreach ($position in @($profileProp.Value)) {
                    if (($position.status ?? "") -eq "open") {
                        $openTakeProfit += 1
                    }
                }
            }
        }
    }

    return [pscustomobject]@{
        settlement_errors_count = $settlementErrors
        pending_settlement_errors_count = $pendingSettlementErrors
        non_pending_settlement_errors_count = $nonPendingSettlementErrors
        settlement_error_samples = @($settlementErrorSamples | Select-Object -Unique -First 10)
        outside_trading_hours_count = $outsideTradingHours
        outside_trading_hours_detected = ($outsideTradingHours -gt 0)
        open_hold_bets = $openHoldBets
        open_take_profit_positions = $openTakeProfit
    }
}

function Get-LastWriteIso {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-Item -LiteralPath $Path).LastWriteTimeUtc.ToString("o")
}

function Get-CompletionAssessment {
    param([object]$Report, [object]$RunnerInfo, [object]$Sums, [array]$ErrMarkers, [array]$OutMarkers)

    if ($null -eq $Report) {
        if ($RunnerInfo.status -eq "not_running") {
            return "partial"
        }
        return "in_progress"
    }

    $status = [string]($Report.status ?? "")
    $seen = [int]($Report.distinct_markets_seen ?? 0)
    $target = [int]($Report.target_markets ?? 0)

    if ($Sums.outside_trading_hours_detected) {
        return "needs_inspection"
    }
    if ($Sums.non_pending_settlement_errors_count -gt 0) {
        return "in_progress_with_settlement_errors"
    }
    if ($status -eq "complete" -and $target -gt 0 -and $seen -ge $target -and $Sums.open_hold_bets -eq 0 -and $Sums.open_take_profit_positions -eq 0) {
        return "complete"
    }
    if ($RunnerInfo.status -eq "not_running" -and $status -ne "complete") {
        return "partial"
    }
    if (($ErrMarkers.Count + $OutMarkers.Count) -gt 0) {
        return "in_progress_with_markers"
    }
    return "in_progress"
}

function Get-IntervalSeconds {
    param([double]$ElapsedMinutes, [bool]$Trouble)
    if ($Trouble -and $TightenOnError) {
        return $Phase1IntervalSeconds
    }
    if ($ElapsedMinutes -lt $Phase1Minutes) {
        return $Phase1IntervalSeconds
    }
    if ($ElapsedMinutes -lt $Phase2Minutes) {
        return $Phase2IntervalSeconds
    }
    return $Phase3IntervalSeconds
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointRoot | Out-Null

# Anchor the cadence schedule; survives a watchdog restart.
$startTime = [DateTime]::UtcNow
if (Test-Path -LiteralPath $StartedPath) {
    try {
        $startedText = (Get-Content -Raw -LiteralPath $StartedPath).Trim()
        if ($startedText) { $startTime = [DateTime]::Parse($startedText).ToUniversalTime() }
    } catch { $startTime = [DateTime]::UtcNow }
} else {
    $startTime.ToString("o") | Set-Content -LiteralPath $StartedPath -Encoding ascii
}

Write-WatchdogLog ("watchdog_started phase1={0}s/<{1}m phase2={2}s/<{3}m phase3={4}s checkpoint_minutes={5} tighten_on_error={6}" -f `
    $Phase1IntervalSeconds, $Phase1Minutes, $Phase2IntervalSeconds, $Phase2Minutes, $Phase3IntervalSeconds, $CheckpointMinutes, $TightenOnError)

if (-not (Test-Path -LiteralPath (Join-Path $CheckpointRoot "initial"))) {
    New-RunCheckpoint -Name "initial" | Out-Null
}

$checkpointEvery = [TimeSpan]::FromMinutes([Math]::Max($CheckpointMinutes, 1))
$lastCheckpoint = [DateTime]::UtcNow
if (Test-Path -LiteralPath $LastCheckpointPath) {
    try {
        $lastCheckpointText = (Get-Content -Raw -LiteralPath $LastCheckpointPath).Trim()
        if ($lastCheckpointText) {
            $lastCheckpoint = [DateTime]::Parse($lastCheckpointText).ToUniversalTime()
        }
    } catch {
        $lastCheckpoint = [DateTime]::UtcNow
    }
}

while ($true) {
    $checkedAt = Now-Iso
    $runner = Get-RunnerInfo
    $reportRead = Read-JsonFile -Path $ReportPath
    $report = $(if ($reportRead.ok) { $reportRead.value } else { $null })
    $sums = Add-ProfileSums -Report $report
    $errMarkers = @(Get-LogMarkers -Path $StderrLog)
    $outMarkers = @(Get-LogMarkers -Path $StdoutLog)
    $assessment = Get-CompletionAssessment -Report $report -RunnerInfo $runner -Sums $sums -ErrMarkers $errMarkers -OutMarkers $outMarkers

    $runnerExitedEarly = ($runner.status -eq "not_running" -and $null -ne $report -and [string]($report.status ?? "") -ne "complete")
    $trouble = (
        $sums.outside_trading_hours_detected -or
        ($sums.non_pending_settlement_errors_count -gt 0) -or
        $runnerExitedEarly -or
        (($errMarkers.Count + $outMarkers.Count) -gt 0)
    )

    $lastAction = "none"
    if ($sums.outside_trading_hours_detected) {
        $lastAction = "needs_inspection_outside_trading_hours"
        Write-WatchdogLog "suspicious outside_trading_hours_count=$($sums.outside_trading_hours_count)"
    } elseif ($sums.non_pending_settlement_errors_count -gt 0) {
        $lastAction = "settlement_errors_observed_needs_inspection"
        Write-WatchdogLog "non_pending_settlement_errors count=$($sums.non_pending_settlement_errors_count) samples=$($sums.settlement_error_samples -join ' || ')"
    } elseif ($runnerExitedEarly) {
        $lastAction = "runner_exited_before_complete_preserve_partial"
        Write-WatchdogLog "runner_not_running status=$($report.status) markets=$($report.distinct_markets_seen)/$($report.target_markets)"
    } elseif (($errMarkers.Count + $outMarkers.Count) -gt 0) {
        $lastAction = "log_markers_observed_continue_monitoring"
    } elseif ($sums.pending_settlement_errors_count -gt 0) {
        $lastAction = "pending_official_resolution_continue_monitoring"
    } elseif ($assessment -eq "complete") {
        $lastAction = "complete_no_open_positions"
    }

    $elapsedMinutes = ([DateTime]::UtcNow - $startTime).TotalMinutes
    $intervalSeconds = Get-IntervalSeconds -ElapsedMinutes $elapsedMinutes -Trouble $trouble
    $phase = $(if ($trouble -and $TightenOnError) { "tightened" } elseif ($elapsedMinutes -lt $Phase1Minutes) { "phase1" } elseif ($elapsedMinutes -lt $Phase2Minutes) { "phase2" } else { "phase3" })

    $statusPayload = [pscustomobject]@{
        checked_at = $checkedAt
        run_root = $RunRoot
        elapsed_minutes = [Math]::Round($elapsedMinutes, 2)
        cadence_phase = $phase
        next_interval_seconds = $intervalSeconds
        trouble_detected = $trouble
        runner = $runner
        report = [pscustomobject]@{
            exists = (Test-Path -LiteralPath $ReportPath)
            last_write_time = Get-LastWriteIso -Path $ReportPath
            parse_ok = $reportRead.ok
            parse_error = $reportRead.error
            status = $(if ($null -eq $report) { $null } else { $report.status })
            distinct_markets_seen = $(if ($null -eq $report) { $null } else { $report.distinct_markets_seen })
            target_markets = $(if ($null -eq $report) { $null } else { $report.target_markets })
            cycles = $(if ($null -eq $report) { $null } else { $report.cycles })
            run_id = $(if ($null -eq $report) { $null } else { $report.run_id })
        }
        settlement_errors_count = $sums.settlement_errors_count
        pending_settlement_errors_count = $sums.pending_settlement_errors_count
        non_pending_settlement_errors_count = $sums.non_pending_settlement_errors_count
        settlement_error_samples = $sums.settlement_error_samples
        outside_trading_hours_detected = $sums.outside_trading_hours_detected
        outside_trading_hours_count = $sums.outside_trading_hours_count
        open_hold_bets = $sums.open_hold_bets
        open_take_profit_positions = $sums.open_take_profit_positions
        runner_stderr_markers = $errMarkers
        runner_stdout_markers = $outMarkers
        last_remediation_action = $lastAction
        completion_assessment = $assessment
    }

    try {
        $statusPayload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $WatchdogStatus -Encoding utf8
        Write-WatchdogLog ("status runner={0} report_status={1} markets={2}/{3} cycles={4} assessment={5} action={6} phase={7} next_interval_s={8} outside_trading_hours={9} non_pending_settlement_errors={10} open_hold={11} open_take_profit={12}" -f @(
            $runner.status,
            $statusPayload.report.status,
            $statusPayload.report.distinct_markets_seen,
            $statusPayload.report.target_markets,
            $statusPayload.report.cycles,
            $assessment,
            $lastAction,
            $phase,
            $intervalSeconds,
            $sums.outside_trading_hours_count,
            $sums.non_pending_settlement_errors_count,
            $sums.open_hold_bets,
            $sums.open_take_profit_positions
        ))
    } catch {
        Write-WatchdogLog "status_write_failed error=$($_.Exception.Message)"
    }

    if (([DateTime]::UtcNow - $lastCheckpoint) -ge $checkpointEvery) {
        $checkpointName = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
        if (New-RunCheckpoint -Name $checkpointName) {
            $lastCheckpoint = [DateTime]::UtcNow
            $lastCheckpoint.ToString("o") | Set-Content -LiteralPath $LastCheckpointPath -Encoding ascii
        }
    }

    if ($assessment -eq "complete") {
        $checkpointName = "complete_" + [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
        New-RunCheckpoint -Name $checkpointName | Out-Null
        Write-WatchdogLog "watchdog_complete status=complete"
        break
    }

    Start-Sleep -Seconds ([Math]::Max($intervalSeconds, 10))
}
