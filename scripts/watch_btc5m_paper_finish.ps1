param(
    [Parameter(Mandatory = $true)]
    [string[]] $RunnerPids,

    [Parameter(Mandatory = $true)]
    [string] $RepoRoot,

    [int] $SettlementWaitSeconds = 360,

    [string] $SettlementSource = "polymarket",

    [string] $LedgerDir = "",

    [string] $Profiles = "",

    [string] $SignalLogPath = ""
)

$ErrorActionPreference = "Continue"

$ParsedRunnerPids = @(
    $RunnerPids |
        ForEach-Object { [string] $_ -split "," } |
        Where-Object { [string]::IsNullOrWhiteSpace($_) -eq $false } |
        ForEach-Object { [int] ([string] $_).Trim() }
)

$LogDir = Join-Path $RepoRoot "data\logs\btc5m_paper"
$WatcherLog = Join-Path $LogDir "paper_finish_watcher.log"
$StatusPath = Join-Path $LogDir "paper_finish_watcher_status.json"
$ReportStdout = Join-Path $LogDir "paper_finish_report.out.log"
$ReportStderr = Join-Path $LogDir "paper_finish_report.err.log"

function Write-WatcherLog {
    param([string] $Message)
    Add-Content -Path $WatcherLog -Value ("{0} {1}" -f (Get-Date -Format o), $Message) -Encoding UTF8
}

function Write-Status {
    param(
        [string] $State,
        [object[]] $Processes
    )

    [ordered]@{
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
        state = $State
        runner_pids = @($Processes | Select-Object -ExpandProperty ProcessId)
        watched_pids = $ParsedRunnerPids
        settlement_wait_seconds = $SettlementWaitSeconds
        settlement_source = $SettlementSource
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Get-RunnerProcesses {
    Get-CimInstance Win32_Process | Where-Object { $ParsedRunnerPids -contains [int] $_.ProcessId }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-WatcherLog "finish watcher started for PIDs $($ParsedRunnerPids -join ',')"

while ($true) {
    $Processes = @(Get-RunnerProcesses)
    Write-Status -State "waiting_for_runner_exit" -Processes $Processes
    if ($Processes.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 60
}

Write-WatcherLog "runner exited; waiting $SettlementWaitSeconds seconds before final settlement/report"
Write-Status -State "waiting_for_settlement_delay" -Processes @()
Start-Sleep -Seconds $SettlementWaitSeconds

Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-WatcherLog "generating final BTC 5m paper report"
Write-Status -State "generating_report" -Processes @()
$ReportArgs = @("scripts\generate_btc5m_paper_report.py", "--settlement-source", $SettlementSource)
if (-not [string]::IsNullOrWhiteSpace($LedgerDir)) {
    $ReportArgs += @("--ledger-dir", $LedgerDir)
}
if (-not [string]::IsNullOrWhiteSpace($Profiles)) {
    $ReportArgs += @("--profiles", $Profiles)
}
if (-not [string]::IsNullOrWhiteSpace($SignalLogPath)) {
    $ReportArgs += @("--signal-log-path", $SignalLogPath)
}
& $Python @ReportArgs > $ReportStdout 2> $ReportStderr
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
    Write-WatcherLog "report generation complete"
    Write-Status -State "complete" -Processes @()
} else {
    Write-WatcherLog "report generation failed with exit code $ExitCode"
    Write-Status -State "report_failed" -Processes @()
}

exit $ExitCode
