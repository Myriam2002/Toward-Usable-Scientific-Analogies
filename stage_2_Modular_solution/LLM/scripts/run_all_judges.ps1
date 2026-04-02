# =============================================================================
# Run All 3 Judge Models in Parallel
# =============================================================================
# Launches 3 PowerShell terminals simultaneously (one per judge model).
# Each terminal runs both modes (3scale + 3scale_fewshot) for its model.
# This script waits for all three to complete and prints a final summary.
#
# Usage:
#   .\run_all_judges.ps1          # full run (~25,159 records × 3 models × 2 modes)
#   .\run_all_judges.ps1 -Test    # test mode (5 records per combo, no files written)
#
# Output files (6 total, in results\upgraded_llm\):
#   upgraded_judge_3scale_gpt-4.1-mini.csv
#   upgraded_judge_3scale_gemini-2.5-flash-lite.csv
#   upgraded_judge_3scale_deepseek-r1.csv
#   upgraded_judge_3scale_fewshot_gpt-4.1-mini.csv
#   upgraded_judge_3scale_fewshot_gemini-2.5-flash-lite.csv
#   upgraded_judge_3scale_fewshot_deepseek-r1.csv
# =============================================================================

param(
    [switch]$Test
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$judgeModels = @("gpt-4.1-mini", "gemini-2.5-flash-lite", "deepseek-r1")
$testLabel   = if ($Test) { " [TEST MODE]" } else { "" }
$markerDir   = Join-Path $scriptDir "..\results\.markers"

# =============================================================================
# STEP 1: Clear old judge marker files
# =============================================================================
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  Launch All Judge Models + Wait$testLabel" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $markerDir) {
    Remove-Item -Path "$markerDir\judge_*.done" -Force -ErrorAction SilentlyContinue
    Write-Host "Cleared old judge completion markers" -ForegroundColor Gray
}
New-Item -ItemType Directory -Path $markerDir -Force | Out-Null

# =============================================================================
# STEP 2: Launch 3 terminals in parallel
# =============================================================================
Write-Host ""
Write-Host "Launching $($judgeModels.Count) judge terminals..." -ForegroundColor Yellow
Write-Host ""

$startTime    = Get-Date
$runScript    = Join-Path $scriptDir "run_single_judge.ps1"

foreach ($model in $judgeModels) {
    Write-Host "  Launching: $model" -ForegroundColor Cyan
    if ($Test) {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $model, "-Test"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $model
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "All $($judgeModels.Count) terminals launched!" -ForegroundColor Green
Write-Host ""

# =============================================================================
# STEP 3: Wait for all models to complete (polls every 10 seconds)
# =============================================================================
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "  Waiting for all judge models to complete..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$totalModels   = $judgeModels.Count
$checkInterval = 10  # seconds

while ($true) {
    $completedModels = @()
    $pendingModels   = @()

    foreach ($model in $judgeModels) {
        $markerFile = Join-Path $markerDir "judge_$model.done"
        if (Test-Path $markerFile) {
            $completedModels += $model
        } else {
            $pendingModels += $model
        }
    }

    $elapsed       = (Get-Date) - $startTime
    $completedCount = $completedModels.Count

    Write-Host "`r  [$completedCount/$totalModels] complete | Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Pending: $($pendingModels -join ', ')" -ForegroundColor Magenta -NoNewline

    if ($completedCount -eq $totalModels) {
        Write-Host ""
        Write-Host ""
        Write-Host "  All judge models completed!" -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds $checkInterval
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
$totalElapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "  COMPLETE!" -ForegroundColor Green
Write-Host "  Total time: $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Output files (results\upgraded_llm\):" -ForegroundColor Cyan
foreach ($model in $judgeModels) {
    Write-Host "    upgraded_judge_3scale_$model.csv" -ForegroundColor Gray
    Write-Host "    upgraded_judge_3scale_fewshot_$model.csv" -ForegroundColor Gray
}
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
