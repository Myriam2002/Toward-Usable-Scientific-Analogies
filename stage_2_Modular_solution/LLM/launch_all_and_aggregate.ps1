# =============================================================================
# Launch All Models + Auto-Aggregate When Complete
# =============================================================================
# This script:
#   1. Clears old marker files
#   2. Launches 12 terminals (one per model)
#   3. Waits for all models to complete
#   4. Automatically runs aggregation
# =============================================================================

param(
    [switch]$Test
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$models = @(
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "grok-4-fast",
    "gemini-2.5-flash-lite",
    "llama-3.1-405b-instruct",
    "meta-llama-3-1-70b-instruct",
    "meta-llama-3-1-8b-instruct",
    "deepseek-r1",
    "qwen3-14b",
    "qwen3-32b"
)

$testLabel = if ($Test) { " [TEST MODE - 2 records]" } else { "" }
$markerDir = Join-Path $scriptDir "results\.markers"

# =============================================================================
# STEP 1: Clear old marker files
# =============================================================================
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Launch All Models + Auto-Aggregate$testLabel" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path $markerDir) {
    Remove-Item -Path "$markerDir\*.done" -Force -ErrorAction SilentlyContinue
    Write-Host "Cleared old completion markers" -ForegroundColor Gray
}
New-Item -ItemType Directory -Path $markerDir -Force | Out-Null

# =============================================================================
# STEP 2: Launch all terminals
# =============================================================================
Write-Host ""
Write-Host "Launching 12 PowerShell terminals..." -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date

foreach ($model in $models) {
    Write-Host "  Launching: $model" -ForegroundColor Cyan
    $runScript = Join-Path $scriptDir "run_single_model.ps1"
    if ($Test) {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $model, "-Test"
    } else {
        Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $model
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "All 12 terminals launched!" -ForegroundColor Green
Write-Host ""

# =============================================================================
# STEP 3: Wait for all models to complete
# =============================================================================
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "Waiting for all models to complete..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$totalModels = $models.Count
$checkInterval = 10  # Check every 10 seconds

while ($true) {
    $completedModels = @()
    $pendingModels = @()
    
    foreach ($model in $models) {
        $markerFile = Join-Path $markerDir "$model.done"
        if (Test-Path $markerFile) {
            $completedModels += $model
        } else {
            $pendingModels += $model
        }
    }
    
    $elapsed = (Get-Date) - $startTime
    $completedCount = $completedModels.Count
    
    # Show progress
    Write-Host "`r[$completedCount/$totalModels] completed | Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Pending: $($pendingModels.Count)" -ForegroundColor Magenta -NoNewline
    
    if ($completedCount -eq $totalModels) {
        Write-Host ""
        Write-Host ""
        Write-Host "All models completed!" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $checkInterval
}

# =============================================================================
# STEP 4: Run aggregation
# =============================================================================
$endTime = Get-Date
$totalElapsed = $endTime - $startTime

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Running Aggregation..." -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

python aggregate_results.py

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "COMPLETE!" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Total time: $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  - results\all_results_targetonly.csv" -ForegroundColor Gray
Write-Host "  - results\all_results_withsub.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
