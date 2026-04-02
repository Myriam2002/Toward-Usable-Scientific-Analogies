# =============================================================================
# Rerun Failed WithSub Models (Parallel)
# =============================================================================
# Checks existing withsub CSV files and reruns only failed models in parallel
# Each failed model runs in a separate terminal (withsub mode + evaluation)
# =============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Rerun Failed WithSub Models (Parallel)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check which models failed
Write-Host "Checking for failed models..." -ForegroundColor Yellow

# Run check and capture stdout (model names only) and stderr (status messages) separately
$listOutput = python utilities\rerun_withsub_failed.py --list-only 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error checking models. Please run manually:" -ForegroundColor Red
    Write-Host "  python utilities\rerun_withsub_failed.py" -ForegroundColor Yellow
    exit 1
}

# Parse failed models from output (one model name per line)
$failedModels = @()
foreach ($line in $listOutput) {
    $line = $line.Trim()
    if ($line -and $line -match '^[a-z0-9\-]+$') {
        $failedModels += $line
    }
}

if ($failedModels.Count -eq 0) {
    Write-Host "`nNo failed models found. All withsub results are OK!" -ForegroundColor Green
    exit
}

Write-Host "`nFound $($failedModels.Count) failed model(s):" -ForegroundColor Yellow
foreach ($model in $failedModels) {
    Write-Host "  - $model" -ForegroundColor Gray
}

Write-Host ""
$response = Read-Host "Rerun these models in parallel? (y/n)"
if ($response -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

# Step 2: Clear old markers for these models
$markerDir = Join-Path $scriptDir "results\.markers"
if (Test-Path $markerDir) {
    foreach ($model in $failedModels) {
        $markerFile = Join-Path $markerDir "$model.withsub.done"
        if (Test-Path $markerFile) {
            Remove-Item -Path $markerFile -Force -ErrorAction SilentlyContinue
        }
    }
}
New-Item -ItemType Directory -Path $markerDir -Force | Out-Null

# Step 3: Launch terminals in parallel
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "Launching $($failedModels.Count) terminals (one per failed model)..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date
$runScript = Join-Path $scriptDir "run_withsub_only.ps1"

foreach ($model in $failedModels) {
    Write-Host "  Launching: $model" -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $model
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "All terminals launched!" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for completion
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "Waiting for all models to complete..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$checkInterval = 10  # Check every 10 seconds

while ($true) {
    $completedModels = @()
    $pendingModels = @()
    
    foreach ($model in $failedModels) {
        $markerFile = Join-Path $markerDir "$model.withsub.done"
        if (Test-Path $markerFile) {
            $completedModels += $model
        } else {
            $pendingModels += $model
        }
    }
    
    $elapsed = (Get-Date) - $startTime
    $completedCount = $completedModels.Count
    
    # Show progress
    Write-Host "`r[$completedCount/$($failedModels.Count)] completed | Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Pending: $($pendingModels.Count)" -ForegroundColor Magenta -NoNewline
    
    if ($completedCount -eq $failedModels.Count) {
        Write-Host ""
        Write-Host ""
        Write-Host "All models completed!" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $checkInterval
}

# Final summary
$endTime = Get-Date
$totalElapsed = $endTime - $startTime

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "Rerun Complete!" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Total time: $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "Rerun models: $($failedModels.Count)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: results/" -ForegroundColor Cyan
Write-Host "Check individual terminal windows for detailed output." -ForegroundColor Gray
