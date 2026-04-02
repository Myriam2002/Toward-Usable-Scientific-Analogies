# =============================================================================
# Rerun Specific Targets for Single Model/Mode
# =============================================================================
# Usage: .\rerun_single_model_targets.ps1 -Model "gpt-4.1-mini" -Mode "withsub" -TargetsFile "targets_gpt-4.1-mini_withsub.txt"
# Reruns specific targets from a pre-computed target list file
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [Parameter(Mandatory=$true)]
    [string]$Mode,
    [Parameter(Mandatory=$true)]
    [string]$TargetsFile
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Rerunning Targets: $Model ($Mode)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$startTime = Get-Date

# Read targets from file
if (-not (Test-Path $TargetsFile)) {
    Write-Host "ERROR: Targets file not found: $TargetsFile" -ForegroundColor Red
    exit 1
}

$targets = Get-Content $TargetsFile
$targetCount = $targets.Count
Write-Host "Found $targetCount targets to rerun" -ForegroundColor Yellow

if ($targetCount -eq 0) {
    Write-Host "No targets to rerun. Exiting." -ForegroundColor Green
    exit 0
}

# Join targets with comma for command line
$targetsString = $targets -join ","

# Run generation with target filter (run_model.py will merge results)
Write-Host "`n[1/2] GENERATING $Model - $Mode ($targetCount records)" -ForegroundColor Yellow
python core\run_model.py --model $Model --mode $Mode --targets $targetsString

$genResult = $LASTEXITCODE

# Run evaluation
if ($genResult -eq 0) {
    $evalFile = "results\LLM_$Model`_$Mode.csv"
    if (Test-Path $evalFile) {
        Write-Host "`n[2/2] EVALUATING $Model - $Mode" -ForegroundColor Magenta
        python core\evaluate_model.py --input $evalFile
        $evalResult = $LASTEXITCODE
    } else {
        Write-Host "  Warning: Generation file not found" -ForegroundColor Yellow
        $evalResult = 1
    }
} else {
    Write-Host "  Generation failed, skipping evaluation" -ForegroundColor Red
    $evalResult = 1
}

# Summary
$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "Model $Model ($Mode) completed!" -ForegroundColor Green
Write-Host "Total time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "Generation: $(if ($genResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($genResult -eq 0) { 'Green' } else { 'Red' })
Write-Host "Evaluation: $(if ($evalResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($evalResult -eq 0) { 'Green' } else { 'Red' })
Write-Host "=========================================================================" -ForegroundColor Green

# Create completion marker file
$markerDir = Join-Path $scriptDir "results\.markers"
if (-not (Test-Path $markerDir)) { New-Item -ItemType Directory -Path $markerDir -Force | Out-Null }
$markerFile = Join-Path $markerDir "$Model.$Mode.rerun.done"
Get-Date | Out-File -FilePath $markerFile
Write-Host "`nCompletion marker created" -ForegroundColor Gray

# Clean up targets file
Remove-Item -Path $TargetsFile -Force -ErrorAction SilentlyContinue
