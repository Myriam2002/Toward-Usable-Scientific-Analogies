# =============================================================================
# Run Single Model with Both Modes + Evaluation (Sequential)
# =============================================================================
# Usage: .\run_single_model.ps1 -Model "gpt-4.1-mini"
# Runs generation for both modes, then evaluates each
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [switch]$Test
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$testLabel = if ($Test) { " [TEST MODE - 5 records]" } else { "" }

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Running Model: $Model (Generation + Evaluation)$testLabel" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$startTime = Get-Date

# ============================================================================
# STEP 1: GENERATION
# ============================================================================

# Run targetonly mode - Generation
Write-Host "`n[1/4] GENERATING $Model - targetonly" -ForegroundColor Yellow
if ($Test) {
    python run_model.py --model $Model --mode targetonly --test
} else {
    python run_model.py --model $Model --mode targetonly
}
$targetonlyGenResult = $LASTEXITCODE

# Run withsub mode - Generation
Write-Host "`n[2/4] GENERATING $Model - withsub" -ForegroundColor Yellow
if ($Test) {
    python run_model.py --model $Model --mode withsub --test
} else {
    python run_model.py --model $Model --mode withsub
}
$withsubGenResult = $LASTEXITCODE

# ============================================================================
# STEP 2: EVALUATION
# ============================================================================

# Evaluate targetonly mode
$targetonlyFile = "results\LLM_$Model`_targetonly.csv"
Write-Host "`n[3/4] EVALUATING $Model - targetonly" -ForegroundColor Magenta
if (Test-Path $targetonlyFile) {
    python evaluate_model.py --input $targetonlyFile
    $targetonlyEvalResult = $LASTEXITCODE
} else {
    Write-Host "  Skipping: Generation file not found" -ForegroundColor Red
    $targetonlyEvalResult = 1
}

# Evaluate withsub mode
$withsubFile = "results\LLM_$Model`_withsub.csv"
Write-Host "`n[4/4] EVALUATING $Model - withsub" -ForegroundColor Magenta
if (Test-Path $withsubFile) {
    python evaluate_model.py --input $withsubFile
    $withsubEvalResult = $LASTEXITCODE
} else {
    Write-Host "  Skipping: Generation file not found" -ForegroundColor Red
    $withsubEvalResult = 1
}

# ============================================================================
# SUMMARY
# ============================================================================

$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "Model $Model completed!" -ForegroundColor Green
Write-Host "Total time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "Generation Results:" -ForegroundColor Cyan
Write-Host "  targetonly: $(if ($targetonlyGenResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($targetonlyGenResult -eq 0) { 'Green' } else { 'Red' })
Write-Host "  withsub:    $(if ($withsubGenResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($withsubGenResult -eq 0) { 'Green' } else { 'Red' })
Write-Host ""
Write-Host "Evaluation Results:" -ForegroundColor Cyan
Write-Host "  targetonly: $(if ($targetonlyEvalResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($targetonlyEvalResult -eq 0) { 'Green' } else { 'Red' })
Write-Host "  withsub:    $(if ($withsubEvalResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($withsubEvalResult -eq 0) { 'Green' } else { 'Red' })
Write-Host ""
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  results\LLM_$Model`_targetonly.csv" -ForegroundColor Gray
Write-Host "  results\LLM_$Model`_targetonly_eval.csv" -ForegroundColor Gray
Write-Host "  results\LLM_$Model`_withsub.csv" -ForegroundColor Gray
Write-Host "  results\LLM_$Model`_withsub_eval.csv" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Green

# Create completion marker file
$markerDir = Join-Path $scriptDir "results\.markers"
if (-not (Test-Path $markerDir)) { New-Item -ItemType Directory -Path $markerDir -Force | Out-Null }
$markerFile = Join-Path $markerDir "$Model.done"
Get-Date | Out-File -FilePath $markerFile
Write-Host "`nCompletion marker created: $markerFile" -ForegroundColor Gray
