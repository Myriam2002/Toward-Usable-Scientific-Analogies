# =============================================================================
# Run Single Model - WithSub Mode Only (Generation + Evaluation)
# =============================================================================
# Usage: .\run_withsub_only.ps1 -Model "gpt-4.1-mini"
# Runs only withsub mode (generation + evaluation)
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Model
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Running Model: $Model - WithSub Mode Only" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$startTime = Get-Date

# ============================================================================
# STEP 1: GENERATION
# ============================================================================

Write-Host "`n[1/2] GENERATING $Model - withsub" -ForegroundColor Yellow
python core\run_model.py --model $Model --mode withsub
$genResult = $LASTEXITCODE

# ============================================================================
# STEP 2: EVALUATION
# ============================================================================

$withsubFile = "results\LLM_$Model`_withsub.csv"
Write-Host "`n[2/2] EVALUATING $Model - withsub" -ForegroundColor Magenta
if (Test-Path $withsubFile) {
    python core\evaluate_model.py --input $withsubFile
    $evalResult = $LASTEXITCODE
} else {
    Write-Host "  Skipping: Generation file not found" -ForegroundColor Red
    $evalResult = 1
}

# ============================================================================
# SUMMARY
# ============================================================================

$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "`n=========================================================================" -ForegroundColor Green
Write-Host "Model $Model (withsub) completed!" -ForegroundColor Green
Write-Host "Total time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "Generation: $(if ($genResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($genResult -eq 0) { 'Green' } else { 'Red' })
Write-Host "Evaluation: $(if ($evalResult -eq 0) { 'SUCCESS' } else { 'FAILED' })" -ForegroundColor $(if ($evalResult -eq 0) { 'Green' } else { 'Red' })
Write-Host ""
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  results\LLM_$Model`_withsub.csv" -ForegroundColor Gray
Write-Host "  results\LLM_$Model`_withsub_eval.csv" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Green

# Create completion marker file
$markerDir = Join-Path $scriptDir "results\.markers"
if (-not (Test-Path $markerDir)) { New-Item -ItemType Directory -Path $markerDir -Force | Out-Null }
$markerFile = Join-Path $markerDir "$Model.withsub.done"
Get-Date | Out-File -FilePath $markerFile
Write-Host "`nCompletion marker created: $markerFile" -ForegroundColor Gray
