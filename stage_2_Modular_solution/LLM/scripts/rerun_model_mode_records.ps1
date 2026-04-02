# =============================================================================
# Rerun Problematic Records for Single Model/Mode
# =============================================================================
# Usage: .\rerun_model_mode_records.ps1 -Model "gpt-4.1-mini" -Mode "withsub"
# Reruns only problematic records for a specific model/mode combination
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [Parameter(Mandatory=$true)]
    [string]$Mode
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Rerunning Problematic Records: $Model ($Mode)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$startTime = Get-Date

# Run the Python script to rerun problematic records for this model/mode
python utilities\rerun_problematic_records.py --model $Model --mode $Mode --rerun

$genResult = $LASTEXITCODE

# Run evaluation if generation succeeded
if ($genResult -eq 0) {
    $evalFile = "results\LLM_$Model`_$Mode.csv"
    if (Test-Path $evalFile) {
        Write-Host "`nEvaluating results..." -ForegroundColor Magenta
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
Write-Host "`nCompletion marker created: $markerFile" -ForegroundColor Gray
