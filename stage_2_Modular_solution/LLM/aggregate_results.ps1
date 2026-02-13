# =============================================================================
# Aggregate All Results by Mode
# =============================================================================
# Run this AFTER all model terminals have completed
# Creates: all_results_targetonly.csv and all_results_withsub.csv
# =============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Aggregating All Model Results" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

python aggregate_results.py

Write-Host "`nAggregation complete!" -ForegroundColor Green
Write-Host "Check results/ folder for:" -ForegroundColor Cyan
Write-Host "  - all_results_targetonly.csv" -ForegroundColor Gray
Write-Host "  - all_results_withsub.csv" -ForegroundColor Gray
