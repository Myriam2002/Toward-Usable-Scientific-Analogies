# =============================================================================
# Rerun Judge Errors - Re-evaluate records that returned errors
# =============================================================================
# Usage:
#   .\rerun_judge_errors.ps1 -Model "gpt-4.1-mini" -Mode "3scale"
#   .\rerun_judge_errors.ps1 -Model "gpt-4.1-mini" -Mode "3scale_fewshot"
#   .\rerun_judge_errors.ps1 -Model "gpt-4.1-mini" -Mode "both"
#
# Reads the completed output CSV, finds rows with status=error, and re-runs
# only those records. Successful records are preserved unchanged.
# Output overwrites the same file: results\upgraded_llm\upgraded_judge_{mode}_{model}.csv
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("gpt-4.1-mini", "gemini-2.5-flash-lite", "deepseek-r1", "claude-sonnet-4.6", "mimo-v2-pro")]
    [string]$Model,

    [Parameter(Mandatory=$true)]
    [ValidateSet("3scale", "3scale_fewshot", "both")]
    [string]$Mode
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  Rerun Judge Errors: $Model / $Mode" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

$startTime     = Get-Date
$result3scale  = 0
$resultFewshot = 0

# =============================================================================
# 3scale mode
# =============================================================================
if ($Mode -eq "3scale" -or $Mode -eq "both") {
    Write-Host "[3scale] $Model - rerunning errors" -ForegroundColor Yellow
    python ..\core\run_judge.py --model $Model --mode 3scale --rerun-errors
    $result3scale = $LASTEXITCODE
    Write-Host ""
}

# =============================================================================
# 3scale_fewshot mode
# =============================================================================
if ($Mode -eq "3scale_fewshot" -or $Mode -eq "both") {
    Write-Host "[3scale_fewshot] $Model - rerunning errors" -ForegroundColor Yellow
    python ..\core\run_judge.py --model $Model --mode 3scale_fewshot --rerun-errors
    $resultFewshot = $LASTEXITCODE
    Write-Host ""
}

# =============================================================================
# SUMMARY
# =============================================================================
$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "  Done!  Total time: $($elapsed.ToString('hh\\:mm\\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "  Results:" -ForegroundColor Cyan

if ($Mode -eq "3scale" -or $Mode -eq "both") {
    if ($result3scale -eq 0) {
        Write-Host "    3scale:         SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "    3scale:         FAILED" -ForegroundColor Red
    }
}

if ($Mode -eq "3scale_fewshot" -or $Mode -eq "both") {
    if ($resultFewshot -eq 0) {
        Write-Host "    3scale_fewshot: SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "    3scale_fewshot: FAILED" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  Output (overwritten in place):" -ForegroundColor Cyan
if ($Mode -eq "3scale" -or $Mode -eq "both") {
    Write-Host "    results\upgraded_llm\upgraded_judge_3scale_$Model.csv" -ForegroundColor Gray
}
if ($Mode -eq "3scale_fewshot" -or $Mode -eq "both") {
    Write-Host "    results\upgraded_llm\upgraded_judge_3scale_fewshot_$Model.csv" -ForegroundColor Gray
}
Write-Host "=========================================================================" -ForegroundColor Green
