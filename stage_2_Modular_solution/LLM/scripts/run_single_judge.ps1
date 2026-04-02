# =============================================================================
# Run Single Judge Model - 3scale + 3scale_fewshot (Sequential)
# =============================================================================
# Usage:
#   .\run_single_judge.ps1 -Model "gpt-4.1-mini"
#   .\run_single_judge.ps1 -Model "gemini-2.5-flash-lite" -Test
#   .\run_single_judge.ps1 -Model "deepseek-r1"
#
# Runs both modes for the given model sequentially.
# Output files:
#   results\upgraded_llm\upgraded_judge_3scale_<model>.csv
#   results\upgraded_llm\upgraded_judge_3scale_fewshot_<model>.csv
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("gpt-4.1-mini", "gemini-2.5-flash-lite", "deepseek-r1", "claude-sonnet-4.6", "mimo-v2-pro")]
    [string]$Model,
    [switch]$Test,
    [switch]$FewShotOnly
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$testLabel = if ($Test) { " [TEST MODE]" } else { "" }

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  LLM Judge: $Model$testLabel" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan

$startTime = Get-Date

# =============================================================================
# [1/2] 3scale (no few-shot) — skipped when -FewShotOnly is set
# =============================================================================
$result3scale = 0
if (-not $FewShotOnly) {
    Write-Host ""
    Write-Host "[1/2] $Model - 3scale (no few-shot)" -ForegroundColor Yellow
    if ($Test) {
        python ..\core\run_judge.py --model $Model --mode 3scale --test
    } else {
        python ..\core\run_judge.py --model $Model --mode 3scale
    }
    $result3scale = $LASTEXITCODE
}

# =============================================================================
# [2/2] 3scale_fewshot  (shown as [1/1] when -FewShotOnly)
# =============================================================================
$fewshotLabel = if ($FewShotOnly) { "[1/1]" } else { "[2/2]" }
Write-Host ""
Write-Host "$fewshotLabel $Model - 3scale_fewshot (with calibration examples)" -ForegroundColor Yellow
if ($Test) {
    python ..\core\run_judge.py --model $Model --mode 3scale_fewshot --test
} else {
    python ..\core\run_judge.py --model $Model --mode 3scale_fewshot
}
$resultFewshot = $LASTEXITCODE

# =============================================================================
# SUMMARY
# =============================================================================
$endTime = Get-Date
$elapsed = $endTime - $startTime

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "  $Model - Done!" -ForegroundColor Green
Write-Host "  Total time: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "  Results:" -ForegroundColor Cyan

if (-not $FewShotOnly) {
    if ($result3scale -eq 0) {
        Write-Host "    3scale:         SUCCESS" -ForegroundColor Green
    } else {
        Write-Host "    3scale:         FAILED" -ForegroundColor Red
    }
}

if ($resultFewshot -eq 0) {
    Write-Host "    3scale_fewshot: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "    3scale_fewshot: FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "  Output files:" -ForegroundColor Cyan
if (-not $FewShotOnly) {
    Write-Host "    results\upgraded_llm\upgraded_judge_3scale_$Model.csv" -ForegroundColor Gray
}
Write-Host "    results\upgraded_llm\upgraded_judge_3scale_fewshot_$Model.csv" -ForegroundColor Gray
Write-Host "=========================================================================" -ForegroundColor Green

# Create completion marker file
$markerDir = Join-Path $scriptDir "..\results\.markers"
if (-not (Test-Path $markerDir)) {
    New-Item -ItemType Directory -Path $markerDir -Force | Out-Null
}
$markerFile = Join-Path $markerDir "judge_$Model.done"
Get-Date | Out-File -FilePath $markerFile
Write-Host ""
Write-Host "Completion marker: $markerFile" -ForegroundColor Gray
