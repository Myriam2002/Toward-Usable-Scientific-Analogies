# =============================================================================
# Rerun Problematic Records (Parallel)
# =============================================================================
# Identifies records with problems across all models and reruns only those records
# Runs in parallel for each model/mode combination (one terminal per model/mode)
# Pre-computes all targets BEFORE launching terminals to avoid file contention
# =============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Rerun Problematic Records (Parallel)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check for problematic records AND export targets to files
Write-Host "Checking for problematic records and exporting targets..." -ForegroundColor Yellow
$checkOutput = python utilities\rerun_problematic_records.py --export-targets 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error checking records." -ForegroundColor Red
    Write-Host $checkOutput
    exit 1
}

# Parse output to identify model/mode combinations with problems
# Format: "model-name (mode): N records -> targets_model_mode.txt"
$problematicCombos = @()

foreach ($line in $checkOutput) {
    $line = $line.Trim()
    # Match pattern: "model-name (targetonly|withsub): N records -> filename"
    if ($line -match '^([a-z0-9\-\.]+)\s+\((targetonly|withsub)\):\s+(\d+)\s+records\s+->\s+(.+\.txt)$') {
        $model = $matches[1]
        $mode = $matches[2]
        $count = [int]$matches[3]
        $targetsFile = $matches[4]
        if ($count -gt 0) {
            $problematicCombos += @{
                Model = $model
                Mode = $mode
                Count = $count
                TargetsFile = $targetsFile
            }
        }
    }
}

if ($problematicCombos.Count -eq 0) {
    Write-Host "`nNo problematic records found!" -ForegroundColor Green
    exit
}

# Show summary
Write-Host ""
Write-Host "Found $($problematicCombos.Count) model/mode combinations with problems:" -ForegroundColor Yellow
foreach ($combo in $problematicCombos) {
    Write-Host "  - $($combo.Model) ($($combo.Mode)): $($combo.Count) records" -ForegroundColor Gray
}

Write-Host ""
$response = Read-Host "Rerun all problematic records in parallel? (y/n)"
if ($response -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    # Clean up target files
    foreach ($combo in $problematicCombos) {
        $targetsFile = Join-Path $scriptDir $combo.TargetsFile
        if (Test-Path $targetsFile) {
            Remove-Item -Path $targetsFile -Force -ErrorAction SilentlyContinue
        }
    }
    exit
}

# Step 2: Clear old markers
$markerDir = Join-Path $scriptDir "results\.markers"
if (Test-Path $markerDir) {
    foreach ($combo in $problematicCombos) {
        $markerFile = Join-Path $markerDir "$($combo.Model).$($combo.Mode).rerun.done"
        if (Test-Path $markerFile) {
            Remove-Item -Path $markerFile -Force -ErrorAction SilentlyContinue
        }
    }
}
New-Item -ItemType Directory -Path $markerDir -Force | Out-Null

# Step 3: Launch terminals in parallel (each reads its own pre-computed targets file)
Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "Launching $($problematicCombos.Count) terminals (one per model/mode)..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date
$runScript = Join-Path $scriptDir "rerun_single_model_targets.ps1"

foreach ($combo in $problematicCombos) {
    Write-Host "  Launching: $($combo.Model) ($($combo.Mode)) - $($combo.Count) records" -ForegroundColor Cyan
    $targetsFile = Join-Path $scriptDir $combo.TargetsFile
    Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $runScript, "-Model", $combo.Model, "-Mode", $combo.Mode, "-TargetsFile", $targetsFile
    Start-Sleep -Milliseconds 300
}

Write-Host ""
Write-Host "All terminals launched!" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for completion
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host "Waiting for all model/mode combinations to complete..." -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Yellow
Write-Host ""

$checkInterval = 10  # Check every 10 seconds

while ($true) {
    $completedCombos = @()
    $pendingCombos = @()
    
    foreach ($combo in $problematicCombos) {
        $markerFile = Join-Path $markerDir "$($combo.Model).$($combo.Mode).rerun.done"
        if (Test-Path $markerFile) {
            $completedCombos += $combo
        } else {
            $pendingCombos += $combo
        }
    }
    
    $elapsed = (Get-Date) - $startTime
    $completedCount = $completedCombos.Count
    
    # Show progress
    $pendingList = ($pendingCombos | ForEach-Object { "$($_.Model)($($_.Mode))" }) -join ", "
    if ($pendingList.Length -gt 80) {
        $pendingList = $pendingList.Substring(0, 77) + "..."
    }
    Write-Host "`r[$completedCount/$($problematicCombos.Count)] completed | Elapsed: $($elapsed.ToString('hh\:mm\:ss')) | Pending: $($pendingCombos.Count)" -ForegroundColor Magenta -NoNewline
    
    if ($completedCount -eq $problematicCombos.Count) {
        Write-Host ""
        Write-Host ""
        Write-Host "All model/mode combinations completed!" -ForegroundColor Green
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
Write-Host "Rerun combinations: $($problematicCombos.Count)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: results/" -ForegroundColor Cyan
Write-Host "Check individual terminal windows for detailed output." -ForegroundColor Gray
