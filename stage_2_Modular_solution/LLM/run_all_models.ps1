# =============================================================================
# Run All Models with Both Modes in Parallel
# =============================================================================
# This script runs all 12 models with both targetonly and withsub modes
# Uses 12 concurrent workers to process jobs in parallel
# =============================================================================

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "Running All Models with Both Modes (Parallel Execution)" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

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

$modes = @("targetonly", "withsub")

$maxConcurrent = 12  # Run 12 jobs at a time
$allJobs = @()

# Create all job definitions
foreach ($model in $models) {
    foreach ($mode in $modes) {
        $allJobs += @{
            Model = $model
            Mode = $mode
            Name = "$model-$mode"
        }
    }
}

$runningJobs = @()
$completed = 0
$total = $allJobs.Count
$startTime = Get-Date
$lastStatusUpdate = Get-Date
$statusUpdateInterval = 30  # Show status every 30 seconds

Write-Host "Total jobs: $total (running $maxConcurrent at a time)" -ForegroundColor Yellow
Write-Host "Started at: $startTime" -ForegroundColor Yellow
Write-Host "Status updates every $statusUpdateInterval seconds`n" -ForegroundColor Yellow

# Function to show progress
function Show-Progress {
    param($runningJobs, $completed, $total, $startTime)
    
    $elapsed = (Get-Date) - $startTime
    $running = $runningJobs.Count
    
    Write-Host "`n--- Progress Update ---" -ForegroundColor Magenta
    Write-Host "Completed: $completed/$total | Running: $running | Elapsed: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
    
    if ($running -gt 0) {
        Write-Host "Currently running:" -ForegroundColor Cyan
        foreach ($j in $runningJobs) {
            $jobElapsed = (Get-Date) - $j.PSBeginTime
            Write-Host "  - $($j.Name) (running for $($jobElapsed.ToString('mm\:ss')))" -ForegroundColor Gray
        }
    }
    Write-Host "---`n" -ForegroundColor Magenta
}

foreach ($jobDef in $allJobs) {
    # Wait if we're at max concurrent
    while ($runningJobs.Count -ge $maxConcurrent) {
        $finished = $runningJobs | Where-Object { $_.State -ne 'Running' }
        foreach ($j in $finished) {
            $runningJobs.Remove($j) | Out-Null
            $completed++
            $elapsed = (Get-Date) - $startTime
            $status = if ($j.State -eq 'Completed') { "[OK] SUCCESS" } else { "[X] FAILED" }
            Write-Host "[$completed/$total] $status : $($j.Name) (Elapsed: $($elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor $(if ($j.State -eq 'Completed') { "Green" } else { "Red" })
            
            if ($j.State -ne 'Completed') {
                $errorOutput = Receive-Job -Job $j 2>&1
                Write-Host "  Error details: $errorOutput" -ForegroundColor Red
            }
        }
        
        # Show periodic progress update
        $now = Get-Date
        if (($now - $lastStatusUpdate).TotalSeconds -ge $statusUpdateInterval) {
            Show-Progress -runningJobs $runningJobs -completed $completed -total $total -startTime $startTime
            $lastStatusUpdate = $now
        }
        
        Start-Sleep -Seconds 2
    }
    
    # Start new job
    Write-Host "Starting: $($jobDef.Name)" -ForegroundColor Cyan
    $job = Start-Job -Name $jobDef.Name -ScriptBlock {
        param($model, $mode, $workDir)
        Set-Location $workDir
        python run_model.py --model $model --mode $mode 2>&1
    } -ArgumentList $jobDef.Model, $jobDef.Mode, $scriptDir
    $runningJobs += $job
}

# Wait for remaining jobs
Write-Host "`nWaiting for remaining jobs to complete...`n" -ForegroundColor Yellow
while ($runningJobs.Count -gt 0) {
    $finished = $runningJobs | Where-Object { $_.State -ne 'Running' }
    foreach ($j in $finished) {
        $runningJobs.Remove($j) | Out-Null
        $completed++
        $elapsed = (Get-Date) - $startTime
        $status = if ($j.State -eq 'Completed') { "[OK] SUCCESS" } else { "[X] FAILED" }
        Write-Host "[$completed/$total] $status : $($j.Name) (Elapsed: $($elapsed.ToString('hh\:mm\:ss')))" -ForegroundColor $(if ($j.State -eq 'Completed') { "Green" } else { "Red" })
        
        if ($j.State -ne 'Completed') {
            $errorOutput = Receive-Job -Job $j 2>&1
            Write-Host "  Error details: $errorOutput" -ForegroundColor Red
        }
    }
    
    # Show periodic progress update
    $now = Get-Date
    if (($now - $lastStatusUpdate).TotalSeconds -ge $statusUpdateInterval) {
        Show-Progress -runningJobs $runningJobs -completed $completed -total $total -startTime $startTime
        $lastStatusUpdate = $now
    }
    
    Start-Sleep -Seconds 2
}

$endTime = Get-Date
$totalElapsed = $endTime - $startTime

Write-Host ""
Write-Host "=========================================================================" -ForegroundColor Green
Write-Host "All jobs completed!" -ForegroundColor Green
Write-Host "Total time: $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "Started: $startTime" -ForegroundColor Green
Write-Host "Finished: $endTime" -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Green

# Clean up
$runningJobs | Remove-Job

Write-Host "`nResults saved in: results/" -ForegroundColor Cyan
Write-Host "Check the results directory for CSV files." -ForegroundColor Cyan
