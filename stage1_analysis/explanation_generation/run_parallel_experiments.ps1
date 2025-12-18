# Parallel Experiment Runner (PowerShell)
# This script runs all 6 experiment settings in parallel
# Each setting will use ALL available models

Write-Host "Starting parallel experiments..." -ForegroundColor Green
Write-Host "Each setting will run ALL models on the full dataset"
Write-Host "============================================"

# Array to hold all jobs
$jobs = @()

# Start each setting as a background job
Write-Host "Starting: none" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting none 
}

Write-Host "Starting: none_description" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting none_description 
}

Write-Host "Starting: unpaired_properties" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting unpaired_properties 
}

Write-Host "Starting: unpaired_properties_description" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting unpaired_properties_description 
}

Write-Host "Starting: paired_properties" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting paired_properties 
}

Write-Host "Starting: paired_properties_description" -ForegroundColor Cyan
$jobs += Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python run_experiments.py --setting paired_properties_description 
}

Write-Host "============================================"
Write-Host "All experiments started!" -ForegroundColor Green
Write-Host "Waiting for all to complete..."
Write-Host ""

# Wait for all jobs and show progress
$jobs | ForEach-Object {
    $_ | Wait-Job | Out-Null
    $jobName = $_.Name
    Write-Host "✅ Completed job: $jobName" -ForegroundColor Green
    
    # Show output
    Receive-Job -Job $_
    
    # Clean up
    Remove-Job -Job $_
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "🎉 All experiments completed!" -ForegroundColor Green
Write-Host "Results saved in: checkpoints/explanation_generation/"

