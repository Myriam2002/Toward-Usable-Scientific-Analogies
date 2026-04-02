$scriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $scriptDir
& .\run_single_model.ps1 -Model "meta-llama-3-1-8b-instruct"
