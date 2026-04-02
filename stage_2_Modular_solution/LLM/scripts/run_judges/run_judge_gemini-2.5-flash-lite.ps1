$scriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $scriptDir
& .\run_single_judge.ps1 -Model "gemini-2.5-flash-lite"
