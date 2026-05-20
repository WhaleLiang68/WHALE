param(
    [string]$Python = "",
    [string]$Instance = "Du62",
    [int]$G = 1000,
    [int]$TMax = 300,
    [double]$TInitial = 10000.0,
    [double]$KHist = 10.0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$defaultPython = "C:\Users\17122\AppData\Local\conda\conda\envs\tensorflow\python.exe"
if ([string]::IsNullOrWhiteSpace($Python)) {
    if (Test-Path $defaultPython) {
        $Python = $defaultPython
    } else {
        $Python = "python"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $seeds = @(
        1103, 1693, 2437, 3203, 4021
    )
    $seedCsv = ($seeds -join ",")

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logRoot = Join-Path "files/aba_screen_logs" $timestamp
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

    $profiles = @("ABA_S2")
    foreach ($profile in $profiles) {
        $env:ELP_RL_AGENT = "dqn"
        $env:ELP_TUNE_PROFILE = $profile
        $env:ELP_FIXED_SEEDS = $seedCsv
        $env:ELP_EXP_NUMBER = "$($seeds.Count)"
        $env:ELP_IS_EXP = "1"
        $env:ELP_EXP_INSTANCE = $Instance
        $env:ELP_G = "$G"
        $env:ELP_T_MAX = "$TMax"
        $env:ELP_T_INITIAL = "$TInitial"
        $env:ELP_K_HIST = "$KHist"
        $env:ELP_EXP_ALGORITHM = "ELP_RL_Standard_$profile"
        $env:ELP_EXP_REMARK = "ABA-screen profile=$profile seeds=$($seeds.Count)"

        $logPath = Join-Path $logRoot "$profile.log"
        $startStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $header = "[ABA-SCREEN] start profile=$profile | seeds=$($seeds.Count) | ts=$startStamp"
        Write-Host "$header | log=$logPath"
        Add-Content -Path $logPath -Value $header -Encoding UTF8

        if ($DryRun) {
            $dryLine = "[ABA-SCREEN] DryRun enabled, skip execution for profile=$profile"
            Write-Host $dryLine
            Add-Content -Path $logPath -Value $dryLine -Encoding UTF8
            continue
        }

        $savedErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python "-m" "src.algorithms.ELP_DRL_Standard" 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) {
                    $line = $_.ToString()
                } else {
                    $line = $_.ToString()
                }
                Write-Host $line
                Add-Content -Path $logPath -Value $line -Encoding UTF8
            }
        }
        finally {
            $ErrorActionPreference = $savedErrorAction
        }

        $exitCode = $LASTEXITCODE
        $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $tail = "[ABA-SCREEN] end profile=$profile | exit_code=$exitCode | ts=$endStamp"
        Write-Host $tail
        Add-Content -Path $logPath -Value $tail -Encoding UTF8

        if ($exitCode -ne 0) {
            throw "Profile $profile failed with exit code $exitCode"
        }
    }

    Write-Host "[ABA-SCREEN] Done. Logs saved to: $logRoot"
}
finally {
    Pop-Location
}

