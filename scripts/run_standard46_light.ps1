param(
    [string]$Python = "",
    [string]$Instance = "Du62",
    [int]$G = 600,
    [int]$TMax = 240,
    [double]$TInitial = 5000.0,
    [double]$KHist = 20.0,
    [string]$FixedSeeds = "20260328,20260329,20260330,20260331,20260332",
    [switch]$RandomPolicy,
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
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logRoot = Join-Path "files/standard46_logs" $timestamp
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

    $algorithmName = "ELP_RL_Standard_standard46_light"
    if ($RandomPolicy) {
        $algorithmName = "ELP_RL_Standard_standard46_light_random"
    }

    $env:ELP_STANDARD46_PROFILE = "light"
    $env:ELP_IS_EXP = "1"
    $env:ELP_EXP_INSTANCE = $Instance
    $env:ELP_G = [string]$G
    $env:ELP_T_MAX = [string]$TMax
    $env:ELP_T_INITIAL = [string]$TInitial
    $env:ELP_K_HIST = [string]$KHist
    $env:ELP_FIXED_SEEDS = $FixedSeeds
    $env:ELP_EXP_NUMBER = [string]($FixedSeeds.Split(",").Count)
    $env:ELP_EXP_ALGORITHM = $algorithmName
    $env:ELP_EXP_REMARK = "Standard4.6 light core | random_policy=" + [string][bool]$RandomPolicy + " | seeds=" + $FixedSeeds + " | fixed_steps=" + [string]($G * $TMax)
    $env:ELP_PRINT_TELEMETRY = "1"
    $env:ELP_TWO_STAGE_HEAVY_ACTIONS_ENABLED = "0"
    $env:ELP_REHEAT_ENABLED = "0"
    $env:ELP_SEGMENT_INSERT_LIGHT_ENABLED = "0"
    $env:ELP_FINAL_ELITE_PUSH_ENABLED = "0"
    $env:ELP_MID_STRUCTURAL_SHOT_ENABLED = "0"

    if ($RandomPolicy) {
        $env:ELP_STANDARD46_RANDOM_POLICY = "1"
        $env:ELP_RL_AGENT = "qlearning"
        $env:ELP_DQN_EPSILON_START = "1"
        $env:ELP_DQN_EPSILON_MIN = "1"
        $env:ELP_DQN_EPSILON_DECAY = "1"
    } else {
        Remove-Item -Path "Env:ELP_STANDARD46_RANDOM_POLICY" -ErrorAction SilentlyContinue
        $env:ELP_RL_AGENT = "dqn"
        Remove-Item -Path "Env:ELP_DQN_EPSILON_START" -ErrorAction SilentlyContinue
        Remove-Item -Path "Env:ELP_DQN_EPSILON_MIN" -ErrorAction SilentlyContinue
        Remove-Item -Path "Env:ELP_DQN_EPSILON_DECAY" -ErrorAction SilentlyContinue
    }

    $logPath = Join-Path $logRoot ($algorithmName + ".log")
    $header = "[STANDARD46] start algorithm=" + $algorithmName + " | G=" + $G + " | TMax=" + $TMax + " | seeds=" + $FixedSeeds
    Write-Host $header
    Add-Content -Path $logPath -Value $header -Encoding UTF8

    if ($DryRun) {
        $dryLine = "[STANDARD46] DryRun enabled, skip execution"
        Write-Host $dryLine
        Add-Content -Path $logPath -Value $dryLine -Encoding UTF8
        return
    }

    $savedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python "src/algorithms/ELP_DRL_Standard4.6.py" 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -Path $logPath -Value $line -Encoding UTF8
        }
    }
    finally {
        $ErrorActionPreference = $savedErrorAction
    }

    $exitCode = $LASTEXITCODE
    $tail = "[STANDARD46] end algorithm=" + $algorithmName + " | exit_code=" + $exitCode
    Write-Host $tail
    Add-Content -Path $logPath -Value $tail -Encoding UTF8

    if ($exitCode -ne 0) {
        throw ("Standard4.6 run failed with exit code " + $exitCode)
    }
}
finally {
    Pop-Location
}
