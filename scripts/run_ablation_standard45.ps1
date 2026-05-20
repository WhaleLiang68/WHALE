param(
    [string]$Python = "",
    [string]$Instance = "Du62",
    [int]$G = 600,
    [int]$TMax = 240,
    [double]$TInitial = 5000.0,
    [double]$KHist = 20.0,
    [string]$FixedSeeds = "20260328,20260329,20260330,20260331,20260332",
    [string]$LogRoot = "",
    [string]$StartGroup = "",
    [string]$ResumeBaselineLog = "",
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
    $groups = @(
        @{ Name = "B0_baseline"; Env = @{} },
        @{ Name = "A1_no_topk_guided"; Env = @{ ELP_TOPK_GUIDED_ENABLED = "0" } },
        @{ Name = "A2_no_high_flow_warmstart"; Env = @{ ELP_HIGH_FLOW_WARMSTART_ENABLED = "0" } },
        @{ Name = "A3_no_reheat"; Env = @{ ELP_REHEAT_ENABLED = "0" } },
        @{ Name = "A4_no_mid_structural_shot"; Env = @{ ELP_MID_STRUCTURAL_SHOT_ENABLED = "0" } },
        @{ Name = "A5_no_elite_archive"; Env = @{ ELP_ELITE_ARCHIVE_ENABLED = "0" } },
        @{ Name = "A6_no_archive_switch"; Env = @{ ELP_ARCHIVE_SWITCH_ENABLED = "0" } },
        @{ Name = "A7_no_final_elite_push"; Env = @{ ELP_FINAL_ELITE_PUSH_ENABLED = "0" } },
        @{ Name = "A8_no_two_stage_heavy_actions"; Env = @{ ELP_TWO_STAGE_HEAVY_ACTIONS_ENABLED = "0" } },
        @{ Name = "A9_no_local_search_on_feasible_accept"; Env = @{ ELP_LOCAL_SEARCH_ON_ANY_FEASIBLE_ACCEPT = "0" } },
        @{ Name = "A10_no_segment_insert_light"; Env = @{ ELP_SEGMENT_INSERT_LIGHT_ENABLED = "0" } },
        @{ Name = "A11_random_policy_no_dqn"; Env = @{ ELP_RL_AGENT = "qlearning"; ELP_DQN_EPSILON_START = "1"; ELP_DQN_EPSILON_MIN = "1"; ELP_DQN_EPSILON_DECAY = "1" } }
    )

    $clearKeys = @(
        "ELP_RL_AGENT",
        "ELP_TOPK_GUIDED_ENABLED",
        "ELP_HIGH_FLOW_WARMSTART_ENABLED",
        "ELP_REHEAT_ENABLED",
        "ELP_MID_STRUCTURAL_SHOT_ENABLED",
        "ELP_ELITE_ARCHIVE_ENABLED",
        "ELP_ARCHIVE_SWITCH_ENABLED",
        "ELP_FINAL_ELITE_PUSH_ENABLED",
        "ELP_TWO_STAGE_HEAVY_ACTIONS_ENABLED",
        "ELP_LOCAL_SEARCH_ON_ANY_FEASIBLE_ACCEPT",
        "ELP_SEGMENT_INSERT_LIGHT_ENABLED",
        "ELP_DQN_EPSILON_START",
        "ELP_DQN_EPSILON_MIN",
        "ELP_DQN_EPSILON_DECAY"
    )

    if ([string]::IsNullOrWhiteSpace($LogRoot)) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $logRoot = Join-Path "files/ablation_logs" $timestamp
    } else {
        $logRoot = $LogRoot
    }
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $allSeeds = @($FixedSeeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    $startReached = [string]::IsNullOrWhiteSpace($StartGroup)

    foreach ($group in $groups) {
        if (-not $startReached) {
            if ($group.Name -eq $StartGroup) {
                $startReached = $true
            } else {
                continue
            }
        }

        foreach ($k in $clearKeys) {
            Remove-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue
        }

        $runSeeds = @($allSeeds)
        if (($group.Name -eq "B0_baseline") -and (-not [string]::IsNullOrWhiteSpace($ResumeBaselineLog))) {
            if (Test-Path $ResumeBaselineLog) {
                $completedCount = (Get-Content -Encoding UTF8 $ResumeBaselineLog | Select-String -Pattern "Experiment\s+\d+\s+complete\s+\|\s+best energy").Count
                if ($completedCount -ge $runSeeds.Count) {
                    $runSeeds = @()
                } elseif ($completedCount -gt 0) {
                    $runSeeds = $runSeeds[$completedCount..($runSeeds.Count - 1)]
                }
            } else {
                throw ("Resume baseline log not found: " + $ResumeBaselineLog)
            }
        }
        if ($group.Name -ne "B0_baseline") {
            $groupLogPath = Join-Path $logRoot ($group.Name + ".log")
            if (Test-Path $groupLogPath) {
                $completedCount = (Get-Content -Encoding UTF8 $groupLogPath | Select-String -Pattern "Experiment\s+\d+\s+complete\s+\|\s+best energy").Count
                if ($completedCount -ge $runSeeds.Count) {
                    $runSeeds = @()
                } elseif ($completedCount -gt 0) {
                    $runSeeds = $runSeeds[$completedCount..($runSeeds.Count - 1)]
                }
            }
        }

        if ($runSeeds.Count -eq 0) {
            $skipLine = "[ABLATION] skip group=" + $group.Name + " | reason=no pending seeds"
            Write-Host $skipLine
            Add-Content -Path (Join-Path $logRoot ($group.Name + ".log")) -Value $skipLine -Encoding UTF8
            continue
        }

        $runSeedCsv = [string]::Join(",", $runSeeds)
        $env:ELP_IS_EXP = "1"
        $env:ELP_EXP_INSTANCE = $Instance
        $env:ELP_G = [string]$G
        $env:ELP_T_MAX = [string]$TMax
        $env:ELP_T_INITIAL = [string]$TInitial
        $env:ELP_K_HIST = [string]$KHist
        $env:ELP_FIXED_SEEDS = $runSeedCsv
        $env:ELP_EXP_NUMBER = [string]$runSeeds.Count
        $env:ELP_RL_AGENT = "dqn"
        $env:ELP_EXP_ALGORITHM = "ablation_standard45_" + $group.Name
        $env:ELP_EXP_REMARK = "Ablation standard4.5 | group=" + $group.Name + " | seeds=" + $runSeedCsv + " | fixed_steps=" + [string]($G * $TMax)

        foreach ($key in $group.Env.Keys) {
            Set-Item -Path ("Env:" + [string]$key) -Value ([string]$group.Env[$key])
        }

        $logPath = Join-Path $logRoot ($group.Name + ".log")
        $startStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $header = "[ABLATION] start group=" + $group.Name + " | ts=" + $startStamp + " | G=" + $G + " | TMax=" + $TMax + " | seeds=" + $runSeedCsv
        Write-Host $header
        Add-Content -Path $logPath -Value $header -Encoding UTF8

        if ($DryRun) {
            $dryLine = "[ABLATION] DryRun enabled, skip execution for group=" + $group.Name
            Write-Host $dryLine
            Add-Content -Path $logPath -Value $dryLine -Encoding UTF8
            continue
        }

        $savedErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python "src/algorithms/ELP_DRL_Standard4.5.py" 2>&1 | ForEach-Object {
                $line = $_.ToString()
                Write-Host $line
                Add-Content -Path $logPath -Value $line -Encoding UTF8
            }
        }
        finally {
            $ErrorActionPreference = $savedErrorAction
        }

        $exitCode = $LASTEXITCODE
        $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $tail = "[ABLATION] end group=" + $group.Name + " | exit_code=" + $exitCode + " | ts=" + $endStamp
        Write-Host $tail
        Add-Content -Path $logPath -Value $tail -Encoding UTF8

        if ($exitCode -ne 0) {
            throw ("Group " + $group.Name + " failed with exit code " + $exitCode)
        }
    }

    Write-Host ("[ABLATION] Done. Logs saved to: " + $logRoot)
}
finally {
    Pop-Location
}
