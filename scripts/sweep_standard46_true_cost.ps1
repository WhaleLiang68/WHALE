param(
    [string]$Python = "",
    [string]$Instance = "Du62",
    [int]$G = 200,
    [int]$TMax = 120,
    [double]$TInitial = 5000.0,
    [double]$KHist = 20.0,
    [object]$FixedSeeds = "20260328,20260329,20260330,20260331,20260332",
    [object]$KPenaltyList = "0.35,0.5,0.7",
    [object]$TauList = "0.2",
    [object]$AlphaList = "0.5,0.7,1.0",
    [object]$BetaList = "8,10,12",
    [object]$EmaAlphaList = "0.05",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Parse-DoubleList {
    param([object]$Raw)
    $values = @()
    $tokens = @()
    if ($Raw -is [System.Array]) {
        foreach ($item in $Raw) {
            if ($null -ne $item) {
                $tokens += $item.ToString()
            }
        }
    } elseif ($null -ne $Raw) {
        $tokens = $Raw.ToString() -split ","
    }
    foreach ($token in $tokens) {
        $trimmed = $token.Trim()
        if ($trimmed -eq "") {
            continue
        }
        $values += [double]$trimmed
    }
    if ($values.Count -eq 0) {
        throw "参数列表不能为空: $Raw"
    }
    return $values
}

function Normalize-SeedCsv {
    param([object]$Raw)
    $tokens = @()
    if ($Raw -is [System.Array]) {
        foreach ($item in $Raw) {
            if ($null -eq $item) {
                continue
            }
            $trimmed = $item.ToString().Trim()
            if ($trimmed -ne "") {
                $tokens += $trimmed
            }
        }
    } elseif ($null -ne $Raw) {
        foreach ($token in ($Raw.ToString() -split '[,\\s]+')) {
            $trimmed = $token.Trim()
            if ($trimmed -ne "") {
                $tokens += $trimmed
            }
        }
    }
    if ($tokens.Count -eq 0) {
        throw "FixedSeeds 不能为空"
    }
    return ($tokens -join ",")
}

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
    $kPenaltyValues = Parse-DoubleList $KPenaltyList
    $tauValues = Parse-DoubleList $TauList
    $alphaValues = Parse-DoubleList $AlphaList
    $betaValues = Parse-DoubleList $BetaList
    $emaAlphaValues = Parse-DoubleList $EmaAlphaList
    $fixedSeedCsv = Normalize-SeedCsv $FixedSeeds

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logRoot = Join-Path "files/standard46_true_cost_sweeps" $timestamp
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

    $combos = @()
    foreach ($kPenalty in $kPenaltyValues) {
        foreach ($tau in $tauValues) {
            foreach ($alpha in $alphaValues) {
                foreach ($beta in $betaValues) {
                    foreach ($emaAlpha in $emaAlphaValues) {
                        $combos += [PSCustomObject]@{
                            KPenalty = $kPenalty
                            Tau = $tau
                            Alpha = $alpha
                            Beta = $beta
                            EmaAlpha = $emaAlpha
                        }
                    }
                }
            }
        }
    }

    $total = $combos.Count
    $index = 0
    foreach ($combo in $combos) {
        $index += 1
        $env:ELP_TRUE_COST_K_PENALTY = [string]$combo.KPenalty
        $env:ELP_TRUE_COST_TAU = [string]$combo.Tau
        $env:ELP_TRUE_COST_ALPHA = [string]$combo.Alpha
        $env:ELP_TRUE_COST_BETA = [string]$combo.Beta
        $env:ELP_TRUE_COST_EMA_ALPHA = [string]$combo.EmaAlpha

        $tag = "k{0}_tau{1}_a{2}_b{3}_ema{4}" -f `
            $combo.KPenalty.ToString([System.Globalization.CultureInfo]::InvariantCulture), `
            $combo.Tau.ToString([System.Globalization.CultureInfo]::InvariantCulture), `
            $combo.Alpha.ToString([System.Globalization.CultureInfo]::InvariantCulture), `
            $combo.Beta.ToString([System.Globalization.CultureInfo]::InvariantCulture), `
            $combo.EmaAlpha.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $safeTag = $tag.Replace(".", "p")
        $logPath = Join-Path $logRoot ($safeTag + ".log")

        $header = "[TRUE_COST_SWEEP] start {0}/{1} | {2} | seeds={3}" -f $index, $total, $tag, $fixedSeedCsv
        Write-Host $header
        Add-Content -Path $logPath -Value $header -Encoding UTF8

        if ($DryRun) {
            $dryLine = "[TRUE_COST_SWEEP] DryRun enabled, skip execution"
            Write-Host $dryLine
            Add-Content -Path $logPath -Value $dryLine -Encoding UTF8
            continue
        }

        & .\scripts\run_standard46_a8.ps1 `
            -Python $Python `
            -Instance $Instance `
            -G $G `
            -TMax $TMax `
            -TInitial $TInitial `
            -KHist $KHist `
            -FixedSeeds $fixedSeedCsv 2>&1 | ForEach-Object {
                $line = $_.ToString()
                Write-Host $line
                Add-Content -Path $logPath -Value $line -Encoding UTF8
            }

        $exitCode = $LASTEXITCODE
        $tail = "[TRUE_COST_SWEEP] end {0}/{1} | {2} | exit_code={3}" -f $index, $total, $tag, $exitCode
        Write-Host $tail
        Add-Content -Path $logPath -Value $tail -Encoding UTF8

        if ($exitCode -ne 0) {
            throw ("True-cost sweep failed for " + $tag + " with exit code " + $exitCode)
        }
    }
}
finally {
    Remove-Item Env:ELP_TRUE_COST_K_PENALTY -ErrorAction SilentlyContinue
    Remove-Item Env:ELP_TRUE_COST_TAU -ErrorAction SilentlyContinue
    Remove-Item Env:ELP_TRUE_COST_ALPHA -ErrorAction SilentlyContinue
    Remove-Item Env:ELP_TRUE_COST_BETA -ErrorAction SilentlyContinue
    Remove-Item Env:ELP_TRUE_COST_EMA_ALPHA -ErrorAction SilentlyContinue
    Pop-Location
}
