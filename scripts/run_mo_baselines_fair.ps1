Param(
    [string]$Instance = "Du62",
    [int]$Runs = 10,
    [int]$BaseSeed = 20260428,
    [int]$Pop = 64,
    [int]$Gen = 80,
    [int]$SeqLen = 300,
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "== Fair MO Baseline Batch Run =="
Write-Host "Instance=$Instance Runs=$Runs BaseSeed=$BaseSeed Pop=$Pop Gen=$Gen SeqLen=$SeqLen"
Write-Host "Python=$PythonExe"

$algorithms = @("nsga2", "moead", "spea2")

foreach ($algo in $algorithms) {
    $algoUpper = $algo.ToUpper()
    Write-Host ""
    Write-Host ">>> Running $algoUpper ..."

    $env:ELP_MO_BASELINE_ALGO = $algo
    $env:ELP_IS_EXP = "true"
    $env:ELP_EXP_NUMBER = "$Runs"
    $env:ELP_BASE_SEED = "$BaseSeed"
    $env:ELP_EXP_INSTANCE = $Instance
    $env:ELP_EXP_ALGORITHM = "MO_BASELINE_$algoUpper"
    $env:ELP_EXP_REMARK = "FairBudget-Pop${Pop}-Gen${Gen}-Seq${SeqLen}-Runs${Runs}-Seed${BaseSeed}"
    $env:ELP_MO_BASELINE_POP = "$Pop"
    $env:ELP_MO_BASELINE_GEN = "$Gen"
    $env:ELP_MO_BASELINE_SEQ_LEN = "$SeqLen"

    & $PythonExe ".\src\algorithms\ELP_DRL_MO.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $algoUpper with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All baseline runs completed."
Write-Host "Result files are under .\files\expresults"
