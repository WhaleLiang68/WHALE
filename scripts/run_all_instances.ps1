<#
Batch run all instances with G=500, TMax=200
#>

$instances = @(
    'AB20-ar3', 'AB20-ar50', 'AB20-ar7', 'AEG20', 'AML4',
    'BA12-maoyan', 'BA12', 'BA14', 'BME15', 'D10',
    'D12', 'D6', 'D8', 'Du62', 'FO10',
    'FO11', 'FO7', 'FO8', 'FO9', 'LW11',
    'LW5', 'MB12', 'O10', 'O12', 'O7-maoyan',
    'O7', 'O8', 'O9-maoyan', 'O9', 'P12',
    'P15', 'P20', 'P30', 'P4', 'P6',
    'P62', 'S8', 'S8H', 'S9', 'S9H',
    'SC30', 'SC35-maoyan', 'SC35', 'TAM20', 'TAM30',
    'TL12', 'TL15', 'TL20', 'TL30', 'TL5',
    'TL6', 'TL7', 'TL8', 'VC10-maoyan', 'VC10'
)

$runScript = "run_standard46_a8.ps1"

foreach ($instance in $instances) {
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "Running instance: $instance" -ForegroundColor Cyan
    Write-Host "Parameters: G=500, TMax=200" -ForegroundColor Cyan

    $seeds = 1..10 | ForEach-Object { Get-Random -Minimum 1000 -Maximum 9999 }
    $seedStr = $seeds -join ","

    & .\$runScript -Instance $instance -FixedSeeds $seedStr -G 500 -TMax 200 -TInitial 5000.0 -KHist 20.0
}

Write-Host "=====================================" -ForegroundColor Green
Write-Host "All instances completed!" -ForegroundColor Green