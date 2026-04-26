param(
    [string]$PythonBin = "python",
    [int]$TrainingLoops = 1,
    [int]$LookbackHours = 168,
    [int]$WebcamDays = 7,
    [switch]$SkipWebcam,
    [switch]$SkipLabeller
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Import-EnvFile {
    param([Parameter(Mandatory = $true)][string]$EnvFilePath)

    if (-not (Test-Path $EnvFilePath)) {
        return
    }

    Get-Content $EnvFilePath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")

        if ($key -and -not $env:$key) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Test-StartupConfiguration {
    if (-not $env:OPENWEATHERMAP_API_KEY) {
        Write-Warning "OPENWEATHERMAP_API_KEY is not set. PM2.5 and forecast enrichment will be unavailable."
    }

    if (-not $env:PHOTO_LAT -or -not $env:PHOTO_LON) {
        Write-Warning "PHOTO_LAT/PHOTO_LON not fully set. Default location values from module arguments will be used."
    }

    if (-not $SkipWebcam) {
        $template = $env:WEBCAM_ARCHIVE_URL_TEMPLATE
        if ($template -and -not $template.Contains("{timestamp}")) {
            Write-Warning "WEBCAM_ARCHIVE_URL_TEMPLATE is set but missing {timestamp} placeholder; scraper may fail."
        }
        if (-not $template) {
            Write-Warning "WEBCAM_ARCHIVE_URL_TEMPLATE is not set. Default template will be used and may return many 404 responses."
        }
    }
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [switch]$ContinueOnError
    )

    & $VenvPython @Args
    if ($LASTEXITCODE -ne 0) {
        if ($ContinueOnError) {
            Write-Warning "$StepName failed with exit code $LASTEXITCODE; continuing."
            return
        }
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[PhotometricAI] Creating virtual environment"
    & $PythonBin -m venv $VenvDir
}

Import-EnvFile -EnvFilePath (Join-Path $RootDir ".env")
Test-StartupConfiguration

Write-Host "[PhotometricAI] Installing dependencies"
Invoke-PythonStep -StepName "pip upgrade" -Args @("-m", "pip", "install", "--upgrade", "pip")
Invoke-PythonStep -StepName "dependency install" -Args @("-m", "pip", "install", "-r", "requirements.txt")

$dirs = @(
    "data/raw",
    "data/processed",
    "models/artifacts",
    "logs"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

Write-Host "[PhotometricAI] Collecting weather + solar ground truth"
Invoke-PythonStep -StepName "weather collector" -Args @("-m", "data.collector", "--lookback-hours", "$LookbackHours", "--storage", "parquet")

if (-not $SkipWebcam) {
    Write-Host "[PhotometricAI] Scraping webcam reference frames"
    $webcamArgs = @("-m", "data.webcam_scraper", "--days", "$WebcamDays")
    if ($env:WEBCAM_ARCHIVE_URL_TEMPLATE) {
        $webcamArgs += @("--archive-url-template", "$env:WEBCAM_ARCHIVE_URL_TEMPLATE")
    }
    Invoke-PythonStep -StepName "webcam scraper" -Args $webcamArgs -ContinueOnError
}

if (-not $SkipLabeller) {
    Write-Host "[PhotometricAI] Computing ALQS labels"
    Invoke-PythonStep -StepName "ALQS labeller" -Args @("-m", "data.labeller", "--input-dir", "data/raw/webcam", "--output-path", "data/processed/alqs_labels.parquet") -ContinueOnError
}

Write-Host "[PhotometricAI] Building ML features"
Invoke-PythonStep -StepName "feature engineering" -Args @("-m", "data.feature_engineer", "--weather-path", "data/raw/weather", "--label-path", "data/processed/alqs_labels.parquet", "--output-dir", "data/processed")

for ($i = 1; $i -le $TrainingLoops; $i++) {
    Write-Host "[PhotometricAI] Training loop $i/$TrainingLoops`: LSTM"
    Invoke-PythonStep -StepName "lstm training (loop $i)" -Args @("-m", "models.lstm_predictor", "--data-path", "data/processed/sequence_dataset.npz", "--artifact-dir", "models/artifacts")

    Write-Host "[PhotometricAI] Training loop $i/$TrainingLoops`: MLP recommender"
    Invoke-PythonStep -StepName "mlp training (loop $i)" -Args @("-m", "models.mlp_recommender", "--features-path", "data/processed/classifier_features.parquet", "--artifact-dir", "models/artifacts")
}

Write-Host "[PhotometricAI] Pipeline complete"
