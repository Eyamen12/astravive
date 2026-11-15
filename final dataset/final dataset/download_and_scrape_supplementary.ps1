# ASTRAVIVE — Supplementary Dataset Downloader & Web Scraper (PowerShell)
# NASA Space Apps Challenge 2026

$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " ASTRAVIVE Supplementary Dataset Web Downloader" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Fetch live NASA DONKI Space Weather
$donkiDir = Join-Path $outputDir "04_Space_Environment_And_Radiation"
if (-not (Test-Path $donkiDir)) { New-Item -ItemType Directory -Path $donkiDir | Out-Null }

$endpoints = @{
    "FLR" = "https://api.nasa.gov/DONKI/FLR?startDate=2024-01-01&endDate=2024-12-31&api_key=DEMO_KEY"
    "SEP" = "https://api.nasa.gov/DONKI/SEP?startDate=2024-01-01&endDate=2024-12-31&api_key=DEMO_KEY"
    "CME" = "https://api.nasa.gov/DONKI/CME?startDate=2024-01-01&endDate=2024-12-31&api_key=DEMO_KEY"
    "GST" = "https://api.nasa.gov/DONKI/GST?startDate=2024-01-01&endDate=2024-12-31&api_key=DEMO_KEY"
}

foreach ($key in $endpoints.Keys) {
    $url = $endpoints[$key]
    $outPath = Join-Path $donkiDir "DONKI_Live_$key.json"
    Write-Host "[*] Fetching DONKI $key..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $outPath -TimeoutSec 15
        Write-Host "  [+] Saved: $outPath" -ForegroundColor Green
    } catch {
        Write-Host "  [-] Failed: $_" -ForegroundColor Yellow
    }
}

# 2. WESAD URL Information
Write-Host "`nTo download WESAD (2.1 GB archive):" -ForegroundColor Magenta
Write-Host "  URL: https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
Write-Host "  Run: curl.exe -L -o `"$outputDir\01_Autonomic_Stress_Wearables\WESAD.zip`" `"https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip`""

# 3. PhysioNet CVES URL Information
Write-Host "`nTo download PhysioNet CVES:" -ForegroundColor Magenta
Write-Host "  URL: https://physionet.org/content/cves/1.0.0/"
Write-Host "  (Sign in on PhysioNet and download the published zip)"
