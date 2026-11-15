"""
ASTRAVIVE — Web Scraping & Supplementary Dataset Downloader
NASA Space Apps Challenge 2026

Downloads supplementary datasets:
1. NASA DONKI API (Solar Flares, CMEs, SEPs, Geomagnetic Storms)
2. UCI WESAD (Wearable Stress and Affect Detection)
3. PhysioNet CVES (Cerebral Vasoregulation and Tilt)
"""

import os
import sys
import json
import urllib.request
import urllib.error

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def fetch_nasa_donki(api_key="DEMO_KEY", start_date="2024-01-01", end_date="2024-12-31"):
    """Fetches solar flare and SEP event streams from NASA DONKI REST API."""
    print(f"[*] Fetching NASA DONKI Space Weather ({start_date} to {end_date})...")
    endpoints = {
        "FLR": f"https://api.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key={api_key}",
        "SEP": f"https://api.nasa.gov/DONKI/SEP?startDate={start_date}&endDate={end_date}&api_key={api_key}",
        "CME": f"https://api.nasa.gov/DONKI/CME?startDate={start_date}&endDate={end_date}&api_key={api_key}",
        "GST": f"https://api.nasa.gov/DONKI/GST?startDate={start_date}&endDate={end_date}&api_key={api_key}",
    }
    
    out_dir = os.path.join(OUTPUT_DIR, "04_Space_Environment_And_Radiation")
    os.makedirs(out_dir, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    for name, url in endpoints.items():
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                target_file = os.path.join(out_dir, f"DONKI_Live_{name}_{start_date}_{end_date}.json")
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"  [+] Saved {name}: {len(data)} records -> {target_file}")
        except Exception as e:
            print(f"  [-] Error fetching {name}: {e}")

def download_file(url, destination_path):
    """Downloads a file with streaming and progress reporting."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)
    print(f"[*] Downloading {url} -> {destination_path}")
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(destination_path, 'wb') as out_file:
            total_size = response.getheader('Content-Length')
            total_size = int(total_size) if total_size else None
            downloaded = 0
            chunk_size = 1024 * 1024 # 1MB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(f"    Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
            print(f"\n  [+] Successfully saved {destination_path}")
    except Exception as e:
        print(f"  [-] Failed to download: {e}")

def main():
    print("=" * 60)
    print("ASTRAVIVE Data Downloader & Web Scraper")
    print("=" * 60)
    
    # 1. Fetch live NASA DONKI telemetry
    fetch_nasa_donki()
    
    # 2. UCI WESAD (2.1 GB archive)
    wesad_url = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
    wesad_target = os.path.join(OUTPUT_DIR, "01_Autonomic_Stress_Wearables", "WESAD.zip")
    print(f"\nTo download WESAD (2.1 GB) manually or via script:")
    print(f"  Source URL: {wesad_url}")
    print(f"  Target: {wesad_target}")
    
    # 3. PhysioNet CVES
    cves_url = "https://physionet.org/static/published-projects/cves/cerebrovascular-and-cardiovascular-regulation-during-tilt-head-up-and-head-down-tilt-in-healthy-humans-1.0.0.zip"
    cves_target = os.path.join(OUTPUT_DIR, "01_Autonomic_Stress_Wearables", "CVES_Tilt_1.0.0.zip")
    print(f"\nTo download PhysioNet CVES (Requires PhysioNet sign-in):")
    print(f"  Source URL: {cves_url}")
    print(f"  Target: {cves_target}")

if __name__ == "__main__":
    main()
