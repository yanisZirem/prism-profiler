<#
===============================================================================
Profiler Desktop v1.2 - Environment Setup Script
Developed by PRISM U1192 Laboratory, Universite de Lille
Author: Yanis Zirem (PhD Candidate, 2025)
===============================================================================
Steps:
  1 - Verify Conda is installed
  2 - Accept Conda Terms of Service (conda >= 24.x only)
  3 - Create 'profiler' Conda environment (Python 3.8.20)
  4 - Install Python dependencies from requirements.txt
  5 - Install ProteoWizard / msconvert (optional - for RAW file conversion only)
  6 - Create desktop shortcut (skipped if already exists)
===============================================================================
#>

Write-Host "Starting Profiler Desktop installation..." -ForegroundColor Cyan
Start-Sleep -Seconds 1

# Resolve script root reliably (works via .bat or direct PS call)
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# ------------------------------------------------------------------------------
# Step 1: Verify Conda
# ------------------------------------------------------------------------------
Write-Host "`nStep 1/5 - Checking for Anaconda / Miniconda..." -ForegroundColor White

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "? Conda not found. Please install Anaconda or Miniconda first:" -ForegroundColor Red
    Write-Host "  https://www.anaconda.com/download" -ForegroundColor Cyan
    exit 1
}
Write-Host "? Conda detected." -ForegroundColor Green

# ------------------------------------------------------------------------------
# Step 2: Accept Conda Terms of Service
# 'conda tos' only exists in conda >= 24.x - silently skip on older versions
# ------------------------------------------------------------------------------
Write-Host "`nStep 2/5 - Checking Conda Terms of Service..." -ForegroundColor White

try {
    $condaTosHelp = & conda tos accept --help 2>&1
    $tosSupported = ($condaTosHelp -notmatch "No command|is not a conda command|unknown command")
} catch {
    $tosSupported = $false
}

if ($tosSupported) {
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  2>$null | Out-Null
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     2>$null | Out-Null
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null | Out-Null
    Write-Host "? Conda Terms of Service accepted." -ForegroundColor Green
} else {
    Write-Host "  (conda tos not required on this conda version - skipping)" -ForegroundColor DarkGray
}

# ------------------------------------------------------------------------------
# Step 3: Create Conda environment (Python 3.8.20)
# ------------------------------------------------------------------------------
Write-Host "`nStep 3/5 - Creating Conda environment 'profiler' (Python 3.8.20)..." -ForegroundColor White

$envExists = conda env list | Select-String "^profiler\s"
if ($envExists) {
    Write-Host "? Conda environment 'profiler' already exists." -ForegroundColor Yellow
} else {
    conda create -y -n profiler python=3.8.20
    $envExists = conda env list | Select-String "^profiler\s"
    if (-not $envExists) {
        Write-Host "? Failed to create Conda environment 'profiler'." -ForegroundColor Red
        exit 1
    }
    Write-Host "? Conda environment 'profiler' created successfully." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# Step 4: Install Python dependencies
# protobuf is set to >=3.20,<6 to satisfy streamlit 1.39.0
# ------------------------------------------------------------------------------
Write-Host "`nStep 4/5 - Installing Python dependencies..." -ForegroundColor White

$reqFile = Join-Path $ScriptRoot "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Host "? requirements.txt not found at: $reqFile" -ForegroundColor Red
    exit 1
}

# Locate pip directly inside the conda environment (avoids conda run issues
# when conda shell integration is not initialised in the current session)
$condaBase = & conda info --base 2>$null
$condaBase = $condaBase.Trim()
$pipExe    = "$condaBase\envs\profiler\Scripts\pip.exe"
$pythonExe = "$condaBase\envs\profiler\python.exe"

if (-not (Test-Path $pipExe)) {
    Write-Host "? pip not found at: $pipExe" -ForegroundColor Red
    Write-Host "  Trying conda run fallback..." -ForegroundColor Yellow
    # Fallback: conda run (works when conda is properly initialised)
    conda run -n profiler python -m pip install --upgrade pip --quiet
    conda run -n profiler pip install -r $reqFile --upgrade-strategy=only-if-needed
} else {
    Write-Host "  Using pip at: $pipExe" -ForegroundColor DarkGray
    # Upgrade pip
    & $pythonExe -m pip install --upgrade pip --quiet

    # ── Step 4a: Install all dependencies except tensorflow ──────────────────
    # tensorflow==2.10.1 requires protobuf<3.20 which conflicts with
    # streamlit==1.39.0 (needs protobuf>=3.20). We solve this by:
    #   1. Installing everything else first (with protobuf>=3.20)
    #   2. Installing tensorflow with --no-deps (ignores its protobuf constraint)
    Write-Host "  Installing main dependencies..." -ForegroundColor DarkGray
    & $pipExe install -r $reqFile --upgrade-strategy=only-if-needed

    # ── Step 4b: Install tensorflow without dependency resolution ─────────────
    # tensorflow==2.10.1 requires protobuf<3.20 which conflicts with
    # streamlit==1.39.0. We install with --no-deps to bypass that constraint.
    Write-Host "  Installing TensorFlow 2.10.1 (--no-deps to bypass protobuf conflict)..." -ForegroundColor DarkGray
    & $pipExe install tensorflow==2.10.1 --no-deps

    # ── Step 4c: Install Keras 2.10.1 explicitly ─────────────────────────────
    # tensorflow.keras requires keras==2.10.x to be installed separately when
    # tensorflow is installed with --no-deps. Without this, imports like
    # 'from tensorflow.keras.callbacks import Callback' will fail at runtime.
    Write-Host "  Installing Keras 2.10.1 (required for tensorflow.keras imports)..." -ForegroundColor DarkGray
    & $pipExe install keras==2.10.0 --no-deps

    # ── Step 4d: Install tensorflow's other deps manually ────────────────────
    Write-Host "  Installing TensorFlow required packages..." -ForegroundColor DarkGray
    & $pipExe install `
        "absl-py>=1.0.0" `
        "astunparse>=1.6.0" `
        "flatbuffers>=2.0" `
        "gast>=0.2.1,<=0.4.0" `
        "google-pasta>=0.1.1" `
        "h5py>=2.9.0" `
        "keras-preprocessing>=1.1.1" `
        "libclang>=13.0.0" `
        "opt-einsum>=2.3.2" `
        "tensorflow-estimator==2.10.0" `
        "termcolor>=1.1.0" `
        "wrapt>=1.11.0,<1.15" `
        --upgrade-strategy=only-if-needed
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "? Some dependencies may not have installed correctly." -ForegroundColor Yellow
    Write-Host "  You can retry manually:" -ForegroundColor Yellow
    Write-Host "    conda activate profiler" -ForegroundColor White
    Write-Host "    pip install -r requirements.txt" -ForegroundColor White
    Write-Host "    pip install tensorflow==2.10.1 --no-deps" -ForegroundColor White
} else {
    Write-Host "? Dependencies installed successfully." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# Step 5: ProteoWizard / msconvert (OPTIONAL - only needed for RAW conversion)
# ------------------------------------------------------------------------------
Write-Host "`nStep 5/5 - Checking ProteoWizard (msconvert)..." -ForegroundColor White
Write-Host "  INFO: ProteoWizard is OPTIONAL. It is only needed if you want to convert" -ForegroundColor DarkGray
Write-Host "        RAW mass-spectrometry files (Thermo .raw, Waters .raw, Bruker .d)." -ForegroundColor DarkGray
Write-Host "        All other Profiler features work without it." -ForegroundColor DarkGray

# Check if msconvert is already on PATH
$msconvertFound = Get-Command msconvert -ErrorAction SilentlyContinue
$pwizVersion    = "3.0.24143"
$pwizPath       = "$env:LOCALAPPDATA\ProteoWizard\ProteoWizard $pwizVersion"
$pwizExe        = Join-Path $pwizPath "msconvert.exe"

if ($msconvertFound) {
    Write-Host "? msconvert already available on PATH." -ForegroundColor Green
} elseif (Test-Path $pwizExe) {
    Write-Host "? ProteoWizard already installed locally." -ForegroundColor Green
    # Add to PATH if not already there
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$pwizPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$pwizPath", [EnvironmentVariableTarget]::User)
        Write-Host "? ProteoWizard added to user PATH (restart terminal to apply)." -ForegroundColor Green
    }
} else {
    # Try to download and install automatically
    Write-Host "  ProteoWizard not found. Attempting automatic download (~150 MB)..." -ForegroundColor White
    $pwizUrl       = "https://github.com/ProteoWizard/pwiz/releases/download/$pwizVersion/ProteoWizard-$pwizVersion-x86_64.msi"
    $pwizInstaller = "$env:TEMP\ProteoWizard.msi"

    $downloaded = $false
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $pwizUrl -OutFile $pwizInstaller -TimeoutSec 180 -ErrorAction Stop
        $downloaded = $true
    } catch {
        Write-Host "  Download failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    if ($downloaded -and (Test-Path $pwizInstaller)) {
        Write-Host "  Installing ProteoWizard (no admin required)..." -ForegroundColor White
        try {
            Start-Process msiexec.exe -Wait -ArgumentList @(
                "/i", "`"$pwizInstaller`"",
                "/quiet", "/norestart",
                "TARGETDIR=`"$pwizPath`""
            ) -ErrorAction Stop

            if (Test-Path $pwizExe) {
                # Add to user PATH
                $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
                if ($userPath -notlike "*$pwizPath*") {
                    [Environment]::SetEnvironmentVariable("Path", "$userPath;$pwizPath", [EnvironmentVariableTarget]::User)
                }
                Write-Host "? ProteoWizard installed and added to user PATH." -ForegroundColor Green
                Write-Host "  Restart your terminal to apply PATH changes." -ForegroundColor Yellow
            } else {
                throw "msconvert.exe not found after install"
            }
        } catch {
            Write-Host "? ProteoWizard installation failed: $($_.Exception.Message)" -ForegroundColor Yellow
            $downloaded = $false
        }
    }

    if (-not $downloaded -or -not (Test-Path $pwizExe)) {
        Write-Host ""
        Write-Host "  --------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "  ? ProteoWizard could not be installed automatically." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  To install it manually:" -ForegroundColor White
        Write-Host "    1. Go to: https://proteowizard.sourceforge.io/download.html" -ForegroundColor Cyan
        Write-Host "    2. Download the Windows 64-bit installer" -ForegroundColor White
        Write-Host "    3. Run the installer (no administrator rights required)" -ForegroundColor White
        Write-Host "    4. Make sure 'msconvert' is added to your PATH during install" -ForegroundColor White
        Write-Host ""
        Write-Host "  ? This only affects RAW file conversion (Thermo/Waters/Bruker)." -ForegroundColor Yellow
        Write-Host "    All other Profiler features (ML, stats, enrichment, etc.)" -ForegroundColor Yellow
        Write-Host "    work perfectly without ProteoWizard." -ForegroundColor Yellow
        Write-Host "  --------------------------------------------------------" -ForegroundColor Yellow
        Write-Host ""
    }
}

# ------------------------------------------------------------------------------
# Step 6: Desktop shortcut
# Skip if shortcut already exists to avoid overwriting user customisations
# Use profiler.ico from project root (fallback: no icon)
# ------------------------------------------------------------------------------
Write-Host "`nCreating desktop shortcut..." -ForegroundColor White

$DesktopPath   = [Environment]::GetFolderPath("Desktop")
$ShortcutPath  = "$DesktopPath\Profiler Desktop.lnk"
$IconPath      = "$ScriptRoot\app\assets\profiler_logo.png"

# .lnk only supports .ico — use profiler.ico at root if available
$IcoPath = "$ScriptRoot\profiler.ico"
if (-not (Test-Path $IcoPath)) { $IcoPath = "" }

if (Test-Path $ShortcutPath) {
    Write-Host "? Desktop shortcut already exists - skipping." -ForegroundColor Yellow
} else {
    try {
        $WshShell              = New-Object -ComObject WScript.Shell
        $Shortcut              = $WshShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath   = "$ScriptRoot\run_profiler.bat"
        $Shortcut.WorkingDirectory = $ScriptRoot
        $Shortcut.Description  = "Launch Profiler Desktop - Multi-Omics Analysis"
        if ($IcoPath -ne "") { $Shortcut.IconLocation = "$IcoPath,0" }
        $Shortcut.Save()
        Write-Host "? Desktop shortcut created." -ForegroundColor Green
    } catch {
        Write-Host "? Could not create shortcut: $_" -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ? Installation complete!" -ForegroundColor Cyan
Write-Host " - Launch Profiler: double-click 'Profiler Desktop' on your desktop" -ForegroundColor White
Write-Host "                    or run run_profiler.bat" -ForegroundColor White
Write-Host " - Documentation:   docs\README.md" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
