<#
===============================================================================
Profiler Desktop v1 - Automated Installer
Developed by PRISM U1192 Laboratory, Université de Lille
Author: Yanis Zirem (PhD Candidate, 2025)
===============================================================================
#>

Write-Host "Starting Profiler Desktop installation..." -ForegroundColor Cyan
Start-Sleep -Seconds 1

# ----------------------------------------------------------------------
# Step 1: Check Conda installation
# ----------------------------------------------------------------------
Write-Host "`nChecking for Anaconda / Miniconda installation..."

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda not found. Please install Anaconda or Miniconda first:" -ForegroundColor Red
    Write-Host "https://www.anaconda.com/download"
    exit 1
}
Write-Host "✔ Conda detected." -ForegroundColor Green

# ----------------------------------------------------------------------
# Step 2: Accept Conda Terms of Service automatically (required since 2024)
# ----------------------------------------------------------------------
Write-Host "`nChecking Conda Terms of Service acceptance..."

Write-Host "Conda Terms of Service must be accepted to continue." -ForegroundColor Yellow
Write-Host "Automatically accepting Conda Terms of Service..." -ForegroundColor Yellow

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main | Out-Null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r | Out-Null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 | Out-Null

Write-Host "✔ Conda Terms of Service accepted." -ForegroundColor Green

# ----------------------------------------------------------------------
# Step 3: Create Conda environment
# ----------------------------------------------------------------------
Write-Host "`nCreating Conda environment 'profiler' (Python 3.8.20)..."

$envExists = conda env list | Select-String "^profiler\s"
if ($envExists) {
    Write-Host "✔ Conda environment 'profiler' already exists." -ForegroundColor Yellow
}
else {
    conda create -y -n profiler python=3.8.20
}

# Verify environment creation
$envExists = conda env list | Select-String "^profiler\s"
if (-not $envExists) {
    Write-Host "❌ Failed to create Conda environment 'profiler'." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------
# Step 4: Install Python dependencies
# ----------------------------------------------------------------------
Write-Host "`nInstalling Python dependencies into 'profiler' environment..."

if (Test-Path "$PSScriptRoot\requirements.txt") {
    conda run -n profiler pip install -r "$PSScriptRoot\requirements.txt"
    Write-Host "✔ Dependencies installed successfully." -ForegroundColor Green
}
else {
    Write-Host "❌ requirements.txt not found in script directory." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------
# Step 5: Install ProteoWizard (user-only)
# ----------------------------------------------------------------------
Write-Host "`nInstalling ProteoWizard (msconvert)..."

$pwizVersion   = "3.0.24143"
$pwizUrl       = "https://github.com/ProteoWizard/pwiz/releases/download/$pwizVersion/ProteoWizard-$pwizVersion-x86_64.msi"
$pwizInstaller = "$env:TEMP\ProteoWizard.msi"
$pwizPath      = "$env:LOCALAPPDATA\ProteoWizard\ProteoWizard $pwizVersion"

if (Test-Path $pwizPath) {
    Write-Host "✔ ProteoWizard already installed." -ForegroundColor Green
}
else {
    Write-Host "Downloading ProteoWizard installer..."
    Invoke-WebRequest -Uri $pwizUrl -OutFile $pwizInstaller

    Write-Host "Installing ProteoWizard (no admin required)..."
    Start-Process msiexec.exe -Wait -ArgumentList `
        "/i `"$pwizInstaller`" /quiet /norestart TARGETDIR=`"$pwizPath`""
}

# ----------------------------------------------------------------------
# Step 6: Add ProteoWizard to USER PATH
# ----------------------------------------------------------------------
if (Test-Path $pwizPath) {
    Write-Host "`nAdding ProteoWizard to USER PATH..."

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$pwizPath*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$userPath;$pwizPath",
            [EnvironmentVariableTarget]::User
        )
        Write-Host "✔ ProteoWizard added to user PATH." -ForegroundColor Green
        Write-Host "⚠ Restart your terminal to apply PATH changes." -ForegroundColor Yellow
    }
    else {
        Write-Host "✔ ProteoWizard already in PATH." -ForegroundColor Green
    }
}
else {
    Write-Host "⚠ ProteoWizard not detected. You can install it manually if needed." -ForegroundColor Yellow
}

# ----------------------------------------------------------------------
# Step 7: Desktop shortcut
# ----------------------------------------------------------------------
Write-Host "`nCreating desktop shortcut..."

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\Profiler Desktop.lnk")

$Shortcut.TargetPath       = "$PSScriptRoot\run_profiler.bat"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.IconLocation     = "$PSScriptRoot\profiler.ico"
$Shortcut.Save()

Write-Host "✔ Desktop shortcut created." -ForegroundColor Green

# ----------------------------------------------------------------------
# Completion
# ----------------------------------------------------------------------
Write-Host "`n✔ Installation complete!" -ForegroundColor Cyan
Write-Host "You can now launch Profiler Desktop using the desktop shortcut."
