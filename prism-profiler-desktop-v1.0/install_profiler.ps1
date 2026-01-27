<#
===============================================================================
Profiler Desktop v1 - Automated Installer
Developed by PRISM U1192 Laboratory, Université de Lille
Author: Yanis Zirem (PhD Candidate, 2025)
===============================================================================
#>

Write-Host "Starting Profiler Desktop installation..." -ForegroundColor Cyan
Start-Sleep -Seconds 1

# Step 1: Check Anaconda installation
Write-Host "`nChecking for Anaconda installation..."
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Anaconda not found. Please install it first:" -ForegroundColor Red
    Write-Host "?? https://www.anaconda.com/download"
    exit 1
}
else {
    Write-Host "? Anaconda detected." -ForegroundColor Green
}

# Step 2: Create Conda environment
Write-Host "`n?? Creating Conda environment 'profiler' (Python 3.8.20)..."
conda create -y -n profiler python=3.8.20

# Step 3: Activate environment using conda run for script compatibility
Write-Host "`n?? Installing Python dependencies into 'profiler' environment..."
if (Test-Path ".\requirements.txt") {
    conda run -n profiler pip install -r requirements.txt
    Write-Host "? Dependencies installed successfully." -ForegroundColor Green
}
else {
    Write-Host "? requirements.txt not found in current directory." -ForegroundColor Red
    exit 1
}

# Step 4: Install ProteoWizard
Write-Host "`nInstalling ProteoWizard (msconvert)..."
$pwizVersion = "3.0.24143"
$pwizUrl = "https://github.com/ProteoWizard/pwiz/releases/download/$pwizVersion/ProteoWizard-$pwizVersion-x86_64.msi"
$pwizInstaller = "$env:TEMP\ProteoWizard.msi"
$pwizPath = "$env:LOCALAPPDATA\ProteoWizard\ProteoWizard $pwizVersion"

if (Test-Path $pwizPath) {
    Write-Host "? ProteoWizard already installed." -ForegroundColor Green
}
else {
    Write-Host "?? Downloading ProteoWizard installer..."
    Invoke-WebRequest -Uri $pwizUrl -OutFile $pwizInstaller
    Write-Host "?? Installing ProteoWizard silently..."
    Start-Process msiexec.exe -Wait -ArgumentList "/i `"$pwizInstaller`" /quiet TARGETDIR=`"$pwizPath`" /norestart"

}

# Step 5: Add ProteoWizard to PATH
if (Test-Path $pwizPath) {
    Write-Host "`n?? Adding ProteoWizard to PATH..."
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$pwizPath", [EnvironmentVariableTarget]::User)
    Write-Host "? ProteoWizard added to system PATH." -ForegroundColor Green
}
else {
    Write-Host "?? ProteoWizard path not found. Please check installation manually." -ForegroundColor Yellow
}

# Step 6: Final verification
Write-Host "`n?? Verifying msconvert installation..."
if (Get-Command msconvert -ErrorAction SilentlyContinue) {
    Write-Host "? msconvert is accessible from PATH." -ForegroundColor Green
}
else {
    Write-Host "?? msconvert not found in PATH. Please restart your terminal or add manually:" -ForegroundColor Yellow
    Write-Host "   $pwizPath"
}

# Step 7: Completion message
Write-Host "`n?? Installation complete!"
Write-Host "To start Profiler Desktop:"
Write-Host "1?? Open Anaconda Prompt"
Write-Host "2?? Run: conda activate profiler"
Write-Host "3?? Navigate to the Profiler Desktop folder"
Write-Host "4?? Launch with: python profiler_desktop.py"
Write-Host "`nProfiler Desktop is now ready to use!" -ForegroundColor Cyan

Write-Host "Creating desktop shortcut..."

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\Profiler Desktop.lnk")

$Shortcut.TargetPath = "$PSScriptRoot\run_profiler.bat"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.IconLocation = "$PSScriptRoot\profiler.ico"
$Shortcut.Save()

Write-Host "Profiler Desktop shortcut created on Desktop."
