<#
===============================================================================
Profiler Desktop v1 - Automated Installer
Developed by PRISM U1192 Laboratory, Université de Lille
Author: Yanis Zirem (PhD Candidate, 2025)
===============================================================================
#>

Write-Host " Starting Profiler Desktop installation..." -ForegroundColor Cyan
Start-Sleep -Seconds 1

# Step 1: Check Anaconda installation
Write-Host "`n Checking for Anaconda installation..."
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host " Anaconda not found. Please install it first:" -ForegroundColor Red
    Write-Host "👉 https://www.anaconda.com/download"
    exit 1
}
else {
    Write-Host "✅ Anaconda detected." -ForegroundColor Green
}

# Step 2: Create Conda environment
Write-Host "`n📦 Creating Conda environment 'profiler' (Python 3.8.20)..."
conda create -y -n profiler python=3.8.20

# Step 3: Activate environment
Write-Host "`n🔄 Activating environment..."
conda activate profiler
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Could not auto-activate environment from script. You may need to run:" -ForegroundColor Yellow
    Write-Host "   conda activate profiler"
}

# Step 4: Install dependencies
Write-Host "`n📦 Installing Python dependencies from requirements.txt..."
if (Test-Path ".\requirements.txt") {
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed successfully." -ForegroundColor Green
}
else {
    Write-Host "❌ requirements.txt not found in current directory." -ForegroundColor Red
    exit 1
}

# Step 5: Install ProteoWizard
Write-Host "`n Installing ProteoWizard (msconvert)..."

$pwizVersion = "3.0.24143"
$pwizUrl = "https://github.com/ProteoWizard/pwiz/releases/download/$pwizVersion/ProteoWizard-$pwizVersion-x86_64.msi"
$pwizInstaller = "$env:TEMP\ProteoWizard.msi"
$pwizPath = "C:\Program Files\ProteoWizard\ProteoWizard $pwizVersion"

if (Test-Path $pwizPath) {
    Write-Host "✅ ProteoWizard already installed." -ForegroundColor Green
}
else {
    Write-Host "⬇️  Downloading ProteoWizard installer..."
    Invoke-WebRequest -Uri $pwizUrl -OutFile $pwizInstaller
    Write-Host "📦 Installing ProteoWizard silently..."
    Start-Process msiexec.exe -Wait -ArgumentList "/i `"$pwizInstaller`" /quiet /norestart"
}

# Step 6: Add to PATH
if (Test-Path $pwizPath) {
    Write-Host "`n🔧 Adding ProteoWizard to PATH..."
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$pwizPath", [EnvironmentVariableTarget]::Machine)
    Write-Host " ProteoWizard added to system PATH." -ForegroundColor Green
}
else {
    Write-Host "⚠️  ProteoWizard path not found. Please check installation manually." -ForegroundColor Yellow
}

# Step 7: Final verification
Write-Host "`n🔎 Verifying msconvert installation..."
if (Get-Command msconvert -ErrorAction SilentlyContinue) {
    Write-Host " msconvert is accessible from PATH." -ForegroundColor Green
}
else {
    Write-Host "⚠️  msconvert not found in PATH. Please restart your terminal or add manually:" -ForegroundColor Yellow
    Write-Host "   $pwizPath"
}

# Step 8: Completion message
Write-Host "`n🎉 Installation complete!"
Write-Host "To start Profiler Desktop:"
Write-Host "1️⃣ Open Anaconda Prompt"
Write-Host "2️⃣ Run: conda activate profiler"
Write-Host "3️⃣ Navigate to the Profiler Desktop folder"
Write-Host "4️⃣ Launch with: python profiler_desktop.py"
Write-Host "`n Profiler Desktop is now ready to use!" -ForegroundColor Cyan
