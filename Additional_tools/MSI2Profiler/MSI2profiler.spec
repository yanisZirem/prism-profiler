# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 10)  # Augmente la limite de récursion
from PyInstaller.utils.hooks import collect_data_files
# Collecte tous les fichiers de ttkthemes
ttkthemes_datas = collect_data_files('ttkthemes')
# Bloc de chiffrement si nécessaire (None pour la plupart des cas)
block_cipher = None

a = Analysis(
    ['MSI2profiler.py'],      # ton script principal
    pathex=[],                 # chemin supplémentaire si nécessaire
    binaries=[],
    datas=ttkthemes_datas,
    hiddenimports=[
        'plotly', 
        'plotly.graph_objs',
        'pandas',
        'numpy',
        'pyimzml',
	'ttkthemes',
        'tkinter'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MSI2Profiler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False   # mettre True si tu veux voir la console pour debug
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MSI2Profiler'
)
