# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all files for heavy packages
torch_datas,    torch_binaries,    torch_hiddens    = collect_all("torch")
numpy_datas,    numpy_binaries,    numpy_hiddens    = collect_all("numpy")
sklearn_datas,  sklearn_binaries,  sklearn_hiddens  = collect_all("sklearn")
matplotlib_datas, matplotlib_binaries, matplotlib_hiddens = collect_all("matplotlib")
pygame_datas,   pygame_binaries,   pygame_hiddens   = collect_all("pygame")

all_datas    = (torch_datas + numpy_datas + sklearn_datas +
                matplotlib_datas + pygame_datas +
                [("driftsync", "driftsync")])   # bundle the whole package
all_binaries = torch_binaries + numpy_binaries + sklearn_binaries + matplotlib_binaries + pygame_binaries
all_hiddens  = (torch_hiddens + numpy_hiddens + sklearn_hiddens +
                matplotlib_hiddens + pygame_hiddens +
                collect_submodules("driftsync") +
                ["scipy", "scipy.special", "PIL", "PIL.Image"])

a = Analysis(
    ["launch.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddens,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # use COLLECT for a folder-based dist
    name="DriftSync",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX breaks torch DLLs on Windows
    console=True,            # keep console for log output during training
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DriftSync",
)
