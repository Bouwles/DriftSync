# -*- mode: python ; coding: utf-8 -*-
"""Portable Windows build. Hooks collect dependencies, DLLs and MathText fonts.
Never copy the working package as data: it can contain private session recordings.
"""
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["launch.py"], pathex=["."], binaries=[], datas=[],
    hiddenimports=collect_submodules("driftsync"),
    hookspath=[], hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "notebook", "pytest", "tensorboard", "tensorflow"],
    noarchive=False, optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="DriftSync",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=True, disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DriftSync")
