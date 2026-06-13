# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


def collect_tree(source, target):
    source_path = Path(source)
    return [
        (str(path), str(Path(target) / path.relative_to(source_path).parent))
        for path in source_path.rglob('*')
        if path.is_file()
    ]


datas = []
datas += collect_tree('eve_sim/res', 'eve_sim/res')
datas += collect_tree('eve_sim/translations', 'eve_sim/translations')
datas += collect_tree('eve_sim/scenario/library', 'eve_sim/scenario/library')
datas += [('eve_sim/version_manifest.json', 'eve_sim')]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['logbook', 'sqlalchemy', 'bs4', 'yaml', 'cryptography', 'cryptography.fernet', 'sqlalchemy.ext.associationproxy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    upx=True,
    upx_exclude=[],
    name='main',
)
