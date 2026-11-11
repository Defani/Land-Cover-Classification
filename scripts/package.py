#!/usr/bin/env python3
"""Builds the plugin, checks it as the registry will, and packages it.

    python scripts/package.py                 # build, check, package into dist/
    python scripts/package.py --tag v0.1.0    # also check the tag matches plugin.json's version
    python scripts/package.py --no-build      # package the module cargo already built

It writes:

    dist/<id>/                    plugin.json and plugin.wasm: a plugin folder to load while you
                                  work (Plugin Manager > Settings > Add Plugin Folder...)
    dist/<id>-<version>.rgplugin  the package to attach to a release
    dist/<id>.json                the registry entry to submit (registry/<id>.json)

The checks are those of the registry's own scripts/build.py, downloaded from
opengeos/rustgis-plugins (--offline skips them).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
TARGET = ROOT / "target" / "wasm32-unknown-unknown" / "release"
BUILD_PY = "https://raw.githubusercontent.com/opengeos/rustgis-plugins/main/scripts/build.py"
# A fixed timestamp, so the same plugin.json and module make the same package (and SHA-256).
ZIP_TIME = (2024, 1, 1, 0, 0, 0)


def registry_checks():
    """The registry's scripts/build.py, as a module.

    Returns:
        The module, or None when it can't be downloaded.
    """
    path = ROOT / "target" / "registry-build.py"
    try:
        with urllib.request.urlopen(BUILD_PY, timeout=30) as r:  # noqa: S310 - a fixed https address
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.read())
    except OSError as e:
        if not path.is_file():
            print(f"warning: can't download the registry's checks ({e}); skipping them", file=sys.stderr)
            return None
    spec = importlib.util.spec_from_file_location("registry_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(manifest: bytes, wasm: bytes, entry: str) -> bytes:
    """A reproducible .rgplugin: plugin.json and the module at the root of a zip archive.

    Args:
        manifest: plugin.json's bytes.
        wasm: The module's bytes.
        entry: The module's file name in the package (plugin.json's `entry`).

    Returns:
        The package's bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in (("plugin.json", manifest), (entry, wasm)):
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return buf.getvalue()


def main() -> int:
    """Builds, checks and packages the plugin.

    Returns:
        The exit status: 1 when a check failed.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true", help="don't run cargo; use the module already built")
    ap.add_argument("--tag", help="a release tag (v1.2.3) that must match plugin.json's version")
    ap.add_argument("--url", help="where the package will be downloaded from (default: the GitHub release's asset)")
    ap.add_argument("--offline", action="store_true", help="skip the registry's checks")
    args = ap.parse_args()

    manifest_bytes = (ROOT / "plugin.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    plugin_id, version = manifest.get("id", ""), manifest.get("version", "")
    problems: list[str] = []
    if args.tag and args.tag.removeprefix("v") != version:
        problems.append(f"the tag {args.tag} doesn't match plugin.json's version {version} (tag v{version})")
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    if cargo["package"]["version"] != version:
        print(f"warning: Cargo.toml's version {cargo['package']['version']} isn't plugin.json's {version}", file=sys.stderr)

    if not args.no_build:
        subprocess.run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"], cwd=ROOT, check=True)
    module = TARGET / (cargo["package"]["name"].replace("-", "_") + ".wasm")
    if not module.is_file():
        print(f"error: no module at {module}; build it with cargo build --release --target wasm32-unknown-unknown", file=sys.stderr)
        return 1
    wasm = module.read_bytes()
    entry = manifest.get("entry", "plugin.wasm")
    data = package(manifest_bytes, wasm, entry)

    checks = None if args.offline else registry_checks()
    if checks:
        where = f"{plugin_id} {version}"
        read = checks.read_package(data, where)
        if read:
            checks.check_manifest(read[0], where)
            for name in ("check_tab", "check_panel"):
                if hasattr(checks, name):
                    getattr(checks, name)(read[0], where)
            checks.check_module(read[1], where)
        problems += checks.problems
        checks.problems.clear()
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1

    DIST.mkdir(exist_ok=True)
    folder = DIST / plugin_id
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    (folder / "plugin.json").write_bytes(manifest_bytes)
    (folder / entry).write_bytes(wasm)
    name = f"{plugin_id}-{version}.rgplugin"
    (DIST / name).write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo and re.match(r"^https://github\.com/[^/]+/[^/]+$", manifest.get("homepage", "")):
        repo = manifest["homepage"].removeprefix("https://github.com/")
    url = args.url or (f"https://github.com/{repo}/releases/download/v{version}/{name}" if repo else f"https://<where you publish it>/{name}")
    registry_entry = {"id": plugin_id, "source": {"url": url, "sha256": sha}, "categories": ["Utilities"]}
    (DIST / f"{plugin_id}.json").write_text(json.dumps(registry_entry, indent=2) + "\n", encoding="utf-8")

    print(f"dist/{name}  {len(data):,} bytes  sha256 {sha}")
    print(f"dist/{plugin_id}/  the plugin folder, to load while you work")
    print(f"dist/{plugin_id}.json  the registry entry (set its categories before you submit it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
