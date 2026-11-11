#!/usr/bin/env python3
"""Runs the packaged plugin in rustgis-cli, on RustGIS's World Atlas sample.

    python scripts/smoke.py [--cli path/to/rustgis-cli] [--strict]

Run scripts/package.py first. RustGIS must load the plugin and list its commands. Each tool
runs with its defaults (layer parameters get a sample layer of the right kind, field
parameters one of its fields) and must succeed. Each command that needs no parameters runs too;
a failure there is reported but only fails the run with --strict, since a command may need a
selection or a layer the sample doesn't have. This is what the registry's CI runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A sample layer for each layer kind, and a field of it.
SAMPLE = {"polygonLayer": "Countries", "pointLayer": "Cities", "vectorLayer": "Countries", "layer": "Countries"}
FIELDS = {"field": "name", "numericField": "pop_est"}


def run(cli: str, folder: Path, cmds: list[str], plugins_dir: str) -> subprocess.CompletedProcess:
    """Runs `cmds` in rustgis-cli with only this plugin loaded, on the sample project.

    Args:
        cli: The rustgis-cli executable.
        folder: The plugin folder (dist/<id>).
        cmds: Commands to run, as `--cmd` takes them.
        plugins_dir: An empty plugins folder, so installed plugins stay out of it.

    Returns:
        The finished process.
    """
    args = [cli, "run", "--no-plugins", "--plugin", str(folder), "--sample"]
    for c in cmds:
        args += ["--cmd", c]
    env = dict(os.environ, RUSTGIS_PLUGINS_DIR=plugins_dir)
    return subprocess.run(args, capture_output=True, text=True, timeout=600, env=env)


def main() -> int:
    """Loads the plugin and runs its tools and commands.

    Returns:
        The exit status: 1 when anything failed.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default=os.environ.get("RUSTGIS_CLI") or shutil.which("rustgis-cli") or "rustgis-cli")
    ap.add_argument("--strict", action="store_true", help="fail when a command fails too")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    plugin_id = manifest["id"]
    folder = ROOT / "dist" / plugin_id
    if not (folder / "plugin.json").is_file():
        print("error: no dist/<id>/: run python scripts/package.py first", file=sys.stderr)
        return 1
    failures = 0
    with tempfile.TemporaryDirectory() as plugins_dir:
        listed = run(args.cli, folder, ["info.commands"], plugins_dir)
        if listed.returncode != 0:
            print(f"error: RustGIS couldn't load the plugin:\n{listed.stderr}")
            return 1
        ids = {c["id"] for c in json.loads(listed.stdout)}
        for c in manifest.get("commands", []):
            cid = f"plugin:{plugin_id}/{c['id']}"
            if cid not in ids:
                print(f"error: {cid} isn't listed")
                failures += 1
                continue
            params = c.get("params", "{}")
            if params not in ("{}", "") and ":" in params and "?" not in params:
                print(f"   command {c['id']}: skipped (it needs parameters)")
                continue
            r = run(args.cli, folder, [cid], plugins_dir)
            if r.returncode == 0:
                print(f"   command {c['id']}: ok")
            else:
                print(f"{'error' if args.strict else 'warning'}: command {c['id']} failed: {r.stderr.strip()}")
                failures += args.strict
        for t in manifest.get("tools", []):
            params, skip = {}, False
            for p in t.get("params", []):
                if p["kind"] in SAMPLE:
                    params[p["name"]] = SAMPLE[p["kind"]]
                elif p["kind"] in FIELDS:
                    params[p["name"]] = FIELDS[p["kind"]]
                elif p.get("required") and "default" not in p:
                    skip = True
            if skip:
                print(f"   tool {t['id']}: skipped (a required parameter has no default)")
                continue
            r = run(args.cli, folder, ["gp.run=" + json.dumps({"tool": f"plugin:{plugin_id}/{t['id']}", "params": params})], plugins_dir)
            if r.returncode == 0:
                print(f"   tool {t['id']}: ok")
            else:
                print(f"error: tool {t['id']} failed: {r.stderr.strip()}")
                failures += 1
    print(f"{failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
