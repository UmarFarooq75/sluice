#!/usr/bin/env python3
"""HEAD-check every model URL and stamp the result into packaging/models.json.

Born from a real defect: cli/sluice's olmoe-7b entry carried a stale lowercase
filename that 404'd. Nobody noticed because nothing ever checked — the URL only
gets exercised when a user pulls that model, and by then it is their problem.

Downloads nothing: HEAD only, following redirects (HF resolve/ URLs 302 to a CDN).
Records status, the reported size, and the check date, so a stale entry is visible
in the manifest itself rather than discovered by a user mid-pull.

  python3 scripts/check_urls.py            # check + report, exit 1 on any failure
  python3 scripts/check_urls.py --write    # also stamp results into models.json
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging" / "models.json"
CLI = ROOT / "cli" / "sluice"


def head(url, timeout=30):
    """Return (status:int|None, bytes:int|None, note:str). HEAD, redirects followed."""
    try:
        r = subprocess.run(
            ["curl", "-sIL", "--max-time", str(timeout), "-o", "-", "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    if r.returncode != 0:
        return None, None, f"curl rc={r.returncode}"
    body = r.stdout
    code = body[-3:] if body[-3:].isdigit() else None
    # last Content-Length wins: the CDN hop after the redirect is the real one
    lens = re.findall(r"(?im)^content-length:\s*(\d+)", body)
    return (int(code) if code else None), (int(lens[-1]) if lens else None), ""


def main():
    write = "--write" in sys.argv
    mf = json.loads(MANIFEST.read_text())
    cli_urls = set(re.findall(r"https://huggingface\.co/\S+?\.gguf", CLI.read_text()))
    today = time.strftime("%Y-%m-%d")

    failures = 0
    checked = 0
    for m in mf["models"]:
        url = m.get("url")
        name = m["name"]
        if not url:
            print(f"  {name:20} SKIP  no url pinned in this manifest")
            continue
        code, size, note = head(url)
        checked += 1
        ok = code == 200
        if not ok:
            failures += 1
        size_s = f"{size/1e9:.2f} GB" if size else "size unreported"
        print(f"  {name:20} {code or note:<8} {size_s}")

        # a size that disagrees with the manifest is a different file, not a typo
        if ok and size and m.get("bytes"):
            drift = abs(size - m["bytes"]) / m["bytes"]
            if drift > 0.02:
                print(f"  {'':20} WARN  manifest says {m['bytes']/1e9:.2f} GB, "
                      f"server says {size/1e9:.2f} GB ({drift*100:.0f}% off)")
        if write:
            m["url_checked"] = {"date": today, "status": code or note,
                                "content_length": size}
            if ok and size:
                m["bytes"] = size  # the server is the authority on its own file

        # the manifest and the CLI must not drift apart; that drift IS the olmoe bug
        if url not in cli_urls:
            print(f"  {'':20} NOTE  this url is not the one in cli/sluice REGISTRY")

    for u in sorted(cli_urls - {m.get('url') for m in mf["models"]}):
        print(f"  {'(cli only)':20} not in manifest: {u}")
        failures += 1

    if write:
        mf["urls_last_checked"] = today
        MANIFEST.write_text(json.dumps(mf, indent=2, ensure_ascii=False) + "\n")
        print(f"\nstamped {MANIFEST.relative_to(ROOT)} ({today})")

    print(f"\n{checked} checked, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
