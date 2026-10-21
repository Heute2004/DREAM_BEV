"""Verify the release's source/evidence bindings without loading data or weights."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path = ROOT) -> dict:
    from build_reproducibility_package import validate_paper_result_contract

    contract = validate_paper_result_contract(root)
    unavailable = [item["path"] for item in contract["sources"]
                   if item.get("distribution") == "on_request" and not (root / item["path"]).is_file()]
    binding = json.loads((root / "contracts/paper_binding.json").read_text(encoding="utf-8"))
    if binding["main_pdf_sha256"] != contract["paper_pdf_sha256"]:
        raise ValueError("manuscript identity differs between contracts")
    checks = 0
    for item in binding["implementation_bindings"]:
        path = root / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"implementation mismatch: {item['path']}")
        checks += 1
    for folder in ("src", "scripts", "external", "tests"):
        for path in (root / folder).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            checks += 1
    manifest = root / "RELEASE_MANIFEST.json"
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for relative, expected in records.items():
            if digest(root / relative) != expected:
                raise ValueError(f"release checksum mismatch: {relative}")
        checks += len(records)
    return {
        "status": "PASS", "checks": checks,
        "scope": "source syntax and frozen evidence hashes; not a retraining verification",
        "unavailable_on_request_assets": unavailable,
        "full_data_contract_verified": not unavailable,
        "main_pdf_sha256": binding["main_pdf_sha256"],
        "locked_test_data_reads": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
