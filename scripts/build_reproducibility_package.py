"""Build the current-mainline DREAM-BEV reproducibility package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


TOP_LEVEL_FILES = (
    ".gitignore",
    ".gitattributes",
    "RIGHTS.md",
    "THIRD_PARTY.md",
    "RELEASE_MANIFEST.json",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)

TOP_LEVEL_DIRECTORIES = (
    "artifacts",
    "configs",
    "contracts",
    "docs",
    "evidence",
    "protocols",
    "scripts",
    "src",
    "tests",
    "third_party",
    "external",
    "paper",
)

EXCLUDED_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignore_generated(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED_NAMES or Path(name).suffix.lower() in EXCLUDED_SUFFIXES
    }


def validate_paper_result_contract(project: Path) -> dict[str, object]:
    contract_path = project / "contracts" / "paper_result_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for source in contract["sources"]:
        relative_path = Path(source["path"])
        artifact = project / relative_path
        if not artifact.is_file():
            if source.get("distribution") == "on_request":
                continue
            failures.append(f"missing: {relative_path.as_posix()}")
            continue
        actual = sha256(artifact)
        if actual != source["sha256"]:
            failures.append(
                f"hash mismatch: {relative_path.as_posix()} "
                f"expected={source['sha256']} actual={actual}"
            )
    if failures:
        raise RuntimeError("paper result contract failed:\n" + "\n".join(failures))
    return contract


def build(project: Path, output: Path, archive: Path) -> None:
    project = project.resolve()
    output = output.resolve()
    archive = archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output package or archive already exists")
    if output == project or project in output.parents:
        raise ValueError("output must be outside the source project")

    paper_contract = validate_paper_result_contract(project)

    release = json.loads((project / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    # Copy the sealed public allowlist, not ignored local data beside the code.
    for name in [*release["files"], "RELEASE_MANIFEST.json"]:
        source = (project / name).resolve()
        if not source.is_relative_to(project):
            raise ValueError(f"path outside project: {name}")
        if name in release["files"] and sha256(source) != release["files"][name]:
            raise ValueError(f"release checksum mismatch: {name}")
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    forbidden_components = {
        "archive",
        "backup",
        "history",
        "retired",
        "superseded",
    }
    bad_paths = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if any(part.lower() in forbidden_components for part in path.relative_to(output).parts)
    ]
    if bad_paths:
        raise RuntimeError(f"non-mainline paths entered package: {bad_paths[:10]}")

    payload_files = sorted(path for path in output.rglob("*") if path.is_file())
    payload_hashes = {
        path.relative_to(output).as_posix(): sha256(path) for path in payload_files
    }
    manifest = {
        "project": "DREAM-BEV Information Fusion",
        "scope": "current paper mainline only",
        "runtime_inputs": ["RA", "RA-MTI", "RGB"],
        "sequence_contract": {"train": 87, "development": 34, "locked_test": 8},
        "paper_pdf_sha256": paper_contract["paper_pdf_sha256"],
        "coordinate_contract": paper_contract["coordinate_contract"],
        "headline_locked_result": paper_contract["locked_result"],
        "mainline": [
            "DREAM-BEV radar anchor",
            "five mechanism-specific radar experts",
            "RGB ConditionPrompt policy",
            "causal RGB bearing evidence",
            "candidate-local Radar-H-forward-coordinate-preserving RVFusion",
            "Train-calibrated causal consistency guard",
        ],
        "scope_statement": "Only the current manuscript mainline is packaged.",
        "files": payload_hashes,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    all_files = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(output).as_posix()}"
        for path in all_files
        if path.name != "SHA256SUMS.txt"
    ]
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    shutil.make_archive(
        str(archive.with_suffix("")),
        "zip",
        root_dir=output.parent,
        base_dir=output.name,
    )


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument(
        "--output",
        type=Path,
        default=project.parent / "DREAM-BEV_Information_Fusion_Package",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=project.parent / "DREAM-BEV_Information_Fusion_Reproducibility.zip",
    )
    args = parser.parse_args()
    build(args.project, args.output, args.archive)
    print(args.output.resolve())
    print(args.archive.resolve())


if __name__ == "__main__":
    main()
