from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_sources_do_not_reference_nonmainline_project_roots_or_stage_names() -> None:
    runtime_sources = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "README.md",
    )
    forbidden = (
        "code_radar_lockbox",
        "code_rvfusion",
        "code_visual",
        "routeblind",
        "strictv",
        "lockbox_v3",
        "dreame9",
        "dream_if",
        "legacy",
        "m_rodnet_lite",
        "wsl_mamba_env.md",
        "locked held-out eval serves validation and test",
    )
    text_files: list[Path] = []
    for source in runtime_sources:
        if source.is_file():
            text_files.append(source)
        else:
            text_files.extend(source.rglob("*.py"))
            text_files.extend(source.rglob("*.md"))
            text_files.extend(source.rglob("*.sh"))

    violations = [
        f"{path.relative_to(PROJECT_ROOT)}: {term}"
        for path in text_files
        for term in forbidden
        if term in path.read_text(encoding="utf-8").lower()
    ]
    assert not violations, "\n".join(violations)


def test_project_declares_an_installable_src_layout() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" in pyproject
    assert 'name = "dream-bev-information-fusion"' in pyproject
    assert "package-dir" in pyproject
    assert 'where = ["src"]' in pyproject


def test_route_expert_cli_exposes_the_dry_complex_specialist() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_experts.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--expert-family {local,recovery,dry_complex}" in result.stdout
    assert "outdoor_dry_complex" in result.stdout


def test_radar_cli_uses_the_final_dream_bev_name() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_radar.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "dream_bev" in result.stdout
    assert "m_rodnet_lite" not in result.stdout
    assert "--selection-metric {ap_0p3m,ap_0p5m}" in result.stdout
