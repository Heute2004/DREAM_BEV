"""Project a frozen segment-admission trace onto a forced-route evidence audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICY_FIELDS = (
    "sequence",
    "frame_id",
    "policy_admitted",
    "radar_x_m",
    "radar_y_m",
    "candidate_x_m",
    "target_x_m",
    "target_y_m",
    "score",
    "signed_innovation",
)
EVIDENCE_FIELDS = (
    "sequence",
    "frame_id",
    "target_x_m",
    "target_y_m",
    "radar_x_m",
    "radar_y_m",
    "candidate_x_m",
    "candidate_score",
    "calibrated_visual_x_m",
    "cross_modal_innovation",
)


def _read_rows(path: Path, required: tuple[str, ...]) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")

    indexed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row["sequence"]), int(row["frame_id"]))
        if key in indexed:
            raise ValueError(f"{path} contains duplicate frame {key}")
        indexed[key] = row
    return indexed


def _require_close(
    policy: dict[str, str],
    evidence: dict[str, str],
    policy_field: str,
    evidence_field: str,
) -> None:
    if abs(float(policy[policy_field]) - float(evidence[evidence_field])) > 1e-6:
        raise ValueError(
            f"policy/evidence mismatch for {policy_field}/{evidence_field}"
        )


def project_policy_trace(
    policy_causal_replay: Path,
    evidence_audit: Path,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    policy_rows = _read_rows(policy_causal_replay, POLICY_FIELDS)
    evidence_rows = _read_rows(evidence_audit, EVIDENCE_FIELDS)
    if set(policy_rows) != set(evidence_rows):
        raise ValueError("policy trace and evidence audit must cover identical frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(POLICY_FIELDS))
        writer.writeheader()
        for key in sorted(policy_rows):
            policy = policy_rows[key]
            evidence = evidence_rows[key]
            _require_close(policy, evidence, "target_x_m", "target_x_m")
            _require_close(policy, evidence, "target_y_m", "target_y_m")
            # A forced expert route can legitimately alter the candidate score.
            # The projected trace therefore retains segment admission from the
            # frozen policy but evaluates AP with the forced-route score.
            innovation = abs(float(evidence["cross_modal_innovation"]))
            if float(evidence["calibrated_visual_x_m"]) < float(evidence["radar_x_m"]):
                innovation = -innovation
            writer.writerow(
                {
                    "sequence": key[0],
                    "frame_id": key[1],
                    "policy_admitted": policy["policy_admitted"],
                    "radar_x_m": evidence["radar_x_m"],
                    "radar_y_m": evidence["radar_y_m"],
                    "candidate_x_m": evidence["candidate_x_m"],
                    "target_x_m": evidence["target_x_m"],
                    "target_y_m": evidence["target_y_m"],
                    "score": evidence["candidate_score"],
                    "signed_innovation": innovation,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-causal-replay", type=Path, required=True)
    parser.add_argument("--evidence-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_policy_trace(
        args.policy_causal_replay,
        args.evidence_audit,
        args.output,
    )


if __name__ == "__main__":
    main()
