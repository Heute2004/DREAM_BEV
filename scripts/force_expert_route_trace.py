from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_counterfactual_module():
    module_path = PROJECT_ROOT / "src" / "rvfusion" / "route_counterfactual.py"
    if not module_path.exists():
        module_path = PROJECT_ROOT / "rvfusion" / "route_counterfactual.py"
    spec = importlib.util.spec_from_file_location("route_counterfactual", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load route counterfactual module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_counterfactual = _load_counterfactual_module()
SUPPORTED_EXPERT_ROUTES = _counterfactual.SUPPORTED_EXPERT_ROUTES
force_policy_trace = _counterfactual.force_policy_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a materialized route-only policy counterfactual trace."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=SUPPORTED_EXPERT_ROUTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_policy_trace(args.input, args.output, args.route)


if __name__ == "__main__":
    main()
