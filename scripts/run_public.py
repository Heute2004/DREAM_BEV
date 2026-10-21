"""Run a packaged public-data entry point with explicit import roots."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {'cruw': 'CRUW_PedestrianFusion', 'carrada': 'CARRADA_DREAM_BEV'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=STUDIES)
    parser.add_argument('entrypoint')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    study = ROOT / 'external' / STUDIES[args.study]
    if Path(args.entrypoint).name != args.entrypoint:
        parser.error('entrypoint must be a packaged script name')
    script = study / 'scripts' / (args.entrypoint.removesuffix('.py') + '.py')
    if not script.is_file():
        parser.error(f'entrypoint is not packaged: {script.name}')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(study), str(study / 'scripts'), env.get('PYTHONPATH', '')])
    raise SystemExit(subprocess.call([sys.executable, str(script), *args.args], env=env))


if __name__ == '__main__':
    main()
