"""Fetch pinned public fusion or policy sources, never model weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess

SOURCES = {
    'rcrosnet': ('https://github.com/Zhuanglong2/RC-ROSNet.git', '8ff57d49efa90934dca2032adf7f6625b5ad9248'),
    'crn': ('https://github.com/youngskkim/CRN.git', '5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d'),
    'rcbevdet': ('https://github.com/VDIGPKU/RCBEVDet.git', '15c83ccdd5a8cfd3b7c0390eacd8664cf842d513'),
    'internvl': ('https://github.com/OpenGVLab/InternVL.git', '2410d1dbf208f0e799459aff9376e5747dbf41a2'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=SOURCES)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error('destination exists; existing checkouts will not change')
    url, commit = SOURCES[args.name]
    subprocess.run(['git', 'clone', '--no-checkout', url, str(args.destination)], check=True)
    subprocess.run(['git', '-C', str(args.destination), 'checkout', '--detach', commit], check=True)
    actual = subprocess.check_output(['git', '-C', str(args.destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise RuntimeError('source commit mismatch')
    print(f'{args.name}: {actual}')


if __name__ == '__main__':
    main()
