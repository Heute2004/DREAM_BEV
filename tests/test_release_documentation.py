"""Check publication bindings and aggregate provenance, without private data."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def test_current_paper_bindings_and_table_exports():
    binding = read('contracts/paper_binding.json')
    tables = read('paper/table_manifest.json')
    assert binding['supplement_pages'] == 7
    assert binding['main_pages'] == 16
    assert binding['main_pdf_sha256'] == tables['main_pdf_sha256']
    assert binding['supplement_pdf_sha256'] == tables['supplement_pdf_sha256']
    assert binding['dream_expansion'] == 'Deep Radar Experts for Adaptive Multimodal BEV Localization'
    assert len(tables['tables']) == 21
    assert {p.name for p in (ROOT / 'paper/tables').glob('*.tex')} == set(tables['tables'])
    for name, expected in tables['tables'].items():
        assert hashlib.sha256((ROOT / 'paper/tables' / name).read_bytes()).hexdigest() == expected


def test_training_stop_counts_from_aggregate_curves():
    folder = ROOT / 'evidence/training_stopping'
    with (folder / 'epoch_metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (folder / 'stopping_summary.csv').open(encoding='utf-8', newline='') as handle:
        summary = list(csv.DictReader(handle))
    assert len(rows) == 82 and len(summary) == 12
    for run in summary:
        values = sorted((r for r in rows if r['model'] == run['model']), key=lambda r: int(r['epoch']))
        best, best_epoch, count = None, None, 0
        for i, record in enumerate(values):
            value = float(record['value'])
            improved = best is None or (value > best if run['direction'] == 'max' else value < best)
            if improved:
                best, best_epoch, count = value, int(record['epoch']), 0
            else:
                count += 1
            assert count == int(record['consecutive_non_improving'])
            if i < len(values) - 1:
                assert count < int(run['patience'])
        assert best_epoch == int(run['best_epoch'])
        assert len(values) == int(run['stopped_epoch'])
        assert count == int(run['patience']) == int(run['trailing_non_improving'])


def test_clean_replays_remain_distinct_and_bound():
    for name in ('operator_clean', 'stress_clean'):
        receipt = read(f'evidence/development_replays/{name}.json')
        assert receipt['frames'] == 9969
        assert receipt['replay_checkpoint']['loaded'] is False
        assert receipt['upstream_candidate_checkpoint']['sha256'] is None
        assert receipt['new_inference'] is False and receipt['locked_test_access'] is False
        for path, expected in receipt['implementation_bindings'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    operator = read('evidence/development_replays/operator_clean.json')
    report = read(operator['report']['path'])
    assert operator['metrics']['protected'] == report['arms']['local_posterior_guarded']
    stress = read('evidence/development_replays/stress_clean.json')
    for arm, source in stress['reports'].items():
        assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
        assert stress['metrics'][arm]['ap_0p3m'] == read(source['path'])['guarded_fusion']['ap_0p3m']
    assert operator['metrics']['protected']['ap_0p3m'] != stress['metrics']['protected']['ap_0p3m']
