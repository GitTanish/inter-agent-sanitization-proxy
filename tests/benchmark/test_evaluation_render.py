from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_evaluation

DATASET = render_evaluation.DatasetCounts(
    total=16,
    benign=7,
    categories={
        "prompt_injection": 5,
        "data_exfiltration": 4,
    },
)


def _matrix(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
) -> dict[str, int | float]:
    positive_calls = true_positives + false_positives
    positives = true_positives + false_negatives
    negatives = false_positives + true_negatives
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "support": positives,
        "precision": true_positives / positive_calls if positive_calls else 0.0,
        "recall": true_positives / positives if positives else 0.0,
        "false_positive_rate": false_positives / negatives if negatives else 0.0,
    }


def _report(
    *,
    model: str,
    inspected: int,
    overall: dict[str, int | float],
    prompt_injection: dict[str, int | float],
    data_exfiltration: dict[str, int | float],
) -> dict[str, object]:
    return {
        "dataset": "dataset.jsonl",
        "generated_at": "2026-10-24T15:12:23+00:00",
        "inspected": inspected,
        "inspection_coverage": inspected / 16,
        "model": model,
        "overall": overall,
        "per_category": {
            "prompt_injection": prompt_injection,
            "data_exfiltration": data_exfiltration,
        },
        "prompt_version": {
            "prompt_injection": "v1",
            "data_exfiltration": "v1",
        },
        "provider": "openai-compatible-http",
        "total": 16,
        "uninspected": 16 - inspected,
    }


def _runs() -> tuple[render_evaluation.RunSummary, render_evaluation.RunSummary]:
    reports = [
        _report(
            model="nvidia/nemotron-3.5-bolt-e4b",
            inspected=16,
            overall=_matrix(9, 0, 7, 0),
            prompt_injection=_matrix(5, 0, 11, 0),
            data_exfiltration=_matrix(4, 0, 12, 0),
        ),
        _report(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            inspected=12,
            overall=_matrix(6, 0, 6, 0),
            prompt_injection=_matrix(4, 0, 8, 0),
            data_exfiltration=_matrix(2, 0, 10, 0),
        ),
    ]
    return render_evaluation.summarize_reports(reports, DATASET)


def test_lightning_uninspected_threats_and_end_to_end_recall() -> None:
    _, lightning = _runs()

    assert lightning.inspected == 12
    assert lightning.uninspected == 4
    assert lightning.uninspected_threats == 3
    assert lightning.uninspected_benign == 1
    assert lightning.inspected_recall == render_evaluation.Proportion(6, 6)
    assert lightning.end_to_end_recall == render_evaluation.Proportion(6, 9)
    assert lightning.category_recall("prompt_injection") == render_evaluation.Proportion(4, 5)
    assert lightning.category_recall("data_exfiltration") == render_evaluation.Proportion(2, 4)


def test_render_evaluation_keeps_compact_sections_only() -> None:
    runs = _runs()
    svg = render_evaluation.render_evaluation(runs, DATASET)
    root = ElementTree.fromstring(svg)

    assert root.get("role") == "img"
    assert "Overall results" in svg
    assert "Inspection coverage" in svg
    assert "Threats detected by category" in svg
    assert "9/9" in svg
    assert "6/9" in svg
    assert "12/16" in svg
    assert "4 uninspected (3 threats)" in svg
    assert "Message outcomes" not in svg
    assert "Metric comparison" not in svg
    assert "95% CI" not in svg
    assert "Wilson" not in svg
    assert "\ufffd" not in svg


def test_report_invariant_violation_is_rejected() -> None:
    report = _report(
        model="nvidia/nemotron-3.5-bolt-e4b",
        inspected=16,
        overall=_matrix(8, 0, 7, 0),
        prompt_injection=_matrix(5, 0, 11, 0),
        data_exfiltration=_matrix(4, 0, 12, 0),
    )

    with pytest.raises(ValueError, match="outcome counts"):
        render_evaluation.RunSummary.from_report(
            report,
            DATASET,
            render_evaluation.ROLES[0],
        )


def test_dataset_counts_match_labeled_dataset() -> None:
    dataset = render_evaluation.load_dataset_counts()

    assert dataset.total == 16
    assert dataset.threats == 9
    assert dataset.benign == 7
    assert dataset.categories == {
        "prompt_injection": 5,
        "data_exfiltration": 4,
    }
