from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate

from inter_agent_sanitization_proxy.classifier import (
    LLMClassifier,
    PermanentProviderError,
)
from inter_agent_sanitization_proxy.models import ThreatCategory


CLEAN_OUTPUT = json.dumps({"finding": None})


def _finding(category: ThreatCategory) -> str:
    return json.dumps(
        {
            "finding": {
                "category": category.value,
                "severity": "high",
                "reason": "benchmark fake provider",
            }
        }
    )


class OracleProvider:
    provider_name = "fake"
    model = "oracle-v1"

    def __init__(self, examples: tuple[evaluate.BenchmarkExample, ...]) -> None:
        self.labels = {
            example.message.message_id: example.expected_category
            for example in examples
        }

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> str:
        context = json.loads(user_prompt)
        expected = self.labels[context["message_id"]]

        if expected is None:
            return CLEAN_OUTPUT
        if expected is ThreatCategory.DATA_EXFILTRATION:
            if "data_exfiltration" in system_prompt:
                return _finding(expected)
            return CLEAN_OUTPUT
        if "data_exfiltration" in system_prompt:
            return CLEAN_OUTPUT
        return _finding(expected)


class AlwaysPromptInjectionProvider:
    provider_name = "fake"
    model = "always-prompt-injection"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> str:
        if "data_exfiltration" in system_prompt:
            return CLEAN_OUTPUT
        return _finding(ThreatCategory.PROMPT_INJECTION)


class FailingProvider:
    provider_name = "fake"
    model = "always-fail"

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> str:
        raise PermanentProviderError("provider unavailable")


def test_dataset_is_labeled_and_covers_both_content_shapes() -> None:
    examples = evaluate.load_dataset()

    assert len(examples) >= 12
    assert len({example.example_id for example in examples}) == len(examples)
    assert {example.expected_category for example in examples} == {
        None,
        ThreatCategory.PROMPT_INJECTION,
        ThreatCategory.DATA_EXFILTRATION,
    }
    assert any(isinstance(example.message.content, str) for example in examples)
    assert any(isinstance(example.message.content, dict) for example in examples)
    message_ids = {example.message.message_id for example in examples}
    assert len(message_ids) == len(examples)
    assert all(
        message_id is not None and message_id.startswith("benchmark-")
        for message_id in message_ids
    )
    assert all(
        example.message.message_id != example.example_id for example in examples
    )

    by_id = {example.example_id: example for example in examples}
    assert by_id["clean-004"].expected_category is None
    assert by_id["clean-007"].expected_category is None
    assert (
        by_id["pi-002"].expected_category is ThreatCategory.PROMPT_INJECTION
    )
    assert (
        by_id["de-002"].expected_category is ThreatCategory.DATA_EXFILTRATION
    )


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    example = {
        "id": "same-id",
        "sender": "a",
        "recipient": "b",
        "content": "hello",
        "expected_category": None,
    }
    path.write_text(
        "\n".join((json.dumps(example), json.dumps(example))),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate id"):
        evaluate.load_dataset(path)


def test_dataset_rejects_unknown_keys_and_labels(tmp_path: Path) -> None:
    unknown_key_path = tmp_path / "unknown-key.jsonl"
    unknown_key_path.write_text(
        json.dumps(
            {
                "id": "bad-key",
                "sender": "a",
                "recipient": "b",
                "content": "hello",
                "expected_category": None,
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate.load_dataset(unknown_key_path)

    unknown_label_path = tmp_path / "unknown-label.jsonl"
    unknown_label_path.write_text(
        json.dumps(
            {
                "id": "bad-label",
                "sender": "a",
                "recipient": "b",
                "content": "hello",
                "expected_category": "malware",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown expected_category"):
        evaluate.load_dataset(unknown_label_path)


def test_env_file_loader_parses_values_and_preserves_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            (
                "# local model configuration",
                'export MODEL_API_URL="https://example.test/v1/chat/completions"',
                "MODEL_NAME='test-model'",
                "MODEL_API_KEY=secret-value # ignored comment",
                "HASH_ONLY=abc#123",
                'QUOTED_HASH="abc#123"',
                "EMPTY=",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(os, "environ", {"MODEL_NAME": "existing"})

    loaded = evaluate.load_env_file(path)

    assert loaded == {
        "MODEL_API_URL": "https://example.test/v1/chat/completions",
        "MODEL_NAME": "test-model",
        "MODEL_API_KEY": "secret-value",
        "HASH_ONLY": "abc#123",
        "QUOTED_HASH": "abc#123",
        "EMPTY": "",
    }
    assert os.environ["MODEL_NAME"] == "existing"
    assert os.environ["MODEL_API_URL"] == "https://example.test/v1/chat/completions"
    assert os.environ["MODEL_API_KEY"] == "secret-value"
    assert os.environ["EMPTY"] == ""


def test_env_file_loader_ignores_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(os, "environ", {})

    assert evaluate.load_env_file(tmp_path / "missing.env") == {}


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ("NOT_A_PAIR\n", "expected KEY=VALUE"),
        ("1INVALID=value\n", "invalid environment variable name"),
        ('KEY="unterminated\n', "invalid quoted value"),
        ('KEY="value" trailing\n', "unexpected text after quoted value"),
    ),
)
def test_env_file_loader_rejects_invalid_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(os, "environ", {})

    with pytest.raises(ValueError, match=message):
        evaluate.load_env_file(path)


def test_benchmark_metrics_are_correct_for_oracle_provider() -> None:
    examples = evaluate.load_dataset()
    provider = OracleProvider(examples)
    classifier = LLMClassifier(provider)

    report = evaluate.run_benchmark(
        examples,
        classifier,
        provider_name=provider.provider_name,
        model_name=provider.model,
        dataset_path="synthetic-dataset",
    )

    assert report.dataset == "synthetic-dataset"
    assert report.provider == "fake"
    assert report.model == "oracle-v1"
    assert report.prompt_version == {
        "prompt_injection": "v1",
        "data_exfiltration": "v1",
    }
    assert report.total == len(examples)
    assert report.inspected == len(examples)
    assert report.uninspected == 0

    for matrix in (report.overall, *report.per_category.values()):
        assert matrix.precision == 1.0
        assert matrix.recall == 1.0
        assert matrix.false_positive_rate == 0.0
        assert matrix.false_negatives == 0
        assert matrix.false_positives == 0

    payload = report.to_dict()
    json.dumps(payload)
    assert payload["overall"]["precision"] == 1.0
    assert payload["inspection_coverage"] == 1.0
    assert set(payload["per_category"]) == {
        "prompt_injection",
        "data_exfiltration",
    }


def test_benchmark_metrics_capture_false_positives_and_category_errors() -> None:
    examples = evaluate.load_dataset()
    provider = AlwaysPromptInjectionProvider()
    classifier = LLMClassifier(provider)

    report = evaluate.run_benchmark(
        examples,
        classifier,
        provider_name=provider.provider_name,
        model_name=provider.model,
    )

    positives = sum(
        1 for example in examples if example.expected_category is not None
    )
    negatives = len(examples) - positives
    injection = sum(
        1
        for example in examples
        if example.expected_category is ThreatCategory.PROMPT_INJECTION
    )
    exfiltration = sum(
        1
        for example in examples
        if example.expected_category is ThreatCategory.DATA_EXFILTRATION
    )

    assert report.inspected == len(examples)
    assert report.uninspected == 0
    assert report.overall == evaluate.ConfusionMatrix(
        true_positives=positives,
        false_positives=negatives,
        true_negatives=0,
        false_negatives=0,
    )
    assert report.per_category["prompt_injection"] == evaluate.ConfusionMatrix(
        true_positives=injection,
        false_positives=exfiltration + negatives,
        true_negatives=0,
        false_negatives=0,
    )
    assert report.per_category["data_exfiltration"] == evaluate.ConfusionMatrix(
        true_positives=0,
        false_positives=0,
        true_negatives=injection + negatives,
        false_negatives=exfiltration,
    )
    assert report.overall.recall == 1.0
    assert report.overall.precision == pytest.approx(positives / len(examples))
    assert report.overall.false_positive_rate == 1.0


def test_uninspected_examples_are_reported_separately() -> None:
    examples = evaluate.load_dataset()
    provider = FailingProvider()
    classifier = LLMClassifier(provider)

    report = evaluate.run_benchmark(
        examples,
        classifier,
        provider_name=provider.provider_name,
        model_name=provider.model,
    )

    assert report.total == len(examples)
    assert report.inspected == 0
    assert report.uninspected == len(examples)
    assert report.overall == evaluate.ConfusionMatrix()
    assert all(matrix == evaluate.ConfusionMatrix() for matrix in report.per_category.values())
    assert report.overall.precision == 0.0
    assert report.overall.recall == 0.0
    assert report.overall.false_positive_rate == 0.0
    assert report.prompt_version == {
        "prompt_injection": "v1",
        "data_exfiltration": "v1",
    }
    assert report.to_dict()["inspection_coverage"] == 0.0
