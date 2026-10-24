from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from inter_agent_sanitization_proxy.classifier import (
    HTTPChatProvider,
    LLMClassifier,
)
from inter_agent_sanitization_proxy.models import (
    InspectionStatus,
    Message,
    ThreatCategory,
    Verdict,
)
from inter_agent_sanitization_proxy.proxy import inspect_message

DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_TIMEOUT_SECONDS = 20.0
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INLINE_COMMENT = re.compile(r"\s+#")


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    message: Message
    expected_category: ThreatCategory | None


@dataclass(frozen=True, slots=True)
class BenchmarkPrediction:
    example_id: str
    expected_category: ThreatCategory | None
    inspection_status: InspectionStatus
    verdict: Verdict
    predicted_category: ThreatCategory | None
    prompt_version: dict[str, str]


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        denominator = self.false_positives + self.true_negatives
        return self.false_positives / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "support": self.true_positives + self.false_negatives,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset: str
    provider: str
    model: str
    prompt_version: dict[str, str]
    generated_at: str
    total: int
    inspected: int
    uninspected: int
    overall: ConfusionMatrix
    per_category: dict[str, ConfusionMatrix]

    def to_dict(self) -> dict[str, object]:
        coverage = self.inspected / self.total if self.total else 0.0
        return {
            "dataset": self.dataset,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": dict(self.prompt_version),
            "generated_at": self.generated_at,
            "total": self.total,
            "inspected": self.inspected,
            "uninspected": self.uninspected,
            "inspection_coverage": round(coverage, 6),
            "overall": self.overall.as_dict(),
            "per_category": {
                category: matrix.as_dict()
                for category, matrix in self.per_category.items()
            },
        }


def _dataset_error(line_number: int, message: str) -> ValueError:
    return ValueError(f"dataset line {line_number}: {message}")


def load_dataset(path: str | Path | None = None) -> tuple[BenchmarkExample, ...]:
    dataset_path = Path(path) if path is not None else DATASET_PATH
    examples: list[BenchmarkExample] = []
    seen_ids: set[str] = set()
    required_keys = {"id", "sender", "recipient", "content"}
    allowed_keys = required_keys | {"metadata", "expected_category"}

    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue

        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise _dataset_error(line_number, "invalid JSON") from exc

        if not isinstance(data, dict):
            raise _dataset_error(line_number, "example must be a JSON object")

        missing = required_keys - set(data)
        if missing:
            raise _dataset_error(
                line_number,
                f"missing required keys: {', '.join(sorted(missing))}",
            )
        unknown = set(data) - allowed_keys
        if unknown:
            raise _dataset_error(
                line_number,
                f"unknown keys: {', '.join(sorted(unknown))}",
            )

        example_id = data["id"]
        if not isinstance(example_id, str) or not example_id:
            raise _dataset_error(line_number, "id must be a non-empty string")
        if example_id in seen_ids:
            raise _dataset_error(line_number, f"duplicate id {example_id!r}")
        seen_ids.add(example_id)

        raw_category = data.get("expected_category")
        if raw_category is None:
            expected_category = None
        elif isinstance(raw_category, str):
            try:
                expected_category = ThreatCategory(raw_category)
            except ValueError as exc:
                raise _dataset_error(
                    line_number,
                    f"unknown expected_category {raw_category!r}",
                ) from exc
        else:
            raise _dataset_error(
                line_number,
                "expected_category must be a string or null",
            )

        neutral_message_id = "benchmark-" + hashlib.sha256(
            example_id.encode("utf-8")
        ).hexdigest()[:16]

        try:
            message = Message(
                sender=data["sender"],
                recipient=data["recipient"],
                content=data["content"],
                metadata=data.get("metadata", {}),
                message_id=neutral_message_id,
            )
        except ValueError as exc:
            raise _dataset_error(line_number, str(exc)) from exc

        examples.append(
            BenchmarkExample(
                example_id=example_id,
                message=message,
                expected_category=expected_category,
            )
        )

    return tuple(examples)


def evaluate_predictions(
    predictions: Sequence[BenchmarkPrediction],
) -> tuple[ConfusionMatrix, dict[str, ConfusionMatrix]]:
    inspected = [
        prediction
        for prediction in predictions
        if prediction.inspection_status is InspectionStatus.INSPECTED
    ]

    overall = ConfusionMatrix(
        true_positives=sum(
            1
            for prediction in inspected
            if prediction.expected_category is not None
            and prediction.predicted_category is not None
        ),
        false_positives=sum(
            1
            for prediction in inspected
            if prediction.expected_category is None
            and prediction.predicted_category is not None
        ),
        true_negatives=sum(
            1
            for prediction in inspected
            if prediction.expected_category is None
            and prediction.predicted_category is None
        ),
        false_negatives=sum(
            1
            for prediction in inspected
            if prediction.expected_category is not None
            and prediction.predicted_category is None
        ),
    )

    per_category: dict[str, ConfusionMatrix] = {}
    for category in ThreatCategory:
        per_category[category.value] = ConfusionMatrix(
            true_positives=sum(
                1
                for prediction in inspected
                if prediction.expected_category is category
                and prediction.predicted_category is category
            ),
            false_positives=sum(
                1
                for prediction in inspected
                if prediction.expected_category is not category
                and prediction.predicted_category is category
            ),
            true_negatives=sum(
                1
                for prediction in inspected
                if prediction.expected_category is not category
                and prediction.predicted_category is not category
            ),
            false_negatives=sum(
                1
                for prediction in inspected
                if prediction.expected_category is category
                and prediction.predicted_category is not category
            ),
        )

    return overall, per_category


def run_benchmark(
    examples: Sequence[BenchmarkExample],
    classifier: LLMClassifier,
    *,
    provider_name: str,
    model_name: str,
    dataset_path: str | Path | None = None,
) -> BenchmarkReport:
    example_list = tuple(examples)
    predictions: list[BenchmarkPrediction] = []

    for example in example_list:
        result = inspect_message(example.message, classifier)
        finding = result.primary_finding
        predictions.append(
            BenchmarkPrediction(
                example_id=example.example_id,
                expected_category=example.expected_category,
                inspection_status=result.inspection_status,
                verdict=result.verdict,
                predicted_category=(
                    finding.category if finding is not None else None
                ),
                prompt_version=dict(result.prompt_version),
            )
        )

    overall, per_category = evaluate_predictions(predictions)
    inspected = sum(
        1
        for prediction in predictions
        if prediction.inspection_status is InspectionStatus.INSPECTED
    )
    prompt_version = next(
        (
            dict(prediction.prompt_version)
            for prediction in predictions
            if prediction.prompt_version
        ),
        {},
    )

    return BenchmarkReport(
        dataset=str(dataset_path) if dataset_path is not None else str(DATASET_PATH),
        provider=provider_name,
        model=model_name,
        prompt_version=prompt_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total=len(predictions),
        inspected=inspected,
        uninspected=len(predictions) - inspected,
        overall=overall,
        per_category=per_category,
    )


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if value.startswith('"'):
        try:
            decoded, end = json.JSONDecoder().raw_decode(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid quoted value") from exc
        if not isinstance(decoded, str):
            raise ValueError("invalid quoted value")
        remainder = value[end:].strip()
        if remainder and not remainder.startswith("#"):
            raise ValueError("unexpected text after quoted value")
        return decoded

    if value.startswith("'"):
        end = value.find("'", 1)
        if end == -1:
            raise ValueError("invalid quoted value")
        remainder = value[end + 1 :].strip()
        if remainder and not remainder.startswith("#"):
            raise ValueError("unexpected text after quoted value")
        return value[1:end]

    comment = _INLINE_COMMENT.search(value)
    if comment is not None:
        value = value[: comment.start()].rstrip()
    return value


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    env_path = Path(path) if path is not None else DEFAULT_ENV_PATH
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(
                f"{env_path}:{line_number}: expected KEY=VALUE"
            )

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(
                f"{env_path}:{line_number}: invalid environment variable name"
            )

        try:
            value = _parse_env_value(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{env_path}:{line_number}: {exc}"
            ) from exc

        os.environ.setdefault(key, value)
        loaded[key] = value

    return loaded


def build_parser() -> argparse.ArgumentParser:
    env_timeout = os.environ.get("MODEL_TIMEOUT_SECONDS")
    parser = argparse.ArgumentParser(
        description="Run the labeled sanitization benchmark against an external model."
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--endpoint", default=os.environ.get("MODEL_API_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(env_timeout) if env_timeout else DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.endpoint or not args.model:
        parser.error(
            "an endpoint and model are required via arguments or "
            "MODEL_API_URL and MODEL_NAME"
        )

    provider = HTTPChatProvider(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
    )
    classifier = LLMClassifier(provider, timeout_seconds=args.timeout)
    examples = load_dataset(args.dataset)
    report = run_benchmark(
        examples,
        classifier,
        provider_name=provider.provider_name,
        model_name=provider.model,
        dataset_path=args.dataset,
    )
    output = json.dumps(report.to_dict(), indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
