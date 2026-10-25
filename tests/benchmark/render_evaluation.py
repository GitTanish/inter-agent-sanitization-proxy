from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")
DEFAULT_REPORTS = (
    REPO_ROOT / "benchmark-report.json",
    REPO_ROOT / "benchmark-report-nemotron-3.5-lightning-30b-a3b.json",
)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "evaluation.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
CATEGORY_ORDER = ("prompt_injection", "data_exfiltration")
CATEGORY_LABELS = {
    "prompt_injection": "Prompt injection",
    "data_exfiltration": "Data exfiltration",
}
MODEL_LABELS = {
    "nvidia/nemotron-3.5-bolt-e4b": "Nemotron 3.5 Bolt e4b",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning 30b-a3b",
}
MODEL_COLORS = ("#0F172A", "#6D28D9")
ROLES = ("Primary configuration", "Alternate trial")

INK = "#0F172A"
MUTED = "#475569"
SUBTLE = "#64748B"
TRACK = "#EDF2F7"
BORDER = "#E2E8F0"
GRID = "#EEF2F7"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
TEAL = "#0F766E"
AMBER = "#B45309"


@dataclass(frozen=True)
class DatasetCounts:
    total: int
    benign: int
    categories: dict[str, int]

    @property
    def threats(self) -> int:
        return sum(self.categories.values())


@dataclass(frozen=True)
class Proportion:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("proportion denominator must be positive")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("proportion numerator must be between zero and denominator")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    def fraction(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def percent(self) -> str:
        return f"{self.value * 100:.1f}%"


@dataclass(frozen=True)
class RunSummary:
    role: str
    model: str
    display_name: str
    provider: str
    prompt_version: dict[str, str]
    generated_at: datetime
    total: int
    inspected: int
    uninspected: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    dataset: DatasetCounts
    per_category: dict[str, dict[str, int]]

    @classmethod
    def from_report(
        cls,
        report: dict[str, object],
        dataset: DatasetCounts,
        role: str,
    ) -> "RunSummary":
        overall = _mapping(report, "overall")
        model = _string(report, "model")
        provider = _string(report, "provider")
        generated_at = _datetime(report, "generated_at")
        prompt_version = _string_mapping(report, "prompt_version")
        total = _integer(report, "total")
        inspected = _integer(report, "inspected")
        uninspected = _integer(report, "uninspected")

        true_positives = _integer(overall, "true_positives")
        false_positives = _integer(overall, "false_positives")
        true_negatives = _integer(overall, "true_negatives")
        false_negatives = _integer(overall, "false_negatives")
        support = _integer(overall, "support")

        if total != dataset.total:
            raise ValueError(
                f"{model}: report total {total} does not match dataset total {dataset.total}"
            )
        if inspected + uninspected != total:
            raise ValueError(f"{model}: inspected and uninspected counts do not sum to total")
        if support != true_positives + false_negatives:
            raise ValueError(f"{model}: overall support does not match true positives plus false negatives")
        if true_positives + false_positives + true_negatives + false_negatives != inspected:
            raise ValueError(f"{model}: outcome counts do not sum to inspected count")

        per_category_raw = _mapping(report, "per_category")
        per_category: dict[str, dict[str, int]] = {}
        category_supports = 0
        for category in CATEGORY_ORDER:
            values = _mapping(per_category_raw, category)
            category_summary = {
                key: _integer(values, key)
                for key in (
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                    "support",
                )
            }
            if category_summary["support"] != (
                category_summary["true_positives"] + category_summary["false_negatives"]
            ):
                raise ValueError(f"{model}: {category} support does not match TP plus FN")
            if sum(
                category_summary[key]
                for key in (
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                )
            ) != inspected:
                raise ValueError(f"{model}: {category} outcomes do not sum to inspected count")
            if category_summary["support"] > dataset.categories[category]:
                raise ValueError(f"{model}: {category} support exceeds labeled threats")
            category_supports += category_summary["support"]
            per_category[category] = category_summary

        if category_supports != support:
            raise ValueError(f"{model}: category support does not match overall support")

        uninspected_threats = dataset.threats - support
        uninspected_benign = dataset.benign - (
            false_positives + true_negatives
        )
        if uninspected_threats < 0 or uninspected_benign < 0:
            raise ValueError(f"{model}: inspected counts exceed labeled totals")
        if uninspected_threats + uninspected_benign != uninspected:
            raise ValueError(f"{model}: uninspected class counts do not sum to uninspected total")

        display_name = MODEL_LABELS.get(
            model,
            model.split("/")[-1].replace("-", " ").replace("_", " "),
        )

        return cls(
            role=role,
            model=model,
            display_name=display_name,
            provider=provider,
            prompt_version=prompt_version,
            generated_at=generated_at,
            total=total,
            inspected=inspected,
            uninspected=uninspected,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            dataset=dataset,
            per_category=per_category,
        )

    @property
    def coverage(self) -> Proportion:
        return Proportion(self.inspected, self.total)

    @property
    def inspected_threats(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def inspected_benign(self) -> int:
        return self.false_positives + self.true_negatives

    @property
    def uninspected_threats(self) -> int:
        return self.dataset.threats - self.inspected_threats

    @property
    def uninspected_benign(self) -> int:
        return self.dataset.benign - self.inspected_benign

    @property
    def end_to_end_recall(self) -> Proportion:
        return Proportion(self.true_positives, self.dataset.threats)

    @property
    def inspected_recall(self) -> Proportion:
        return Proportion(self.true_positives, self.inspected_threats)

    @property
    def precision(self) -> Proportion:
        return Proportion(
            self.true_positives,
            self.true_positives + self.false_positives,
        )

    @property
    def false_positive_rate(self) -> Proportion:
        return Proportion(
            self.false_positives,
            self.false_positives + self.true_negatives,
        )

    def category_recall(self, category: str) -> Proportion:
        return Proportion(
            self.per_category[category]["true_positives"],
            self.dataset.categories[category],
        )

    def category_inspected(self, category: str) -> int:
        return self.per_category[category]["support"]


def load_dataset_counts(path: str | Path = DEFAULT_DATASET) -> DatasetCounts:
    dataset_path = Path(path)
    counts = {category: 0 for category in CATEGORY_ORDER}
    benign = 0

    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"dataset line {line_number}: expected a JSON object")
        category = value.get("expected_category")
        if category is None:
            benign += 1
        elif category in counts:
            counts[category] += 1
        else:
            raise ValueError(
                f"dataset line {line_number}: unknown category {category!r}"
            )

    return DatasetCounts(
        total=benign + sum(counts.values()),
        benign=benign,
        categories=counts,
    )


def summarize_reports(
    reports: Sequence[dict[str, object]],
    dataset: DatasetCounts,
) -> tuple[RunSummary, RunSummary]:
    if len(reports) != 2:
        raise ValueError("evaluation visual requires exactly two reports")
    if not all(isinstance(report, dict) for report in reports):
        raise ValueError("each report must be a JSON object")
    return (
        RunSummary.from_report(reports[0], dataset, ROLES[0]),
        RunSummary.from_report(reports[1], dataset, ROLES[1]),
    )


def render_evaluation(
    runs: tuple[RunSummary, RunSummary],
    dataset: DatasetCounts,
) -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" role="img" aria-labelledby="evaluation-title evaluation-desc">',
        '  <title id="evaluation-title">Threat detection comparison: coverage and end-to-end recall</title>',
        f'  <desc id="evaluation-desc">{escape(_description(runs, dataset))}</desc>',
        "  <defs>",
        '    <pattern id="evaluation-dot-grid" width="28" height="28" patternUnits="userSpaceOnUse">',
        '      <circle cx="2" cy="2" r="1.1" fill="#CBD5E1" opacity="0.4"/>',
        "    </pattern>",
        "  </defs>",
        f'  <rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="#F8FAFC"/>',
        f'  <rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="url(#evaluation-dot-grid)"/>',
        f'  <rect width="{CANVAS_WIDTH}" height="6" fill="{INK}"/>',
        '  <g font-family="Segoe UI, Inter, Arial, sans-serif">',
    ]

    parts.extend(_header(runs, dataset))
    parts.extend(_overall_section(runs))
    parts.extend(_coverage_section(runs))
    parts.extend(_category_section(runs, dataset))
    parts.extend(_footer(runs, dataset))
    parts.extend(["  </g>", "</svg>"])
    return "\n".join(parts) + "\n"


def _description(
    runs: tuple[RunSummary, RunSummary],
    dataset: DatasetCounts,
) -> str:
    primary, alternate = runs
    return (
        f"Comparison of two models on {dataset.total} labeled messages. "
        f"{primary.display_name} inspected {primary.inspected} of {primary.total} "
        f"and detected {primary.true_positives} of {dataset.threats} threats end-to-end. "
        f"{alternate.display_name} inspected {alternate.inspected} of {alternate.total} "
        f"and detected {alternate.true_positives} of {dataset.threats} threats; "
        f"{alternate.uninspected} messages were uninspected, including "
        f"{alternate.uninspected_threats} threats. "
        "Precision and false-positive rate are conditional on completed inspections."
    )


def _header(
    runs: tuple[RunSummary, RunSummary],
    dataset: DatasetCounts,
) -> list[str]:
    prompt_labels = sorted(
        {
            version
            for run in runs
            for version in run.prompt_version.values()
        }
    )
    prompt_label = "PROMPT " + " / ".join(prompt_labels)
    date_label = _date_label(runs)
    badges = [
        (f"{dataset.total} MESSAGES", 136),
        (prompt_label, 110),
        (date_label, 126),
    ]

    parts = [
        _text(60, 50, "INTER-AGENT SANITIZATION PROXY / BENCHMARK", 12, BLUE, 700, spacing=2.1),
        _text(60, 96, "Coverage and end-to-end threat recall", 40, INK, 700, spacing=-0.9),
        _text(
            60,
            126,
            f"Two saved model runs on {dataset.total} labeled messages. Uninspected messages count as undetected.",
            15,
            MUTED,
        ),
    ]

    x = 1540
    for label, width in reversed(badges):
        x -= width
        parts.append(_badge(x, 46, width, 31, label, WHITE, BORDER, MUTED))
        x -= 10

    parts.append(_line(60, 152, 1540, 152, BORDER))
    return parts


def _overall_section(
    runs: tuple[RunSummary, RunSummary],
) -> list[str]:
    x = 60
    y = 176
    width = 1480
    height = 252
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{WHITE}" stroke="{BORDER}"/>',
        _text(x + 28, y + 32, "Overall results", 19, INK, 700, spacing=-0.2),
        _line(x + 28, 284, x + width - 28, 284, GRID),
        _line(640, 228, 640, 414, GRID),
        _line(1080, 228, 1080, 414, GRID),
        _text(x + 28, 258, "METRIC", 8.6, SUBTLE, 700, spacing=1.0),
    ]

    for index, run in enumerate(runs):
        column_x = 680 + index * 440
        parts.append(
            _text(
                column_x,
                238,
                run.role.upper(),
                8.8,
                MODEL_COLORS[index],
                700,
                spacing=1.1,
            )
        )
        parts.append(
            _text(column_x, 258, run.display_name, 14.5, INK, 700, spacing=-0.2)
        )
        parts.append(
            _text(
                column_x,
                273,
                run.model,
                8.2,
                SUBTLE,
                family="Cascadia Code, Consolas, monospace",
            )
        )

    rows = (
        ("Inspection coverage", runs[0].coverage, runs[1].coverage, True, True),
        ("Threats detected, end-to-end", runs[0].end_to_end_recall, runs[1].end_to_end_recall, True, True),
        ("Precision on inspected messages", runs[0].precision, runs[1].precision, False, False),
        ("False-positive rate on inspected messages", runs[0].false_positive_rate, runs[1].false_positive_rate, False, False),
    )
    for row_index, (label, primary, alternate, emphasis, alert) in enumerate(rows):
        row_y = 306 + row_index * 28
        parts.append(_text(88, row_y, label, 10.8, MUTED, 600))
        parts.extend(
            _metric_value(
                680,
                row_y,
                primary,
                emphasis=emphasis,
                color=INK,
            )
        )
        parts.extend(
            _metric_value(
                1120,
                row_y,
                alternate,
                emphasis=emphasis,
                color=AMBER if alert else INK,
            )
        )

    uninspected_y = 306 + 4 * 28
    parts.append(_text(88, uninspected_y, "Uninspected threats", 10.8, MUTED, 600))
    parts.append(_text(680, uninspected_y, str(runs[0].uninspected_threats), 15, INK, 700))
    parts.append(
        _text(
            1120,
            uninspected_y,
            str(runs[1].uninspected_threats),
            15,
            AMBER,
            700,
        )
    )
    return parts


def _metric_value(
    x: float,
    y: float,
    proportion: Proportion,
    *,
    emphasis: bool,
    color: str,
) -> list[str]:
    count_size = 15.5 if emphasis else 12.8
    percent_size = 9.8 if emphasis else 9.2
    percent_x = x + (62 if emphasis else 54)
    return [
        _text(x, y, proportion.fraction(), count_size, color, 700, spacing=-0.2),
        _text(percent_x, y, proportion.percent(), percent_size, SUBTLE),
    ]


def _coverage_section(
    runs: tuple[RunSummary, RunSummary],
) -> list[str]:
    x = 60
    y = 452
    width = 1480
    height = 180
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{WHITE}" stroke="{BORDER}"/>',
        _text(x + 28, y + 32, "Inspection coverage", 19, INK, 700, spacing=-0.2),
        _text(
            x + 28,
            y + 53,
            "One cell per message. Amber cells were never inspected.",
            11.3,
            MUTED,
        ),
    ]

    legend_items = (
        (1160, TEAL, "Inspected threats"),
        (1300, BLUE, "Inspected benign"),
        (1440, AMBER, "Uninspected"),
    )
    for legend_x, color, label in legend_items:
        parts.append(
            f'<rect x="{legend_x}" y="{y + 23}" width="9" height="9" rx="2" fill="{color}"/>'
        )
        parts.append(_text(legend_x + 14, y + 32, label, 9.0, MUTED))

    for index, run in enumerate(runs):
        row_y = y + 88 + index * 52
        strip_y = row_y - 13
        parts.append(
            _text(x + 28, row_y, _short_name(run), 11.4, INK, 700)
        )
        parts.append(
            _text(
                x + 28,
                row_y + 15,
                f"{run.coverage.fraction()} inspected  \u00b7  {run.coverage.percent()}",
                8.8,
                SUBTLE,
            )
        )
        parts.extend(_coverage_strip(280, strip_y, run))
        if run.uninspected:
            status = f"{run.uninspected} uninspected ({run.uninspected_threats} threats)"
            status_color = AMBER
        else:
            status = "0 uninspected"
            status_color = TEAL
        parts.append(
            _text(1540, row_y, status, 9.6, status_color, 700, anchor="end")
        )

    return parts


def _coverage_strip(
    x: float,
    y: float,
    run: RunSummary,
) -> list[str]:
    cell_width = 62
    gap = 6
    fills = (
        [TEAL] * run.inspected_threats
        + [BLUE] * run.inspected_benign
        + [AMBER] * run.uninspected
    )
    if len(fills) != run.total:
        raise ValueError("coverage strip does not account for every message")

    parts: list[str] = []
    for cell_index, fill in enumerate(fills):
        cell_x = x + cell_index * (cell_width + gap)
        parts.append(
            f'<rect x="{cell_x}" y="{y}" width="{cell_width}" height="18" rx="3" fill="{fill}" stroke="{WHITE}"/>'
        )
    return parts


def _category_section(
    runs: tuple[RunSummary, RunSummary],
    dataset: DatasetCounts,
) -> list[str]:
    x = 60
    y = 656
    width = 1480
    height = 184
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{WHITE}" stroke="{BORDER}"/>',
        _text(x + 28, y + 32, "Threats detected by category", 19, INK, 700, spacing=-0.2),
        _text(
            x + 28,
            y + 53,
            "Share of labeled threats detected end-to-end.",
            11.3,
            MUTED,
        ),
    ]

    for category_index, category in enumerate(CATEGORY_ORDER):
        group_x = 88 if category_index == 0 else 820
        track_left = 200 if category_index == 0 else 932
        track_right = 660 if category_index == 0 else 1392
        value_x = 674 if category_index == 0 else 1406
        total = dataset.categories[category]

        parts.append(
            _text(group_x, y + 88, CATEGORY_LABELS[category], 13.2, INK, 700)
        )
        parts.append(
            _text(group_x, y + 104, f"{total} labeled threats", 8.8, SUBTLE)
        )

        for model_index, run in enumerate(runs):
            row_y = y + 138 + model_index * 30
            metric = run.category_recall(category)
            track_y = row_y - 11
            color = MODEL_COLORS[model_index]
            bar_width = (track_right - track_left) * metric.value

            parts.append(
                f'<circle cx="{group_x + 6}" cy="{row_y - 4}" r="4" fill="{color}"/>'
            )
            parts.append(
                _text(
                    group_x + 18,
                    row_y,
                    _short_name(run),
                    10.2,
                    MUTED,
                    600,
                )
            )
            parts.append(
                f'<rect x="{track_left}" y="{track_y}" width="{track_right - track_left}" height="12" rx="6" fill="{TRACK}"/>'
            )
            parts.append(
                f'<rect x="{track_left}" y="{track_y}" width="{bar_width:.2f}" height="12" rx="6" fill="{color}"/>'
            )
            parts.append(
                _text(
                    value_x,
                    row_y,
                    metric.fraction(),
                    11.6,
                    color,
                    700,
                )
            )

    return parts


def _footer(
    runs: tuple[RunSummary, RunSummary],
    dataset: DatasetCounts,
) -> list[str]:
    date_label = _date_label(runs)
    prompt_labels = sorted(
        {
            version
            for run in runs
            for version in run.prompt_version.values()
        }
    )
    category_text = ", ".join(
        f"{dataset.categories[category]} {CATEGORY_LABELS[category].lower()}"
        for category in CATEGORY_ORDER
    )
    source = (
        f"Source: saved primary and alternate reports, {date_label}  \u00b7  "
        f"{dataset.total} messages ({dataset.threats} threats, {dataset.benign} benign; {category_text})  \u00b7  "
        f"prompt {' / '.join(prompt_labels)}  \u00b7  provider {runs[0].provider}"
    )
    return [
        _line(60, 864, 1540, 864, BORDER),
        _text(60, 888, source, 8.6, SUBTLE),
    ]


def _short_name(run: RunSummary) -> str:
    if "bolt" in run.model:
        return "Bolt"
    if "lightning" in run.model:
        return "Lightning"
    return run.display_name


def _date_label(runs: tuple[RunSummary, RunSummary]) -> str:
    dates = sorted({run.generated_at.date() for run in runs})
    if len(dates) == 1:
        return dates[0].strftime("%d %b %Y").upper()
    return f"{dates[0].strftime('%d %b %Y').upper()} TO {dates[-1].strftime('%d %b %Y').upper()}"


def _badge(
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    fill: str,
    stroke: str,
    color: str,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{height / 2}" fill="{fill}" stroke="{stroke}"/>'
        f'<text x="{x + width / 2}" y="{y + height / 2 + 3.5}" text-anchor="middle" font-size="8.6" font-weight="700" letter-spacing="0.8" fill="{color}">{escape(label)}</text>'
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: float,
    fill: str,
    weight: int = 400,
    *,
    anchor: str = "start",
    spacing: float | None = None,
    family: str | None = None,
) -> str:
    attributes = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
        f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
    ]
    if spacing is not None:
        attributes.append(f'letter-spacing="{spacing}"')
    if family is not None:
        attributes.append(f'font-family="{family}"')
    return f"<text {' '.join(attributes)}>{escape(value)}</text>"


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}"/>'


def _mapping(value: dict[str, object], key: str) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"report field {key!r} must be an object")
    return item


def _string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"report field {key!r} must be a string")
    return item


def _string_mapping(value: dict[str, object], key: str) -> dict[str, str]:
    item = _mapping(value, key)
    if not all(isinstance(entry, str) for entry in item.values()):
        raise ValueError(f"report field {key!r} values must be strings")
    return {str(entry_key): str(entry) for entry_key, entry in item.items()}


def _integer(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"report field {key!r} must be an integer")
    return item


def _datetime(value: dict[str, object], key: str) -> datetime:
    raw = _string(value, key)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"report field {key!r} must be an ISO timestamp") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the static benchmark evaluation SVG."
    )
    parser.add_argument(
        "--report",
        action="append",
        help="Report JSON path; pass twice. Defaults to the Bolt and Lightning reports.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report_paths = (
        tuple(Path(path) for path in args.report)
        if args.report
        else DEFAULT_REPORTS
    )
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in report_paths
    ]
    dataset = load_dataset_counts(args.dataset)
    runs = summarize_reports(reports, dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_evaluation(runs, dataset), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
