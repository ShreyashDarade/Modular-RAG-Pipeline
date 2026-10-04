"""Dataset parsing is strict: a typo in an evaluation set must not silently weaken it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.core.errors import ConfigError
from src.evaluation.dataset import EvalCase, Label, build_dataset, load_dataset, parse_case, write_dataset


def write(tmp_path: Path, *lines: object) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )
    return path


def test_a_full_case_round_trips():
    raw = {
        "id": "q1",
        "query": "How did revenue change?",
        "relevant": [{"source": "r.pdf", "pages": [3, 4]}, {"contains": "twelve percent", "grade": 2}],
        "reference_answer": "Up twelve percent.",
        "tags": ["finance"],
        "collection": "alpha",
        "kinds": ["text", "table"],
    }
    case = parse_case(raw, "t")
    assert case.labels == (Label(source="r.pdf", pages=(3, 4)), Label(contains="twelve percent", grade=2))
    assert case.to_dict() == raw


def test_load_skips_blank_lines_and_fingerprints_the_content(tmp_path):
    a = load_dataset(write(tmp_path, {"id": "1", "query": "a"}, "", {"id": "2", "query": "b"}))
    assert len(a) == 2
    again = load_dataset(write(tmp_path, {"id": "1", "query": "a"}, {"id": "2", "query": "b"}))
    changed = load_dataset(write(tmp_path, {"id": "1", "query": "a"}, {"id": "2", "query": "B"}))
    assert a.fingerprint == again.fingerprint != changed.fingerprint


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"id": "1"', "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ({"query": "q"}, "`id` is required"),
        ({"id": "1", "query": "  "}, "`query` is required"),
        ({"id": "1", "query": "q", "reliable": []}, "unknown key"),
        ({"id": "1", "query": "q", "relevant": "a.pdf"}, "`relevant` must be a list"),
        ({"id": "1", "query": "q", "relevant": [{"pages": [1]}]}, "needs `source`, `chunk_id` or `contains`"),
        (
            {"id": "1", "query": "q", "relevant": [{"source": "a", "grade": 0}]},
            "`grade` must be an integer >= 1",
        ),
        (
            {"id": "1", "query": "q", "relevant": [{"source": "a", "pages": ["3"]}]},
            "`pages` must be a list of integers",
        ),
        ({"id": "1", "query": "q", "relevant": [{"source": "a", "colour": 1}]}, "unknown label key"),
        ({"id": "1", "query": "q", "relevant": [{"source": ""}]}, "non-empty string"),
        ({"id": "1", "query": "q", "answerable": "no"}, "`answerable` must be true or false"),
        ({"id": "1", "query": "q", "answerable": False, "relevant": [{"source": "a"}]}, "unanswerable"),
        ({"id": "1", "query": "q", "tags": "x"}, "`tags` must be a list"),
    ],
)
def test_malformed_cases_are_rejected_with_the_file_and_line(tmp_path, line, message):
    with pytest.raises(ConfigError, match=rf"cases.jsonl:2: .*{message}"):
        load_dataset(write(tmp_path, {"id": "ok", "query": "fine"}, line))


def test_empty_and_duplicate_datasets_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="no cases"):
        load_dataset(write(tmp_path, ""))
    with pytest.raises(ConfigError, match="duplicate case id 'x'"):
        load_dataset(write(tmp_path, {"id": "x", "query": "a"}, {"id": "x", "query": "b"}))
    with pytest.raises(ConfigError, match="cannot read dataset"):
        load_dataset(tmp_path / "missing.jsonl")


def test_write_then_load_preserves_everything(tmp_path):
    cases = [
        EvalCase("a", "q1", (Label(source="x.pdf", pages=(2,)),), "ans", tags=("t",)),
        EvalCase("b", "q2", answerable=False),
    ]
    path = tmp_path / "out.jsonl"
    write_dataset(cases, path)
    assert list(load_dataset(path).cases) == cases
    assert build_dataset(cases).fingerprint == load_dataset(path).fingerprint
