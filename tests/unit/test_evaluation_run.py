"""Runner, report, comparison, gates, judges, answer evaluation, generation and variants."""

from __future__ import annotations

import json

import pytest
from src.core.config import Settings
from src.core.errors import ConfigError, ModelError
from src.core.specs import RagConfig
from src.core.types import AnswerResult, RetrievalResult, RetrievalScope, RetrievedDocument
from src.evaluation.answers import AnswerEvaluator, citation_is_valid, citations
from src.evaluation.dataset import EvalCase, Label, build_dataset
from src.evaluation.generate import generate_cases
from src.evaluation.judge import JudgeError, LlmJudge, extract_json
from src.evaluation.report import (
    EvalReport,
    build_report,
    check_report,
    compare_reports,
    render_comparison,
    render_report,
)
from src.evaluation.runner import RetrievalEvaluator
from src.evaluation.variants import apply_overrides, evaluation_settings, parse_variant

from tests.fake_plugin import ScriptedChat
from tests.unit.fakes import MemorySearcher


def doc(source: str, page: int = 1, content: str = "text", score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(
        content=content,
        metadata={"source": source, "page": page},
        score=score,
        collection="c",
        kind="text",
        index="i",
    )


class StubRetrieval:
    """What `RetrievalPipeline` exposes to the evaluator: scope() and retrieve()."""

    def __init__(self, table: dict[str, list[RetrievedDocument]], fail: set[str] = frozenset()) -> None:
        self.table, self.fail, self.queries = table, set(fail), []

    def scope(self, collections=None, kinds=None, sources=None):
        return RetrievalScope(tuple(collections or ("c",)))

    async def retrieve(self, query, scope):
        self.queries.append(query)
        if query in self.fail:
            raise ModelError("reranker is down")
        return RetrievalResult(query=query, expanded_queries=[query], documents=self.table[query])


CASES = [
    EvalCase("q1", "first", (Label(source="a.pdf"),), tags=("easy",)),
    EvalCase("q2", "second", (Label(source="b.pdf"),), tags=("hard",)),
    EvalCase("q3", "third", (Label(source="c.pdf"),), tags=("hard",)),
    EvalCase("q4", "no evidence", answerable=False),
]
DATASET = build_dataset(list(CASES))
GOOD = {
    "first": [doc("a.pdf")],
    "second": [doc("x.pdf"), doc("b.pdf")],
    "third": [doc("x.pdf")],
    "no evidence": [doc("x.pdf")],
}


async def evaluate(table=GOOD, *, fail=frozenset(), **kw):
    stub = StubRetrieval(table, set(fail))
    evaluator = RetrievalEvaluator(stub, ks=(1, 3), default_collection="c", **kw)  # type: ignore[arg-type]
    return evaluator, await evaluator.run(DATASET)


# --- runner ------------------------------------------------------------------------------------
async def test_every_query_is_scored_and_unlabelled_ones_have_no_retrieval_metrics():
    _, results = await evaluate()
    by_id = {r.id: r for r in results}
    assert by_id["q1"].metrics["hit@1"] == 1.0 and by_id["q1"].metrics["mrr@3"] == 1.0
    assert by_id["q2"].metrics["hit@1"] == 0.0 and by_id["q2"].metrics["hit@3"] == 1.0
    assert by_id["q2"].metrics["mrr@3"] == 0.5
    assert by_id["q3"].metrics["recall@3"] == 0.0 and by_id["q3"].missed == [0]
    assert by_id["q4"].metrics == {} and by_id["q4"].missed == []
    assert [item.matched for item in by_id["q2"].ranked] == [(), (0,)]


async def test_a_failed_query_scores_zero_and_is_reported_not_dropped():
    _, results = await evaluate(fail={"first"})
    failed = next(r for r in results if r.id == "q1")
    assert failed.error and "model_error" in failed.error and "reranker is down" in failed.error
    assert failed.metrics["recall@3"] == 0.0 and failed.metrics["mrr@3"] == 0.0
    report = build_report("r", DATASET, results, {})
    assert report.errors == 1
    assert report.metrics["hit@3"].n == 3, "the failure still counts as one of the labelled queries"
    assert check_report(report) == ["1 queries failed to run"]


async def test_document_granularity_scores_documents_not_chunks():
    table = {
        **GOOD,
        "second": [doc("x.pdf", 1, "1"), doc("x.pdf", 2, "2"), doc("x.pdf", 3, "3"), doc("b.pdf")],
    }
    _, chunk = await evaluate(table)
    _, document = await evaluate(table, granularity="document")
    q2 = lambda results: next(r for r in results if r.id == "q2").metrics  # noqa: E731
    assert q2(chunk)["hit@3"] == 0.0, "three chunks of one wrong document fill the top 3"
    assert q2(document)["hit@3"] == 1.0, "collapsed: x.pdf, then b.pdf"


async def test_the_scope_comes_from_the_case_then_the_run_default():
    stub = StubRetrieval(GOOD)
    seen = []
    stub.scope = lambda collections=None, kinds=None, sources=None: (
        seen.append((collections, kinds)) or RetrievalScope(("c",))
    )  # type: ignore[method-assign]
    evaluator = RetrievalEvaluator(stub, ks=(1,), default_collection="fallback")  # type: ignore[arg-type]
    case = EvalCase("z", "first", (Label(source="a.pdf"),), collection="special", kinds=("table",))
    await evaluator.run_case(case)
    await evaluator.run_case(EvalCase("y", "second", (Label(source="b.pdf"),)))
    assert seen == [(["special"], ("table",)), (["fallback"], None)]


def test_invalid_cutoffs_are_refused():
    with pytest.raises(ValueError, match="cutoffs"):
        RetrievalEvaluator(StubRetrieval({}), ks=[0])  # type: ignore[arg-type]


# --- report -----------------------------------------------------------------------------------
async def test_report_aggregates_with_intervals_tags_and_latency_and_round_trips(tmp_path):
    _, results = await evaluate()
    report = build_report("baseline", DATASET, results, {"config": {"reranker": "identity"}})
    assert report.metrics["hit@1"].mean == pytest.approx(1 / 3) and report.metrics["hit@1"].n == 3
    assert report.metrics["hit@1"].low <= report.metrics["hit@1"].mean <= report.metrics["hit@1"].high
    assert report.by_tag["hard"]["hit@3"] == (0.5, 2) and report.by_tag["easy"]["hit@3"] == (1.0, 1)
    assert report.meta["cases"] == 4 and report.meta["labelled_cases"] == 3 and report.errors == 0
    path = tmp_path / "r.json"
    report.save(path)
    loaded = EvalReport.load(path)
    assert (
        loaded.metrics == report.metrics and loaded.cases == report.cases and loaded.by_tag == report.by_tag
    )
    text = render_report(loaded)
    assert "baseline" in text and "ndcg@3" in text and "reranker=identity" in text


@pytest.mark.parametrize(
    "content", ["[]", '"hi"', "null", '{"schema": 1, "name": "x", "meta": null}', '{"schema": 1}']
)
def test_garbage_report_files_are_config_errors_not_tracebacks(tmp_path, content):
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    with pytest.raises(ConfigError, match="cannot read report|unsupported report schema"):
        EvalReport.load(bad)


def test_loading_a_bad_report_is_a_config_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="cannot read report"):
        EvalReport.load(bad)
    bad.write_text(json.dumps({"schema": 99}))
    with pytest.raises(ConfigError, match="unsupported report schema"):
        EvalReport.load(bad)


async def test_comparison_pairs_queries_flags_real_gains_and_refuses_different_datasets():
    _, base_results = await evaluate()
    better_table = {**GOOD, "second": [doc("b.pdf")], "third": [doc("c.pdf")]}
    _, better_results = await evaluate(better_table)
    base = build_report("base", DATASET, base_results, {})
    better = build_report("better", DATASET, better_results, {})
    comparison = {c.metric: c for c in compare_reports([base, better])}
    assert comparison["hit@1"].baseline == pytest.approx(1 / 3) and comparison["hit@1"].values == [1.0]
    assert comparison["hit@1"].deltas[0].diff == pytest.approx(2 / 3) and comparison["hit@1"].deltas[0].n == 3
    text = render_comparison([base, better], list(comparison.values()))
    assert "baseline: base" in text and "better" in text and "not a verdict" in text

    other = build_report("other", build_dataset([EvalCase("only", "q", (Label(source="a"),))]), [], {})
    with pytest.raises(ConfigError, match="different datasets"):
        compare_reports([base, other])
    with pytest.raises(ConfigError, match="at least two"):
        compare_reports([base])


async def test_gates_fail_on_a_missing_metric_a_low_score_or_a_regression():
    _, base_results = await evaluate()
    _, worse_results = await evaluate({**GOOD, "first": [doc("x.pdf")]})
    base = build_report("base", DATASET, base_results, {})
    worse = build_report("worse", DATASET, worse_results, {})
    assert check_report(base, minimums={"hit@3": 0.5}) == []
    assert "below the minimum" in check_report(base, minimums={"hit@3": 0.9})[0]
    assert "not in the report" in check_report(base, minimums={"nope@1": 0.1})[0]
    regression = check_report(worse, baseline=base, max_drop=0.05)
    assert any(p.startswith("hit@1: fell") for p in regression)
    assert check_report(base, baseline=base) == [], "a report never regresses against itself"
    assert check_report(worse, baseline=base, max_drop=0.9) == [], "within the allowed drop"


# --- judge --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here is the JSON: {"a": 1} Hope that helps.',
        ' \n{"a": 1}\n',
    ],
)
def test_extract_json_tolerates_fences_and_chatter(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", '{"a": ', "[1, 2]", ""])
def test_extract_json_rejects_everything_else(text):
    with pytest.raises(JudgeError):
        extract_json(text)


class CannedChat:
    model_id = "judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def complete(self, messages):
        return self.replies.pop(0)

    def stream(self, messages): ...


async def test_judge_scores_faithfulness_correctness_and_abstention():
    claims = {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
    judge = LlmJudge(
        CannedChat(
            json.dumps(claims), '{"claims": []}', '{"verdict": "partially_correct"}', '{"abstained": true}'
        )
    )
    faith = await judge.faithfulness("q", [doc("a.pdf")], "answer")
    assert faith.score == 0.5 and faith.claims == (("a", True), ("b", False))
    assert (await judge.faithfulness("q", [], "I do not know")).score is None, "no claims: nothing to support"
    assert await judge.correctness("q", "ref", "ans") == 0.5
    assert await judge.abstained("q", "ans") is True


@pytest.mark.parametrize(
    ("reply", "call"),
    [
        ('{"claims": "none"}', lambda j: j.faithfulness("q", [], "a")),
        ('{"claims": [{"claim": "x"}]}', lambda j: j.faithfulness("q", [], "a")),
        ('{"verdict": "great"}', lambda j: j.correctness("q", "r", "a")),
        ('{"abstained": "yes"}', lambda j: j.abstained("q", "a")),
    ],
)
async def test_unusable_judge_replies_raise_instead_of_scoring_zero(reply, call):
    with pytest.raises(JudgeError):
        await call(LlmJudge(CannedChat(reply)))


# --- answers ------------------------------------------------------------------------------------
def test_citation_parsing_and_validation():
    answer = "Revenue grew [Source: r.pdf, Page: 3]. See also [Source: made-up.pdf] and [source: notes.txt, page: 1]."
    assert citations(answer) == [("r.pdf", "3"), ("made-up.pdf", None), ("notes.txt", "1")]
    retrieved = [doc("/data/r.pdf", page=3), doc("/data/notes.txt", page=2)]
    assert citation_is_valid("r.pdf", "3", retrieved)
    assert not citation_is_valid("r.pdf", "9", retrieved), "right file, page that was not retrieved"
    assert citation_is_valid("r.pdf", None, retrieved)
    assert not citation_is_valid("made-up.pdf", None, retrieved), "a hallucinated source"
    assert not citation_is_valid("notes.txt", "1", retrieved)


class StubAnswers:
    def __init__(
        self, replies: dict[str, tuple[str, list[RetrievedDocument]]], fail: set[str] = frozenset()
    ) -> None:
        self.replies, self.fail = replies, set(fail)

    async def ask(self, query, scope, model=None):
        if query in self.fail:
            raise ModelError("chat is down")
        text, docs = self.replies[query]
        return AnswerResult(
            query=query, expanded_queries=[query], answer=text, model=model or "m", documents=docs
        )


def evaluator_for(*queries: str) -> RetrievalEvaluator:
    table = {q: [doc("a.pdf")] for q in queries}
    return RetrievalEvaluator(StubRetrieval(table), ks=(1, 3), default_collection="c")  # type: ignore[arg-type]


async def test_answer_evaluation_scores_citations_faithfulness_correctness_and_abstention():
    cases = [
        EvalCase("a1", "q-good", (Label(source="a.pdf"),), reference_answer="twelve percent"),
        EvalCase("a2", "q-unknown", answerable=False),
        EvalCase("a3", "q-down", (Label(source="a.pdf"),)),
    ]
    dataset = build_dataset(cases)
    retrieval = evaluator_for("q-good", "q-unknown", "q-down")
    answers = StubAnswers(
        {
            "q-good": (
                "Up twelve percent [Source: a.pdf, Page: 1] and [Source: ghost.pdf].",
                [doc("/x/a.pdf")],
            ),
            "q-unknown": ("The documents do not say.", []),
        },
        fail={"q-down"},
    )
    judge = LlmJudge(
        CannedChat(
            '{"claims": [{"claim": "x", "supported": true}, {"claim": "y", "supported": true}]}',
            '{"verdict": "correct"}',
            '{"abstained": true}',
        )
    )
    results = await retrieval.run(dataset)
    await AnswerEvaluator(answers, judge, retrieval, concurrency=1).run(dataset, results)  # type: ignore[arg-type]
    by_id = {r.id: r.answer for r in results}
    good, unknown, down = by_id["a1"], by_id["a2"], by_id["a3"]
    assert good["metrics"] == {
        "citation_rate": 1.0,
        "citation_valid": 0.5,
        "faithfulness": 1.0,
        "correctness": 1.0,
    }
    assert unknown["metrics"] == {"abstained": 1.0}
    assert "chat is down" in down["error"] and "metrics" not in down
    report = build_report("r", dataset, results, {})
    assert report.metrics["faithfulness"].n == 1 and report.metrics["abstained"].mean == 1.0
    assert report.values("faithfulness") == {"a1": 1.0}, "paired comparison can read answer metrics too"


async def test_a_judge_that_returns_garbage_leaves_the_case_out_of_that_metric_and_says_so():
    case = EvalCase("a1", "q", (Label(source="a.pdf"),), reference_answer="ref")
    dataset = build_dataset([case])
    retrieval = evaluator_for("q")
    judge = LlmJudge(CannedChat("this is not json", '{"verdict": "incorrect"}'))
    results = await retrieval.run(dataset)
    answers = StubAnswers({"q": ("Some answer.", [doc("a.pdf")])})
    await AnswerEvaluator(answers, judge, retrieval, concurrency=1).run(dataset, results)  # type: ignore[arg-type]
    answer = results[0].answer
    assert "faithfulness" not in answer["metrics"], "unparsed: not scored 0, not counted"
    assert answer["metrics"]["correctness"] == 0.0 and "judge_errors" in answer
    assert "faithfulness" in answer["judge_errors"][0]


# --- generation ---------------------------------------------------------------------------------
def chunk(i: int, text: str, *, source="/data/r.pdf", page: int | None = 1):
    return {"chunk_id": f"c{i}", "content": text, "source": source, "page": page, "content_vector": [0.0]}


async def test_generation_drafts_labelled_questions_and_reports_what_it_skipped():
    searcher = MemorySearcher()
    long_text = "Cloud revenue grew twelve percent in the second quarter across every region we serve. " * 4
    for i in range(6):
        searcher.add("alpha-text", chunk(i, long_text + str(i), page=i + 1))
    searcher.add("alpha-text", chunk(90, "too short"))
    config = RagConfig.model_validate(
        {
            "default_chat_model": "c",
            "default_collection": "alpha",
            "chat_models": {"c": {"provider": "fake", "model": "c"}},
            "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 8}},
            "collections": {"alpha": {"embedding_model": "e"}},
        }
    )
    result = await generate_cases(searcher, config, ScriptedChat("c"), collection="alpha", count=4, seed=1)  # type: ignore[arg-type]
    assert len(result.cases) == 4 and result.skipped.get("passage too short") == 1
    case = result.cases[0]
    assert case.tags == ("synthetic",) and case.collection == "alpha" and case.reference_answer
    assert case.labels[0].source == "/data/r.pdf" and case.labels[0].pages and case.query.endswith("?")
    again = await generate_cases(searcher, config, ScriptedChat("c"), collection="alpha", count=4, seed=1)  # type: ignore[arg-type]
    assert [c.query for c in again.cases] == [c.query for c in result.cases], "same seed, same questions"


async def test_generation_skips_unusable_model_replies_without_inventing_questions():
    searcher = MemorySearcher()
    for i in range(3):
        searcher.add("alpha-text", chunk(i, "x " * 200))
    config = RagConfig.model_validate(
        {
            "default_chat_model": "c",
            "default_collection": "alpha",
            "chat_models": {"c": {"provider": "fake", "model": "c"}},
            "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 8}},
            "collections": {"alpha": {"embedding_model": "e"}},
        }
    )
    result = await generate_cases(
        searcher,
        config,
        CannedChat("nope", '{"question": 3}', '{"question": "ok?", "answer": "a"}'),
        collection="alpha",
        count=3,
    )  # type: ignore[arg-type]
    assert [c.query for c in result.cases] == ["ok?"]
    assert result.skipped == {"model did not return JSON": 1, "model reply missing question/answer": 1}


# --- variants -----------------------------------------------------------------------------------
def test_variant_syntax():
    v = parse_variant("ce:reranker=precise, hybrid_alpha=0.7")
    assert v.name == "ce" and dict(v.overrides) == {"reranker": "precise", "hybrid_alpha": "0.7"}
    assert parse_variant("baseline").overrides == {}
    for bad in (":x=1", "n:oops"):
        with pytest.raises(ConfigError):
            parse_variant(bad)


CONFIG = RagConfig.model_validate(
    {
        "default_chat_model": "c",
        "default_collection": "alpha",
        "chat_models": {"c": {"provider": "fake", "model": "c"}},
        "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 8}},
        "collections": {"alpha": {"embedding_model": "e"}},
        "reranker_models": {"precise": {"provider": "fake", "model": "m"}},
    }
)


def test_overrides_are_validated_by_the_real_models_and_never_silently_ignored():
    settings = Settings(_env_file=None, plugins=[])
    new_settings, new_config = apply_overrides(
        settings, CONFIG, {"hybrid_alpha": "0.9", "reranker": "precise", "query_expander": "identity"}
    )
    assert (
        new_settings.hybrid_alpha == 0.9
        and new_config.reranker == "precise"
        and new_config.query_expander == "identity"
    )
    assert settings.hybrid_alpha == 0.5, "the original is untouched"
    with pytest.raises(ConfigError, match="unknown override"):
        apply_overrides(settings, CONFIG, {"hybird_alpha": "0.9"})
    with pytest.raises(ConfigError, match="invalid override"):
        apply_overrides(settings, CONFIG, {"hybrid_alpha": "7"})
    with pytest.raises(ConfigError, match="invalid override"):
        apply_overrides(
            settings, CONFIG, {"reranker": "precise", "query_expander": ""}
        ) if False else apply_overrides(settings, CONFIG, {"rerank_candidates": "2"})


def test_evaluation_always_uses_a_private_cache_and_a_wide_enough_result():
    settings = Settings(_env_file=None, plugins=[], cache_backend="redis", retriever_top_k=5)
    forced = evaluation_settings(settings, [1, 10], {})
    assert forced["cache_backend"] == "memory" and forced["retriever_top_k"] == "10"
    assert "retriever_top_k" not in evaluation_settings(settings, [1, 3], {}), "already wide enough"
    assert evaluation_settings(settings, [1, 10], {"retriever_top_k": "7"})["retriever_top_k"] == "7", (
        "an explicit choice wins"
    )


def test_a_cutoff_beyond_the_default_candidate_pool_widens_both_together():
    """Regression: top-k was raised first and the (unchanged) pool of 30 then failed validation."""
    settings = Settings(_env_file=None, plugins=[])
    forced = evaluation_settings(settings, [1, 5, 50], {})
    assert forced["retriever_top_k"] == "50" and forced["rerank_candidates"] == "50"
    new_settings, _ = apply_overrides(settings, CONFIG, forced)
    assert new_settings.retriever_top_k == 50 == new_settings.rerank_candidates
    with pytest.raises(ConfigError, match="RERANK_CANDIDATES"):
        apply_overrides(settings, CONFIG, evaluation_settings(settings, [1, 50], {"rerank_candidates": "20"}))


def test_document_scoring_asks_for_more_chunks_than_documents():
    settings = Settings(_env_file=None, plugins=[])
    forced = evaluation_settings(settings, [1, 10], {}, "document")
    assert forced["retriever_top_k"] == "30", "10 documents need more than 10 chunks"
    assert "retriever_top_k" not in evaluation_settings(settings, [1, 3], {}, "document"), (
        "9 chunks fit in the default 10"
    )


def test_an_unusable_override_value_is_a_config_error():
    settings = Settings(_env_file=None, plugins=[])
    with pytest.raises(ConfigError, match="invalid override"):
        evaluation_settings(settings, [10], {"retriever_top_k": "many"})


def test_variants_are_validated_up_front_including_component_names():
    from src.core.errors import UnknownComponentError
    from src.evaluation.variants import validate_variant

    settings = Settings(_env_file=None, plugins=["tests.fake_plugin"])
    validate_variant(settings, CONFIG, parse_variant("ok:reranker=identity,hybrid_alpha=0.8"), [10])
    with pytest.raises(UnknownComponentError, match="Unknown reranker 'nonexistent'"):
        validate_variant(settings, CONFIG, parse_variant("bad:reranker=nonexistent"), [10])
    with pytest.raises(UnknownComponentError, match="Unknown query expander 'wishful'"):
        validate_variant(settings, CONFIG, parse_variant("bad:query_expander=wishful"), [10])
    with pytest.raises(ConfigError, match="unknown override"):
        validate_variant(settings, CONFIG, parse_variant("bad:hybird_alpha=0.1"), [10])


async def test_failed_answers_and_judge_outages_are_errors_not_silence():
    """Regression: a failed answer was only recorded on the case, so the run still exited 0, and a judge
    transport error aborted the whole evaluation."""
    cases = [
        EvalCase("ok", "q-ok", (Label(source="a.pdf"),), reference_answer="ref"),
        EvalCase("down", "q-down", (Label(source="a.pdf"),)),
        EvalCase("judged-out", "q-out", (Label(source="a.pdf"),), reference_answer="ref"),
    ]
    dataset = build_dataset(cases)
    retrieval = evaluator_for("q-ok", "q-down", "q-out")
    answers = StubAnswers(
        {"q-ok": ("Fine.", [doc("a.pdf")]), "q-out": ("Fine.", [doc("a.pdf")])}, fail={"q-down"}
    )

    class FlakyJudge(CannedChat):
        async def complete(self, messages):
            reply = self.replies.pop(0)
            if reply == "BOOM":
                raise ModelError("judge is rate limited")
            return reply

    judge = LlmJudge(
        FlakyJudge(
            '{"claims": [{"claim": "x", "supported": true}]}', '{"verdict": "correct"}', "BOOM", "BOOM"
        )
    )
    results = await retrieval.run(dataset)
    await AnswerEvaluator(answers, judge, retrieval, concurrency=1).run(dataset, results)  # type: ignore[arg-type]
    by_id = {r.id: r.answer for r in results}
    assert by_id["ok"]["metrics"]["correctness"] == 1.0 and "failed" not in by_id["ok"]
    assert by_id["down"]["error"] and by_id["judged-out"]["failed"] is True
    assert "rate limited" in by_id["judged-out"]["judge_errors"][0]
    report = build_report("r", dataset, results, {})
    assert report.errors == 2, "the failed answer and the judge outage both fail the run"
    assert check_report(report) == ["2 queries failed to run"]
    assert "FAILED" in render_report(report)


async def test_latency_and_short_ranking_caveats_are_printed():
    _, results = await evaluate()
    report = build_report("r", DATASET, results, {"ks": [1, 3], "concurrency": 4, "short_rankings": 3})
    text = render_report(report)
    assert "concurrency 4: includes queueing" in text
    assert "3 queries returned fewer than 3 results" in text
    quiet = render_report(build_report("r", DATASET, results, {"ks": [1, 3], "concurrency": 1}))
    assert "queueing" not in quiet and "fewer than" not in quiet
