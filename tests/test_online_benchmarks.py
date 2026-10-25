from copy import deepcopy

import pytest

from src.evaluation.online_data import (
    check_manifest, load_suite, materialize_roles, normalize_row, remote_rows,
)


def small_suite(selection="reservoir"):
    return {"version": 1, "seed": 42, "protocol": "pid-grpo-zero-shot-v1",
            "aggregation": "sample_weighted", "sources": {
                "mmlu": {"path": "test/mmlu", "name": "all", "revision": "a" * 40,
                         "split": "test", "kind": "mmlu", "selection": selection,
                         "allocations": {"control": 6, "forgetting": 6}}}}


def fake_rows(source):
    return [{"question": f"q{i}", "subject": f"subject_{i % 3}",
             "choices": ["one", "two"], "answer": i % 2} for i in range(30)]


def test_fixed_selection_disjoint_roles_and_manifest_content_guard():
    suite = small_suite()
    rows, manifest = materialize_roles(suite, ["control", "forgetting"], row_loader=fake_rows)
    again, second = materialize_roles(suite, ["control"], row_loader=fake_rows)
    assert rows["control"] == again["control"]
    assert not {r["id"] for r in rows["control"]} & {r["id"] for r in rows["forgetting"]}
    check_manifest(manifest, second)
    changed = deepcopy(second)
    changed["roles"]["control"][0]["content_hash"] = "changed"
    with pytest.raises(ValueError, match="content changed"):
        check_manifest(manifest, changed)


def test_stratified_control_and_heldout_cover_all_subjects():
    rows, _ = materialize_roles(small_suite("stratified"), ["control", "forgetting"], row_loader=fake_rows)
    for role in rows:
        subjects = [r["raw"]["subject"] for r in rows[role]]
        assert [subjects.count(f"subject_{i}") for i in range(3)] == [2, 2, 2]


def test_duplicate_questions_and_undersized_sources_fail_closed():
    with pytest.raises(ValueError, match="too small"):
        materialize_roles(small_suite(), ["control"], row_loader=lambda _: fake_rows(None)[:2])
    with pytest.raises(ValueError, match="duplicate question"):
        materialize_roles(small_suite(), ["control"], row_loader=lambda _: [fake_rows(None)[0]] * 30)


def test_final_only_request_does_not_read_training_or_control_sources():
    suite = small_suite()
    suite["sources"]["gsm8k"] = {"path": "test/gsm8k", "split": "test", "kind": "math",
                                   "selection": "reservoir", "allocations": {"final_math": None},
                                   "expected_rows": 1, "revision": "a" * 40}
    def loader(source):
        assert source["path"] == "test/gsm8k"
        return [{"question": "How many?", "answer": "work\n#### 42"}]
    rows, _ = materialize_roles(suite, ["final_math"], row_loader=loader)
    assert rows["final_math"][0]["answer"] == "42"


def test_buffered_training_is_bounded_and_reserves_validation():
    suite = small_suite()
    source = suite["sources"]["mmlu"]
    source.update(selection="buffered", buffer_size=3, allocations={"train": 4, "domain_validation": 2})
    reads = []
    def loader(_):
        for row in fake_rows(None):
            reads.append(row)
            yield row
    rows, _ = materialize_roles(suite, ["train", "domain_validation"], row_loader=loader)
    assert len(reads) <= 9
    assert len(rows["train"]) == 4 and len(rows["domain_validation"]) == 2


def test_remote_reader_uses_streaming_and_parquet_column_projection(monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return iter([])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    source = load_suite("configs/online_benchmarks.yaml")["sources"]["deepmath"]
    list(remote_rows(source))
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["revision"] == source["revision"]
    assert calls[0][1]["columns"] == ["question", "final_answer", "difficulty", "topic"]


def test_arc_numeric_labels_and_hellaswag_normalization():
    arc = normalize_row("arc", {"kind": "arc"}, {
        "question": "Which?", "choices": {"label": ["1", "2"], "text": ["yes", "no"]},
        "answerKey": "2"})
    assert arc["gold"] == 1 and arc["metric"] == "acc_norm"
    hs = normalize_row("hs", {"kind": "hellaswag"}, {
        "ctx_a": "A", "ctx_b": "person", "activity_label": "Test",
        "endings": ["A [title] go [artifact]", "stop"], "label": "0"})
    assert hs["choices"][0] == "A. go "


def test_shipped_online_configuration_counts_and_revisions():
    suite = load_suite("configs/online_benchmarks.yaml")
    assert sum(s["allocations"].get("control", 0) for s in suite["sources"].values()) == 1000
    from src.utils.config import load_config
    cfg = load_config("configs/online_pid_eval.yaml")
    assert not cfg["training"]["enabled"]
    assert cfg["evaluation"]["backend"] == "online_suite"


def test_evaluator_sample_weighting_and_restores_training_after_failure():
    torch = pytest.importorskip("torch")
    from src.evaluation.benchmark_evaluator import OnlineBenchmarkEvaluator

    class Scorer:
        def prepare(self, rows):
            pass
        def score(self, model, row):
            assert not model.training
            return {"score": row["score"]}

    model = torch.nn.Linear(1, 1)
    rows = [{"id": str(i), "task": "mmlu" if i < 2 else "arc", "score": float(i < 2)} for i in range(3)]
    evaluator = OnlineBenchmarkEvaluator(Scorer())
    result = evaluator.evaluate(model, rows)
    assert result["overall_score"] == pytest.approx(2/3)
    assert result["subscores"] == {"mmlu": 1, "arc": 0}
    assert model.training
    rows[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="invalid benchmark"):
        evaluator.evaluate(model, rows)
    assert model.training


def test_loglikelihood_scores_only_continuation_and_rejects_truncation():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    import math
    from src.evaluation.benchmark_evaluator import HFBenchmarkScorer

    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 128)))

    model = UniformModel()
    scorer = HFBenchmarkScorer(Tokenizer(), max_context_length=20)
    assert scorer._loglikelihood(model, "Question:", "A") == pytest.approx(-2 * math.log(128))
    with pytest.raises(ValueError, match="context limit"):
        scorer._loglikelihood(model, "Q" * 20, "A")
