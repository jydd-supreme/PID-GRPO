from copy import deepcopy
import json

import pytest

from src.evaluation.online_data import (
    check_manifest, load_suite, materialize_roles, normalize_row, remote_rows,
)
from src.evaluation.selection_cache import SelectedSourceCache


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
    with pytest.raises(ValueError, match="too small"):
        materialize_roles(small_suite(), ["control"], row_loader=lambda _: [fake_rows(None)[0]] * 30)


@pytest.mark.parametrize("selection", ["buffered", "reservoir", "stratified"])
def test_repeated_source_questions_are_replaced_before_fixed_size_splits(selection):
    suite = small_suite()
    source = suite["sources"]["mmlu"]
    source.update(selection=selection, buffer_size=3)
    raw = fake_rows(None)
    repeated = [raw[0], raw[0], *raw[:15]]
    rows, manifest = materialize_roles(suite, ["control", "forgetting"],
                                       row_loader=lambda _: iter(repeated))
    again, second = materialize_roles(suite, ["control", "forgetting"],
                                      row_loader=lambda _: iter(repeated))
    assert [len(rows[role]) for role in ("control", "forgetting")] == [6, 6]
    assert len({row["question_hash"] for role in rows for row in rows[role]}) == 12
    assert rows == again and manifest == second


def test_full_split_still_rejects_duplicate_questions():
    suite = {"version": 1, "seed": 42, "protocol": "pid-grpo-zero-shot-v1",
             "aggregation": "sample_weighted", "sources": {
                 "math": {"path": "test/math", "revision": "a" * 40,
                          "kind": "math", "selection": "reservoir",
                          "allocations": {"final_math": None}, "expected_rows": 2}}}
    raw = {"question": "same question", "answer": "42"}
    with pytest.raises(ValueError, match="duplicate question"):
        materialize_roles(suite, ["final_math"], row_loader=lambda _: iter([raw, raw]))


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


def test_selected_source_cache_survives_later_source_failure(tmp_path):
    suite = small_suite()
    suite["sources"]["mmlu2"] = {
        **suite["sources"]["mmlu"], "path": "test/mmlu2", "revision": "b" * 40,
    }
    cache = SelectedSourceCache(tmp_path / "selected")
    calls = []

    def interrupted_loader(source):
        calls.append(source["path"])
        if source["path"] == "test/mmlu2":
            raise ConnectionError("network reset")
        return fake_rows(source)

    with pytest.raises(ConnectionError, match="network reset"):
        materialize_roles(suite, ["control"], row_loader=interrupted_loader,
                          source_cache=cache)
    assert calls == ["test/mmlu", "test/mmlu2"]
    assert cache._path(suite, "mmlu").is_file()

    def remaining_loader(source):
        calls.append(source["path"])
        assert source["path"] == "test/mmlu2"
        return [{**row, "question": "second " + row["question"]}
                for row in fake_rows(source)]

    rows, manifest = materialize_roles(suite, ["control"], row_loader=remaining_loader,
                                       source_cache=cache)
    assert calls[-1:] == ["test/mmlu2"]
    again, second = materialize_roles(
        suite, ["control"], source_cache=cache,
        row_loader=lambda _: pytest.fail("cache hit must not request the network"),
    )
    assert rows == again and manifest == second


def test_selected_source_cache_rejects_modified_content(tmp_path):
    suite = small_suite()
    cache = SelectedSourceCache(tmp_path)
    materialize_roles(suite, ["control"], row_loader=fake_rows, source_cache=cache)
    path = cache._path(suite, "mmlu")
    payload = json.loads(path.read_text())
    payload["selected"]["control"][0]["prompt"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cached row content mismatch"):
        materialize_roles(suite, ["control"], source_cache=cache,
                          row_loader=lambda _: pytest.fail("must not refetch silently"))


def test_selected_source_cache_does_not_persist_manifest_mismatch(tmp_path):
    suite = small_suite()
    _, frozen = materialize_roles(suite, ["control"], row_loader=fake_rows)
    cache = SelectedSourceCache(tmp_path, expected_manifest=frozen)

    def changed_rows(source):
        return [{**row, "question": "changed " + row["question"]}
                for row in fake_rows(source)]

    with pytest.raises(ValueError, match="differs from frozen manifest"):
        materialize_roles(suite, ["control"], row_loader=changed_rows,
                          source_cache=cache)
    assert not cache._path(suite, "mmlu").exists()


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
