"""Tests des metriques de tokens LLM."""

from datamind.core.metrics import MetricsStore


def test_record_updates_totals(tmp_path):
    store = MetricsStore(path=str(tmp_path / "m.jsonl"))
    store.record(prompt_tokens=100, completion_tokens=20)
    store.record(prompt_tokens=50, completion_tokens=10)

    snap = store.snapshot()
    assert snap["requests"] == 2
    assert snap["prompt_tokens"] == 150
    assert snap["completion_tokens"] == 30
    assert snap["total_tokens"] == 180


def test_jsonl_persistence(tmp_path):
    path = tmp_path / "nested" / "m.jsonl"
    store = MetricsStore(path=str(path))
    store.record(prompt_tokens=10, completion_tokens=5)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"prompt_tokens": 10' in lines[0]


def test_global_store_is_shared():
    from datamind.core import metrics

    before = metrics.store.snapshot()["requests"]
    metrics.store.record(prompt_tokens=1, completion_tokens=1)
    after = metrics.store.snapshot()["requests"]
    assert after == before + 1
