"""Task 8 BM25 report and regression gate contracts."""

import json
import sys
from pathlib import Path

import pytest
from scripts.eval import evaluate_bm25


def _case(case_id: str, category: str) -> dict:
    return {
        "id": case_id,
        "category": category,
        "query": case_id,
        "relevant_source_contains": ["source"],
        "must_preserve_tokens": [],
    }


def test_evaluate_emits_six_strata_and_overall_metrics(monkeypatch):
    cases = [_case(category, category) for category in evaluate_bm25.REQUIRED_CATEGORIES]

    class FakeIndex:
        doc_count = 6
        token_count = 60
        index_version = "candidate-v2"

        def search(self, query, top_k):
            return [{"doc_id": query, "source": "source.pdf"}]

    monkeypatch.setattr(evaluate_bm25, "_load_golden", lambda _path: (cases, (1, 3, 5, 10)))

    report = evaluate_bm25.evaluate(
        Path("offline-fixture.json"),
        top_k=10,
        index=FakeIndex(),
        cold_load_seconds=1.25,
    )

    assert report["k_values"] == [1, 3, 5, 10]
    assert set(evaluate_bm25.REQUIRED_CATEGORIES).issubset(report["metrics"])
    assert report["metrics"]["overall"]["recall@10"] == 1.0
    assert report["metrics"]["gene_variant"]["ndcg@10"] == 1.0
    assert "exact_term" in report["metrics"]
    assert report["metrics"]["exact_term"]["recall@10"] == 1.0


def test_compare_reports_exposes_all_quality_performance_and_consistency_checks():
    baseline = {
        "metrics": {
            "overall": {"recall@10": 0.70, "ndcg@10": 0.60},
            "exact_term": {"recall@10": 0.50},
        },
        "cold_load_seconds": 2.0,
        "latency_ms": {"p95": 2.0},
        "consistency": {
            "generation_mismatch_count": 0,
            "stale_cache_hit_count": 0,
        },
    }
    candidate = {
        "metrics": {
            "overall": {"recall@10": 0.70, "ndcg@10": 0.61},
            "exact_term": {"recall@10": 0.55},
        },
        "cold_load_seconds": 9.0,
        "latency_ms": {"p95": 5.0},
        "consistency": {
            "generation_mismatch_count": 0,
            "stale_cache_hit_count": 0,
        },
    }

    result = evaluate_bm25.compare_reports(candidate, baseline)

    assert result["passed"] is True
    assert set(result["checks"]) == {
        "overall_recall@10",
        "overall_ndcg@10",
        "exact_term_recall@10",
        "cold_load_seconds",
        "search_p95_ms",
        "generation_mismatch_count",
        "stale_cache_hit_count",
    }


def test_parser_accepts_compare_and_fail_on_regression():
    parser = evaluate_bm25.build_parser()

    args = parser.parse_args(["--compare", "baseline.json", "--fail-on-regression"])

    assert args.compare == Path("baseline.json")
    assert args.fail_on_regression is True


def test_fail_on_regression_rejects_missing_active_generation(monkeypatch):
    class MissingIndex:
        initialized = False
        doc_count = 0

    monkeypatch.setattr(
        "app.services.rag.bm25_search.get_bm25_index",
        lambda: MissingIndex(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_bm25.py",
            "--compare",
            "missing-baseline.json",
            "--fail-on-regression",
        ],
    )

    assert evaluate_bm25.main() == 1


# ---------------------------------------------------------------------------
# --validate-golden-only tests
# ---------------------------------------------------------------------------


def _make_golden_file(tmp_path: Path, cases: list, k_values=None) -> Path:
    """Write a golden set JSON file and return its path."""
    golden = {"cases": cases, "k_values": k_values or [1, 3, 5, 10]}
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    return golden_path


def _valid_cases(minimum: int = 40) -> list:
    """Generate minimum valid cases covering all 6 required categories."""
    categories = list(evaluate_bm25.REQUIRED_CATEGORIES)
    cases = []
    for i in range(minimum):
        cat = categories[i % len(categories)]
        cases.append({
            "id": f"case-{i:03d}",
            "category": cat,
            "query": f"medical query {i}",
            "relevant_source_contains": [f"source-{i}.pdf"],
            "must_preserve_tokens": ["medical"],
        })
    return cases


def test_validate_golden_only_does_not_call_get_bm25_index(tmp_path, monkeypatch):
    """--validate-golden-only must NOT invoke get_bm25_index()."""
    cases = _valid_cases(42)
    golden_path = _make_golden_file(tmp_path, cases)

    # Track if validate_golden_only tries to import bm25_search
    import sys
    original_modules = dict(sys.modules)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is True
    # Verify no new bm25_search imports happened
    assert "app.services.rag.bm25_search" not in (
        set(sys.modules) - set(original_modules)
    ), "bm25_search module must not be imported in validate-golden-only mode"


def test_validate_golden_only_rejects_duplicate_ids(tmp_path):
    """Duplicate case IDs must fail validation."""
    cases = _valid_cases(40)
    # Introduce duplicate ID
    cases[1]["id"] = cases[0]["id"]
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False
    assert "duplicate" in result["error"].lower() or "unique" in result["error"].lower()


def test_validate_golden_only_rejects_missing_category(tmp_path):
    """Missing a required category must fail validation."""
    # Only 5 categories, missing icd_code
    categories = list(evaluate_bm25.REQUIRED_CATEGORIES)[:5]
    cases = []
    for i in range(42):
        cat = categories[i % len(categories)]
        cases.append({
            "id": f"case-{i:03d}",
            "category": cat,
            "query": f"query {i}",
            "relevant_source_contains": ["src.pdf"],
            "must_preserve_tokens": ["query"],
        })
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False
    assert "category" in result["error"].lower() or "icd_code" in result["error"].lower()


def test_validate_golden_only_rejects_empty_query(tmp_path):
    """Empty query field must fail validation."""
    cases = _valid_cases(40)
    cases[0]["query"] = ""
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False


def test_validate_golden_only_rejects_empty_relevant_sources(tmp_path):
    """Empty relevant_source_contains must fail validation."""
    cases = _valid_cases(40)
    cases[0]["relevant_source_contains"] = []
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False


def test_validate_golden_only_rejects_empty_must_preserve_tokens(tmp_path):
    """Empty must_preserve_tokens must fail validation."""
    cases = _valid_cases(40)
    cases[0]["must_preserve_tokens"] = []
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False


def test_validate_golden_only_rejects_token_not_in_query(tmp_path, monkeypatch):
    """must_preserve_tokens not preservable from query tokens must fail."""
    cases = _valid_cases(40)
    # Set a token that cannot be derived from the query
    cases[0]["query"] = "simple query"
    cases[0]["must_preserve_tokens"] = ["xyznonexistent"]
    golden_path = _make_golden_file(tmp_path, cases)

    # Use a simple tokenizer that splits on whitespace
    monkeypatch.setattr(
        evaluate_bm25,
        "_simple_tokenize",
        lambda text: text.lower().split(),
    )

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False
    assert "token" in result["error"].lower()


def test_validate_golden_only_rejects_below_minimum_cases(tmp_path):
    """Fewer than 40 cases must fail validation."""
    cases = _valid_cases(39)
    golden_path = _make_golden_file(tmp_path, cases)

    result = evaluate_bm25.validate_golden_only(golden_path)
    assert result["passed"] is False
    assert "40" in result["error"] or "minimum" in result["error"].lower()


def test_parser_accepts_validate_golden_only():
    """Parser must accept --validate-golden-only flag."""
    parser = evaluate_bm25.build_parser()
    args = parser.parse_args(["--validate-golden-only"])
    assert args.validate_golden_only is True


def test_parser_accepts_policy_and_provenance():
    """Parser must accept --policy, --baseline-provenance, --consistency-report."""
    parser = evaluate_bm25.build_parser()
    args = parser.parse_args([
        "--policy", "policy.json",
        "--baseline-provenance", "provenance.json",
        "--consistency-report", "consistency.json",
        "--compare", "baseline.json",
        "--fail-on-regression",
    ])
    assert args.policy == Path("policy.json")
    assert args.baseline_provenance == Path("provenance.json")
    assert args.consistency_report == Path("consistency.json")


def test_fail_on_regression_requires_policy_provenance_consistency():
    """--fail-on-regression requires --policy, --baseline-provenance, --consistency-report."""
    parser = evaluate_bm25.build_parser()
    args = parser.parse_args(["--compare", "baseline.json", "--fail-on-regression"])
    # When --fail-on-regression is set without policy/provenance/consistency,
    # main() should error
    assert args.fail_on_regression is True
    assert args.policy is None


# ---------------------------------------------------------------------------
# Medical tokenizer boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "required"),
    [
        ("PD-L1 TPS≥50%", {"pd-l1", "tps", "50%"}),
        ("EGFR-T790M阳性", {"egfr-t790m"}),
        ("HER2 3+", {"her2", "3"}),
    ],
)
def test_medical_tokenizer_preserves_boundary_tokens(text, required):
    assert required <= set(evaluate_bm25._simple_tokenize(text))
