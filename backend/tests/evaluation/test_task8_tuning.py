"""Task 8 split-safe joint BM25/RRF tuning contracts."""

import pytest

from scripts.eval import tune_weights


def test_joint_grid_matches_task8_brief():
    assert tune_weights.BM25_GRID == {
        "k1": [0.9, 1.2, 1.5],
        "b": [0.5, 0.7, 0.8],
        "heading_boost": [1, 2],
        "entity_boost": [1, 2, 3],
    }
    assert tune_weights.RRF_K_GRID == [30, 35, 60]
    assert len(list(tune_weights.iter_parameter_grid())) == 162


@pytest.mark.asyncio
async def test_dev_selects_once_then_test_is_evaluated_once():
    calls = []

    async def fake_evaluate(params, cases, split):
        calls.append((params, cases, split))
        # Make the final grid member the deterministic winner.
        score = 1.0 if params["rrf_k"] == 60 else 0.0
        return {"ndcg@10": score, "recall@10": score}

    result = await tune_weights.tune_splits(
        dev_cases=[{"id": "dev-1"}],
        test_cases=[{"id": "test-1"}],
        evaluate_combo=fake_evaluate,
    )

    dev_calls = [call for call in calls if call[2] == "dev"]
    test_calls = [call for call in calls if call[2] == "test"]
    assert len(dev_calls) == 162
    assert len(test_calls) == 1
    assert result["selected_params"]["rrf_k"] == 60
    assert result["test_result"]["split"] == "test"


@pytest.mark.asyncio
async def test_test_split_cannot_be_used_as_tuning_split():
    async def fake_evaluate(params, cases, split):
        return {"ndcg@10": 0.0}

    with pytest.raises(ValueError, match="dev"):
        await tune_weights.tune_splits(
            dev_cases=[{"id": "dev-1"}],
            test_cases=[{"id": "test-1"}],
            evaluate_combo=fake_evaluate,
            tuning_split="test",
        )
