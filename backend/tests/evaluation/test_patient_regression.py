# -*- coding: utf-8 -*-
"""回归阈值 check_thresholds 破线检测与退出码测试"""

from evaluation.patient_regression import check_thresholds, extract_arm_metrics


def _report(agent_tool_summaries):
    return {
        "cases": [
            {"case_id": f"c{i}", "agent_tool": {"summary": s}}
            for i, s in enumerate(agent_tool_summaries)
        ]
    }


def test_extract_arm_metrics_means_across_cases():
    report = _report([
        {"disclosure_rate": 0.4, "judge_overall_avg": 3.0},
        {"disclosure_rate": 0.6, "judge_overall_avg": 4.0},
    ])
    metrics = extract_arm_metrics(report, "agent_tool")
    assert metrics["disclosure_rate"] == 0.5
    assert metrics["judge_overall_avg"] == 3.5


def test_check_thresholds_pass():
    report = _report([{"disclosure_rate": 0.7, "judge_overall_avg": 4.0}])
    thresholds = {"agent_tool": {"disclosure_rate_min": 0.5, "judge_overall_avg_min": 3.2}}
    results, ok = check_thresholds(report, thresholds)
    assert ok is True
    assert all(r["status"] == "PASS" for r in results)


def test_check_thresholds_fail_on_breach():
    report = _report([{"disclosure_rate": 0.3, "judge_overall_avg": 4.0}])
    thresholds = {"agent_tool": {"disclosure_rate_min": 0.5}}
    results, ok = check_thresholds(report, thresholds)
    assert ok is False
    breach = next(r for r in results if r["metric"] == "disclosure_rate")
    assert breach["status"] == "FAIL"
    assert breach["actual"] == 0.3
    assert breach["bound"] == 0.5


def test_check_thresholds_max_bound():
    report = _report([{"tool_degrade_rate": 0.4}])
    thresholds = {"agent_tool": {"tool_degrade_rate_max": 0.2}}
    results, ok = check_thresholds(report, thresholds)
    assert ok is False
    assert results[0]["status"] == "FAIL"


def test_check_thresholds_skip_missing_metric():
    report = _report([{"disclosure_rate": 0.7}])
    thresholds = {"agent_tool": {"nonexistent_metric_min": 1.0}}
    results, ok = check_thresholds(report, thresholds)
    # 缺失指标标 SKIP，不算破线
    assert ok is True
    assert results[0]["status"] == "SKIP"
