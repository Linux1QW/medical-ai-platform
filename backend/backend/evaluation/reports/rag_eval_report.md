# RAG / Tool Use 评估报告

## 概述

- **评估时间**：2026-07-21T04:47:04.380497+00:00
- **模式**：mock
- **数据集**：backend\evaluation\rag_cases\rag_gold_cases.jsonl
- **样本数**：5
- **正常案例**：5
- **拒答案例**：0

**合规率**: ❌ 62%

## 核心门槛

| 门槛 | 当前值 | 要求 | 等级 | 结果 |
|---|---:|---:|:---:|---|
| Citation Validity | 0.0% | 100% | P0 | ❌ 未通过 |
| Hallucinated Citation Rate | 100.0% | ≤5% | P0 | ❌ 未通过 |
| False Acceptance Rate | 0.0% | ≤5% | P0 | ✅ 通过 |
| Refusal Accuracy | 100.0% | ≥80% | P1 | ✅ 通过 |
| Stance Accuracy | 100.0% | ≥70% | P1 | ✅ 通过 |
| Score Range Accuracy | 100.0% | ≥60% | P1 | ✅ 通过 |

## 检索指标

| 指标 | 值 |
|---|---:|
| R@1 | 0.0% |
| R@3 | 0.0% |
| R@5 | 0.0% |
| MRR | 0.0% |
| NDCG@5 | 0.0% |

## 拒答指标

| 指标 | 值 |
|---|---:|
| Refusal Accuracy | 100.0% |
| Refusal Precision | 0.0% |
| Refusal Recall | 0.0% |
| Refusal F1 | 0.0% |
| False Refusal Rate | 0.0% |
| False Acceptance Rate | 0.0% |

## 引用指标

| 指标 | 值 |
|---|---:|
| Citation Validity | 0.0% |
| Citation Hallucination Rate | 100.0% |
| Citation Coverage | 20.0% |

## Tool Use 指标

- **平均工具调用次数**：1.0
- **平均耗时**：0.10s
- **工具成功率**：100.0%
- **最终回答关键词覆盖率**：0.0%
- **工具调用准确率**：100.0%
- **分数范围准确率**：100.0%

### 工具调用明细

| 工具名称 | 调用次数 | 成功率 | 平均耗时(ms) |
|---|---:|---:|---:|
| mock_tool | 5 | 100.0% | 100 |

## 按难度分组

| 难度 | 拒答准确率 | 错误接受率 | 错误拒绝率 |
|---|---:|---:|---:|
| easy | 100.0% | 0.0% | 0.0% |

## 失败样本

| case_id | 类型 | 说明 |
|---|---|---|
| mock_case_001 | hallucinated_citation | Hallucinated citation: mock-citation-1 |
| mock_case_002 | hallucinated_citation | Hallucinated citation: mock-citation-1 |
| mock_case_003 | hallucinated_citation | Hallucinated citation: mock-citation-1 |
| mock_case_004 | hallucinated_citation | Hallucinated citation: mock-citation-1 |
| mock_case_005 | hallucinated_citation | Hallucinated citation: mock-citation-1 |

## 改进建议

- [P0] 引用有效性不足 (0.0% < 100.0%)，需检查检索管道与引用映射逻辑
- [P0] 引用幻觉率偏高 (100.0% > 5.0%)，需加强引用来源校验
- [P2] Recall@5 偏低 (0.0% < 50.0%)，建议优化检索查询或增加召回数量
