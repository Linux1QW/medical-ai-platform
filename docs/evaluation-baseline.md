# 评估系统基线记录（Task 0 冻结）

> 冻结日期：2026-08-01  
> 提交：24e89c8（fix(eval-regression): 小样本报告不作为 pre-push 门禁）  
> 目的：固定当前行为语义，后续迭代不得破坏本文档记录的不变量。

---

## 1. pre-push 门禁行为

| 场景 | 退出码 | pre-push 行为 |
|------|--------|--------------|
| 3 例冒烟报告（< min_cases=18） | 2 (SKIP) | **放行** |
| 17 例报告（< 18） | 2 (SKIP) | **放行** |
| 无报告 / 报告路径不存在 | 2 (SKIP) | **放行** |
| 18+ 例报告，全部指标 PASS | 0 (PASS) | **放行** |
| 18+ 例报告，存在指标 FAIL | 1 (FAIL) | **拦截 push** |
| 基础设施异常（非 0/1/2） | 其它 | **放行**（钩子兜底 exit 0） |

- 门控配置：`evaluation/patient_ab_thresholds.json` → `_gate.min_cases = 18`
- 钩子源：`scripts/hooks/pre-push`，仅 `$CODE -eq 1` 时 exit 1
- 临时跳过：`SKIP_EVAL_REGRESSION=1 git push`

## 2. 阈值配置

| 臂 | 指标 | 约束 | 阈值 | 基线均值 |
|----|------|------|------|----------|
| agent_ledger | disclosure_rate | min | 0.45 | 0.5009 |
| agent_ledger | judge_overall_avg | min | 4.5 | 4.9486 |
| agent_tool | disclosure_rate | min | 0.40 | 0.5287 |
| agent_tool | judge_overall_avg | min | 4.4 | 4.8708 |
| agent_tool | tool_degrade_rate | max | 0.30 | 0.0 |

基线报告：`ab_20260731_171321.json`（qwen3.7-plus，18 例分层抽样，--judge，0 失败）

## 3. 五维分数语义

| 状态 | total_score | 含义 | 前端渲染 |
|------|-------------|------|----------|
| completed | float (0-100) | 正常完成 | 显示分数 |
| needs_review | **None** | 安全/反思/证据不足触发复核 | 不得渲染为 0 |
| insufficient (维度级) | score=None | 该维度数据不足无法评分 | 不得渲染为 0 |
| error (维度级) | score=None | Agent 执行异常 | 不得渲染为 0 |

**不变量**：
- `ScoreCalculator` 只对 `status="scored"` 的维度加权，None 维度不参与、不重分配权重
- `finalize_needs_review` 节点显式设置 `"total_score": None`
- 缺失必需维度时 total_score = None，不降级为 0

## 4. RAG / Citation 字段

### AgentResultEnvelope 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| agent_name | str | — | Agent 标识 |
| status | str | — | success / error / skipped |
| score | float \| None | None | 维度分数 |
| analysis | str | "" | 分析文本 |
| skip_reason | str \| None | None | 跳过原因 |
| human_review_needed | bool | False | 是否需人工复核 |
| review_reason | str \| None | None | 复核原因 |
| citations | list | [] | 引用列表 |
| trace | dict | {} | 执行追踪（含 rag_trace） |

### 检索状态标签

- `sufficient`：检索充分
- `insufficient`：检索不足
- `error`：检索异常

## 5. 人工复核字段

### Evaluation ORM 模型

- `human_review_needed: bool`
- `review_reason: str | None`

### 触发路径

1. Safety Gate → high/undetermined → `finalize_needs_review`
2. Reflection Agent → `needs_review=True` → `review_gate_router` → `finalize_needs_review`
3. Knowledge Agent → evidence insufficient → `human_review_needed=True`

## 6. 图节点清单（15 个）

| 节点 | 波次 | 功能 |
|------|------|------|
| load_context | — | 上下文加载 |
| classify_consultation | — | 问诊类型分类 |
| safety_check | — | 安全门控 |
| plan_evaluation | — | 生成评估计划 |
| validate_plan | — | 校验计划 |
| run_agent_wave1 | Wave 1 | knowledge/inquiry/humanistic 并行 |
| extract_knowledge_citations | 桥接 | 提取 knowledge 证据给 Wave 2 |
| run_agent | Wave 2 | diagnosis/treatment 并行（携带 citations） |
| dispatch_and_run | 兼容 | 旧版 asyncio.gather 路径 |
| aggregate_results | — | Fan-in 汇聚 |
| deterministic_scoring | — | 确定性加权评分 |
| reflection_check | — | ReAct 反思验证 |
| review_gate_node | — | 复核门控 |
| generate_suggestion | — | 建议生成 |
| finalize_completed | — | 正常完成 |
| finalize_needs_review | — | 复核终止 |

## 7. 测试基线

| 范围 | 结果 | 日期 |
|------|------|------|
| tests/evaluation + tests/orchestration | **265 passed, 2 skipped** | 2026-08-01 |
| test_iteration_baseline.py | **16 passed** | 2026-08-01 |
| eval_regression.py（最新 3 例报告） | exit=2 (SKIP) | 2026-08-01 |

## 8. 已知限制（后续 Task 解决）

| 限制 | 对应 Task |
|------|-----------|
| 报告无 manifest / 版本绑定 | Task 1 |
| 门禁依赖文件名字典序选报告 | Task 2 |
| 五维只有单分数，无原子 rubric | Task 3 |
| Judge 无重复性 / 位置偏差校准 | Task 4 |
| 安全规则无结构化 RiskFinding | Task 5 |
| 复核无状态机，可覆盖原结果 | Task 6 |
| citation ID 依赖列表 index | Task 7 |
| 无 Claim-Evidence 链路 | Task 8 |
