# RAG Gold Cases

This directory contains gold standard test cases for evaluating RAG (Retrieval-Augmented Generation) performance.

## File Structure

- `rag_gold_cases.jsonl`: JSON Lines file containing gold standard test cases for RAG evaluation
- `dataset_gold_cases.jsonl`: 从根目录 `dataset/` 真实门诊病例转换的 regression 回归集（15 例），
  `gold_relevant_sources` 由规则表自动生成（notes 含 `gold=auto-suggested` 标记，欢迎人工修正）

## Case Format

Each line in the `rag_gold_cases.jsonl` file contains a JSON object with the following fields:

### Identification Fields
- `case_id`: Unique identifier for the test case
- `split`: Dataset split (dev/test/regression)
- `department`: Medical department
- `domain_expertise`: Specific medical domain expertise
- `difficulty`: Difficulty level (easy/medium/hard)

### Case Information Fields
- `chief_complaint`: Patient's main complaint
- `patient_info`: Patient information
- `conversation_text`: Doctor-patient conversation
- `doctor_diagnosis`: Doctor's diagnosis
- `treatment_plan`: Doctor's treatment plan

### Expected Retrieval Results
- `gold_queries`: Expected queries for retrieval
- `gold_doc_ids`: Expected document IDs
- `gold_citation_ids`: Expected citation IDs
- `gold_relevant_sources`: Expected sources
- `gold_citation_keywords`: Expected keywords in citations
- `gold_relevance_grades`: Relevance grades for documents

### Expected Evaluation Results
- `expected_stance`: Expected stance (supports/contradicts/mixed/undetermined)
- `should_refuse`: Whether the system should refuse to answer
- `expected_score_range`: Expected score range
- `expected_review_reason`: Expected reason for review

### Tool Use Expectations
- `expected_tool_calls`: Expected tool calls
- `expected_tool_params`: Expected tool parameters
- `expected_final_answer_keywords`: Expected keywords in final answer

### Metadata
- `notes`: Additional notes about the case

## Usage

These cases are used by the RAG evaluation system to measure:
- Retrieval accuracy
- Citation validity
- Refusal accuracy
- Tool use effectiveness
- Overall RAG performance

## 真实病例回归集（dataset_gold_cases.jsonl）

### 重新生成

从 `dataset/` 转换 regression split 并自动补 gold 来源建议（幂等，已有标注不覆盖）：

```bash
cd backend
python -m evaluation.rag_eval --from-dataset \
    --export-cases evaluation/rag_cases/dataset_gold_cases.jsonl \
    --export-split regression --bootstrap-gold
```

### 本地跑真实回归

需要可用的 DASHSCOPE embedding/LLM 额度与已建好的向量知识库：

```bash
cd backend
python -m evaluation.rag_eval --cases evaluation/rag_cases/dataset_gold_cases.jsonl \
    --split regression --mode legacy --output-dir evaluation/reports
```

关注指标 `source_hit_rate`：citation 来源文件名包含任一 `gold_relevant_sources`
子串即算命中（与 `scripts/eval/golden_set.json` 的 `relevant_source_contains` 口径一致）。
真实链路的 `rag_trace` 不含 `retrieved_docs`，因此 recall@k / MRR 仅在 mock 模式有值。

### CI 阻断门

CI 的 backend-test 作业中运行 `--mode mock --fail-on-threshold`：mock 结果与
gold 期望对齐（绿路径），任何使指标管道/阈值配置失灵的改动都会让该步骤失败。