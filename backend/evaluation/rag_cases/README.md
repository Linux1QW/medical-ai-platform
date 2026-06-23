# RAG Gold Cases

This directory contains gold standard test cases for evaluating RAG (Retrieval-Augmented Generation) performance.

## File Structure

- `rag_gold_cases.jsonl`: JSON Lines file containing gold standard test cases for RAG evaluation

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