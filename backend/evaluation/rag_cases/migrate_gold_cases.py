"""
Migration script: Convert rag_gold_cases.jsonl from old format to new RagGoldCase schema.
"""
import json
import re
from pathlib import Path

CASES_FILE = Path(__file__).parent / "rag_gold_cases.jsonl"
OUTPUT_FILE = CASES_FILE  # overwrite in place

# Mapping tables
DIFFICULTY_MAP = {
    "简单": "easy",
    "中等": "medium",
    "困难": "hard",
}

STANCE_MAP = {
    "提供信息": "supports",
    "建议就医": "mixed",
    "拒绝回答": "contradicts",
}


# Question-start patterns that indicate the transition from patient description to question
QUESTION_STARTERS = [
    r'最可能的诊断',
    r'请分析',
    r'请问',
    r'如何处理',
    r'怎么处理',
    r'如何调整',
    r'下一步如何',
    r'需要哪些检查',
    r'需要注意',
    r'紧急处理',
    r'急诊如何处理',
    r'一线治疗如何选择',
    r'治疗方案',
    r'的用量',
    r'分别',
]


def split_query_into_fields(query: str):
    """
    Split a query into patient_info and conversation_text.
    If the query contains patient case description, extract patient_info.
    The remaining question part becomes conversation_text.
    """
    # Only attempt to split if query starts with patient case indicators
    if not re.match(r'^(患者|儿童|2型糖尿病)', query):
        return "", query
    
    # Try to find where the question part starts
    for starter in QUESTION_STARTERS:
        starter_match = re.search(re.escape(starter), query)
        if starter_match:
            pos = starter_match.start()
            prefix = query[:pos]
            # Prefer splitting at sentence boundary (。or ；) before the starter
            last_period = max(prefix.rfind('。'), prefix.rfind('；'))
            if last_period >= 0:
                patient_info = prefix[:last_period].strip()
                conversation = query[last_period+1:].strip()
                if patient_info and conversation:
                    return patient_info, conversation
            # Fallback: split at last comma before the starter
            last_comma = max(prefix.rfind('，'), prefix.rfind(','))
            if last_comma >= 0:
                patient_info = prefix[:last_comma].strip()
                conversation = query[last_comma+1:].strip()
                if patient_info and conversation:
                    return patient_info, conversation
    
    # Fallback: if query contains 。or ；, split there
    period_pattern = re.compile(r'^(.*?[。；])\s*(.*)', re.DOTALL)
    match = period_pattern.match(query)
    if match:
        patient_info = match.group(1).strip().rstrip('。').rstrip('；')
        conversation = match.group(2).strip()
        if conversation:
            return patient_info, conversation
    
    # No clear split point found
    return "", query


def determine_should_refuse(old_stance: str) -> bool:
    return old_stance == "拒绝回答"


# Split assignment: distribute cases across dev/test/regression
SPLIT_CYCLE = ["dev", "test", "regression"]


def convert_case(old: dict, case_index: int = 0) -> dict:
    """Convert a single old-format case to new RagGoldCase schema."""
    patient_info, conversation_text = split_query_into_fields(old.get("query", ""))
    
    old_stance = old.get("expected_stance", "提供信息")
    old_difficulty = old.get("difficulty", "中等")
    
    new_case = {
        # Identification
        "case_id": old.get("id", ""),
        "split": SPLIT_CYCLE[case_index % len(SPLIT_CYCLE)],
        "department": old.get("department", ""),
        "domain_expertise": None,
        "difficulty": DIFFICULTY_MAP.get(old_difficulty, "medium"),
        
        # Case information
        "chief_complaint": None,
        "patient_info": patient_info,
        "conversation_text": conversation_text,
        "doctor_diagnosis": None,
        "treatment_plan": None,
        
        # Expected retrieval results
        "gold_queries": [],
        "gold_doc_ids": [],
        "gold_citation_ids": [],
        "gold_relevant_sources": old.get("reference_docs", []),
        "gold_citation_keywords": [],
        "gold_relevance_grades": {},
        
        # Expected queries
        "expected_queries": [],
        
        # Expected evaluation results
        "expected_stance": STANCE_MAP.get(old_stance, "supports"),
        "should_refuse": determine_should_refuse(old_stance),
        "expected_score_range": None,
        "expected_review_reason": None,
        
        # Tool use expectations
        "expected_tool_calls": [],
        "expected_tool_params": {},
        "expected_final_answer_keywords": old.get("tags", []),
        
        # Metadata
        "notes": old.get("expected_answer", ""),
    }
    
    return new_case


def main():
    old_cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                old_cases.append(json.loads(line))
    
    print(f"Read {len(old_cases)} old-format cases")
    
    new_cases = [convert_case(c, i) for i, c in enumerate(old_cases)]
    
    # Write new format
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for case in new_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(new_cases)} new-format cases to {OUTPUT_FILE}")
    
    # Validate with Pydantic
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from evaluation.datasets import RagGoldCase
    
    errors = 0
    for i, case_data in enumerate(new_cases):
        try:
            validated = RagGoldCase(**case_data)
            print(f"  ✓ {validated.case_id}: difficulty={validated.difficulty}, stance={validated.expected_stance}, refuse={validated.should_refuse}")
        except Exception as e:
            errors += 1
            print(f"  ✗ Case {i+1} ({case_data.get('case_id')}): {e}")
    
    if errors:
        print(f"\n{errors} case(s) failed validation!")
        sys.exit(1)
    else:
        print(f"\nAll {len(new_cases)} cases passed Pydantic validation!")


if __name__ == "__main__":
    main()
