"""
Unit tests for dataset handling.
"""
import unittest
import tempfile
import json
from pathlib import Path
from backend.evaluation.datasets import RagGoldCase, RagEvalResult, load_gold_cases, save_gold_cases, StanceType


class TestDatasets(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_cases_path = Path(self.temp_dir) / "test_cases.jsonl"
    
    def test_rag_gold_case_creation(self):
        """Test creation of RagGoldCase instances."""
        case = RagGoldCase(
            case_id="test_case_001",
            split="dev",
            department="内科",
            domain_expertise="高血压",
            difficulty="easy",
            chief_complaint="头晕头痛2天",
            patient_info="患者，男，55岁，头晕头痛2天",
            conversation_text="医生: 您哪里不舒服？\n患者: 最近总是头晕头痛",
            doctor_diagnosis="高血压病",
            treatment_plan="建议降压药物治疗，监测血压",
            gold_queries=["高血压 诊断 治疗 指南"],
            gold_doc_ids=["htn-guidelines-2023"],
            gold_citation_ids=["htn-cit-001"],
            gold_relevant_sources=["高血压防治指南"],
            gold_citation_keywords=["高血压", "降压药", "血压监测"],
            gold_relevance_grades={"htn-guidelines-2023": 3},
            expected_stance=StanceType.SUPPORTS,
            should_refuse=False,
            expected_score_range=[80, 95],
            notes="标准高血压诊断案例",
            expected_tool_calls=[{"name": "search_tool", "params": {"query": "hypertension"}}],
            expected_tool_params={"search_tool": {"param": "value"}},
            expected_final_answer_keywords=["高血压", "治疗", "监测"]
        )
        
        self.assertEqual(case.case_id, "test_case_001")
        self.assertEqual(case.department, "内科")
        self.assertEqual(case.expected_stance, StanceType.SUPPORTS)
        self.assertFalse(case.should_refuse)
        self.assertEqual(case.expected_score_range, [80, 95])
        self.assertEqual(len(case.expected_tool_calls), 1)
        self.assertIn("高血压", case.expected_final_answer_keywords)
    
    def test_rag_eval_result_creation(self):
        """Test creation of RagEvalResult instances."""
        result = RagEvalResult(
            case_id="test_case_001",
            mode="tooluse",
            knowledge_score=88.5,
            evaluation_status="completed",
            human_review_needed=False,
            review_reason=None,
            retrieval_status="sufficient",
            evidence_stance=StanceType.SUPPORTS,
            citation_data=[{"id": "cit1", "text": "test citation"}],
            rag_trace_data={"query": "test query"},
            tool_trace=[{"name": "search", "status": "success"}],
            latency_ms=1250,
            error=None,
            actual_tool_calls=[{"name": "search_tool", "result": "result"}],
            final_answer_text="This is the final answer"
        )
        
        self.assertEqual(result.case_id, "test_case_001")
        self.assertEqual(result.knowledge_score, 88.5)
        self.assertEqual(result.evidence_stance, StanceType.SUPPORTS)
        self.assertEqual(result.latency_ms, 1250)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.actual_tool_calls), 1)
        self.assertIn("final", result.final_answer_text)
    
    def test_save_and_load_gold_cases(self):
        """Test saving and loading gold cases."""
        # Create test cases
        original_cases = [
            RagGoldCase(
                case_id="test_case_001",
                split="dev",
                department="内科", 
                difficulty="easy",
                patient_info="Patient info",
                conversation_text="Conversation",
                expected_stance=StanceType.SUPPORTS,
                should_refuse=False,
                expected_tool_calls=[],
                expected_tool_params={},
                expected_final_answer_keywords=[]
            ),
            RagGoldCase(
                case_id="test_case_002",
                split="test",
                department="外科",
                difficulty="medium", 
                patient_info="Another patient",
                conversation_text="More conversation",
                expected_stance=StanceType.CONTRADICTS,
                should_refuse=True,
                expected_tool_calls=[],
                expected_tool_params={},
                expected_final_answer_keywords=[]
            )
        ]
        
        # Save cases
        save_gold_cases(original_cases, self.test_cases_path)
        
        # Verify file was created and has correct content
        self.assertTrue(self.test_cases_path.exists())
        
        lines = self.test_cases_path.read_text(encoding='utf-8').strip().split('\n')
        self.assertEqual(len(lines), 2)
        
        # Load cases back
        loaded_cases = load_gold_cases(self.test_cases_path)
        
        # Verify loaded cases match original
        self.assertEqual(len(loaded_cases), 2)
        self.assertEqual(loaded_cases[0].case_id, "test_case_001")
        self.assertEqual(loaded_cases[0].department, "内科")
        self.assertEqual(loaded_cases[0].expected_stance, StanceType.SUPPORTS)
        self.assertFalse(loaded_cases[0].should_refuse)
        
        self.assertEqual(loaded_cases[1].case_id, "test_case_002")
        self.assertEqual(loaded_cases[1].department, "外科")
        self.assertEqual(loaded_cases[1].expected_stance, StanceType.CONTRADICTS)
        self.assertTrue(loaded_cases[1].should_refuse)
    
    def test_load_gold_cases_file_not_found(self):
        """Test loading gold cases from non-existent file."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.jsonl"
        
        with self.assertRaises(FileNotFoundError):
            load_gold_cases(nonexistent_path)
    
    def test_load_gold_cases_invalid_json(self):
        """Test loading gold cases with invalid JSON."""
        # Create a file with invalid JSON
        invalid_json_path = Path(self.temp_dir) / "invalid.jsonl"
        with open(invalid_json_path, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }\n')
        
        with self.assertRaises(json.JSONDecodeError):
            load_gold_cases(invalid_json_path)


if __name__ == '__main__':
    unittest.main()