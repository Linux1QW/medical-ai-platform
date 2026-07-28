"""
Unit tests for dataset/ real case conversion (dataset_cases).
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset_cases import (
    DEFAULT_DATASET_DIR,
    convert_patient_case,
    load_dataset_cases,
)


def _make_case_data(rounds: int = 2) -> dict:
    """构造 dataset 主 JSON 的最小结构（中文键名，含零宽字符）"""
    return {
        "基础信息": {"姓名": "任xx", "性别": "女", "年龄": "29"},
        "门诊病历": {
            "主诉": "看检查结果",
            "现病史": "大便不成形1年",
            "既往史": "既往体健",
            "药物过敏史": "无",
            "处理": "无",
        },
        "主诊断": "慢性胃炎",
        "处方单1(西成方）": [
            {
                "药品名称": "米曲菌胰酶片",
                "给药方式": "口服",
                "频次": "TID",
                "天数": "7",
            }
        ],
        "门诊对话": [
            {"轮次": i + 1, "医生": f"问题{i + 1}？\u200b", "患者": f"回答{i + 1}。\u200b"}
            for i in range(rounds)
        ],
    }


def _write_case(root: Path, dirname: str, data: dict) -> Path:
    case_dir = root / dirname
    case_dir.mkdir(parents=True)
    with open(case_dir / f"{dirname}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return case_dir


class TestConvertPatientCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_convert_full_fields(self):
        case_dir = _write_case(self.temp_dir, "patient1_5", _make_case_data())
        case = convert_patient_case(case_dir, split="dev")

        self.assertIsNotNone(case)
        self.assertEqual(case.case_id, "patient1_5")
        self.assertEqual(case.split, "dev")
        self.assertEqual(case.difficulty, "easy")
        self.assertEqual(case.chief_complaint, "看检查结果")
        self.assertEqual(case.doctor_diagnosis, "慢性胃炎")
        # 对话逐轮展开且零宽字符已清理
        self.assertIn("医生: 问题1？", case.conversation_text)
        self.assertIn("患者: 回答2。", case.conversation_text)
        self.assertNotIn("\u200b", case.conversation_text)
        # 患者描述：性别年龄 + 现病史（"无"的药物过敏史被过滤）
        self.assertIn("患者女，29岁", case.patient_info)
        self.assertIn("现病史：大便不成形1年", case.patient_info)
        self.assertNotIn("药物过敏史", case.patient_info)
        # 治疗方案：处方明细（"无"的处理被过滤）
        self.assertIn("米曲菌胰酶片 口服 TID 7天", case.treatment_plan)
        self.assertNotIn("处理：无", case.treatment_plan)
        # 默认期望值与 gold 留空
        self.assertEqual(case.expected_stance, "supports")
        self.assertFalse(case.should_refuse)
        self.assertEqual(case.gold_relevant_sources, [])

    def test_difficulty_mapping_by_suffix(self):
        for dirname, expected in (
            ("patient1_5", "easy"),
            ("patient2_10", "easy"),
            ("patient3_21", "medium"),
            ("patient4_26.5", "hard"),
            ("patient5_30", "hard"),
        ):
            case_dir = _write_case(self.temp_dir, dirname, _make_case_data())
            case = convert_patient_case(case_dir)
            self.assertIsNotNone(case, dirname)
            self.assertEqual(case.difficulty, expected, dirname)

    def test_skip_invalid_suffix(self):
        case_dir = _write_case(self.temp_dir, "patient_notes", _make_case_data())
        self.assertIsNone(convert_patient_case(case_dir))

    def test_skip_missing_main_json(self):
        case_dir = self.temp_dir / "patient9_9"
        case_dir.mkdir()
        # 仅有人格变体文件，无主 JSON
        with open(case_dir / "patient9_9_合作_平稳_高_高_中_正常.json", "w", encoding="utf-8") as f:
            json.dump(_make_case_data(), f, ensure_ascii=False)
        self.assertIsNone(convert_patient_case(case_dir))

    def test_skip_empty_dialogue(self):
        data = _make_case_data()
        data["门诊对话"] = []
        case_dir = _write_case(self.temp_dir, "patient8_8", data)
        self.assertIsNone(convert_patient_case(case_dir))

    def test_skip_broken_json(self):
        case_dir = self.temp_dir / "patient7_7"
        case_dir.mkdir()
        (case_dir / "patient7_7.json").write_text("{not valid json", encoding="utf-8")
        self.assertIsNone(convert_patient_case(case_dir))


class TestLoadDatasetCases(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset_cases(self.temp_dir / "not_exist")

    def test_split_assignment_deterministic(self):
        # 10 个有效病例：7 dev / 2 test / 1 regression，按目录名排序确定性划分
        for i in range(10):
            _write_case(self.temp_dir, f"patient{i:02d}_5", _make_case_data())
        cases = load_dataset_cases(self.temp_dir)

        self.assertEqual(len(cases), 10)
        splits = [c.split for c in cases]
        self.assertEqual(splits.count("dev"), 7)
        self.assertEqual(splits.count("test"), 2)
        self.assertEqual(splits.count("regression"), 1)
        # 二次加载结果一致
        cases_again = load_dataset_cases(self.temp_dir)
        self.assertEqual([(c.case_id, c.split) for c in cases],
                         [(c.case_id, c.split) for c in cases_again])

    def test_invalid_dirs_skipped(self):
        _write_case(self.temp_dir, "patient1_5", _make_case_data())
        _write_case(self.temp_dir, "readme_dir", _make_case_data())  # 后缀非法
        empty = _make_case_data()
        empty["门诊对话"] = []
        _write_case(self.temp_dir, "patient2_6", empty)  # 空对话

        cases = load_dataset_cases(self.temp_dir)
        self.assertEqual([c.case_id for c in cases], ["patient1_5"])

    def test_limit(self):
        for i in range(5):
            _write_case(self.temp_dir, f"patient{i}_5", _make_case_data())
        cases = load_dataset_cases(self.temp_dir, limit=3)
        self.assertEqual(len(cases), 3)


@unittest.skipUnless(DEFAULT_DATASET_DIR.exists(), "repo dataset/ not present")
class TestRealDatasetIntegration(unittest.TestCase):
    """真实 dataset/ 目录冒烟：150+ 病例可全部通过 RagGoldCase 校验"""

    def test_load_real_dataset(self):
        cases = load_dataset_cases()
        self.assertGreaterEqual(len(cases), 100)
        for case in cases:
            self.assertTrue(case.conversation_text)
            self.assertNotIn("\u200b", case.conversation_text)
            self.assertIn(case.difficulty, ("easy", "medium", "hard"))
            self.assertIn(case.split, ("dev", "test", "regression"))


if __name__ == "__main__":
    unittest.main()
