# -*- coding: utf-8 -*-
"""RAG 离线评测框架

用法：
    python -m backend.tests.rag.eval_offline --input test_cases.json --output results.json

此脚本用于评估 RAG 检索管线的质量，对比不同配置下的召回效果。
需要预先准备的测试用例文件（test_cases.json）。
"""

import argparse
import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


class OfflineRAGEvaluator:
    """RAG 离线评测器"""

    def __init__(self):
        self.results = []

    async def evaluate_single(
        self,
        case: dict,
    ) -> dict:
        """评估单个测试用例"""
        from app.services.agents.knowledge_agent import build_queries
        from app.services.rag.types import ClinicalFacts

        # 构建查询
        facts = ClinicalFacts(
            age=case.get("age"),
            gender=case.get("gender"),
            chief_complaint=case.get("chief_complaint", ""),
            symptoms=case.get("symptoms", []),
            timeline=case.get("timeline", []),
            red_flags=case.get("red_flags", []),
            comorbidities=case.get("comorbidities", []),
            medications=case.get("medications", []),
            allergies=case.get("allergies", []),
            doctor_diagnoses=case.get("doctor_diagnoses", []),
            treatment_items=case.get("treatment_items", []),
        )

        queries = build_queries(facts)

        start = time.perf_counter()

        # 尝试检索（不强制要求成功，评测就是在真实环境下跑）
        try:
            from app.services.rag.retriever import tiered_retrieve
            bundle = await tiered_retrieve(
                queries=queries,
                top_k_per_query=10,
                candidate_limit=20,
            )
            retrieval_ms = (time.perf_counter() - start) * 1000

            result = {
                "case_id": case.get("id", "unknown"),
                "query_count": len(queries),
                "query_types": [q.query_type for q in queries],
                "retrieval_level": bundle.level_used,
                "candidate_count": len(bundle.candidates),
                "retrieval_status": bundle.status,
                "retrieval_ms": round(retrieval_ms, 1),
                "degraded": bundle.degraded,
                "sources": list(set(c.source for c in bundle.candidates)),
                "unique_sources": len(set(c.source for c in bundle.candidates)),
            }

        except Exception as e:
            retrieval_ms = (time.perf_counter() - start) * 1000
            result = {
                "case_id": case.get("id", "unknown"),
                "error": str(e),
                "retrieval_ms": round(retrieval_ms, 1),
            }

        self.results.append(result)
        return result

    def summary(self) -> dict:
        """生成评测汇总"""
        if not self.results:
            return {"total": 0}

        total = len(self.results)
        errors = sum(1 for r in self.results if "error" in r)
        success = total - errors

        levels = {}
        statuses = {}
        candidate_counts = []
        retrieval_times = []

        for r in self.results:
            if "error" not in r:
                level = r.get("retrieval_level", "unknown")
                levels[level] = levels.get(level, 0) + 1

                status = r.get("retrieval_status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1

                candidate_counts.append(r.get("candidate_count", 0))
                retrieval_times.append(r.get("retrieval_ms", 0))

        avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
        avg_retrieval_ms = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0

        return {
            "total": total,
            "success": success,
            "errors": errors,
            "success_rate": round(success / total, 3) if total > 0 else 0,
            "level_distribution": levels,
            "status_distribution": statuses,
            "avg_candidate_count": round(avg_candidates, 1),
            "avg_retrieval_ms": round(avg_retrieval_ms, 1),
            "p50_retrieval_ms": round(sorted(retrieval_times)[len(retrieval_times) // 2], 1) if retrieval_times else 0,
            "p95_retrieval_ms": round(sorted(retrieval_times)[int(len(retrieval_times) * 0.95)], 1) if retrieval_times else 0,
        }


async def main():
    parser = argparse.ArgumentParser(description="RAG 离线评测")
    parser.add_argument("--input", "-i", help="测试用例 JSON 文件路径")
    parser.add_argument("--output", "-o", help="结果输出文件路径")
    args = parser.parse_args()

    evaluator = OfflineRAGEvaluator()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            cases = json.load(f)
    else:
        # 内置示例测试用例
        cases = [
            {
                "id": "case_001",
                "age": 45,
                "gender": "男",
                "chief_complaint": "持续咳嗽两周",
                "symptoms": ["咳嗽", "发热", "胸闷", "乏力"],
                "timeline": ["2周"],
                "red_flags": ["咯血"],
                "comorbidities": ["高血压"],
                "medications": ["氨氯地平"],
                "allergies": [],
                "doctor_diagnoses": ["社区获得性肺炎"],
                "treatment_items": ["阿莫西林 0.5g tid 7天", "对症治疗"],
            },
            {
                "id": "case_002",
                "age": 32,
                "gender": "女",
                "chief_complaint": "反复头痛",
                "symptoms": ["头痛", "恶心", "畏光"],
                "timeline": ["3个月，近1周加重"],
                "red_flags": [],
                "comorbidities": [],
                "medications": [],
                "allergies": ["青霉素"],
                "doctor_diagnoses": ["偏头痛"],
                "treatment_items": ["布洛芬 400mg prn", "预防性用药"],
            },
        ]

    print(f"开始评测，共 {len(cases)} 个用例...")
    for case in cases:
        result = await evaluator.evaluate_single(case)
        status = result.get("retrieval_status", "error")
        count = result.get("candidate_count", 0)
        ms = result.get("retrieval_ms", 0)
        print(f"  [{case.get('id', '?')}] status={status}, candidates={count}, time={ms:.0f}ms")

    summary = evaluator.summary()
    print("\n=== 评测汇总 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        output = {"summary": summary, "results": evaluator.results}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
