-- 迁移脚本：从 v1 升级到 v2（新增诊断/治疗方案字段 + 五维度评估）
USE medical_ai;

-- 问诊表新增诊断和治疗方案字段
ALTER TABLE consultations
    ADD COLUMN diagnosis TEXT DEFAULT NULL COMMENT '医生提交的诊断结果' AFTER summary,
    ADD COLUMN treatment_plan TEXT DEFAULT NULL COMMENT '医生提交的治疗方案' AFTER diagnosis;

-- 评估表新增诊断和治疗方案评估字段
ALTER TABLE evaluations
    ADD COLUMN diagnosis_score FLOAT DEFAULT 0 COMMENT '诊断结果评分' AFTER humanistic_analysis,
    ADD COLUMN diagnosis_analysis TEXT COMMENT '诊断结果分析' AFTER diagnosis_score,
    ADD COLUMN treatment_score FLOAT DEFAULT 0 COMMENT '治疗方案评分' AFTER diagnosis_analysis,
    ADD COLUMN treatment_analysis TEXT COMMENT '治疗方案分析' AFTER treatment_score;
