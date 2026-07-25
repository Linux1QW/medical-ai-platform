-- Migration v5: Plan-Execute 模式字段（幂等版本）
-- 为 evaluation_runs 表添加评估计划和执行结果字段
-- 可重复执行，已存在的列会自动跳过

DELIMITER $$
DROP PROCEDURE IF EXISTS migrate_v5$$
CREATE PROCEDURE migrate_v5()
BEGIN
    -- evaluation_plan: 评估计划（Plan-Execute 模式）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluation_runs' AND column_name='evaluation_plan') THEN
        ALTER TABLE evaluation_runs ADD COLUMN evaluation_plan JSON NULL COMMENT '评估计划（Plan-Execute 模式）';
    END IF;

    -- execution_results: 计划步骤执行结果
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluation_runs' AND column_name='execution_results') THEN
        ALTER TABLE evaluation_runs ADD COLUMN execution_results JSON NULL COMMENT '计划步骤执行结果';
    END IF;
END$$
DELIMITER ;

CALL migrate_v5();
DROP PROCEDURE IF EXISTS migrate_v5;
