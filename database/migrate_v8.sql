-- Migration v8: 人工复核记录表 + 评估检查点表
-- 支持 LangGraph checkpoint 恢复与教师复核流程

DELIMITER $$
DROP PROCEDURE IF EXISTS migrate_v8$$
CREATE PROCEDURE migrate_v8()
BEGIN
    -- 人工复核记录表
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='review_records') THEN
        CREATE TABLE review_records (
            id VARCHAR(36) PRIMARY KEY,
            evaluation_id VARCHAR(36) NOT NULL,
            reviewer_id VARCHAR(36) NOT NULL,
            feedback TEXT NOT NULL,
            review_reason VARCHAR(255),
            score_adjustments JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_review_evaluation_id (evaluation_id),
            INDEX idx_review_reviewer_id (reviewer_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    END IF;

    -- 评估检查点表（用于 LangGraph checkpoint 持久化）
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='evaluation_checkpoints') THEN
        CREATE TABLE evaluation_checkpoints (
            id VARCHAR(36) PRIMARY KEY,
            evaluation_id VARCHAR(36) NOT NULL UNIQUE,
            state_json JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_checkpoint_evaluation_id (evaluation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    END IF;

    -- 为 evaluations 表添加复核完成相关字段
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='review_completed_by') THEN
        ALTER TABLE evaluations ADD COLUMN review_completed_by VARCHAR(36) NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='review_completed_at') THEN
        ALTER TABLE evaluations ADD COLUMN review_completed_at DATETIME NULL;
    END IF;
END$$
DELIMITER ;

CALL migrate_v8();
DROP PROCEDURE IF EXISTS migrate_v8;
