-- Migration v4: RAG 审计字段（幂等版本）
-- 可重复执行，已存在的列会自动跳过

DELIMITER $$
DROP PROCEDURE IF EXISTS migrate_v4$$
CREATE PROCEDURE migrate_v4()
BEGIN
    -- citation_data
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='citation_data') THEN
        ALTER TABLE evaluations ADD COLUMN citation_data JSON NULL;
    END IF;

    -- retrieval_status
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='retrieval_status') THEN
        ALTER TABLE evaluations ADD COLUMN retrieval_status VARCHAR(20) NOT NULL DEFAULT 'not_run';
    END IF;

    -- evidence_stance
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='evidence_stance') THEN
        ALTER TABLE evaluations ADD COLUMN evidence_stance VARCHAR(20) NOT NULL DEFAULT 'undetermined';
    END IF;

    -- human_review_needed
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='human_review_needed') THEN
        ALTER TABLE evaluations ADD COLUMN human_review_needed BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;

    -- review_reason
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='review_reason') THEN
        ALTER TABLE evaluations ADD COLUMN review_reason TEXT NULL;
    END IF;

    -- rag_trace_data
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='rag_trace_data') THEN
        ALTER TABLE evaluations ADD COLUMN rag_trace_data JSON NULL;
    END IF;

    -- evaluation_status
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='evaluations' AND column_name='evaluation_status') THEN
        ALTER TABLE evaluations ADD COLUMN evaluation_status VARCHAR(20) NOT NULL DEFAULT 'completed';
    END IF;

    -- 允许 knowledge_score 和 total_score 为 NULL，并去掉默认值
    ALTER TABLE evaluations MODIFY COLUMN knowledge_score FLOAT NULL DEFAULT NULL;
    ALTER TABLE evaluations MODIFY COLUMN total_score FLOAT NULL DEFAULT NULL;
END$$
DELIMITER ;

CALL migrate_v4();
DROP PROCEDURE IF EXISTS migrate_v4;
