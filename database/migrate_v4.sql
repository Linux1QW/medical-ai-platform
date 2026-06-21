-- Migration v4: RAG 审计字段
-- 支持拒答、引用追溯和人工复核

ALTER TABLE evaluations
    ADD COLUMN citation_data JSON NULL,
    ADD COLUMN retrieval_status VARCHAR(20)
        NOT NULL DEFAULT 'not_run',
    ADD COLUMN evidence_stance VARCHAR(20)
        NOT NULL DEFAULT 'undetermined',
    ADD COLUMN human_review_needed BOOLEAN
        NOT NULL DEFAULT FALSE,
    ADD COLUMN review_reason TEXT NULL,
    ADD COLUMN rag_trace_data JSON NULL,
    ADD COLUMN evaluation_status VARCHAR(20)
        NOT NULL DEFAULT 'completed';

-- 允许 knowledge_score 和 total_score 为 NULL，并去掉默认值
ALTER TABLE evaluations MODIFY COLUMN knowledge_score FLOAT NULL DEFAULT NULL;
ALTER TABLE evaluations MODIFY COLUMN total_score FLOAT NULL DEFAULT NULL;
