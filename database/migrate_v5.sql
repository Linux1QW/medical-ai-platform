-- migrate_v5.sql: LangGraph 编排支持
-- 新增 consultation_type、evaluation_runs、evaluation_node_results
-- 扩展 evaluations 审计字段

USE medical_ai;

-- 1. consultations 增加 consultation_type
ALTER TABLE consultations
ADD COLUMN consultation_type VARCHAR(20) NOT NULL DEFAULT 'initial'
COMMENT '问诊类型: initial/follow_up/communication';

-- 2. 新增 evaluation_runs 表
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id CHAR(36) PRIMARY KEY COMMENT 'run_id (UUID)',
    consultation_id INT NOT NULL,
    evaluation_id INT NULL,
    graph_version VARCHAR(50) NOT NULL DEFAULT 'evaluation-graph-v1',
    scoring_policy_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    checkpoint_thread_id VARCHAR(100) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'running'
        COMMENT 'running/completed/failed/needs_review',
    selected_agents JSON NULL
        COMMENT '本次运行选中的Agent列表',
    attempt INT NOT NULL DEFAULT 1,
    error_type VARCHAR(100) NULL,
    error_message TEXT NULL,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_consultation_id (consultation_id),
    INDEX idx_status (status),
    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. 新增 evaluation_node_results 表
CREATE TABLE IF NOT EXISTS evaluation_node_results (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id CHAR(36) NOT NULL,
    node_name VARCHAR(50) NOT NULL,
    attempt INT NOT NULL DEFAULT 1,
    status VARCHAR(20) NOT NULL
        COMMENT 'success/skipped/error/insufficient',
    duration_ms INT NULL,
    result_summary JSON NULL
        COMMENT '脱敏摘要：状态、分数、关键标志',
    error_type VARCHAR(100) NULL,
    started_at DATETIME NULL,
    finished_at DATETIME NULL,
    INDEX idx_run_id (run_id),
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. evaluations 增加审计字段
ALTER TABLE evaluations
ADD COLUMN run_id CHAR(36) NULL COMMENT '关联的评估运行ID',
ADD COLUMN safety_data JSON NULL COMMENT 'Safety检查结果',
ADD COLUMN applicable_dimensions JSON NULL COMMENT '适用维度列表',
ADD COLUMN scoring_policy_version VARCHAR(50) NULL COMMENT '评分策略版本',
ADD COLUMN graph_version VARCHAR(50) NULL COMMENT '编排图版本';
