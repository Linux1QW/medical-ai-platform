-- 评估防重复提交 — 评估锁表
CREATE TABLE IF NOT EXISTS evaluation_locks (
    consultation_id INTEGER PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    run_id VARCHAR(36),
    locked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    heartbeat_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    error_message TEXT,
    INDEX idx_status (status),
    INDEX idx_expires_at (expires_at)
);
