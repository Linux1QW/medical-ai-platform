-- migrate_v6.sql: 审计日志表（幂等迁移）
-- 执行方式: mysql -u root -p medical_ai < database/migrate_v6.sql

USE medical_ai;

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL COMMENT '操作用户ID',
    action ENUM('login', 'create_consultation', 'submit_diagnosis', 'trigger_evaluation', 'admin_action') NOT NULL COMMENT '操作类型',
    resource_id VARCHAR(50) NULL COMMENT '关联资源ID',
    ip_address VARCHAR(45) NULL COMMENT '客户端IP',
    user_agent VARCHAR(500) NULL COMMENT '客户端UA',
    detail TEXT NULL COMMENT '操作详情（脱敏）',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_user_id (user_id),
    INDEX idx_audit_action (action),
    INDEX idx_audit_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
