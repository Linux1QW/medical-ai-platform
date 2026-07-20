-- ============================================
-- migrate_v10: 模型版本注册表 + 用户细粒度权限
-- ============================================

-- 模型版本注册表
CREATE TABLE IF NOT EXISTS model_versions (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    config_json JSON,
    status ENUM('active', 'inactive', 'deprecated') DEFAULT 'active',
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_name_version (name, version)
);

-- 用户细粒度权限字段
ALTER TABLE users ADD COLUMN IF NOT EXISTS permissions JSON DEFAULT NULL;
