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

-- 用户细粒度权限字段（兼容 MySQL 8.0）
DROP PROCEDURE IF EXISTS _add_permissions_column;
DELIMITER //
CREATE PROCEDURE _add_permissions_column()
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'users'
        AND COLUMN_NAME = 'permissions'
    ) THEN
        ALTER TABLE users ADD COLUMN permissions JSON DEFAULT NULL;
    END IF;
END //
DELIMITER ;
CALL _add_permissions_column();
DROP PROCEDURE IF EXISTS _add_permissions_column;
