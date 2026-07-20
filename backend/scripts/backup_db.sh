#!/bin/bash
# ============================================
# MySQL 自动备份脚本
# ============================================
# 用法: ./backup_db.sh [backup_dir]
#
# 功能:
#   - 全量备份（mysqldump + gzip 压缩）
#   - 自动清理超过 RETENTION_DAYS 天的旧备份
#   - 从环境变量读取数据库配置，支持 .env 文件
#
# 环境变量（可选，有默认值）:
#   MYSQL_HOST     - 数据库主机（默认: localhost）
#   MYSQL_PORT     - 数据库端口（默认: 3306）
#   MYSQL_USER     - 数据库用户（默认: root）
#   MYSQL_PASSWORD - 数据库密码（必填）
#   MYSQL_DATABASE - 数据库名称（默认: medical_ai）
# ============================================

set -euo pipefail

BACKUP_DIR=${1:-"./backups"}
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# 从环境变量读取数据库配置
DB_HOST=${MYSQL_HOST:-"localhost"}
DB_PORT=${MYSQL_PORT:-"3306"}
DB_USER=${MYSQL_USER:-"root"}
DB_PASS=${MYSQL_PASSWORD:-""}
DB_NAME=${MYSQL_DATABASE:-"medical_ai"}

# 校验必填参数
if [ -z "$DB_PASS" ]; then
  echo "[ERROR] MYSQL_PASSWORD 环境变量未设置，请先配置。"
  exit 1
fi

# 创建备份目录
mkdir -p "$BACKUP_DIR"

BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz"

echo "[INFO] 开始备份数据库: $DB_NAME @ $DB_HOST:$DB_PORT"
echo "[INFO] 备份文件: $BACKUP_FILE"

# 全量备份（含存储过程、触发器，单事务一致性快照）
mysqldump \
  -h "$DB_HOST" \
  -P "$DB_PORT" \
  -u "$DB_USER" \
  -p"$DB_PASS" \
  --single-transaction \
  --routines \
  --triggers \
  --set-gtid-purged=OFF \
  "$DB_NAME" | gzip > "$BACKUP_FILE"

# 校验备份文件是否生成成功
if [ ! -f "$BACKUP_FILE" ] || [ ! -s "$BACKUP_FILE" ]; then
  echo "[ERROR] 备份失败：文件未生成或为空。"
  exit 1
fi

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[INFO] 备份完成: $BACKUP_FILE (大小: $BACKUP_SIZE)"

# 清理过期备份
DELETED=$(find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -print -delete | wc -l)
echo "[INFO] 已清理 $DELETED 个超过 ${RETENTION_DAYS} 天的旧备份"

echo "[INFO] 备份任务全部完成。"
