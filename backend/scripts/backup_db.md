# MySQL 加密自动备份配置说明 (V1.1)

## 备份脚本

`backup_db.sh` 提供 MySQL 全量加密备份能力：

- **流式加密管道**：`mysqldump → gzip → age`，不落地未加密 SQL
- **age 公钥加密**：生产主机仅持有 public recipient，不保存解密 identity
- **Off-host 上传**：通过 `rclone` 上传至远端存储，并校验远端 checksum/size
- **原子状态文件**：全部成功后原子写 `BACKUP_STATUS_FILE`（JSON 格式）
- **保留策略**：本地 7 天、off-host 30 天（可配置）
- **最小权限**：使用 `umask 077`，临时文件 `0600`，trap 清理

## 环境变量

### 必填

| 变量 | 说明 |
|------|------|
| `MYSQL_HOST` | 数据库主机（建议 `127.0.0.1`） |
| `MYSQL_USER` | 备份专用账号（仅 SELECT/SHOW VIEW/TRIGGER/LOCK TABLES/EXECUTE） |
| `MYSQL_PASSWORD` | 备份账号密码 |
| `BACKUP_AGE_RECIPIENT` | age 公钥 recipient（例：`age1...`） |
| `BACKUP_REMOTE_URI` | rclone 远端路径（例：`medical-ai-backup:production/mysql`） |
| `BACKUP_STATUS_FILE` | 状态 JSON 文件路径 |

### 可选

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_PORT` | `3306` | 数据库端口 |
| `MYSQL_DATABASE` | `medical_ai` | 数据库名称 |
| `BACKUP_LOCAL_RETENTION_DAYS` | `7` | 本地备份保留天数 |
| `BACKUP_OFFHOST_RETENTION_DAYS` | `30` | 远端备份保留天数 |

## 手动执行

```bash
# 加载环境变量
source /etc/medical-ai/backup.env

# 执行加密备份
bash backend/scripts/backup_db.sh /var/backups/medical-ai
```

## Systemd Timer（推荐）

仓库提供 systemd service/timer，每 12 小时自动执行一次。

### 安装

```bash
# 复制单元文件
sudo cp deploy/systemd/medical-ai-backup.service /etc/systemd/system/
sudo cp deploy/systemd/medical-ai-backup.timer /etc/systemd/system/

# 创建配置目录和环境文件
sudo mkdir -p /etc/medical-ai
sudo tee /etc/medical-ai/backup.env <<'EOF'
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=medical_backup
MYSQL_PASSWORD=<strong-password>
MYSQL_DATABASE=medical_ai
BACKUP_AGE_RECIPIENT=age1...
BACKUP_REMOTE_URI=medical-ai-backup:production/mysql
BACKUP_STATUS_FILE=/opt/medical-ai-platform/.deploy/backup-status/status.json
BACKUP_LOCAL_RETENTION_DAYS=7
BACKUP_OFFHOST_RETENTION_DAYS=30
EOF
sudo chmod 0600 /etc/medical-ai/backup.env

# 创建备份目录
sudo mkdir -p /var/backups/medical-ai
sudo chown medical-backup:medical-backup /var/backups/medical-ai

# 启用并启动 timer
sudo systemctl daemon-reload
sudo systemctl enable --now medical-ai-backup.timer
```

### 管理命令

```bash
# 查看 timer 状态
sudo systemctl status medical-ai-backup.timer

# 查看下次触发时间
systemctl list-timers medical-ai-backup.timer

# 手动触发一次备份
sudo systemctl start medical-ai-backup.service

# 查看备份日志
sudo journalctl -u medical-ai-backup.service --since today

# 禁用 timer
sudo systemctl disable --now medical-ai-backup.timer
```

## 备份恢复

恢复操作需要在**独立恢复主机**上执行（生产主机不持有 age identity）：

```bash
# 1. 从 off-host 下载最新备份
rclone copy medical-ai-backup:production/mysql/latest.sql.gz.age ./restore/

# 2. 使用 age identity 解密 + 解压 + 恢复
age -d -i /path/to/age-identity.txt ./restore/latest.sql.gz.age | \
  gunzip | \
  mysql -h localhost -u root -p medical_ai

# 3. 验证 Alembic revision
alembic current
```

## 备份保留策略

| 位置 | 保留天数 | 说明 |
|------|----------|------|
| 本地 (`/var/backups/medical-ai`) | 7 天 | 快速恢复用 |
| Off-host (rclone remote) | 30 天 | 灾难恢复用 |

## 安全注意事项

- **生产主机不持有 age identity**：只保存 public recipient，解密在独立恢复主机进行
- **备份账号最小权限**：仅 SELECT/SHOW VIEW/TRIGGER/LOCK TABLES/EXECUTE，禁止 INSERT/UPDATE/DDL
- **`/etc/medical-ai/backup.env`** 权限为 `0600`，属主为 `medical-backup`
- **季度恢复演练**：每季度在隔离环境验证备份可恢复性
- **凭据轮换**：定期轮换 `MYSQL_BACKUP_PASSWORD` 和 rclone 凭据
