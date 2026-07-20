# MySQL 自动备份配置说明

## 备份脚本

`backup_db.sh` 提供 MySQL 全量自动备份能力：

- 使用 `mysqldump --single-transaction` 实现 InnoDB 一致性快照（不锁表）
- 备份内容包含：表数据、存储过程（`--routines`）、触发器（`--triggers`）
- 输出文件经 `gzip` 压缩，节省磁盘空间
- 自动清理超过 7 天的旧备份

## 手动执行

```bash
# 设置环境变量（或从 .env 文件加载）
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD="your_password"
export MYSQL_DATABASE=medical_ai

# 执行备份（默认备份到 ./backups）
bash backend/scripts/backup_db.sh

# 指定备份目录
bash backend/scripts/backup_db.sh /data/mysql_backups
```

## Cron 定时备份（Linux）

### 1. 编辑 crontab

```bash
crontab -e
```

### 2. 添加定时任务

```cron
# 每天凌晨 2:00 执行备份
0 2 * * * cd /path/to/medical-ai-platform && MYSQL_PASSWORD="your_password" bash backend/scripts/backup_db.sh /data/mysql_backups >> /var/log/medical-ai-backup.log 2>&1
```

### 3. 使用环境变量文件（推荐）

创建 `/etc/medical-ai/backup.env`：

```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=medical_ai
```

crontab 中加载：

```cron
0 2 * * * . /etc/medical-ai/backup.env && cd /path/to/medical-ai-platform && bash backend/scripts/backup_db.sh /data/mysql_backups >> /var/log/medical-ai-backup.log 2>&1
```

### 4. Docker 环境下的备份

若 MySQL 运行在 Docker 容器中，可在宿主机通过端口映射连接：

```bash
# 通过 docker exec 直接在容器内执行（无需暴露端口）
docker exec medical-ai-mysql mysqldump \
  -u root -p"$MYSQL_PASSWORD" \
  --single-transaction --routines --triggers \
  medical_ai | gzip > /data/backups/medical_ai_$(date +%Y%m%d_%H%M%S).sql.gz
```

## 备份恢复

```bash
# 解压并恢复
gunzip < /data/backups/medical_ai_20250101_020000.sql.gz | \
  mysql -h localhost -u root -p"$MYSQL_PASSWORD" medical_ai
```

## 备份保留策略

- 默认保留最近 **7 天**的备份
- 修改脚本中 `RETENTION_DAYS` 变量可调整保留天数
- 建议生产环境保留 30 天，并定期将备份归档至对象存储（如 S3/OSS）

## 注意事项

- 确保执行备份的用户对 `BACKUP_DIR` 有写权限
- 生产环境建议将备份目录挂载到独立磁盘或网络存储
- 定期检查备份文件完整性（可尝试解压验证）
