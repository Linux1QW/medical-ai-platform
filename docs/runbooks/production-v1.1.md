# Production Runbook – Medical AI Platform V1.1

> **Scope:** Single-host controlled production with maintenance windows.  
> **Not** a high-availability architecture.

---

## 1. RTO / RPO Targets

| Scenario | Target | Notes |
|----------|--------|-------|
| Application rollback | RTO ≤ 15 min | Image swap only; no Alembic downgrade |
| Single-host disaster recovery | RTO ≤ 4 h | Full restore from off-host encrypted backup |
| Database disaster | RPO ≤ 24 h | Based on 12-hour backup cadence + margin |

> **If business requires lower RPO**, you must introduce managed MySQL/PITR or binlog replication. This runbook does not cover HA.

---

## 2. One-Time Account Setup

### 2.1 Create dedicated MySQL accounts

```sql
-- Application account (runtime only)
CREATE USER 'medical_app'@'127.0.0.1' IDENTIFIED BY '<strong-password-app>';
GRANT SELECT, INSERT, UPDATE, DELETE ON medical_ai.* TO 'medical_app'@'127.0.0.1';
GRANT EXECUTE ON medical_ai.* TO 'medical_app'@'127.0.0.1';

-- Migration account (used only during deploy)
CREATE USER 'medical_migration'@'127.0.0.1' IDENTIFIED BY '<strong-password-migration>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, INDEX ON medical_ai.* TO 'medical_migration'@'127.0.0.1';

-- Backup account (read-only, no DML)
CREATE USER 'medical_backup'@'127.0.0.1' IDENTIFIED BY '<strong-password-backup>';
GRANT SELECT, SHOW VIEW, TRIGGER, LOCK TABLES, EXECUTE ON medical_ai.* TO 'medical_backup'@'127.0.0.1';

FLUSH PRIVILEGES;
```

### 2.2 Verify minimal privileges

```sql
-- Backup user must NOT be able to write
-- Test (should fail):
-- mysql -h 127.0.0.1 -u medical_backup -p -e "INSERT INTO medical_ai.users VALUES (...)"
```

### 2.3 Create OS user for backup

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin medical-backup
sudo mkdir -p /var/backups/medical-ai
sudo chown medical-backup:medical-backup /var/backups/medical-ai
sudo chmod 0700 /var/backups/medical-ai
```

---

## 3. Credential Setup

### 3.1 age keypair (for backup encryption)

```bash
# Generate on a SEPARATE recovery host (NOT production)
age-keygen -o /path/to/age-identity.txt
# Extract public key
age-keygen -y /path/to/age-identity.txt
# Output: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- Store identity file on **recovery host only** or in organizational key vault
- Production `.env` stores ONLY the public recipient:
  ```
  BACKUP_AGE_RECIPIENT=age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
- Two authorized personnel manage the identity; verify quarterly

### 3.2 rclone configuration

```bash
# Configure on production host (as medical-backup user)
sudo -u medical-backup rclone config
# Follow prompts to configure "medical-ai-backup" remote
```

### 3.3 Backup environment file

```bash
sudo mkdir -p /etc/medical-ai
sudo tee /etc/medical-ai/backup.env <<'EOF'
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=medical_backup
MYSQL_PASSWORD=<backup-password>
MYSQL_DATABASE=medical_ai
BACKUP_AGE_RECIPIENT=age1...
BACKUP_REMOTE_URI=medical-ai-backup:production/mysql
BACKUP_STATUS_FILE=/opt/medical-ai-platform/.deploy/backup-status/status.json
BACKUP_LOCAL_RETENTION_DAYS=7
BACKUP_OFFHOST_RETENTION_DAYS=30
EOF
sudo chown medical-backup:medical-backup /etc/medical-ai/backup.env
sudo chmod 0600 /etc/medical-ai/backup.env
```

---

## 4. Systemd Backup Timer

### 4.1 Installation

```bash
sudo cp deploy/systemd/medical-ai-backup.service /etc/systemd/system/
sudo cp deploy/systemd/medical-ai-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now medical-ai-backup.timer
```

### 4.2 Verification

```bash
# Check timer is active
systemctl list-timers medical-ai-backup.timer

# Check next run time (should be within 12 hours)
systemctl status medical-ai-backup.timer

# Manual trigger
sudo systemctl start medical-ai-backup.service

# View logs
sudo journalctl -u medical-ai-backup.service --since today
```

### 4.3 Backup status check

```bash
cat /opt/medical-ai-platform/.deploy/backup-status/status.json
# Expected: {"last_success_timestamp":1234567890,"result":"success"}

# Deploy preflight: last backup must be < 13 hours old
LAST_TS=$(python3 -c "import json; print(json.load(open('/opt/medical-ai-platform/.deploy/backup-status/status.json'))['last_success_timestamp'])")
NOW=$(date +%s)
AGE=$(( (NOW - LAST_TS) / 3600 ))
if [ "$AGE" -gt 13 ]; then
  echo "ALERT: Last backup is ${AGE} hours old (> 13h threshold)"
fi
```

---

## 5. Deployment Procedure

### 5.1 Pre-flight checklist

- [ ] age recipient configured in `.env`
- [ ] rclone remote configured and tested
- [ ] Last successful backup < 13 hours old
- [ ] Quarterly recovery演练 still in date
- [ ] Account isolation verified (app/migration/backup)
- [ ] Key hosting proof (identity on recovery host only)

### 5.2 Trigger deployment

```
GitHub → Actions → Deploy → Run workflow
  - environment: staging | production
  - image_tag: <40-char-SHA>
  - migration_operator_id: <optional positive integer>
  - allow_first_deploy_without_rollback: false (default)
```

### 5.3 Deployment sequence (automated)

1. **Input validation** – SHA format, no `latest`, valid operator ID
2. **Environment approval** – GitHub Environment protection rules
3. **Image pull** – `IMAGE_TAG=<sha> docker compose pull`
4. **Maintenance mode** – Enable marker, stop frontend
5. **Drain** – Stop beat, wait for worker queue to empty (max 15 min)
6. **Backup** – Encrypted backup with off-host verification
7. **Migrate** – Run Alembic migrations with operator ID
8. **Start new** – `docker compose up -d --wait --remove-orphans`
9. **Smoke tests** – Health checks, login, read-only endpoints
10. **Restore traffic** – Remove maintenance marker, start frontend
11. **Cleanup** – Retain current + previous images only

---

## 6. Rollback Procedure

### 6.1 Automatic rollback (application only)

If smoke tests fail, the workflow automatically:
1. Stops new version services
2. Starts previous image (from `.deploy/previous-image-tag`)
3. Removes maintenance marker
4. Restores frontend

**No Alembic downgrade is performed.**

### 6.2 Manual rollback

```bash
cd /opt/medical-ai-platform

# Read previous tag
cat .deploy/previous-image-tag

# Set to previous image
export IMAGE_TAG=$(cat .deploy/previous-image-tag)

# Restart with previous image
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --wait

# Verify
curl http://127.0.0.1:8000/health/ready
```

### 6.3 V1.0 → V1.1 first deployment notes

- First deploy: no previous image exists
- `allow_first_deploy_without_rollback=true` only for verified empty environments
- If V1.1 receives real traffic, rollback to V1.0 requires:
  1. Enter maintenance mode
  2. Stop new submissions
  3. Drain all pending/leased/published/queued/running/retrying to safe terminal state
  4. Then restore V1.0 image

---

## 7. Backup Recovery Drill (Quarterly)

### 7.1 Procedure

1. Download latest backup from off-host
2. On isolated recovery host with age identity:
   ```bash
   age -d -i /path/to/identity.txt backup.sql.gz.age | gunzip | mysql isolated_db
   ```
3. Verify:
   - Alembic revision matches expected
   - Key table row counts are reasonable (log counts only, not data)
   - Application can start against restored database

### 7.2 Success criteria

- Decryption succeeds
- All tables present
- Alembic `current` matches deployment
- Row counts within expected range (±10%)

---

## 8. Credential Rotation

### 8.1 MySQL backup password

```sql
ALTER USER 'medical_backup'@'127.0.0.1' IDENTIFIED BY '<new-password>';
```

Update `/etc/medical-ai/backup.env` and restart timer.

### 8.2 age recipient

```bash
# Generate new keypair on recovery host
age-keygen -o /new/identity.txt
NEW_RECIPIENT=$(age-keygen -y /new/identity.txt)

# Update .env on production
# BACKUP_AGE_RECIPIENT=$NEW_RECIPIENT

# Keep old identity until all old backups expire (30 days)
```

### 8.3 rclone credentials

Re-run `rclone config` as `medical-backup` user and update remote credentials.

---

## 9. Monitoring & Alerting

| Check | Frequency | Alert if |
|-------|-----------|----------|
| Backup status file | Every deploy | `last_success_timestamp` > 13h |
| Backup script exit code | Every 12h | Non-zero |
| Disk usage `/var/backups` | Hourly | > 80% |
| Off-host storage | Daily | Unreachable |
| age recipient in `.env` | Every deploy | Missing/invalid |

---

## 10. Troubleshooting

### Backup fails with "path not allowed"

Ensure `BACKUP_DIR` is `/var/backups/medical-ai` (no traversal).

### Remote checksum mismatch

1. Check network stability
2. Verify rclone remote configuration
3. Re-run backup manually

### Drain timeout

If tasks don't drain in 15 minutes:
1. Check for stuck tasks: `celery -A app.celery_app inspect reserved`
2. Consider manual task revocation
3. Deployment aborts and restores previous version

### Maintenance mode stuck

```bash
# Check if maintenance marker exists
ls -la .deploy/maintenance/enabled

# Remove manually (only after confirming services are healthy)
rm .deploy/maintenance/enabled
```

---

## 11. Security Boundaries

- **Production host**: Only holds age public recipient, never identity
- **Recovery host**: Holds age identity, managed by 2+ authorized personnel
- **Backup user**: Cannot INSERT/UPDATE/DELETE/CREATE/ALTER/DROP
- **Migration user**: Not available at runtime, only during deploy
- **`.env` file**: No age identity, no rclone credentials on production
