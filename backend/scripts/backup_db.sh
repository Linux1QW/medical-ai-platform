#!/bin/bash
# ============================================
# MySQL Encrypted Backup Script (V1.1)
# ============================================
# Usage: bash backup_db.sh [backup_dir]
#
# Pipeline: mysqldump → gzip → age (streaming encryption)
# - No unencrypted SQL is written to disk
# - Off-host upload via rclone with remote checksum verification
# - Local retention: 7 days (configurable via BACKUP_LOCAL_RETENTION_DAYS)
# - Off-host retention: 30 days (configurable via BACKUP_OFFHOST_RETENTION_DAYS)
# - BACKUP_STATUS_FILE atomically updated on full success only
#
# Required environment variables:
#   MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
#   BACKUP_AGE_RECIPIENT   – age public recipient (NOT identity)
#   BACKUP_REMOTE_URI      – rclone remote prefix (e.g. medical-ai-backup:production/mysql)
#   BACKUP_STATUS_FILE     – path to status JSON file
#
# Optional:
#   BACKUP_LOCAL_RETENTION_DAYS  (default: 7)
#   BACKUP_OFFHOST_RETENTION_DAYS (default: 30)
# ============================================

set -euo pipefail

# ── Strict umask: only owner can read/write ──
OLD_UMASK=$(umask)
umask 077

# ── Configuration ──
BACKUP_DIR="${1:-/var/backups/medical-ai}"
DATE=$(date +%Y%m%d_%H%M%S)
LOCAL_RETENTION="${BACKUP_LOCAL_RETENTION_DAYS:-7}"
OFFHOST_RETENTION="${BACKUP_OFFHOST_RETENTION_DAYS:-30}"

DB_HOST="${MYSQL_HOST:?MYSQL_HOST is required}"
DB_PORT="${MYSQL_PORT:-3306}"
DB_USER="${MYSQL_USER:?MYSQL_USER is required}"
DB_PASS="${MYSQL_PASSWORD:?MYSQL_PASSWORD is required}"
DB_NAME="${MYSQL_DATABASE:-medical_ai}"

AGE_RECIPIENT="${BACKUP_AGE_RECIPIENT:?BACKUP_AGE_RECIPIENT is required}"
REMOTE_URI="${BACKUP_REMOTE_URI:?BACKUP_REMOTE_URI is required}"
STATUS_FILE="${BACKUP_STATUS_FILE:?BACKUP_STATUS_FILE is required}"

# ── Path validation: only allow known safe prefixes ──
ALLOWED_PREFIX="/var/backups/medical-ai"
REAL_BACKUP_DIR="$(cd / 2>/dev/null && cd "$BACKUP_DIR" 2>/dev/null && pwd)" || REAL_BACKUP_DIR="$BACKUP_DIR"
# Resolve to absolute for comparison
case "$BACKUP_DIR" in
  /*) ABS_BACKUP_DIR="$BACKUP_DIR" ;;
  *)  ABS_BACKUP_DIR="$(pwd)/$BACKUP_DIR" ;;
esac

# Check the resolved path starts with the allowed prefix
if [ "$ABS_BACKUP_DIR" != "$ALLOWED_PREFIX" ] && \
   [ "${ABS_BACKUP_DIR#"$ALLOWED_PREFIX"/}" = "$ABS_BACKUP_DIR" ]; then
  echo "[ERROR] Backup directory '$BACKUP_DIR' is not allowed." >&2
  echo "[ERROR] Only '$ALLOWED_PREFIX' (and subdirectories) is permitted." >&2
  umask "$OLD_UMASK"
  exit 1
fi

# ── Create backup directory ──
mkdir -p "$BACKUP_DIR"

# ── Temp MySQL defaults file (0600, cleaned on exit) ──
MY_CNF=$(mktemp "${BACKUP_DIR}/.my.cnf.XXXXXX")
chmod 0600 "$MY_CNF"
cat > "$MY_CNF" <<EOF
[client]
host=${DB_HOST}
port=${DB_PORT}
user=${DB_USER}
password=${DB_PASS}
EOF

cleanup() {
  rm -f "$MY_CNF"
  # Remove any partial unencrypted temp files
  rm -f "${BACKUP_DIR}/.tmp_${DB_NAME}_${DATE}.sql.gz" 2>/dev/null || true
}
trap cleanup EXIT

# ── Encrypted backup file ──
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz.age"
LOCAL_CHECKSUM_FILE="$BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz.age.sha256"

echo "[INFO] Starting encrypted backup of database: $DB_NAME @ $DB_HOST:$DB_PORT"
echo "[INFO] Backup target: $BACKUP_FILE"

# ── Step 1: Streaming pipeline: mysqldump → gzip → age ──
# Uses temp file to avoid partial writes, then atomically moves
TEMP_BACKUP="${BACKUP_DIR}/.tmp_${DB_NAME}_${DATE}.sql.gz.age"

mysqldump \
  --defaults-file="$MY_CNF" \
  --single-transaction \
  --skip-lock-tables \
  --no-tablespaces \
  --routines \
  --triggers \
  --set-gtid-purged=OFF \
  "$DB_NAME" | gzip | age -r "$AGE_RECIPIENT" -o "$TEMP_BACKUP"

# Verify pipefail caught any mid-pipeline failure (set -e + pipefail)
if [ ! -f "$TEMP_BACKUP" ] || [ ! -s "$TEMP_BACKUP" ]; then
  echo "[ERROR] Backup pipeline failed: encrypted file is missing or empty." >&2
  umask "$OLD_UMASK"
  exit 1
fi

# Atomically move temp file to final location
mv "$TEMP_BACKUP" "$BACKUP_FILE"
chmod 0600 "$BACKUP_FILE"

echo "[INFO] Encrypted backup created: $BACKUP_FILE"

# ── Step 2: Generate local SHA-256 checksum ──
LOCAL_SHA256=$(sha256sum "$BACKUP_FILE" | awk '{print $1}')
echo "$LOCAL_SHA256  $(basename "$BACKUP_FILE")" > "$LOCAL_CHECKSUM_FILE"
chmod 0600 "$LOCAL_CHECKSUM_FILE"
echo "[INFO] Local checksum: $LOCAL_SHA256"

# ── Step 3: Upload to off-host via rclone ──
REMOTE_FILENAME="$(basename "$BACKUP_FILE")"
echo "[INFO] Uploading to off-host: $REMOTE_URI/$REMOTE_FILENAME"

rclone copy "$BACKUP_FILE" "$REMOTE_URI"

# ── Step 4: Verify remote checksum and size ──
echo "[INFO] Verifying remote checksum..."
REMOTE_SHA256=$(rclone sha256sum "$REMOTE_URI/$REMOTE_FILENAME" | awk '{print $1}')

if [ "$LOCAL_SHA256" != "$REMOTE_SHA256" ]; then
  echo "[ERROR] Remote checksum mismatch!" >&2
  echo "[ERROR]   Local:  $LOCAL_SHA256" >&2
  echo "[ERROR]   Remote: $REMOTE_SHA256" >&2
  umask "$OLD_UMASK"
  exit 1
fi

# Also verify remote file size matches local
LOCAL_SIZE=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE" 2>/dev/null || wc -c < "$BACKUP_FILE")
REMOTE_SIZE=$(rclone size "$REMOTE_URI/$REMOTE_FILENAME" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | grep -o '[0-9]*' || echo "0")

if [ "$REMOTE_SIZE" != "0" ] && [ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]; then
  echo "[ERROR] Remote size mismatch! Local=$LOCAL_SIZE Remote=$REMOTE_SIZE" >&2
  umask "$OLD_UMASK"
  exit 1
fi

echo "[INFO] Remote verification passed (checksum + size match)"

# ── Step 5: Atomic status file update ──
STATUS_DIR=$(dirname "$STATUS_FILE")
mkdir -p "$STATUS_DIR"
STATUS_TMP=$(mktemp "${STATUS_DIR}/.status.tmp.XXXXXX")
chmod 0600 "$STATUS_TMP"

UNIX_TS=$(date +%s)
cat > "$STATUS_TMP" <<EOF
{"last_success_timestamp":${UNIX_TS},"result":"success"}
EOF

mv "$STATUS_TMP" "$STATUS_FILE"
echo "[INFO] Status file updated: last_success_timestamp=$UNIX_TS"

# ── Step 6: Cleanup old local backups ──
echo "[INFO] Cleaning local backups older than ${LOCAL_RETENTION} days in $BACKUP_DIR"
DELETED_LOCAL=$(find "$BACKUP_DIR" -name "*.sql.gz.age" -mtime +${LOCAL_RETENTION} -print -delete | wc -l)
find "$BACKUP_DIR" -name "*.sql.gz.age.sha256" -mtime +${LOCAL_RETENTION} -delete 2>/dev/null || true
echo "[INFO] Removed $DELETED_LOCAL old local backup(s)"

# ── Step 7: Cleanup old off-host backups ──
echo "[INFO] Cleaning off-host backups older than ${OFFHOST_RETENTION} days"
rclone delete "$REMOTE_URI" --min-age "${OFFHOST_RETENTION}d" 2>/dev/null || \
  echo "[WARN] Off-host cleanup skipped (rclone delete not supported or failed)"

# ── Done ──
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[INFO] Backup complete: $BACKUP_FILE (size: $BACKUP_SIZE)"
echo "[INFO] Off-host: $REMOTE_URI/$REMOTE_FILENAME"
echo "[INFO] All verification passed."

umask "$OLD_UMASK"
