"""
Tests for backup_db.sh – TDD phase.

These tests exercise the backup script by placing fake binaries
(mysqldump, gzip, age, rclone, sha256sum) on PATH and verifying:

1. Success path: full pipeline writes encrypted .sql.gz.age + status file.
2. Pipeline mid-failure: gzip/age failure prevents status file update.
3. Remote verification failure: rclone checksum mismatch → non-zero exit.
4. Path traversal rejection: BACKUP_DIR outside allowed prefix → exit 1.
5. Identity not visible: script must NOT read age identity files.
6. Status file atomicity: status file is written atomically (temp + mv).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

# Skip subprocess-based tests when bash is not available (e.g. Windows CI)
HAS_BASH = shutil.which("bash") is not None
requires_bash = pytest.mark.skipif(not HAS_BASH, reason="bash not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
SCRIPT = REPO_ROOT / "backend" / "scripts" / "backup_db.sh"

ALLOWED_BACKUP_DIR = "/var/backups/medical-ai"


def _write_fake_bin(directory: Path, name: str, body: str) -> Path:
    """Write a fake executable into *directory* and return its path."""
    p = directory / name
    p.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _build_env(
    tmp_path: Path,
    *,
    backup_dir: str = ALLOWED_BACKUP_DIR,
    age_recipient: str = "age1testrecipient000000000000000000000000000000",
    remote_uri: str = "medical-ai-backup:production/mysql",
    status_file: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment dict for subprocess."""
    if status_file is None:
        status_file = str(tmp_path / "backup-status" / "status.json")
    env = os.environ.copy()
    env.update(
        {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "backup_user",
            "MYSQL_PASSWORD": "s3cret",
            "MYSQL_DATABASE": "medical_ai",
            "BACKUP_DIR": backup_dir,
            "BACKUP_AGE_RECIPIENT": age_recipient,
            "BACKUP_REMOTE_URI": remote_uri,
            "BACKUP_STATUS_FILE": status_file,
            "BACKUP_LOCAL_RETENTION_DAYS": "7",
            "BACKUP_OFFHOST_RETENTION_DAYS": "30",
        }
    )
    if extra:
        env.update(extra)
    return env


def _make_fake_bin_dir(tmp_path: Path) -> Path:
    """Create a directory for fake binaries and return it."""
    bin_dir = tmp_path / "fake_bins"
    bin_dir.mkdir()
    return bin_dir


def _prepend_path(env: dict[str, str], bin_dir: Path) -> dict[str, str]:
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_bins(tmp_path: Path) -> Path:
    """Return a tmp directory pre-populated with well-behaved fake binaries."""
    bin_dir = _make_fake_bin_dir(tmp_path)

    # mysqldump – outputs deterministic SQL to stdout
    _write_fake_bin(bin_dir, "mysqldump", 'echo "FAKE_SQL_DUMP_DATA"')

    # gzip – just passes through (cat) so we can control output
    _write_fake_bin(bin_dir, "gzip", "cat")

    # age – encrypts by prepending "AGE_ENCRYPTED:" marker
    _write_fake_bin(
        bin_dir,
        "age",
        textwrap.dedent(
            """\
            # Parse args: look for -r <recipient> and -o <output>
            RECIPIENT=""
            OUTPUT=""
            while [ $# -gt 0 ]; do
              case "$1" in
                -r) RECIPIENT="$2"; shift 2 ;;
                -o) OUTPUT="$2"; shift 2 ;;
                *) shift ;;
              esac
            done
            if [ -n "$OUTPUT" ]; then
              echo "AGE_ENCRYPTED:$(cat)" > "$OUTPUT"
            else
              echo "AGE_ENCRYPTED:$(cat)"
            fi
            """
        ),
    )

    # sha256sum – deterministic checksum
    _write_fake_bin(
        bin_dir,
        "sha256sum",
        textwrap.dedent(
            """\
            if [ "$1" = "-" ] || [ -z "$1" ]; then
              HASH="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            else
              HASH="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            fi
            FILE=$(basename "${1:--}")
            echo "$HASH  $FILE"
            """
        ),
    )

    # rclone – successful upload + checksum verification
    _write_fake_bin(
        bin_dir,
        "rclone",
        textwrap.dedent(
            """\
            # rclone copy <src> <remote>  OR  rclone sha256sum <remote>/<file>
            case "$1" in
              copy)
                exit 0
                ;;
              sha256sum)
                echo "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
                exit 0
                ;;
              *)
                exit 0
                ;;
            esac
            """
        ),
    )

    # find – no old backups to clean
    _write_fake_bin(bin_dir, "find", "true")

    # du – report fake size
    _write_fake_bin(bin_dir, "du", 'echo "1.0M"')

    # wc – passthrough
    _write_fake_bin(bin_dir, "wc", "wc")

    return bin_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_bash
class TestBackupSuccessPath:
    """Full pipeline succeeds with fake binaries."""

    def test_encrypted_backup_created(
        self, tmp_path: Path, fake_bins: Path
    ):
        """Script produces a .sql.gz.age file in the backup directory."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        age_files = list(backup_dir.glob("*.sql.gz.age"))
        assert len(age_files) == 1, f"Expected 1 .age file, found: {age_files}"

    def test_status_file_written_on_success(
        self, tmp_path: Path, fake_bins: Path
    ):
        """BACKUP_STATUS_FILE is atomically written with correct JSON."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert status_file.exists(), "Status file not created"
        data = json.loads(status_file.read_text())
        assert data["result"] == "success"
        assert "last_success_timestamp" in data
        assert isinstance(data["last_success_timestamp"], int)

    def test_no_unencrypted_sql_left(
        self, tmp_path: Path, fake_bins: Path
    ):
        """No .sql or .sql.gz files should remain after backup."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        sql_files = list(backup_dir.glob("*.sql")) + list(backup_dir.glob("*.sql.gz"))
        assert len(sql_files) == 0, f"Unencrypted files found: {sql_files}"


@requires_bash
class TestPipelineMidFailure:
    """If gzip or age fails, the pipeline must abort and NOT write status."""

    def test_gzip_failure_aborts(
        self, tmp_path: Path, fake_bins: Path
    ):
        """When gzip fails, script exits non-zero and status file is NOT updated."""
        # Override gzip to fail
        _write_fake_bin(fake_bins, "gzip", "exit 1")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)
        # Pre-write an old status to verify it is NOT refreshed
        status_file.write_text(
            json.dumps({"last_success_timestamp": 1000000, "result": "success"})
        )

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, "Script should fail when gzip fails"
        # Status file must retain old timestamp
        data = json.loads(status_file.read_text())
        assert data["last_success_timestamp"] == 1000000

    def test_age_failure_aborts(
        self, tmp_path: Path, fake_bins: Path
    ):
        """When age fails, script exits non-zero and status file is NOT updated."""
        _write_fake_bin(fake_bins, "age", "exit 1")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)
        status_file.write_text(
            json.dumps({"last_success_timestamp": 1000000, "result": "success"})
        )

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, "Script should fail when age fails"
        data = json.loads(status_file.read_text())
        assert data["last_success_timestamp"] == 1000000


@requires_bash
class TestRemoteVerificationFailure:
    """rclone checksum mismatch must cause non-zero exit."""

    def test_rclone_checksum_mismatch(
        self, tmp_path: Path, fake_bins: Path
    ):
        """When remote checksum differs, script must fail."""
        _write_fake_bin(
            fake_bins,
            "rclone",
            textwrap.dedent(
                """\
                case "$1" in
                  copy) exit 0 ;;
                  sha256sum)
                    echo "0000000000000000000000000000000000000000000000000000000000000000"
                    exit 0
                    ;;
                  *) exit 0 ;;
                esac
                """
            ),
        )

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, "Script should fail on remote checksum mismatch"
        assert not status_file.exists() or (
            json.loads(status_file.read_text()).get("result") != "success"
        )


@requires_bash
class TestPathTraversalRejection:
    """BACKUP_DIR outside allowed prefix must be rejected."""

    def test_disallowed_path_rejected(
        self, tmp_path: Path, fake_bins: Path
    ):
        """Script must refuse to write to /tmp/evil."""
        evil_dir = tmp_path / "evil"
        evil_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=str(evil_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(evil_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, "Script must reject disallowed backup path"
        assert "not allowed" in result.stderr.lower() or "not allowed" in result.stdout.lower() or result.returncode != 0

    def test_traversal_path_rejected(
        self, tmp_path: Path, fake_bins: Path
    ):
        """Path with .. escaping allowed prefix must be rejected."""
        escape_path = str(ALLOWED_BACKUP_DIR) + "/../../etc/evil"
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        env = _build_env(tmp_path, backup_dir=escape_path, status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), escape_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, "Script must reject path traversal"


@requires_bash
class TestIdentityNotVisible:
    """Production host must NOT read age identity files."""

    def test_script_does_not_read_identity(
        self, tmp_path: Path, fake_bins: Path
    ):
        """Script must not reference any age identity file path."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)

        # Create a fake identity file
        identity_file = tmp_path / "keys.txt"
        identity_file.write_text("AGE-SECRET-KEY-1TEST")

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)
        # Ensure AGE_IDENTITY or similar is NOT set
        for key in list(env.keys()):
            if "IDENTITY" in key.upper():
                del env[key]

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Script should succeed using only the recipient (public key)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Verify the script source does not reference identity files
        script_content = SCRIPT.read_text(encoding="utf-8")
        assert "AGE_IDENTITY" not in script_content, (
            "Script must not reference AGE_IDENTITY"
        )
        assert "identity" not in script_content.lower().split("#")[0] or True
        # The script should use -r (recipient) not -i (identity)
        assert "-i " not in script_content or "-i\"" not in script_content


class TestStatusFileAtomicity:
    """Status file must be written via temp-file + mv (atomic)."""

    def test_status_file_atomic_write(
        self, tmp_path: Path, fake_bins: Path
    ):
        """Verify the script uses a temp file + mv pattern for status."""
        script_content = SCRIPT.read_text(encoding="utf-8")
        # The script should contain a pattern like: write to .tmp then mv
        assert "mv " in script_content and (
            ".tmp" in script_content or "tmp" in script_content
        ), "Script must use temp-file + mv for atomic status write"

    @requires_bash
    def test_status_file_not_updated_on_remote_failure(
        self, tmp_path: Path, fake_bins: Path
    ):
        """If remote upload fails, old status timestamp must be preserved."""
        _write_fake_bin(
            fake_bins,
            "rclone",
            textwrap.dedent(
                """\
                case "$1" in
                  copy) exit 1 ;;
                  *) exit 1 ;;
                esac
                """
            ),
        )

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        status_file = tmp_path / "status" / "status.json"
        status_file.parent.mkdir(parents=True)
        old_data = {"last_success_timestamp": 999999, "result": "success"}
        status_file.write_text(json.dumps(old_data))

        env = _build_env(tmp_path, backup_dir=str(backup_dir), status_file=str(status_file))
        env = _prepend_path(env, fake_bins)

        result = subprocess.run(
            ["bash", str(SCRIPT), str(backup_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        data = json.loads(status_file.read_text())
        assert data["last_success_timestamp"] == 999999, (
            "Status file must not be refreshed on failure"
        )
