import os
import pytest
from src.data.guard import check_path


def test_blind_filename_raises():
    with pytest.raises(RuntimeError, match="BLIND ACCESS BLOCKED"):
        check_path("data/raw/blind.json")


def test_blind_test_filename_raises():
    with pytest.raises(RuntimeError, match="BLIND ACCESS BLOCKED"):
        check_path("/some/path/blind_test.json")


def test_non_blind_filename_passes():
    # Should not raise
    check_path("data/raw/misinfo_SFT_train_for_cot.json")
    check_path("data/raw/misinfo_RL_train_for_cot.json")


def test_blind_allowed_with_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_BLIND_EVAL", "1")
    monkeypatch.chdir(tmp_path)
    # Should not raise; should write audit log
    check_path("data/raw/blind.json")
    audit = (tmp_path / "blind_access_audit.log").read_text()
    assert "BLIND_READ" in audit
    assert "blind.json" in audit
