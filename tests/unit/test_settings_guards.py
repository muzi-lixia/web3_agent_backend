"""缺密钥就不许启动：守卫不在代码里，在字段没有默认值这件事上。"""
import pytest
from pydantic import ValidationError

from web3_agent.config.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # 本机的 .env / 环境变量不能影响这里的判定，只用显式传参。
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_ALGORITHM", raising=False)


def test_missing_secret_fails_at_construction():
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(_env_file=None)


def test_secret_from_env_is_accepted(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "from-env-secret-3f8a19c47b0e526d")
    assert Settings(_env_file=None).jwt_secret == "from-env-secret-3f8a19c47b0e526d"


def test_short_secret_is_rejected():
    """低于 HS256 的 32 字节基线就起不来，不留到线上被弱密钥静默通过。"""
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(_env_file=None, jwt_secret="too-short")


def test_unknown_algorithm_is_rejected():
    """算法名允许配置，但只能是 HS 系列：不能把 none 传进去。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, jwt_secret="x" * 32, jwt_algorithm="none")
