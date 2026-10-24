"""测试进程的配置兜底。"""
import os

# jwt_secret 是必填项，测试里到处 `Settings(_env_file=None, ...)` 逐个传太吵，
# 统一在这里给一个。必须在任何 Settings() 构造之前写入。
os.environ.setdefault("JWT_SECRET", "test-only-secret-3f8a19c47b0e526d")
