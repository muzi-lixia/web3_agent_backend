"""ORM 表定义。只做存储，不含业务方法。

会话、消息、审计、代币注册表（V1a）+ workflow / tx record / quota（写链路）。
领域模型在 common/types.py，与这里的 ORM 严格分离，由 repositories 做转换。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from web3_agent.persistence.pg.base import Base


class ConversationSession(Base):
    __tablename__ = "conversation_session"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    siwe_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    wallet_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ConversationMessage(Base):
    __tablename__ = "conversation_message"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool | summary
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    siwe_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    langsmith_run_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TokenRegistry(Base):
    __tablename__ = "token_registry"

    chain_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token_address: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(64))
    raw_chain_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decimals: Mapped[int] = mapped_column(Integer)
    is_whitelisted: Mapped[bool] = mapped_column(Boolean, default=True)


class WorkflowPersistent(Base):
    """写链路主表。运行态与归档同表；过期靠 expire_at 惰性判定，不做 TTL 自动删除。

    expire_type 区分两种过期（从未广播 / 已广播未核实），这是 PRD 9 章的要求，
    不能只存一个 expire_at 了事。
    """

    __tablename__ = "workflow_persistent"

    workflow_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    siwe_user_id: Mapped[str] = mapped_column(String(64), index=True)
    siwe_address: Mapped[str] = mapped_column(String(64), index=True)
    chain_id: Mapped[int] = mapped_column(Integer)
    intent: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), index=True)
    slots_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    generated_tx_count: Mapped[int] = mapped_column(Integer, default=0)
    max_tx_count: Mapped[int] = mapped_column(Integer, default=1)
    approve_fingerprints: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    transaction_group_id: Mapped[str] = mapped_column(String(36), index=True)
    group_status: Mapped[str] = mapped_column(String(16), default="idle")
    expire_at: Mapped[int] = mapped_column(BigInteger, index=True)  # unix 秒
    expire_type: Mapped[str] = mapped_column(String(24), default="NONE")
    error_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error_msg: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class WorkflowTxRecord(Base):
    """单笔交易。tx_intent_snapshot 是 PREVIEW 幂等的载体：重复查询必须返回同一份，
    不重建、不递增 generated_tx_count、expiresAt 不变。

    nonce 在 broadcast 后回填，REPLACED 判定要用它和链上已用 nonce 比对——
    只靠 getTransactionReceipt 判不出来，被替换的交易那个 hash 永远是 null。
    """

    __tablename__ = "workflow_tx_record"
    __table_args__ = (UniqueConstraint("workflow_id", "index", name="uq_workflow_tx_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(String(36), index=True)
    index: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # approve | stake | transfer
    tx_hash: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    tx_status: Mapped[str] = mapped_column(String(16), default="PENDING")
    nonce: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    chain_id: Mapped[int] = mapped_column(Integer)
    tx_intent_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class QuotaCounter(Base):
    """配额与限流计数，1 分钟分桶，单语句 upsert 累加。

    subject 形如 `{userId}|{metric}`，metric ∈ steps / tokens / rpcCalls / txCount / qps。
    不单开 metric 列，是为了让主键查询一列搞定（PRD 附录B 的 subject/window/count 语义）。
    """

    __tablename__ = "quota_counter"

    subject: Mapped[str] = mapped_column(String(128), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    count: Mapped[int] = mapped_column(BigInteger, default=0)


class SiweNonce(Base):
    """SIWE 一次性 nonce。

    consumed_at 承担"单次消费"语义：核销靠 UPDATE ... WHERE consumed_at IS NULL，
    并发下只有一个请求能拿到那条记录，不需要额外加锁。
    address 非空表示 nonce 已绑定地址，验签时 recovered 必须与之一致。
    """

    __tablename__ = "siwe_nonce"

    nonce: Mapped[str] = mapped_column(String(64), primary_key=True)
    address: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
