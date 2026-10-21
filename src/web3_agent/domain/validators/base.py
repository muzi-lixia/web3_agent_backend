"""校验套件的公共类型与返回枚举。

套件只校验「写链路即将下发的动作参数」。三件不属于它的事：
路由归 IntentRouter（唯一决策点），只读查询的返回结果归 Sanitize，
链上取数归调用方——校验器本身是纯函数，不碰数据库也不碰 RPC。

关于 RETRYABLE：架构 §557 说它是"参数轻微格式错误，允许 LLM 一次自愈"，
但 PRD 8 章明确"失败时不回传 LLM 二次生成，直接返回固定错误码"。
两处对同一件事给了相反的处置，这里按 PRD 执行——参数错误一律 REJECT，
RETRYABLE 留给 agent 层对工具级错误（RPC 抖动）做重试，本套件不产出它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Protocol

from web3_agent.common.exceptions import error_message
from web3_agent.common.types import BudgetSnapshot, TokenMeta
from web3_agent.config.constants import DEFAULT_MAX_AMOUNT_USD, Intent


class Verdict(str, Enum):
    RETRYABLE = "RETRYABLE"
    REJECT = "REJECT"
    TERMINATE = "TERMINATE"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """verdict 为 None 表示通过。"""

    verdict: Verdict | None = None
    code: str | None = None
    message: str = ""
    field: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict is None

    @classmethod
    def ok(cls) -> ValidationResult:
        return cls()

    @classmethod
    def reject(cls, code: str, *, field: str | None = None) -> ValidationResult:
        return cls(Verdict.REJECT, code, error_message(code), field)

    @classmethod
    def escalate(cls, code: str, *, field: str | None = None) -> ValidationResult:
        return cls(Verdict.ESCALATE, code, error_message(code), field)

    @classmethod
    def terminate(
        cls, code: str | None = None, *, message: str | None = None, field: str | None = None
    ) -> ValidationResult:
        """非故障终止：步数超限、配额耗尽这类，给提示但不需要人工介入（架构 §556）。"""
        text = message if message is not None else (error_message(code) if code else "")
        return cls(Verdict.TERMINATE, code, text, field)

    def as_event(self) -> dict[str, str | None]:
        """给 AuditLogger / LangSmith L1 上报用的结构，不含业务内容。"""
        return {
            "verdict": self.verdict.value if self.verdict else "PASS",
            "code": self.code,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class TxDraft:
    """一笔待下发的写动作，字段与写工具的入参一一对应。

    allow_zero_amount 只在状态机内部下发 approve(0)（USDT 清零额度那一步）时为 True；
    LLM 传来的 0 一律按 E104 拒绝，两者不能混。
    """

    intent: Intent
    chain_id: int
    from_address: str
    token_symbol: str
    human_amount: str | None = None
    to_address: str | None = None
    spender: str | None = None
    stake_contract: str | None = None
    allow_zero_amount: bool = False


@dataclass(frozen=True, slots=True)
class StakeBinding:
    """本次 workflow 绑定的质押合约。spender 只能取这里的地址（E204）。"""

    contract_address: str
    adapter_name: str
    supported_tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GuardLimits:
    """资源阈值的显式入参。max_* 为 None 表示该维度不限。"""

    max_steps: int = 5
    max_tokens: int | None = None
    max_rpc_calls: int | None = None
    max_tx_count: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """调用方备好的上下文。所有"要从别处取"的东西都在这里传进来。"""

    allowed_chain_ids: frozenset[int]
    siwe_address: str | None = None
    token: TokenMeta | None = None
    stake: StakeBinding | None = None
    usd_price: Decimal | None = None
    max_amount_usd: Decimal = Decimal(str(DEFAULT_MAX_AMOUNT_USD))
    budget: BudgetSnapshot | None = None
    limits: GuardLimits = field(default_factory=GuardLimits)


class Validator(Protocol):
    def __call__(self, draft: TxDraft, ctx: ValidationContext) -> ValidationResult: ...
