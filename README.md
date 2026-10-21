# Web3 链上智能 Agent 助手 — 后端

对齐文档：`架构 v0.8` / `PRD V1.5` / `Python后端工程目录结构 v2`。

**当前范围：V1a 只读链路 + SIWE 登录 + 限流配额。** 写链路（T4–T7、状态机、workflow 接口）
尚未实现，`route=write` 一律返回"写操作尚未接入"，不会生成任何 TxIntent。

## 已经能跑通的

```
POST /api/v1/chat  →  限流（1 分钟分桶，落 PG）
                   →  意图抽取(Ollama, JSON Schema)
                   →  JsonGuard 熔断
                   →  IntentRouter（唯一路由决策点）
                   →  route=read → LangGraph ReAct 子图（≤5 步）
                   →  3 个只读工具（余额 / 交易记录 / 质押）
                   →  回答 + 步数/RPC 计数回写配额

POST /api/v1/auth/nonce   →  一次性 nonce（5 分钟，绑定地址与 chainId）+ 完整 SIWE 消息
POST /api/v1/auth/verify  →  校验 domain/uri/时效 → 验签反推地址 → 核销 nonce → 签 JWT
```

- 对话历史落 PG `conversation_session` / `conversation_message`，按 `expires_at` 惰性过期；
- 审计落 PG `audit_log`，`trace_id` 贯穿整条链路；登录成功与 SIWE 拒绝各留一条；
- 私钥/助记词命中时**不调用模型、不落库**，直接拒答；
- LangSmith 上报收敛在 `agent/langsmith_tracer.py` 一处，默认关闭。

## 已验证（本地实测，非纸面）

| 场景 | 结果 |
|---|---|
| 查余额 | `我的 ETH 余额是多少` → 调 `get_asset_balance` → Sepolia 返回真实余额并正确转述 |
| 私钥粘贴 | 拒答、`steps=0`（未调模型）、PG 里查不到该会话（未落库） |
| 写意图 | `帮我转 10 USDC 给 0x44…` → 意图识别为 transfer，路由到 write 后被拦为"尚未接入"，不生成 TxIntent |
| 闲聊 / 越界 | 正常回答或说明能力边界，不误触发工具 |
| 落库 | 会话 8 条消息、审计 5 条（4 条 intent_routed + 1 条 secret_leak_blocked） |
| SIWE 登录 | nonce → 钱包签名 → verify 返回 token；同一签名重放被拒（401/E503）；用别的钱包签同一消息被拒 |
| 限流 | 当前分钟匿名计数打满后 `/chat` 在进模型之前返回 429/E501，`retryable=true` |
| 测试 | 72 条全绿（65 单测 + 7 集成，集成需本机 PG） |

> 地址注入：模型曾把 `address` 填成 `0xYourAddressHere`。现在会话地址写进 system prompt，
> 且地址参数不是合法 EVM 地址时由后端回填——**不让"模型编地址"变成用户看到的"地址不合法"**。

## 启动

```bash
cd web3_agent_backend
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env          # 填 RPC_URLS / DATABASE_URL

docker compose -f deployment/docker/docker-compose.yml up -d postgres
.venv/bin/python scripts/migrate_pg.py
.venv/bin/uvicorn web3_agent.main:app --app-dir src --port 8000
```

自检：

```bash
curl localhost:8000/health
curl localhost:8000/health/ready          # 会真的问一次 Ollama
curl -s localhost:8000/api/v1/chat -H 'content-type: application/json' \
  -d '{"sessionId":"s1","wallet":{"address":"0x..","chainId":11155111},"userMessage":"我的 ETH 余额是多少"}'

# 登录：拿 nonce 与消息，用钱包对 message 签 personal_sign，再提交 signature
curl -s localhost:8000/api/v1/auth/nonce -H 'content-type: application/json' \
  -d '{"address":"0x..","chainId":11155111}'
curl -s localhost:8000/api/v1/auth/verify -H 'content-type: application/json' \
  -d '{"message":"<上一步返回的 message>","signature":"0x.."}'
```

测试：

```bash
.venv/bin/pytest -q tests/unit          # 不依赖 PG / Ollama
.venv/bin/pytest -q                     # 集成测试需要本机 PG，不通自动 skip
```

## 这一版刻意没做的事

| 未做 | 原因 |
|---|---|
| 写工具 T4–T7、状态机、workflow / intent / tx 接口 | M2 范围。写工具**不进 `bind_tools`**这条约束已经在 `ToolRegistry` 里用断言锁住 |
| 质押适配器实现 | `get_stake_info` 只做白名单拦截 + 链高度探活，不编造仓位数字 |
| Alembic | V1a 用 `create_all` 起步，写链路开工前替换 |
| 兜底轮询、TTL 双 EXPIRED | 属于 workflow 生命周期，M2 |
| token 用量配额 | 配额维度里 `tokens` 只建了表与读取，`LlmAdapter` 尚未回传 usage |

## 限流与配额的落法

- 1 分钟分桶，`quota_counter` 单语句 upsert 累加（`ON CONFLICT DO UPDATE ... RETURNING`），
  一次往返拿到最新值，不在热路径上先查后改；
- 匿名按 IP、登录按 `siwe_user_id`；`/auth/*` 单独一档额度；
- RPC 次数由 `web3/metering.py` 用 ContextVar 在请求内累加，请求结束时回写——
  `ChainAdapter` 是进程级单例，按用户归属只能在请求上下文里做；
- 超限统一 429/E501。PRD 写的是"超限降级只读"，但降级需要有写链路可降，写链路上线后改判定即可。

## 与文档的差异（都是显式选择，不是遗漏）

1. 新增 `agent/prompts.py`：两段系统提示词集中放，便于评审时逐句核对能力承诺。
2. `main.py` 里的 `Container` 就是组装点，没有额外引入 DI 容器——当前只有一个进程、一份依赖图。
3. 只读工具的 `chainId` 不是入参：后端按 `ALLOWED_CHAIN_IDS` 遍历，符合"LLM 不可见 chainId"。

## 已知缺口（会阻断 M2，不阻断 V1a）

- 真实质押合约地址 + ABI（当前 `STAKE_CONTRACTS` 为空）
- IntentRouter 语义互斥表（`(intent × token × 目标)` 判定矩阵）
- 前端不在本仓库：PRD 16 章的"用户故事全通过"需要改判为接口级验收
