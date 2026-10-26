# Web3 链上智能 Agent 助手 — 后端

FastAPI + LangGraph 的单 Agent 服务：一句话进来，先判意图，再决定调链上只读工具还是查知识库，最后给出带引用的回答。服务端不持私钥，不签名、不广播。

## 架构

```
POST /api/v1/chat
  → 限流配额（1 分钟分桶，落 PG，超限 429/E501）
  → 意图抽取（模型 + 结构化输出，JsonGuard 兜格式故障）
  → IntentRouter（唯一路由决策点）
  → route=read → LangGraph ReAct 子图（≤5 步）
      ├─ 只读工具：余额 / 交易记录 / 质押信息
      └─ 知识检索：改写 → 双路召回 → RRF 融合 → 重排 → 证据装配
  → 回答 + 步数、RPC 次数回写配额
```

模块职责：

| 目录 | 职责 |
|---|---|
| `api/` | HTTP 路由、鉴权依赖、请求响应模型 |
| `agent/` | 意图抽取、路由、LangGraph 状态机、提示词、回答后置校验 |
| `llm/` | 模型调用的唯一出口：结构化输出、绑定工具、纯对话 |
| `rag/` | 知识检索全链路：切块、向量化、召回、融合、重排、证据装配 |
| `web3/` | 链适配器、只读工具、待签名交易构造 |
| `domain/` | 对话记忆、摘要、用户画像 |
| `persistence/` | PG 仓储、迁移、幂等缓存 |
| `middlewares/` | 请求上下文、trace_id、限流 |
| `config/` `common/` | 配置、异常体系、资源预算、日志 |

两条贯穿全局的约定：

- **模型只有一个出口**（`llm/client.py`）。传输、超时、异常归一、调用配额全在这里定义，业务层不直接碰 SDK。
- **下游调用先记账再发**。模型、工具、下游请求、探索器四类额度分开计，调用前 `reserve`、结束结算，取消与异常也走收尾。

## 启动

需要 Python ≥ 3.11、Docker（只跑 PG）、一个可用的模型服务。

```bash
cd web3_agent_backend

# 1. 依赖
uv sync                                  # 或 python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 2. 配置
cp .env.example .env
# 必填：DATABASE_URL、RPC_URLS
# 模型二选一：MODEL_PROVIDER=ollama（本地）或 openai_compat（在线，另需 MODEL_BASE_URL / MODEL_API_KEY）

# 3. 数据库（PostgreSQL + pgvector）
docker compose -f deployment/docker/docker-compose.yml up -d postgres
uv run python scripts/migrate_pg.py

# 4. 起服务
uv run uvicorn web3_agent.main:app --app-dir src --port 8000
```

自检：

```bash
curl localhost:8000/health
curl localhost:8000/health/ready      # 会真调用一次所配模型
```

测试：

```bash
uv run pytest -q tests/unit          # 不依赖 PG 与模型服务
uv run pytest -q                     # 集成用例需要本机 PG，连不上自动 skip
```

知识库（可选，默认关闭）：`RAG_ENABLED=true` 后，先导入并发布文档再检索。

```bash
uv run python scripts/ingest_docs.py
```
