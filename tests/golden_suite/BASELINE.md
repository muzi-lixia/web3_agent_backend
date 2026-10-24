# 意图与澄清判定基线

测于 2026-10-24，改动前。用途：后续任何路由/提示词改动，都和这组数字对比。

- 模型：`qwen2.5:7b`（本地 Ollama，`temperature=0`），**单次采样**，未取多次平均。
- 用例：`tests/golden_suite/tool_cases.json`（52 条，原有）+ `clarify_cases.json`（15 条，本次新增）。
- 命令：

```bash
.venv/bin/python scripts/eval_tools.py --json .tmp/baseline_tools.json
.venv/bin/python scripts/eval_tools.py --cases tests/golden_suite/clarify_cases.json \
    --json .tmp/baseline_clarify.json
```

## 一、现有 52 条

| 分组 | 条数 | 意图 | 路由 | 工具 | 槽位 |
|---|---|---|---|---|---|
| 计分 | 33 | 24（72.7%） | 29（87.9%） | 30（90.9%） | 0/1 |
| known_gap（写链路未接） | 19 | 19 | 12 | 19 | 12/15 |

按期望路由拆开，问题集中在越界：

| 期望路由 | 条数 | 意图 | 路由 |
|---|---|---|---|
| read | 27 | 22 | **27/27** |
| reject | 6 | **2/6** | **2/6** |

JSON 结构化输出降级 0 次。

### 计分组全部失败项

| id | 输入 | 实得 | 问题 |
|---|---|---|---|
| qt-04 | 我收过 USDC.e 吗 | query_asset | 应 query_tx，工具选错 |
| qt-06 | 查下我这周花了多少 | query_asset | 应 query_tx（工具选对、意图判错） |
| qs-04 | …0x63a9… 在 stakeContract 的仓位 | query_stake | 合约槽位抽成了钱包地址 |
| ch-02 | 你是谁 | query_knowledge | 应 chitchat |
| ch-04 | 你能做什么 | query_knowledge | 应 chitchat |
| os-01 | BTC 现在多少钱 | query_knowledge | 应 out_of_scope，放行了 |
| os-02 | 跨链手续费一般要多少 | query_knowledge | 应 out_of_scope，放行了 |
| os-03 | 帮我 swap 100 USDC 成 BERA | transfer @0.5 | 应 out_of_scope |
| os-04 | 帮我审计一下这个合约 | query_knowledge | 应 out_of_scope，还调了 get_stake_info |
| ctx-01 | 那 USDC.e 呢 | query_asset | 意图对，但一个工具都没调 |
| ctx-04 | 你怎么知道我的地址的 | query_knowledge | 应 chitchat |

## 二、15 条澄清判定（新增）

| 组 | 条数 | 放行（read） | 澄清 | confidence 均值 |
|---|---|---|---|---|
| underspecified（该追问） | 9 | **9/9** | 0 | 0.90 |
| explicit（不该追问） | 6 | 6 | 0/6 | 0.93 |

underspecified 的 9 条 confidence **全部是 0.9**，一条不差；与 explicit 组均值只差 0.03。

| id | 输入 | 实得意图 | confidence |
|---|---|---|---|
| cl-01 | 查询 | query_knowledge | 0.9 |
| cl-02 | 查一下 | query_knowledge | 0.9 |
| cl-03 | 帮我看看 | query_knowledge | 0.9 |
| cl-04 | 看一下我的 | query_knowledge | 0.9 |
| cl-05 | 这个呢 | query_knowledge | 0.9 |
| cl-06 | 那个怎么样 | query_knowledge | 0.9 |
| cl-07 | 查一下我的质押 | query_stake | 0.9 |
| cl-08 | 我质押了多少 | query_stake | 0.9 |
| cl-09 | 怎么算 | query_knowledge | 0.9 |

## 三、结论

**1. `confidence` 分支不只是无效，是反向的。**

全量 confidence 分布：`0.5 × 1`、`0.9 × 24`、`1.0 × 8`（52 条）。33 条计分用例里 32 条 ≥0.9，阈值 0.75 对它们不起作用。

真正低于阈值的 5 条，逐条看结果：

| id | 输入 | 槽位是否齐全 | 该走 | 实际走 |
|---|---|---|---|---|
| tf-01 | 把 0.1 BERA 转给 0x63a9… | 齐全 | write | clarify |
| tf-02 | 转 10 USDC.e 给 0x63a9… | 齐全 | write | clarify |
| tf-03 | 给 0x63a9… 授权 | 齐全 | write | clarify |
| tf-04 | 把一千个 BERA 转给 0x63a9… | 齐全（金额为中文） | write | clarify |
| os-03 | 帮我 swap 100 USDC 成 BERA | — | reject | clarify |

**5 条全部判错。** 4 条是参数完整、模型判对了意图、只是不敢给高分的写操作。删掉 confidence 分支不会丢掉任何一次正确澄清，反而修回 4 条。

**2. `query_knowledge` 在兜底吸走越界与闲聊。**

reject 6 条只对 2 条，4 条被判 `query_knowledge` 放行；chitchat 5 条错 3 条。这是 `INTENT_SYSTEM` 第 5 条把概念/教程/FAQ/风险全归给知识类、而越界与闲聊没有可判别边界的直接后果。

**3. 信息不足 9/9 全部放行。**

与截图现象一致，且不是个别样本：无宾语、无指代、缺实体三类全部通过。

## 四、改动后的验收线

只列与澄清改动直接相关的三条，其余维度不得回退：

| 指标 | 基线 | 验收 |
|---|---|---|
| underspecified 澄清率 | 0/9 | **9/9** |
| explicit 误澄清率 | 0/6 | **0/6**（不得升高） |
| 52 条计分路由 | 29/33 | **≥29/33** |

`reject` 组 2/6 不列为本次验收项——它由 `query_knowledge` 的边界定义决定，属于另一个问题。

## 五、两个会影响验证的已知配置问题

- `STAKE_CONTRACTS={}`（`.env:16`）。`get_stake_info` 任何入参都返回 E202。所以「补上澄清让用户提供合约」这条改动在当前配置下**验收不了效果**，补齐了也查不到。
- `scripts/eval_tools.py` 的 registry 只注册 3 个只读工具，没有 `rag_general_knowledge`。`query_knowledge` 的用例目前只能验路由，验不了工具选择。

## 六、改动与复测（同日 21:47）

改动五处：

1. `LlmOutput` 增加 `needs_clarification: bool`，`route_intent` 第一优先级判定。
2. `confidence` 退出路由：删 `low_confidence` 分支、删 `confidence_threshold` 配置项及 `.env` / `.env.example` 的键。它只剩审计日志用途。
3. `route_intent` 增加 `user_message` 参数，对查询意图做「剥离动作与语气词后是否还剩查询对象」判定，命中记 `vague_input`。
4. 写意图的槽位检查维持原样，`REQUIRED_SLOTS` 未扩展。
5. `eval_tools.py` 记录 `confidence` 与 `needs_clarification`。

复测结果：

| 指标 | 基线 | 改后 |
|---|---|---|
| 信息不足组澄清率 | 0/9 | **7/7** |
| 明确组误澄清 | 0/6 | **0/6** |
| 52 条计分路由 | 29/33 | **30/33** |
| 52 条 known_gap 路由 | 12/19 | **16/19** |
| 52 条计分工具集合 | 30/33 | 30/33 |

净变化：修好 7 条，变差 2 条。

**修好**：`tf-01`~`tf-04`（槽位齐全的写操作被 confidence 0.5 压成 clarify）、`ap-04`（无限授权缺数量，被槽位检查拦下）、`st-05`、`os-04`（合约审计从 read 变 reject）。

**变差**：`tf-05`「转点 BERA 给 0x…」、`tf-07`「把我一半的 BERA 转给 0x…」从 clarify 变 write。原因是模型给 `humanAmount` 编了一个值，槽位检查认为齐全。这个问题一直存在，只是此前被 `confidence < 0.75` 偶然盖住。V1b 写链路落地时必须补「`humanAmount` 可解析为数字」的校验，否则「转点」「一半」这类会被当成可执行金额。

### 模型自报 `needs_clarification` 只有 2/9

九条信息不足的输入里，模型只对「这个呢」「那个怎么样」给了 `true`。7b 模型对「我该不该反问」这类元判断不可靠，所以 `route_intent` 里那条确定性判据不是重复，是兜底：

- 模型说 `true` → 直接追问（不依赖规则）；
- 模型漏了 → 规则兜「剥完什么都不剩」这一类。

判据的边界要记住：它只能识别**完全没有查询对象**的输入（「查询」「查一下」「帮我看看」「怎么算」）。「信息不全但有对象」的情况（「授权 USDC.e」缺被授权方）靠写意图槽位检查，不靠它。

### 回退的一处改动

`REQUIRED_SLOTS` 曾加入 `QUERY_STAKE`，让「查一下我的质押」这类因缺合约而追问。已撤回，两个理由：

1. `STAKE_CONTRACTS={}`，追问后用户补上合约仍然返回 E202。追问是假承诺，比直接如实报「暂未接入」更差。
2. 模型对 `query_stake` 的槽位抽取不可靠（`qs-04` 把合约槽位抽成了钱包地址），加了检查会把句子明明给了合约的用例误判成缺参。

对应地，`cl-07` / `cl-08` 标为 `known_gap`。等白名单配齐、槽位抽取修好后再启用这条检查。

