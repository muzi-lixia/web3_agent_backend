# GitHub Actions → GHCR → muzi-server

本目录是独立的生产部署配置，不使用开发期 Compose。前后端各自有工作流：

```text
Pull Request → 测试 / 前端构建 → Docker 构建验证（不发布、不部署）
main push   → 测试通过 → GHCR sha-<完整提交号> 镜像
                        → DEPLOY_ENABLED=true 时 SSH 到服务器
                        → pull 精确 digest → 更新对应服务 → 健康检查
                                                  → 失败则尝试恢复旧镜像
```

服务器不拉代码、不安装项目依赖、不执行 npm build，不运行本地模型。现有博客仍由宿主机 Nginx 托管；本项目单独使用 127.0.0.1:8081。发布不是零停机部署，单实例更新会短暂中断请求。

## 1. 文件与镜像

- 后端镜像：`ghcr.io/muzi-lixia/web3_agent_backend`
- 前端镜像：`ghcr.io/muzi-lixia/web3_agent_frontend`
- 两个工作流都生成 `sha-<40 位提交号>` 标签，部署实际使用 `@sha256:...`。
- `compose.yml`：仅运行 PG/pgvector、后端、前端，不包含 Ollama。
- `production.env`：两个镜像版本、数据库密码与前端监听端口。
- `app.env`：只在服务器保存的模型凭据、JWT、域名、RPC 等运行配置。
- `deploy.sh`：部署单个组件，跨仓库共享文件锁，拉取失败不动现有容器。
- `ssh-deploy.sh`：专用 SSH 公钥的 forced command，只允许本项目部署命令；用当前 Actions job 的短期 token 登录 GHCR，部署结束清理临时凭据。
- `postgres-init/001-vector.sql`：新数据库第一次初始化时启用 vector 扩展。
- `nginx-host.conf.example`：宿主机新站点示例，HTTPS、SSE 和真实 IP 转发。

`.dockerignore` 排除本机 `.env`、Git、虚拟环境等。模型 key、JWT 和数据库密码不参与镜像构建。前端 Reown project ID 是公开的浏览器配置，编译进 JS，并非后端 API key。

## 2. GitHub 仓库配置

两个仓库的 Settings → Actions → General 中允许工作流运行。发布镜像使用仓库自动提供的 `GITHUB_TOKEN`，工作流只为镜像发布 job 请求 `packages: write`，不需要把个人 GitHub token 写进代码。

前端仓库 Settings → Secrets and variables → Actions → **Variables**：

| Repository variable | 值 |
|---|---|
| `VITE_REOWN_PROJECT_ID` | 你的 Reown project ID；main 发布时必填 |
| `DEPLOY_ENABLED` | 首次准备阶段不设或设 `false`；完成服务器初始化后再设 `true` |

后端仓库同样使用 repository variable `DEPLOY_ENABLED`。不要只在 production environment 设置此开关，因为 job 的启动条件在进入 environment 之前判断。

两个仓库均创建 `production` Environment，在其中添加 secrets：

| Secret | 内容 |
|---|---|
| `DEPLOY_HOST` | 服务器的实际公网 IP 或域名，不能填本机的 SSH 别名 `muzi-server` |
| `DEPLOY_USER` | 有权运行 Docker、读写 `/opt/web3-agent` 的部署用户 |
| `DEPLOY_PORT` | SSH 端口；不设则 22 |
| `DEPLOY_SSH_KEY` | 专用于部署的 SSH 私钥；公钥需加入服务器部署用户 |
| `DEPLOY_KNOWN_HOSTS` | 经本机已信任连接核对过的服务器 host key 行，非 22 端口需使用 `[host]:port` 格式 |

部署公钥在服务器 authorized_keys 中应使用：

```text
command="/opt/web3-agent/ssh-deploy.sh",restrict ssh-ed25519 <public-key> web3-agent-github-actions
```

工作流使用严格 host key 校验，不临时信任未核对的 `ssh-keyscan` 输出。可为 Environment 配置人工审批；若希望 main 合并后全自动部署，不启用 required reviewers。两种方式都不会让 PR 部署。

## 3. 首次生成镜像

先保持两个仓库 `DEPLOY_ENABLED=false`，提交并推送当前代码与工作流。main 工作流通过后，在 Actions summary 取得两个镜像 digest。

GHCR 首次发布的包可能默认是 private，即使代码仓库公开。按你的选择：

- 公开镜像：在对应 Package settings 将可见性设为 Public，服务器可以匿名拉取。
- 私有镜像：自动部署使用当前 Actions job 的短期 `GITHUB_TOKEN`（`packages: read`），经 SSH 标准输入传到 `ssh-deploy.sh`。凭据只保存在临时 Docker 配置目录，不持久化到服务器。首次手工拉取若需要登录，可使用只含 `read:packages` 的凭据并以 `--password-stdin` 传入，不要复用模型 key 或把 token 写进 Compose。

服务器需能访问 `ghcr.io` 及实际镜像层下载地址；仅 registry 返回 401 证明仓库入口可达，不等于所有层都可下载。以首次 `compose pull` 为准。

## 4. 初始化服务器一次

确认 `/opt/web3-agent` 是本项目专用的新目录。以下复制操作只传生产模板和脚本，不自动上传本机开发 `.env`。

在本机后端仓库执行：

```bash
ssh muzi-server 'sudo install -d -m 750 /opt/web3-agent'
scp -r deployment/production/. muzi-server:/opt/web3-agent/
ssh muzi-server
```

若 SSH 用户不是 root，需要管理员先把目录授权给部署用户，否则 scp 无权写入。部署用户也需要 Docker 权限；Docker 权限具有较高主机权限，应使用专用凭据。

在服务器执行：

```bash
cd /opt/web3-agent
umask 077
cp production.env.example production.env
cp app.env.example app.env
chmod 600 production.env app.env
chmod +x deploy.sh ssh-deploy.sh
```

编辑两个文件，替换所有 `REPLACE` / `example.com` 占位：

1. `BACKEND_IMAGE`、`FRONTEND_IMAGE` 填第一次 CI 输出的精确镜像 digest。
2. 数据库密码用 `openssl rand -hex 32` 生成；当前连接串拼接要求使用 hex，避免特殊字符需要 URL 编码。
3. JWT 使用独立随机密钥，至少 32 字符。
4. 设置真实 `MODEL_*`、RPC、允许链、`SIWE_DOMAIN` 与 `SIWE_URI`。
5. 首次建议关闭 RAG 与上传；启用前将嵌入服务也外置，使用数据库兼容的 1024 维向量，并重新导入对应文档。
6. 配置 `CHAT_ALLOWLIST` 为允许聊天的完整小写钱包地址 JSON 数组。当前业务代码默认空名单，所有地址都会被拒绝；不要通过关闭白名单绕过这条业务规则。

然后校验、拉取和启动：

```bash
docker compose --env-file production.env -f compose.yml config --quiet
docker compose --env-file production.env -f compose.yml pull
docker compose --env-file production.env -f compose.yml up -d --wait --wait-timeout 180
curl --fail http://127.0.0.1:8081/_frontend_health
curl --fail http://127.0.0.1:8081/health
curl --fail http://127.0.0.1:8081/health/ready
```

`/health/ready` 返回体必须手动确认；它可能用 HTTP 200 返回 not_ready，且模型列表可达不代表结构化输出、工具调用都兼容。还应完成一次实际钱包登录与聊天验收。当前容器健康检查严格检查 `/health` 的 JSON 状态及数据库连接，但不把外部模型临时故障当成反复重启应用的理由。

默认资源上限：后端 512MiB、PG 320MiB、前端 64MiB，适用于当前 2 核、约 1.57GiB 可见内存的演示机，需要观察真实峰值。Compose 不会自动配置 Swap，也没有提供聊天并发信号量；公开演示应关注并发、RSS、OOM 和模型额度。

宿主机 Nginx 使用单独子域名转发到 8081，保留原博客站点。申请该域名的 HTTPS 证书后，再按 `nginx-host.conf.example` 加站点并执行 `nginx -t` 后 reload。Reown 后台同步允许该域名来源。浏览器前端同源请求 `/api`，不需要另开 CORS。

## 5. 启用持续部署

首次启动和域名验收完成后，把两个仓库的 repository variable `DEPLOY_ENABLED` 设为 `true`。

之后 push main 或在 main 上手动运行工作流，测试、构建、发布成功后自动连接服务器执行（私有镜像拉取的认证由 forced command 包装器处理）：

```bash
/opt/web3-agent/deploy.sh backend ghcr.io/muzi-lixia/web3_agent_backend@sha256:<digest>
# 前端仓库对应 frontend
```

GitHub runner 不会自动获得你本机 `~/.ssh/config` 或 `muzi-server` 私钥。Environment secrets 必须配置完整。脚本会用 `flock` 串行前后端部署，只重建对应组件，不重启数据库、不删除数据卷；`--wait` 等待健康检查。保持旧镜像，不自动做全局 prune，避免误删博客镜像或回退版本。

生产 Compose、宿主机 Nginx、脚本、环境变量的变化需单独审查并同步到服务器，镜像发布不会覆盖这些服务器配置。

## 6. 回退、备份与限制

手动回退使用之前 workflow summary 或部署日志记录的 digest：

```bash
cd /opt/web3-agent
./deploy.sh backend ghcr.io/muzi-lixia/web3_agent_backend@sha256:<previous_digest>
```

每次成功部署保留 `production.env.previous`，它包含数据库密码，不可公开。自动回退仅发生在更新命令健康检查失败时，不能撤销业务数据或数据库结构变化，也不能自动恢复已经被 kill 掉的部署进程。

当前应用启动的 `create_all` 只创建缺失表，不升级老表。部署脚本不擅自执行 DDL；有数据库结构变更时，应先备份、评审迁移与回退兼容性，再发布。

新数据库的初始化 SQL 只运行一次；已有数据卷不要通过删除卷来“重跑初始化”。本项目从新卷起步，不能把已有博客或其他 PG 的数据卷挂进来。

备份示例（服务器本地，实际还应复制到服务器之外）：

```bash
cd /opt/web3-agent
mkdir -p backups
chmod 700 backups
docker compose --env-file production.env -f compose.yml exec -T postgres pg_dump -U web3_agent -d web3_agent -Fc > "backups/web3-agent-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

不要执行 `docker compose down -v`。数据库不开放公网端口，后端仅在 Compose 网络中可达，只有前端网关绑定宿主机回环端口。

## 7. 本次验证范围

工作流包含后端单元测试、前端生产构建和 Docker 构建检查；不等于已对在线模型供应商、生产域名、真实钱包及数据库迁移完成验收。首次启用 CD 前按第 4 节完成端到端检查。

参考：[GitHub GHCR 发布](https://docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions)、[容器仓库权限](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)、[Compose 健康等待](https://docs.docker.com/reference/cli/docker/compose/up/)。
