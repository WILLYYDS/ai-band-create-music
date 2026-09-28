# Docker 部署与打包清单

本项目默认按单进程、单张 NVIDIA GPU 部署。`output` 和 RVC 模型位于宿主机，重建镜像不会删除它们。

## 1. 服务器准备

- Docker Engine 和 Docker Compose v2
- NVIDIA 驱动及 NVIDIA Container Toolkit
- 可访问 PyPI、PyTorch wheel 源和 `ghcr.io` 的构建网络
- 建议至少预留 20 GB 磁盘空间；CUDA/PyTorch 镜像较大

首次部署前确认 GPU 容器可用：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. 必须打包的文件

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `pyproject.toml`
- `uv.lock`
- `app/`
- `.env.example`（模板，不含真实密钥）
- `DEPLOYMENT.md`

RVC 功能还需要单独携带这些大文件：

- `assets/rvc/weights/9971-700.pth`
- `assets/rvc/indices/9971-700_added_IVF2037_Flat_nprobe_1_9971-700_v2.index`
- `assets/rvc/base_model/hubert_base.pt`
- `assets/rvc/base_model/rmvpe.pt`
- `assets/rvc/base_model/rmvpe.onnx`

不要打包 `.env`、`.venv`、`.git`、缓存和现有 `output/`；只有迁移历史任务时才单独备份 `output/`。

## 3. 制作发布包

在项目根目录执行：

```bash
tar -czf ai-band-generate-module.tar.gz \
  Dockerfile docker-compose.yml .dockerignore DEPLOYMENT.md \
  pyproject.toml uv.lock app .env.example

tar -czf ai-band-rvc-assets.tar.gz assets/rvc
```

如果需要迁移历史生成结果，再额外执行：

```bash
tar -czf ai-band-output-backup.tar.gz output
```

## 4. 在线构建并启动

将发布包传到服务器后：

```bash
mkdir -p ai-band && cd ai-band
tar -xzf ../ai-band-generate-module.tar.gz
tar -xzf ../ai-band-rvc-assets.tar.gz
cp .env.example .env
chmod 600 .env
mkdir -p output
```

编辑 `.env`，至少设置真实的 `LLM_API_KEY`、音乐 Provider 配置。宿主机端口可用 `APP_PORT` 修改；宿主机用户不是 UID/GID 1000 时，设置 `APP_UID` 和 `APP_GID`，确保容器能写入 `output`。

```bash
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
curl http://127.0.0.1:${APP_PORT:-8010}/api/health
```

查看日志和升级：

```bash
docker compose logs -f --tail=200 api
docker compose build --pull
docker compose up -d
```

### 安全启用歌曲保留策略

`output` 树内（含 `jobs`、`.expired`、`.trash`）不支持符号链接：保留清理拒绝沿链接操作，
跨出 `output` 的音频路径返回 404。多卷部署请直接挂载到 `OUTPUT_DIR` 或真实子目录。
启动时后台检查 `jobs`/`.expired`，不支持的布局会记录 WARNING `unsupported_layout`，
health 中 `unsupportedLayout=true` 并列出相对目录名 `unsupportedLayoutDirectories`。
启用前确认该标志为 `false`；`null` 表示检查尚未完成或无法确认。修复后等待下一次清单刷新。

**正式启用后，第一次启动会永久删除所有超过 `SONG_RETENTION_DAYS`（默认 3 天）的既有历史、
歌曲与回收区内容；策略追溯旧任务，无法恢复。** `output/` 可能是唯一副本，先备份：

```bash
tar -czf ai-band-output-before-retention.tar.gz output
```

先在 `.env` 设置 `SONG_RETENTION_ENABLED=true`，保持默认的 `SONG_RETENTION_DRY_RUN=true`，
重建/重启服务。dry-run 不删除、不移动文件，也不限制历史访问。检查
`output/logs/retention.log` 中的 `wouldDeleteJobIds` 和跳过原因，以及 `/api/health` 的
`retention.orphanJobIds`、`pendingDeletionJobIds`、`expiredActiveJobIds`。日志不可写时检查应用日志。
确认备份与删除范围后，再设置 `SONG_RETENTION_DRY_RUN=false` 并重启，执行正式清理。
正式启用前，必须由 PO 在 PR 或上线工单明确确认追溯删除，并由前端负责人确认新增字段、
SSE `done/status=expired` 无结果载荷及音频 404 的兼容性。策略关闭时不返回新增任务字段，
默认部署保持原有任务响应结构；开启 dry-run 才会暴露新增字段。
关闭开关会停止删除，但 `.expired` 残留仍列在 health/启动日志中，重新正式启用后会重试。

health 的清单来自后台快照（启动、每小时及清理后刷新），先确认 `inventoryUpdatedAt` 非空、
`inventoryStale=false` 且 `inventoryError=null`，并核对快照时间。`inventoryAttemptedAt` 只表示
尝试时间；失败不会推进成功时间，失败类别保留旧数据（或首次扫描时计数为 `null`），
对应的 `*Stale=true`。不能把旧计数为 0 当作本次已确认无残留。
每类最多展示 20 个 id；`Count` 为总数，`Truncated=true` 表示汇总并不完整。审查全部删除范围
时查看逐任务 `job_would_delete` 日志。`unmanagedJobs` 给出元数据异常原因，修复后需重启重新加载。

另核对 `lastCleanupAt/Reason/Result/Error`；它们反映最近一轮清理的结束时间、触发原因、
结果和脱敏错误，与清单扫描状态独立。`cleanupRunning=false` 和 `remainingJobs=0` 不代表清理成功。
dry-run 的 `lastCleanupResult=success` 只表示审计完成，未执行删除；同时核对 `dryRun` 和
`deletedJobIds/wouldDeleteJobIds`。只有正式模式才能实际删除文件。
日志的 `cleanup_failed` 带 `stage`（`pendingDeletionScan` 或 `run`），`cleanup_finished` 带
`result/error/pendingDeletionScanFailed`。暂存区扫描失败仍尝试处理已知过期任务；若暂存区
损坏或不可写，逐任务失败会列入 `failedJobIds`，修复后再重试，不能视为空批次。
关停只等待当前目录删除完成，批次剩余任务保留，结果为 `stopped`；意外循环异常记录
`retention_monitor_failed` 并在下一次刷新间隔重试。

## 5. 离线服务器打包镜像

在可联网、CPU 架构与目标服务器一致的机器上构建：

```bash
docker compose build
docker image save ai-band-generate-module:latest | gzip > ai-band-image.tar.gz
```

将代码发布包、RVC 资源包、镜像包一起传到服务器，然后执行：

```bash
gzip -dc ai-band-image.tar.gz | docker image load
docker compose up -d --no-build
```

`.env` 必须只在服务器上创建和保管。公网部署时再在服务前配置现有的 HTTPS 反向代理；本清单不额外绑定某一种代理。

## 6. ElevenLabs 升级注意事项

- 旧版 `.env` 中的 `ELEVENLABS_MUSIC_OUTPUT_FORMAT=auto` 会自动兼容为
  `pcm_44100`；其他非 `pcm_*` 值会在服务启动时报错。
- ElevenLabs 的双曲生成是串行且按两次调用计费：第二首失败时整个任务失败，
  第一首 WAV 仍留在 `output/jobs/<id>/song_1/`；清理失败任务时应包含该目录。
  请优先使用 `/api/jobs` 异步任务端点；如果使用同步生成端点，反向代理超时必须
  高于两次 ElevenLabs 生成的总时间。
- 44.1 kHz、16-bit 立体声 WAV 约为 10.6 MB/分钟。在 6 分钟、两首歌的上限下，
  单任务约需 127 MB 磁盘和下载流量，部署时需相应规划 `output` 容量与出网带宽。
  前端的 `fullTrack` 已是 `.wav`，播放时返回 `audio/wav`。
