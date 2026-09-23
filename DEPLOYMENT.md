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
