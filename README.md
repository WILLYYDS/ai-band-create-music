# AI Band Generate Module

AI 音乐生成与 RVC 人声替换的纯 FastAPI 后端。项目使用 `uv` 固定 Python 和依赖版本，
不包含原 Vue/Vite 前端。

## 功能

- OpenAI-compatible LLM 音乐 Prompt 结构化扩写与质量校验
- ElevenLabs Music API（Composition Plan / 直接 Prompt）
- Suno sidecar 和 Generic Provider 兼容适配器
- 内置 Demucs `vocal / drums / bass / other` 四轨分离实现（当前生成管线未启用）
- RVC 人声替换、结果下载、软删除与恢复
- `/api/health`、`/api/generate` 和 `/output/*` 音频访问接口
- 本地内联任务执行，以及队列、缓存、事件发布的可替换接口

默认端口为 `8010`，Prompt 上限为 2000 个 Unicode 字符。

## 安装

项目固定使用 Python 3.10：

```bash
uv python install 3.10
uv sync --frozen
cp .env.example .env
```

当前依赖固定为 Linux `torch/torchaudio 2.7.1+cu128`，用于支持 RTX 50 系列
Blackwell `sm_120`；Demucs 和 RVC 共用这一套 CUDA 环境，不再安装 CPU Torch。
运行拆轨或默认的 RVC 配置需要可用的 NVIDIA 驱动和 CUDA GPU。

编辑 `.env`，至少配置 LLM Key。`MUSIC_PROVIDER` 只是请求未传 `provider` 时的默认值；
MiniMax 和 ElevenLabs 的配置可以同时保留。真实 ElevenLabs 模式还需要配置：

```env
MUSIC_API_MODE=real
MUSIC_PROVIDER=elevenlabs_music
LLM_API_KEY=...
LLM_MAX_TOKENS=1024
LLM_TIMEOUT_SECONDS=120
LLM_DISABLE_THINKING=true
ELEVENLABS_API_KEY=...
```

自部署 MiniMax Music 3 使用兼容的 `/v1/audio/jobs` 接口：

```env
MUSIC_API_MODE=real
MUSIC_PROVIDER=minimax_music
MINIMAX_BASE_URL=http://127.0.0.1:8111
MINIMAX_MODEL=MiniMaxAI/MiniMax-Music3
MINIMAX_SEED=42
MINIMAX_NUM_INFERENCE_STEPS=30
MINIMAX_TIMEOUT_SECONDS=7200
LLM_API_KEY=...
```

后端将 LLM 生成的歌词作为 `input`，结构化音乐描述作为 `instructions`，并将返回的
44.1 kHz WAV 直接保存到 `output`。MiniMax 始终使用自动时长，请求不包含
`audio_duration`，由模型选择自然完整的时长；即使客户端传入固定时长也会被忽略。
其他 Provider 在未指定时长时使用 `DEFAULT_DURATION_MINUTES`。
如果两个服务分别运行在 Docker 容器中，请将 `MINIMAX_BASE_URL` 改为可达的容器服务名
或宿主机地址。

音乐 Prompt 扩写建议使用非推理模型。推理模型可能先输出很长的思考过程，增加
延迟并触发读取超时。遇到 LLM `ReadTimeout` 时，应先确认 `LLM_MODEL`，再根据
服务延迟调整 `LLM_TIMEOUT_SECONDS`；后端不会自动重试网络超时，以免产生重复调用。
分类格式的合格扩写结果必须包含 8 个简洁英文制作标签，覆盖风格与年代、速度与拍号、
情绪、配器、人声、编曲结构、制作与混音以及排除项，目标长度为 350–900 字符。
同时兼容 Qwen 返回的单方括号扁平制作标签列表，但至少需要 10 个标签和 280 字符。
若模型以 JSON 数组返回 `styleTags`，其中不带方括号的裸标签（Qwen 常见的
`Category: value` 写法）会在拼接前自动补上方括号，再按上面同一套规则校验。
未指定的要素由 LLM 做协调一致的专业补充，短标签翻译不会再被当作扩写成功。

对于 Qwen3/Qwen3.5，默认通过 `chat_template_kwargs.enable_thinking=false` 关闭推理，
避免思考过程耗尽输出预算。所有模型的实际 `max_tokens` 上限为 4096；健康检查会分别
显示配置值、首轮实际值和重试实际值。

若首轮响应不是纯标签或扩写细节不足，后端会进行一次温度为 0、输出预算至少为
2048 tokens 的严格重试，并可从 `Final Answer`、末尾标签块或 JSON `tags` 中提取结果。
仍然失败时，终端日志会记录 `finish_reason`、标签数、`content_chars` 和
`reasoning_chars`，用于区分输出截断、细节不足、空正文以及供应商返回格式不兼容；
日志不会记录用户 Prompt 和模型正文。

Mock 模式需要把测试母带放到 `output/mock_full.mp3`，或者修改
`MOCK_FULL_SONG_PATH`。

## 启动

```bash
uv run ai-band-api
```

也可以直接启动 Uvicorn：

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8010
```

接口文档：<http://127.0.0.1:8010/docs>

健康检查：

```bash
curl http://127.0.0.1:8010/api/health
```

服务器 Docker 部署、离线镜像和发布包清单见 [`DEPLOYMENT.md`](DEPLOYMENT.md)。

RVC 默认自动查找 `assets/rvc` 中的模型、索引和 HuBERT/RMVPE 基础模型；也可通过
`RVC_MODEL_PATH`、`RVC_INDEX_PATH` 和 `RVC_BASE_MODEL_DIR` 显式指定。RVC 推理在
第一次转换时懒加载到 `cuda:0`；默认 `rms_mix_rate=1`，使替换人声沿用原人声的音量包络：

已完成拆轨的生成任务可直接启动后台人声替换，无需重新上传 vocal 文件：

```bash
curl -X POST 'http://127.0.0.1:8010/api/jobs/<jobId>/replace?song=0'
```

首次启动返回 `202`；重复启动同一任务返回当前状态，已有可读结果时返回 `200`；上一次替换恰好
超时或取消、推理线程仍在安全收尾时返回 `409`。`GET /api/jobs/<jobId>` 和任务 SSE 会返回
`replaceStatus`、`replaceSong`、`replaceError`，运行时顶层 `status/stage/progress/message`
与拆轨任务一致地反映当前操作。完成后歌曲结果增加 `replacedVocal` 音频 URL。转换超过
`RVC_CONVERSION_TIMEOUT_SECONDS`（默认 1800 秒）会立即标记失败；由于 RVC 推理线程不能安全
强停，锁和并发额度会保留到线程实际退出，期间产物不会发布。这种"已经终态但线程还没退出"的
替换数量可以从 `GET /api/health` 的 `replacement.workersHoldingCapacityAfterTerminal` 读到，
日志里也会在进入收尾和线程退出时各记一条 warning。`PATCH /api/jobs/<jobId>` 可取消：
接口会立即把 `replaceStatus` 标记为 `cancelled`，但保留已完成生成任务的顶层
`status=succeeded`；当前推理安全退出后会丢弃产物、释放并发额度。

替换结果与 RVC 模型绑定：`RVC_MODEL_PATH`/`RVC_INDEX_PATH` 内容或 `RVC_MODEL_VERSION` 变化后，
已缓存的结果会被判定为失效并自动重新推理。失效到新产物落位之间旧文件仍保留在磁盘上（不再
被 `replacedVocal` 引用），因为新旧产物同名、成功时会被原子覆盖；这样即使重跑失败、超时或
被取消，上一版可用的替换人声也不会被一并删除。删除该歌曲的 vocal 分轨会同时作废替换结果。

替换结果由 `/output/jobs/<jobId>/song_<n>/<原音轨名>_rvc_vocal.wav` 提供，接口返回的
`replacedVocal` 就是该 URL，前端直接播放或下载。`DELETE /api/voice/result` 软删除替换产物，
`PUT /api/voice/result` 恢复；两者均传 `job_id`、`filename` 和可选的 `song`，删除后可恢复，
重新替换则会覆盖同一路径。

生成音乐。`provider` 可传 `minimax_music` 或 `elevenlabs_music`，不传时使用
`MUSIC_PROVIDER`：

```bash
curl -X POST http://127.0.0.1:8010/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"明亮的普通话摇滚，清晰女声和有力鼓组","durationMinutes":"auto","provider":"minimax_music"}'
```

`durationSeconds`（10–360）用于 30 秒试听这类精确短时长，传了它就会覆盖
`durationMinutes`；MiniMax 仍然忽略任何固定时长。可选的 `title`（最长 100 字符）是
群聊里选定的歌曲名，会随任务持久化，并由 `GET /api/jobs` 和任务 SSE 返回，供历史页显示。

前端使用异步任务接口，以便刷新后恢复任务并显示真实阶段进度：

| 方法 | 路径 | 语义 |
| --- | --- | --- |
| `POST` | `/api/jobs` | 创建生成任务 |
| `GET` | `/api/jobs` | 列出全部历史任务（不含波形数据） |
| `GET` | `/api/jobs/{jobId}` | 查询任务状态与结果 |
| `GET` | `/api/jobs/{jobId}/events` | 通过 SSE 接收任务阶段和终态 |
| `PATCH` | `/api/jobs/{jobId}` | 局部更新任务状态（当前用于取消） |
| `DELETE` | `/api/jobs/{jobId}/stems/{stemId}` | 删除指定分轨及其输出文件 |
| `PUT` | `/api/jobs/{jobId}/stems/{stemId}` | 撤回删除并恢复指定分轨 |

当前没有完整替换任务资源的业务，因此不提供 `PUT`；未来需要整体替换任务配置时再增加。

```bash
# 创建任务（返回 202 和 jobId）
curl -X POST http://127.0.0.1:8010/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"明亮的普通话摇滚","durationMinutes":2}'

# 查询任务状态、阶段、进度和最终 result
curl http://127.0.0.1:8010/api/jobs/<jobId>

# 订阅阶段进度；收到具名 done 事件后客户端应主动关闭 EventSource
curl -N http://127.0.0.1:8010/api/jobs/<jobId>/events

# 对已完成的历史歌曲按需执行四轨分离；已有分轨时直接返回缓存
curl -X POST 'http://127.0.0.1:8010/api/jobs/<jobId>/split?song=0'

# 取消任务
curl -X PATCH http://127.0.0.1:8010/api/jobs/<jobId> \
  -H 'Content-Type: application/json' \
  -d '{"status":"cancelled"}'

# 删除已生成的某条分轨（同时删除 output 中的对应文件）
curl -X DELETE http://127.0.0.1:8010/api/jobs/<jobId>/stems/vocal

# 撤回删除
curl -X PUT http://127.0.0.1:8010/api/jobs/<jobId>/stems/vocal
```

任务元数据写入 `output/jobs/<jobId>/job.json`，完整歌曲、分轨、RVC 人声和最终混音统一
写入 `output/jobs/<jobId>/song_<n>/`。服务重启后会重新加载历史。
重启时仍为 `pending` 或 `running` 的任务会恢复为 `failed`，并标记“服务器重启，生成任务已中断”；
不会自动续跑未完成的音乐生成。执行队列和并发计数仍为单进程状态，因此仍推荐单
Uvicorn worker；需要多 worker 时再接入共享任务存储。

成功响应继续包含：

```text
jobId, prompt, durationMinutes, structuredPrompt, fullTrack,
stems, stemUrls, waveforms, splitEnabled, debug
```

`waveforms` 是每条音轨的归一化 RMS 包络，固定 640 个 bin
（`app/services/waveforms.py` 的 `WAVEFORM_BIN_COUNT`）：多轨编辑器按车道满宽绘制时
640 个点才像波形，且同一个 `waveforms` 字典内的所有音轨共享同一长度，客户端可以用同一
根 x 轴对齐。因此：

- `GET /api/jobs/{jobId}` 返回真实包络；`GET /api/jobs` 是历史列表投影，其中的
  `waveforms` 一律为空对象 `{}`（列表不绘制编辑器车道，避免按任务数放大响应体积）。
- `GET /api/jobs/{jobId}/events` 只在终态 `done` 帧里带真实波形，中间帧的 `waveforms`
  同样是 `{}`：任务结束前 `result` 不会变化（分轨也在收尾那一刻才写入波形），而分轨
  期间的中间帧每 15 秒就会被保活超时重推一次，带上波形等于把同一份 640-bin 包络重复
  发送二十多次。需要随时拿真实包络时走 `GET /api/jobs/{jobId}`。
- 拆轨时会连同 `full` 一起重新提取波形：早期版本把 `full` 存成 64 个 bin，重新提取后
  同一首歌的所有车道都统一为 640 个 bin，不会出现长短不一的车道。

## 按需四轨分离

音乐生成只产出完整混音，不会自动拆轨。从历史记录进入编辑时，调用分轨接口执行
Demucs 四轨分离并通过任务 SSE 推送真实进度；已有 `stems` 时直接复用结果。

## 本地基础设施模式

当前版本不需要 Redis 或消息队列：

```env
TASK_BACKEND=inline
CACHE_BACKEND=none
EVENT_BACKEND=none
```

对应接口位于 `app/infrastructure/`。后续可以新增 Redis/队列实现而不改变
`GenerationOrchestrator` 的音乐生成业务流程。不要在当前版本中把
`TASK_BACKEND` 或 `CACHE_BACKEND` 设置为未实现的值；配置层会在启动时拒绝它们。

## 测试

运行全部本地测试：

```bash
uv run pytest
```

测试分为：

- `tests/unit`：配置、Prompt、Provider、Demucs 命令和基础设施接口
- `tests/integration`：FastAPI API 契约、并发、音频下载
- `tests/functional`：启动真实 Uvicorn TCP 服务完成生成与四轨下载

真实 Demucs 冒烟测试默认跳过，因为首次运行需要下载模型且耗时较长：

```bash
RUN_REAL_DEMUCS_TEST=1 uv run pytest -m slow \
  tests/functional/test_real_demucs_optional.py
```

质量检查：

```bash
uv run ruff check .
uv run pytest --cov=app --cov-report=term-missing
```

## 运行约束

本地内联执行使用进程内并发限制。Demucs 模式建议保持单 Uvicorn worker：

```text
MAX_CONCURRENT_GENERATIONS=1
```

如果未来使用多个 API worker，应先实现共享任务队列或分布式锁，不能依赖每个
进程独立的内存计数器。
