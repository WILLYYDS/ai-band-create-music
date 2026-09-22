# AI Band Generate Module

AI 音乐生成与 RVC 人声替换的纯 FastAPI 后端。项目使用 `uv` 固定 Python 和依赖版本，
不包含原 Vue/Vite 前端。

## 功能

- OpenAI-compatible LLM 音乐 Prompt 结构化扩写与质量校验
- ElevenLabs Music API（直接 Prompt）
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
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_SECONDS=120
LLM_DISABLE_THINKING=true
ELEVENLABS_API_KEY=...
```

ElevenLabs 直接接收完整的结构化歌词、扩充风格标签和固定的 `music_length_ms`，由模型自行
安排段落与演唱节奏。后端不会裁剪歌词；组合后的 Prompt 超过接口的 4100 字符上限时会在
调用前报错。歌词相对目标时长过长时可能无法完整唱完，调用方需要缩短歌词或增加目标时长。
请求两首时后端会使用相同歌词和风格独立调用两次 ElevenLabs。音频固定请求
`pcm_44100`，再在本地封装为 16-bit 立体声 WAV，不保存 MP3。
当前产品只生成中文歌曲，不再根据请求内容检测演唱语言：ElevenLabs 默认追加中文清晰人声
要求，MiniMax 同样直接接收中文歌词与中文人声风格标签。启用
`ELEVENLABS_FORCE_INSTRUMENTAL=true` 时会自动关闭 ElevenLabs 中文人声要求，避免与纯音乐
模式冲突。

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
合格扩写结果必须包含 6–8 个 `[Category: value]` 英文制作标签，整体覆盖风格与年代、
速度与拍号、情绪、配器、中文人声、编曲结构、制作与混音以及排除项；相邻类别可合并。
若模型以 JSON 数组返回 `styleTags`，其中不带方括号的 `Category: value` 项会在拼接前
自动补上方括号，再按同一套规则校验。
未指定的要素由 LLM 做协调一致的专业补充，短标签翻译不会再被当作扩写成功。

对于 Qwen3/Qwen3.5，默认通过 `chat_template_kwargs.enable_thinking=false` 关闭推理，
避免思考过程耗尽输出预算。`LLM_MAX_TOKENS` 直接控制“歌词 + 风格标签”合并请求的输出
预算，默认 2048、最大 4096，不建议低于 2048。自动生成的歌词正文最多 1000 个字符，
并要求模型将每个原始风格标签值控制在 200 个字符以内，避免后端生成超出音乐 Provider
的 Prompt 上限。

歌词与风格在同一次 JSON 请求中生成；响应格式或内容校验失败时，后端会携带失败原因
重试一次。

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
第一次转换时懒加载到 `cuda:0`；任务替换固定使用 `rms_mix_rate=0`，使替换人声沿用原人声的音量包络：

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

## 合轨导出

四轨分离并替换人声后，把替换后的人声与 drums/bass/other 合成一首成品。音频处理步骤与原
实现沿用相同的响度补偿与限幅参数，执行器使用仓库统一的 async 子进程写法。
RVC 单声道人声先显式复制为双声道并做 `equalizer f=3000 t=q w=1 g=2.5`，再四路
`amix=inputs=4:duration=longest:dropout_transition=0:normalize=0`；随后用 `loudnorm`
测出成品与原曲的响度差、按最大 ±12 dB 补偿，最后过一次
`alimiter=limit=0.891251`（−1 dBFS 真峰、latency 补偿）。输出沿用原曲的采样率与声道数
（PCM 原曲沿用位深，其它格式回退 16-bit）。**静音或响度测不出来的输入直接失败**，
不会静默按 0 dB 出成品。

入口与 `/split`、`/replace` 同构：任务级、异步执行、进度走任务状态与 SSE，可取消。
音轨取自任务结果（`fullTrack`、`replacedVocal`、drums/bass/other），不需要请求体。
**合轨必须基于已替换的人声**：没有 `replacedVocal`（未替换，或替换产物已被删除）时返回 409
并提示先完成替换；不提供"用原始 vocal 分轨直接合轨"的回退。

```bash
curl -X POST 'http://127.0.0.1:8010/api/jobs/<jobId>/mix?song=0'
```

- 成品写入 `output/jobs/<jobId>/song_<n>/<人声名>_rvc_mix.wav`（与 `fullTrack`、各分轨
  同一目录），以 `mixedTrack` 暴露，并写入该歌曲 `waveforms` 的 `mix` 车道（640 bin）。
  元数据先落盘、成品文件后原子替换；若进程恰好死在两步之间，重启时会摘掉指向不存在文件的
  引用（不会对外报"有成品"却 404）。重启时还会清理被硬杀留下的 `.mix-*` 中间目录；隐藏目录
  （`.trash`、`.mix-*`）一律不通过 `/output` 对外提供。
  `GET /api/jobs` 的 history 投影里同样带 `mixedTrack`，可直接播放；历史列表按既有约定
  不返回波形，也不返回合轨运行态。
- 文件名固定、同名覆盖：混音在临时目录完成后才原子替换成品，因此**失败、超时、取消都不会
  破坏上一版**。`job.json` 只在成功路径上更新 `mixedTrack` 与波形，顺序是先写元数据、再替换
  文件（这样写盘失败时成品一个字节都没动）；两步之间被杀进程的窗口由上一条的重启修复兜底。
- 音轨路径由任务结果给出，只校验形状（必须在 output 根内、正好在该歌曲目录、不在 `.trash`、
  扩展名属于媒体白名单）；绝对 URL 按 `/output/` 之后的部分定位本地文件，与 `/split`、
  `/replace` 处理分轨 URL 的方式一致，不会发起任何请求。
- 与生成、拆轨、人声替换共享同一并发额度：额度用满返回 429；同一任务已有音频操作在跑返回
  409。合轨进行中会拒绝该任务的拆轨、替换人声、分轨删除/恢复与替换产物删除/恢复。
- 入参不合法时返回 400（路径形状/任务 id）、404（输入文件不存在/不可读、歌曲不存在）、
  409（任务未完成、缺音轨、模型已变更、已有音频操作在跑）、429（额度用满）。超时阈值为
  `RVC_MIX_TIMEOUT_SECONDS`（默认 180 秒），也可从 `GET /api/health` 的
  `mixing.timeoutSeconds` 读取；同处还有 `mixing.modelGuardEnforced`，用于判断当前实例的
  模型指纹守卫是否真的生效（缺资产时为 false）。混音本身的失败或超时（含静音输入）不改变 HTTP
  状态：任务级入口已经返回 202，结果通过 `mixStatus=failed` 与 `mixError` 暴露，
  错误只返回通用文案，细节写日志。`PATCH /api/jobs/<jobId>` 可取消：立即标记
  `mixStatus=cancelled`，回收 FFmpeg 进程并丢弃半成品后再释放并发额度。
- 波形提取失败不影响成品：成品照常落盘可播放，只是该歌曲没有 `mix` 车道，日志里会记一条
  warning。
- 输入变了就作废引用，让"是否过期"机器可判定。以下四种情况都会清掉 `mixedTrack` 与该歌曲的
  `mix` 车道，`mixStatus` 随之由结果推导为 `null`——客户端据此要求用户重新合轨，不必自己猜：
  **重新替换人声成功**、**替换结果被判定失效**（模型指纹不符，见下）、**删除任一分轨**
  （人声/鼓/贝斯/其它都是合轨输入）、以及 **`DELETE /api/voice/result` 软删除替换人声或成品
  本身**。成品文件保留在磁盘上（与替换人声旧文件同一取舍：不删、只是不再被引用），下一次合轨
  会覆盖同名文件。可逆操作（分轨 `PUT`、替换产物 `PUT`）会把作废的成品引用与车道一起还原。
- **替换失败、超时或被取消时，成品引用会还原**：人声引用按既有契约不还原（判定失效即撤下，
  见 `tests/integration/test_replace_history.py`），但成品是已经完成、文件也没被动过的产物，
  还原后 `mixStatus` 仍为 `succeeded`、可继续播放，只是重新合轨会 409（需要先有有效的人声）。
  只有替换**成功**才会真正作废成品，那时才需要重新合轨。
- `DELETE /api/voice/result` 只用于派生音频：删母带（`fullTrack`）返回 409；删成品本身会让
  记录停止宣称该成品，`PUT` 撤回时连引用与 `mix` 车道一起还原。
- **重新拆轨不会作废成品**：重拆只是把同一批分轨重新提取一遍，成品内容仍然成立，`mix` 车道
  也照旧保留。
- `mixedTrack` 与 `mix` 车道成对出现，但有两个例外要按字面理解：波形提取失败时成品照常发布
  而没有 `mix` 车道（见下）；`GET /api/jobs` 的 history 投影按约定不返回波形。客户端不要用
  `waveforms["mix"]` 的存在与否判断"有没有成品"，请直接看 `mixedTrack`。
- 替换执行期间若进程重启：准入阶段已把 `replacedVocal` 与 `mixedTrack` 的引用摘掉并落盘，
  重启后保持"没有有效替换人声"（两份文件都还在盘上），需要重新替换。这是刻意的安全默认：
  不复活一份无法验证的替换人声。
- 替换人声的模型指纹（`RVC_MODEL_PATH`/`RVC_INDEX_PATH`/`RVC_MODEL_VERSION` 变了）与结果里
  记录的不符时，合轨返回 409 并提示先重新替换：否则会用旧模型的人声渲染出一个看起来正常的
  成品。这一点与 `/replace` 判定缓存失效的判据一致。
  **只在模型资产确实存在时才做这层比较**：指纹在文件缺失时把 `missing` 拼进摘要，缺资产的
  实例（`.dockerignore` 排除了 `assets/rvc`，权重靠运行时挂载）算出的指纹与记录值必然不同，
  若据此拒绝，就会把只依赖 ffmpeg 的合轨锁死，还给出一个必然失败的补救动作。此时放行合轨
  并记一条 warning。

任务状态里合轨以 `mixStatus`、`mixSong`、`mixError` 暴露，运行期间顶层
`status/stage/progress/message` 与拆轨、替换人声一致地反映当前操作（混音阶段 `progress`
从 76 起，波形阶段 90，完成 100）。`mixStatus` 与 `replaceStatus` 一样不落盘：重启后由
结果里的 `mixedTrack` 推导为 `succeeded`；此时若只有一首歌有成品才给出 `mixSong`，多首
都有成品时 `mixSong` 为 `null`——客户端应直接读每首歌自己的 `mixedTrack` 判断。

合轨沿用原版滤镜图，最后一步没有按原曲时长截断（`amix=duration=longest`），因此成品可能
比原曲长几十毫秒（Demucs 输出的 mp3 分轨带编码器补零），`mix` 车道的时轴也随之略长于其它
车道。这是原版既有特性，本次未做改动；需要严格对齐时应在最后一步加 `-t <原曲时长>`。

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
