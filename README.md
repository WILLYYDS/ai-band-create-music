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

ElevenLabs 直接接收完整的结构化歌词和扩充风格标签，由模型自行安排段落与演唱节奏。
未传 `durationSeconds`，且 `durationMinutes` 为 `auto`、`null` 或未传时，
不发送 `music_length_ms` 或固定目标时长，由模型根据 Prompt 决定时长。
`durationSeconds` 优先于 `durationMinutes`；显式指定时长时发送 `music_length_ms`。
自动时长返回的音频须至少为 3 秒，低于该下限会报错，不保存为成功歌曲。
后端不会裁剪歌词；组合后的 Prompt 超过接口的 4100 字符上限时会在
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
除 MiniMax 和 ElevenLabs 外，其他 Provider 在未指定时长时使用 `DEFAULT_DURATION_MINUTES`。
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
`replaceStatus`、`replaceSong`、`replaceError`。顶层 `status/stage/progress/message`
沿用既有兼容契约：操作 `pending/running` 期间覆盖为当前操作的四字段；操作进入终态后，
顶层回到主音乐生成任务。历史列表与落盘记录始终表示主生成任务，操作失败不会污染歌曲生成状态。
POST、GET 与 SSE 新增独立操作字段：`operation`（`split` / `replace` / `mix`）、
`operationStatus`、`operationSong`、`operationStage`、`operationProgress`、`operationMessage`。
三种操作的进行中和终态均使用这套字段；`operationSong` 与对应的 `splitSong/replaceSong/mixSong`
一致，都是从 0 起的序号。既有操作 Status/Song/Error 字段保留。
操作字段选择当前操作或本进程最近一次操作，运行态不落盘；重启后复用既有 `replaceStatus/mixStatus`
推导补齐成功状态、`completed` 和 100。同时存在替换与合轨产物时优先报告下游合轨；
同类操作只有唯一一首有可用产物时才恢复歌曲序号；多首都有时为 `null`，应查看各首结果。失败/取消记录不跨重启保留，
没有可推导操作时新字段为 `null`；分轨缓存 POST 仍会补齐成功状态和歌曲序号。
替换依次经过 `operationStage=preparing_vocal`（准备输入音频，0）、
`replacing_vocal`（RVC 整体转换，25）、
`creating_replacement`（替换音轨与波形，50）、`exporting_replacement`（进入试听导出时 75、导出完成后 99）。
RVC 不暴露独立的特征分析回调或内部推理进度，只报告阶段里程碑；
结果保存可用后才报告 `operationStage=completed`、`replaceStatus=succeeded`、
`operationStatus=succeeded`、`operationProgress=100`。完成后歌曲结果增加 `replacedVocal` 音频 URL。转换超过
`RVC_CONVERSION_TIMEOUT_SECONDS`（默认 1800 秒）会立即标记失败；由于 RVC 推理线程不能安全
强停，锁和并发额度会保留到线程实际退出，期间产物不会发布。这种"已经终态但线程还没退出"的
替换数量可以从 `GET /api/health` 的 `replacement.workersHoldingCapacityAfterTerminal` 读到，
日志里也会在进入收尾和线程退出时各记一条 warning。`PATCH /api/jobs/<jobId>` 可取消：
接口会立即把 `operationStatus/operationStage` 和 `replaceStatus` 标记为 `cancelled`，
`operationProgress` 置空；主生成任务的顶层字段及历史保持成功。当前推理安全退出后会丢弃产物、释放并发额度。

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
  `GET /api/jobs` 的 history 投影里同样带 `mixedTrack` WAV 母带；播放优先使用
  `playback.mixedTrack` MP3，缺失时才回退 WAV。历史列表按既有约定不返回波形，
  也不返回合轨运行态。
- 文件名固定、同名覆盖：混音在临时目录完成后才原子替换成品，因此**失败、超时、取消都不会
  破坏上一版**。`job.json` 只在成功路径上更新 `mixedTrack` 与波形，顺序是先写元数据、再替换
  文件（这样写盘失败时成品一个字节都没动）；两步之间被杀进程的窗口由上一条的重启修复兜底。
- 音轨路径由任务结果给出，只校验形状（必须在 output 根内、正好在该歌曲目录、不在 `.trash`、
  扩展名属于媒体白名单）；绝对 URL 按 `/output/` 之后的部分定位本地文件，与 `/split`、
  `/replace` 处理分轨 URL 的方式一致，不会发起任何请求。
- 与生成、拆轨、人声替换共享同一并发额度：额度用满返回 429；同一任务已有音频操作在跑返回
  409。合轨进行中（含试听编码）会拒绝该任务的拆轨、替换人声、分轨删除/恢复与替换产物
  删除/恢复。试听编码进入并发闸门前可能等待，此时仍占用合轨额度；编码本身最长 120 秒。
- `operationStage` 合轨阶段依次为 `mixing`（`operationProgress=76`）、
  `waveform`（`operationProgress=90`）、`preview`（`operationProgress=95`）
  和 `completed`（`operationProgress=100`）。`preview` 时 WAV 已发布，
  但 `mixStatus` 仍为 `running`；试听 MP3 就绪或回退处理完成后才发送 SSE `done`。
  前端应容忍 `operationStage=preview`（运行期顶层也为 `stage=preview`）；试听失败时 `playback.mixedTrack` 缺失，继续使用 WAV。
- 入参不合法时返回 400（路径形状/任务 id）、404（输入文件不存在/不可读、歌曲不存在）、
  409（任务未完成、缺音轨、模型已变更、已有音频操作在跑）、429（额度用满）。超时阈值为
  `RVC_MIX_TIMEOUT_SECONDS`（默认 180 秒），也可从 `GET /api/health` 的
  `mixing.timeoutSeconds` 读取；同处还有 `mixing.modelGuardEnforced`，用于判断当前实例的
  模型指纹守卫是否真的生效（缺资产时为 false）。混音本身的失败或超时（含静音输入）不改变 HTTP
  状态：任务级入口已经返回 202，结果通过 `mixStatus=failed` 与 `mixError` 暴露，
  错误只返回通用文案，细节写日志。`PATCH /api/jobs/<jobId>` 在 `mixing`/`waveform`
  阶段取消会标记 `mixStatus=cancelled`、回收 FFmpeg 进程并丢弃半成品；在 `preview`
  阶段只取消试听编码，已发布的 WAV 保留，任务仍以 `succeeded` 收尾。
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

任务状态里合轨以 `mixStatus`、`mixSong`、`mixError` 暴露，独立的 `operation*` 字段
与拆轨、替换人声使用同一规则，包括成功、失败和取消；只有运行期兼容覆盖顶层字段，
终态回到主生成状态。
合轨保留既有百分比：混音阶段 `operationProgress` 从 76 起，波形阶段 90，完成 100。`mixStatus` 与 `replaceStatus` 一样不落盘：重启后由
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
群聊里选定的歌曲名；未传时会在歌词与风格扩写时生成简短歌名。标题随任务持久化，
并由 `GET /api/jobs` 和任务 SSE 返回，供历史页显示。

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

SSE 终态始终使用具名 `done` 事件，客户端收到后应主动关闭 EventSource。若已连接的任务
在流中到期，`done` 的数据为 `{"success":false,"status":"expired","jobId":"...",
"message":"歌曲已过期。"}`，不包含音频结果；`expired` 仅表示该 SSE 终态的原因，
不会写入任务的持久化状态。连接前已过期的任务仍返回 HTTP 404。

ElevenLabs 优先使用 `/v1/music/stream`，收到首块音频前显示等待状态；接收期间
`stage=receiving_audio`，`receivedAudioSeconds` 表示当前这一首已收到的 PCM 音频秒数，
`expectedAudioSeconds` 表示请求的目标秒数（自动时长为 `null`）。进度更新通常最多每秒一次，
首块立即推送，流结束并原子保存音频后补齐最后一次更新。固定时长的 `progress` 按整个任务
聚合已完成歌曲数和当前歌曲接收比例，后处理及歌曲切换期间保留进度，最高为 99；
自动时长保持 `progress=null`，可直接展示 `message` 中的已接收秒数。
这表示音频接收量，不代表上游模型内部推理进度。双曲切换和后处理阶段清空
`receivedAudioSeconds`/`expectedAudioSeconds`，`progress` 保留当前任务进度；
所有歌曲均已完成或失败后，只要至少一首可用，任务就以 `progress=100` 结束。
MiniMax 继续使用上游回传的 `step/totalSteps`。

**硬性发布门槛：按首发布与部分成功改变了已发布客户端的响应契约，必须与支持这些
行为的前端同版本发布，禁止只升级后端。无法保证配套发布时，本版本不得上线。**
上线前须联调确认：运行中的 `result` 不会被当作任务成功或导致 SSE 关闭；
第二首未完成时不会按当前可用输出数夹取或覆盖用户的歌曲选择；部分成功按
`songNumber` 映射输出与状态，并保留可用歌曲。本仓库的后端测试不能代替前端联调。

双曲任务按首串行生成，每首完成文件保存、试听处理和波形提取后立即发布：
`GET /api/jobs/<jobId>`、历史列表和 SSE 的 `result` 都包含当前已完成的歌曲，
此时顶层 `status` 仍可为 `running`，不能仅凭 `result` 存在就关闭 SSE。
`currentSong` 是当前正在处理的歌曲序号（从 1 开始，终态为 `null`）；
`songStates` 按序号分别提供 `songNumber/status/stage/message/progress/error`，以及
每首自己的 `step/totalSteps` 和音频接收秒数。顶层 `progress` 是任务级进度。
顶层 `stage` 增加 `song_completed`/`song_failed`，两者仍属于任务运行阶段。
`songStates` 长度为实际尝试的曲数，可能小于 `requestedCount`（例如只支持单曲的 provider）；
首个进度上报前可能为空数组，请以 `songStates[].songNumber` 区分歌曲。
`songStates[].progress` 是单首进度（固定时长可用百分比，自动时长为 `null`，完成为 100）。
客户端可立即展示第一首播放器，并按 `currentSong` 显示第二首的进度。
SSE 仅在 `song_completed` 帧及终态帧附带波形，其余中间帧的 `waveforms` 为 `{}`，
不表示已保存的波形被删除。帧可能合并推进到下一阶段；新订阅或未收到歌曲完成帧的
客户端需要立即展示波形时，请查询 `GET /api/jobs/<jobId>` 获取完整结果。

第二首失败不会撤回第一首：任务以 `status=succeeded` 收尾，`warning` 说明部分失败，
`songStates[1].status=failed` 和 `error` 保存第二首的错误。两首都失败才令任务 `failed`。
明确的 provider 全局错误例外：鉴权、套餐、配额或限流错误（HTTP 401/402/403/429）以及
缺少必需配置会终止后续歌曲调用，任务标为 `failed`，但已经发布的歌曲仍可播放和下载。
ElevenLabs 的 402/403 流式端点拒绝仍先单次回退 compose；回退也失败才按全局错误终止。
生成被取消或服务器重启时，已经发布的歌曲也保留，未完成的歌曲分别标为 `cancelled` 或 `failed`。
`result.count` 始终等于可用歌曲数（`1 + alternatives.length`），`requestedCount` 是请求数；
每个输出的 `songNumber` 表示原始序号。因此第一首失败、第二首成功时，`result.count=1`、
`result.songNumber=2`，`alternatives=[]`。编辑接口的 `song` 仍是可用输出列表的零基索引，
这种情况下编辑第二首应传 `song=0`。音频编辑操作仍需等待任务结束。

首块音频到达前，上游不提供推理进度；本服务不会周期更新等待秒数，`message` 保持等待文案。
这段等待期间 SSE 仍约每 15 秒保活；保活表示连接存活，不表示上游生成取得了新进展。
流式端点返回 402/403/404/405 时，仅回退一次到 `/v1/music`，使用相同参数；
超时、断流或其它错误不重试。回退记录写入 `prompts.json` 的
`providerRequests[].streamFallbackStatus`，被拒响应的状态码、响应头与响应体保存在
`providerRequests[].streamFallbackResponse`；条目按 `variation` 区分，`variation=0` 对应第 1 首。
实际使用的端点记录在结果的 `debug.music.endpoint`。compose 回退可能集中返回音频，
因而无法保证持续更新接收量。进度观察回调失败只记日志，取消仍向上传播；完整音频已保存后
发生取消时保留该文件。2026-09-29 使用真实 Key 实测流式端点：`music_v2`、
`pcm_44100`、`music_length_ms=3000`、纯钢琴器乐请求返回 HTTP 200；408 个音频块共
529,200 字节，保存后的 WAV 为 44.1 kHz、16 位双声道、3.000 秒。

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

### 创作保留与清理日志

永久清理默认关闭（`SONG_RETENTION_ENABLED=false`），默认仅审计（`SONG_RETENTION_DRY_RUN=true`）。
**正式启用会追溯全部历史：第一次启动就会永久删除所有超过 N 天的既有任务及音频，无法恢复。**
先备份完整 `output/`，设置 `SONG_RETENTION_ENABLED=true` 并保持 `SONG_RETENTION_DRY_RUN=true`
重启，检查日志 `wouldDeleteJobIds`、跳过的任务以及 health 中的孤立目录；确认影响范围后，
才将 `SONG_RETENTION_DRY_RUN=false` 并再次重启，正式执行删除。

关闭或 dry-run 时不删除、不移动目录、不隐藏历史、不限制访问，试听缓存沿用原有策略。
关闭时任务 HTTP/SSE 响应不返回 `expiresAt`、`retentionState`、`audioAvailable`，保留原有响应结构；
开启 dry-run 时返回这三个字段，`expiresAt=null`。dry-run 启动时及每天凌晨 03:00 仅记录候选
`wouldDeleteJobIds`。正式启用前需 PO 在 PR 或上线工单明确确认追溯删除，并完成前端对新增字段、
`done/status=expired`（无结果）和音频 404 的兼容验证。`SONG_RETENTION_DAYS` 范围为 1–3650 天。

正式启用后，创作从任务 `createdAt` 起可访问连续 72 小时（`SONG_RETENTION_DAYS=3`），播放、
拆轨、换声和合轨不延长访问期。历史、详情和 SSE 响应提供 `expiresAt`，用于前端提前提示。
到期且执行结束后任务不再出现在历史中，详情、编辑和新音频下载请求返回 404；试听 MP3 的缓存时间
不超过剩余访问期。已经开始的下载可完成，已下载或此前已缓存到客户端的内容无法追回。

正式启用后，服务启动时在后台清理一次，不阻塞 API 就绪；随后每天北京时间凌晨 03:00 清理。
正常情况下文件在创建后 72–96 小时内删除，容量规划应按接近 4 天计算；仍在执行的任务
或删除失败的目录可能保留更久。访问期为 72 小时，磁盘释放时间取决于清理执行。
清理包含整个任务目录 `output/jobs/<jobId>/` 和回收区 `output/.trash/<jobId>/`，
一次任务中的所有歌曲一并永久删除。
仍在生成、拆轨、换声或合轨的任务跳过本轮，实际执行结束后在下次清理时删除；换声超时
但推理线程仍在运行时也不会删除。创建时间缺失、无时区或格式异常的历史记录不会被猜测删除，
需要根据日志人工修复。过期但仍在执行的任务继续出现在历史中，详情、SSE 和 PATCH 取消
仍可使用，SSE 不会提前发送 `done/expired`；新编辑和音频下载仍被拒绝，实际执行结束后再清理。
RVC 推理线程无法强制停止，取消仅标记取消意图，并发额度仍等到实际线程退出后释放。
没有任务元数据的孤立文件不会按文件修改时间自动删除；正式启用时，这些任务及无法计算
到期时间的任务音频返回 404。孤立目录列在启动日志和 health 中，需人工修复或处理。

策略开启后的任务响应中，`retentionState` 明确区分 `dry_run`（仅审计）、
`retained`（可访问）、`expired_active`（已过期但仍在执行）和 `unmanaged`（创建时间异常）。
`audioAvailable` 表示任务是否已有结果且保留策略允许音频访问，不代表对每个音频文件做了
磁盘校验。正式清理时，`unmanaged` 任务仍可查看以便修复，但 `audioAvailable=false`；
前端应显示“元数据异常，需修复”，禁用播放/下载，不能把 `expiresAt=null` 当作无限期可播放。
修复 `job.json` 后重启服务重新加载；不会猜测时间，也不会自动删除这些任务。

删除前先将目录移入不可访问的 `output/.expired/<jobId>/`；删除失败会保留目录，下一次清理
（含服务重启）重试，避免把部分删除的任务重新加载到历史中。
暂存前先校验全部源路径、目标路径及目标冲突。重试暂存区条目时，若任务仍被加载且按当前
保留天数尚未过期（或创建时间不可解析），跳过删除；已移走元数据的任务仍可继续重试。

`output` 树内（包括 `jobs`、`.expired`、`.trash` 和任务目录）不支持符号链接。
保留清理拒绝沿符号链接操作；指向 `output` 之外的链接，其 `/output` 音频请求返回 404。
音频放在其它卷时，请将卷直接挂载到 `OUTPUT_DIR` 或其真实子目录，不要用符号链接转接。
后台首轮清单检查 `jobs` 和 `.expired` 布局，发现链接时记录一次 WARNING 的
`unsupported_layout`；重复刷新同一问题不重复该告警。health 的缓存字段 `unsupportedLayout`
为 `true`，`unsupportedLayoutDirectories` 只列 `jobs`/`.expired` 相对名称；错误原因明确包含
`refusing symlinked directory`。首次检查前或无法确认布局时为 `null`，修复后下一次刷新恢复
`false`；这些根目录检查不代表完整扫描了树内所有符号链接。

清理日志保存到 `output/logs/retention.log`，同时接入现有应用日志；每行是一个 JSON 对象，
`time` 使用带时区的 UTC 时间。启动时记录开关、天数、清理时区和下次计划清理时间；
文件日志初始化失败时告警并降级到应用日志，不阻止 API 启动。
日志记录清理开始、删除尝试、每个任务删除成功/失败或跳过的
时间与原因，结束时汇总 `deletedJobIds`、`failedJobIds` 和 `skippedJobIds`。没有过期任务时也
记录开始和结束。单文件最大 5 MiB，保留 3 个轮转备份；日志不会随歌曲清理，也不通过
`/output` 对外提供。仍须保持单 worker、单实例写入该 output 目录。

`cleanup_failed` 记录扫描或整轮错误：`stage=pendingDeletionScan` 表示放弃暂存区残留重试，
仍继续处理内存中的已知过期任务；`stage=run` 表示本轮中止。`cleanup_finished` 包含
`result`、`error` 和 `pendingDeletionScanFailed`。暂存区本身损坏或无法写入时，无法安全移动
目录，相应任务记录 `job_delete_failed` 并进入 `failedJobIds`，保留原文件以便修复后重试。

批次汇总清单最多包含前 20 个 jobId，并附带对应的 `Count` 和 `Truncated`；dry-run 的
`wouldDeleteJobIds` 也遵循这个限制。核对完整范围时查看逐任务 `job_would_delete`、
`job_deleted`、`job_delete_failed` 和 `job_skipped` 记录，不能只依据截断后的汇总。

`GET /api/health` 的 `retention` 返回 `enabled`、`days`、`cleanupTimezone`、`nextCleanupAt`、
`cleanupRunning` 和 `remainingJobs`。关闭时下次清理时间为 `null`；剩余数量包含正在删除的
任务，仅代表本轮待处理数量（本轮跳过/失败的任务见日志）。
`lastCleanupAt` 是上一轮结束的 UTC 时间，`lastCleanupReason` 是触发原因，
`lastCleanupResult` 为 `success`、`partial_failure`（有任务处理成功，但仍有错误）、`failed`
或 `cancelled`；关停时完成当前删除后停止批次，剩余任务留待下次清理，结果为 `stopped`。
`lastCleanupError` 给出脱敏后的首个错误，`pendingDeletionScanFailed`
报告上一轮是否无法扫描暂存区。尚未执行时这些字段为 `null`；本轮执行期间继续显示上一轮结果。
dry-run 下 `success` 仅表示审计完成，未删除或移动任何内容；`partial_failure` 也仅表示部分
审计完成。正式模式的 `success` 表示本轮删除完成或无需删除。判断删除是否已执行必须同时
核对 `dryRun`、`deletedJobIds` 和 `wouldDeleteJobIds`，不能只依据结果枚举。

health 只读取缓存，不在请求中扫描目录或解析全部任务时间。后台线程在启动时、每小时、
以及清理后刷新清单，策略关闭时也会刷新。`inventoryUpdatedAt` 仅在整轮扫描成功时推进，
`inventoryAttemptedAt` 表示最近尝试时间。首次刷新前时间、计数和 `Truncated` 为 `null`，
`inventoryStale=true`。正常刷新时清单最多滞后一小时，运行标志与本轮剩余数量仍实时返回。

health 和清单日志报告 `dryRun`、`expiredActiveJobIds`、`orphanJobIds`、`pendingDeletionJobIds`
及 `unmanagedJobIds`。每类清单最多 20 条，分别附有 `expiredActiveCount/Truncated`、
`orphanCount/Truncated`、`pendingDeletionCount/Truncated` 和 `unmanagedCount/Truncated`。
`unmanagedJobs` 给出前 20 个任务的具体原因，包括创建时间异常、元数据缺失或 `job.json`
JSON/字段校验失败；health 的错误原因不包含原始作品内容或绝对路径，I/O 错误仅报告
异常类型、errno 和标准说明。完整元数据加载错误及路径见应用日志。
JSON 语法错误包含解析说明与行列，字段错误用点分隔字段名（如 `result.fullTrack: missing`），
目录与 `jobId` 不一致明确报告 `jobId does not match directory`。
关闭策略时 `.expired` 残留不会继续删除，但仍报告其摘要；重新正式启用后才重试。
三类扫描（待删除目录、孤立/异常元数据任务、活跃任务）独立进行。失败时 `inventoryError`
给出原因、`inventoryStale=true`，失败类别保留上次成功数据并标记对应的 `*Stale=true`；
没有成功数据的类别，其 `Count` 和 `Truncated` 保持 `null`。其它类别仍刷新。
计数、空列表或非空 `inventoryUpdatedAt` 均不能单独作为“确认没有残留”的依据。

后台循环遇到意外异常会记录 `retention_monitor_failed` 和应用堆栈，在下次清单刷新间隔重试，
不会因单轮异常永久停止调度；正常取消仍向上传递。此功能不更改全局日志或 httpx 日志级别；
文件日志不可写时，仅 retention 自身增加 stderr handler，关停时释放。

音频接口先非阻塞打开并确认普通文件，再恢复阻塞读取；FIFO、目录和 Unix socket 返回 404。
普通文件不存在返回 404，权限不足返回 403，文件描述符耗尽返回 503（`Retry-After: 5`），
其余读取前 I/O 故障返回 500 并记录堆栈。服务端故障不能作为“歌曲已过期”的信号；
响应头已发送后发生的磁盘读取故障只能中断流，客户端应检查下载是否完整。

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

拆轨的 `operationStage` 依次经过 `starting_split`（读取输入，0）、`splitting`（Demucs，25）、
`waveform`（真实波形，50–75）、`preview`（每轨试听 MP3，75–99）和 `completed`。
`operationProgress` 按实际处理波形/试听轨道数更新；只有结果可用且操作成功时才报告 100、
`splitStatus=succeeded` 和 `operationStatus=succeeded`。POST、GET 与 SSE 共用独立操作字段，
SSE 按 `operationStatus` 判断音频操作是否结束，主生成任务成功不会提前关闭操作流。
SSE 保存阶段快照，队列最多 16 帧；慢客户端积压时丢弃最旧更新，失败/取消优先于缓冲帧。
生成进度通知可合并，队列非空表示消费者已有待处理通知；所有发布点使用同一 helper。
后端不人为延时。`preview` 阶段的 `result.stems`、`splitEnabled`、`waveforms` 已经写入，
`splitStatus` 仍为 `running`：

- **`preview` 阶段取消只取消编码器**：`PATCH /api/jobs/{jobId}` 仍会取消该阶段的
  ffmpeg 编码任务，因为 WAV 分轨与波形都已落盘，`splitStatus` 不被改写，任务照常以
  `succeeded` 收尾，只是 `result.playback.stems` 缺少试听 MP3（客户端回落到 WAV）。
  这样 PATCH 的响应与最终状态不会互相矛盾。输入读取、`splitting`/`waveform` 阶段取消才是终态：
  `splitStatus=cancelled`。
- **分轨进行中（`splitStatus` 为 `pending`/`running`，含 `preview` 窗口）拒绝删除/恢复
  分轨**，返回 `409`：此时编码器正在读写 `playtrack/`，删除会丢失播放引用，恢复会永久
  降级为 WAV。`/split`、`/replace` 的同类冲突防护不变。
- 单轨试听编码失败或被取消只影响那一轨（该 key 不出现在 `result.playback.stems`），
  不会把整个分轨判为失败。

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
