# City-Parse
利用小参数LLM从文章标题里提取城市名称

## 为什么做这个项目？
在洗数据的时候往往会需要从文本中提取城市名称，但是内容不规则，不能很好用正则表达式去匹配，目前我想到最简单的方式时通过LLM对文本内容进行提取。直接通过调用OpenAI (or DeepSeek)效果固然好，但是如果有很多的数据要处理，那么费用会很高，而小参数的模型完全能胜任工作。因此该项目旨在将 **从文本（主要是标题）里的城市提取出来** 这一任务SOP，包装成开箱即用的模块以避免重复造轮子。

## 哪些数据用了这个项目
- [ODS-TalentPolicy](https://github.com/statamcp-team/ODS-TalentPolicy): 中国地方人才引进政策整理

## 如何使用？

### 基础安装
克隆项目到本地，然后同步依赖：
```bash
git clone https://github.com/sepinetam/city-parse.git
cd city-parse
uv sync  # 基础依赖（ollama + openai）
```

### 扩展安装
如果你想使用 HuggingFace 或 ModelScope 模型，需要安装可选依赖：
```bash
# 安装 HuggingFace 扩展
uv sync --extra huggingface

# 安装 ModelScope 扩展
uv sync --extra modelscope

# 或者安装所有依赖（包括扩展）
uv sync --extra all
```

### 使用方式
将需要提取的信息构成列表，这里假设你将所有的文本构建成了 `raw_data.csv` 文件，且第一行为索引（数据格式参考 `raw_data.example.csv` ）

然后根据你的模型源运行：

1. **使用 Ollama 模型**（默认）：
   ```bash
   # 确保 ollama 服务正在运行
   uv run main.py
   ```

2. **使用 HuggingFace 模型**：
   ```python
   # 修改 main.py 中的模型配置
   parser = Parse(
       model_id="Qwen/Qwen2.5-0.5B",  # 或其他 HuggingFace 模型
       source=ModelSource.HUGGINGFACE,
       system_prompt="你是一个专业的城市名称提取助手..."
   )
   ```

3. **使用 ModelScope 模型**：
   ```python
   # 修改 main.py 中的模型配置
   parser = Parse(
       model_id="qwen/Qwen2.5-0.5B",  # 或其他 ModelScope 模型
       source=ModelSource.MODELSCOPE,
       system_prompt="你是一个专业的城市名称提取助手..."
   )
   ```

### 作为包安装使用
```bash
# 基础版本
pip install city-parse

# 包含 HuggingFace 扩展
pip install city-parse[huggingface]

# 包含 ModelScope 扩展
pip install city-parse[modelscope]

# 包含所有扩展
pip install city-parse[all]
```

## 支持的模型源

### 1. Ollama（推荐用于本地部署）
```python
from city_parse.core import Parse, ModelSource

parser = Parse(
    model_id="qwen2.5:0.5b",
    source=ModelSource.OLLAMA,
    system_prompt="你是一个城市名称提取助手..."
)
```

### 2. OpenAI（推荐用于生产环境）
```python
parser = Parse(
    model_id="gpt-3.5-turbo",
    source=ModelSource.OPENAI,
    system_prompt="你是一个城市名称提取助手..."
)
```

### 3. HuggingFace（推荐用于研究和定制）
需要安装扩展依赖：`uv sync --extra huggingface`

```python
parser = Parse(
    model_id="Qwen/Qwen2.5-0.5B",  # 或其他 HuggingFace 模型
    source=ModelSource.HUGGINGFACE,
    system_prompt="你是一个城市名称提取助手...",
    device="cpu",  # 或 "cuda", "mps"
    temperature=0.1
)
```

### 4. ModelScope（推荐用于国内用户）
需要安装扩展依赖：`uv sync --extra modelscope`

```python
parser = Parse(
    model_id="qwen/Qwen2.5-0.5B",  # 或其他 ModelScope 模型
    source=ModelSource.MODELSCOPE,
    system_prompt="你是一个城市名称提取助手...",
    device="cpu",  # 或 "cuda", "mps"
    temperature=0.1
)
```

**Transformers 特性（HuggingFace + ModelScope）：**
- 🔄 **智能缓存**：模型下载一次，后续使用缓存
- 🔒 **哈希验证**：确保模型文件完整性
- 🚀 **进程优化**：每个进程只验证一次模型
- ⚡ **支持量化**：4bit/8bit 量化降低内存使用
- 📱 **多设备支持**：CPU/CUDA/MPS
- 🏗️ **共享架构**：统一接口，易于维护
- 📦 **模块化设计**：避免代码重复

### 推荐的模型
| 平台 | 模型 | 参数量 | 内存需求 | 适用场景 |
|------|------|--------|----------|----------|
| HuggingFace | `Qwen/Qwen2.5-0.5B` | 0.5B | ~2GB | 轻量级任务 |
| ModelScope | `qwen/Qwen2.5-0.5B` | 0.5B | ~2GB | 轻量级任务 |
| HuggingFace | `Qwen/Qwen2.5-1.5B` | 1.5B | ~4GB | 平衡性能 |
| ModelScope | `qwen/Qwen2.5-1.5B` | 1.5B | ~4GB | 平衡性能 |

## 最小可修改参数
下面是对main文件的解析


## 硬件配置
> 该部分的配置由ChatGPT提供，项目不对其进行负责。

### macOS
> M-series Chip 指的是2020年后发布的苹果电脑，即购买的时候显示为M1或M2等的设备，如果你不知道可以点击电脑左上角Apple的logo -> 关于本机 进行查看电脑的芯片和内存。

目前仅对Apple Silicon (M-series Chip)进行讨论，过早的Intel系列未进行测试。

| 模型版本        | 参数规模    | 推荐内存 (Unified Memory) |
|-------------|---------|-----------------------|
| Qwen3-0.6B  | 6 亿     | ≥ 8 GB                |
| Qwen3-1.7B  | 17 亿    | ≥ 16 GB               |
| Qwen3-4B    | 40 亿    | ≥ 16 GB               |
| Qwen3-8B    | 80 亿    | ≥ 16 GB（推荐 24 GB ↑）   |
| Qwen3-14B ↑ | 140 亿 ↑ | ≥ 32 GB ↑             |


### Windows & Linux (with NVIDIA GPU)

| 模型版本        | 参数规模    | 最低显存 (VRAM) |
|-------------|---------|-------------|
| Qwen3-0.6B  | 6 亿     | ≥ 4 GB      |
| Qwen3-1.7B  | 17 亿    | ≥ 8 GB      |
| Qwen3-4B    | 40 亿    | ≥ 16 GB     |
| Qwen3-8B    | 80 亿    | ≥ 24 GB     |
| Qwen3-14B ↑ | 140 亿 ↑ | ≥ 48 GB ↑   |

### Windows & Linux (without NVIDIA GPU)
不了解，有待测试

## License
This project is under [MIT LICENSE].

该项目的License基于MIT LICENSE，如有冲突以补充条款为准，以下为补充条款：

- 该项目底层基于各种开源模型本地运行或闭源模型API服务，优先遵循模型提供商的条款，包括但不限于是否被授权商用，是否可以进行二次打包分发等；  
- 基于该项目进行二次开发或分发的，有权声明与本项目之间的关系并在README或LICENSE中著明，且本项目与再分发版本免责；  
- 禁止任何个人或公司以任何形式进行营利，所有人有权免费使用该项目；
- 最终解释权归原作者所有。
