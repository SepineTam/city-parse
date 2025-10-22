# City-Parse
利用小参数LLM从文章标题里提取城市名称

## 为什么做这个项目？
在洗数据的时候往往会需要从文本中提取城市名称，但是内容不规则，不能很好用正则表达式去匹配，目前我想到最简单的方式时通过LLM对文本内容进行提取。直接通过调用OpenAI (or DeepSeek)效果固然好，但是如果有很多的数据要处理，那么费用会很高，而小参数的模型完全能胜任工作。因此该项目旨在将 **从文本（主要是标题）里的城市提取出来** 这一任务SOP，包装成开箱即用的模块以避免重复造轮子。

## 如何使用？
克隆项目到本地，然后同步依赖：
```bash
git clone https://github.com/sepinetam/city-parse.git
cd city-parse
uv sync  # 或者根据你自己的需求来安装依赖
```

然后你需要将需要提取的信息构成列表，这里假设你将所有的文本构建成了 `raw_data.csv` 文件，且第一行为索引（数据格式参考 `raw_data.example.csv` ）

接下来通过 `uv run mian.py` 运行文件即可。

或者你可以通过pip或者uv进行安装然后自己进行探索和使用（目前还未上传至pypi，暂时不可用）：
```bash
pip install city-parse

# or
uv add city-parse
```

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
