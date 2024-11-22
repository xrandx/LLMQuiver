# LLMQuiver

该辅助工具用于调用在线LLM服务API，支持本地缓存、提示渲染和配置管理。

## 支持功能
- 支持Openai/Azure LLM服务提供商（或兼容Openai的服务提供商）。
- 支持vllm服务。
- 本地缓存（基于sqlite）。
- 提示渲染（基于toml文件）。
- 配置管理（基于toml文件）。

## 待完成
- 多服务提供商支持。

## 安装

```bash
pip install llm-quiver
```

## 基本用法

### 1. 直接调用模式

假设有配置文件`path/to/gpt.toml`，API_KEY 需要自己填具体的，内容如下：

```toml
API_TYPE = "azure_openai"
API_BASE = "https://endpoint.openai.azure.com/"
API_VERSION = "2023-05-15"
API_KEY = "********************************"
MODEL_NAME = "gpt-4o-20240513"
temperature = 0.0
max_tokens = 4096
enable_cache = true
cache_dir = "oai_cache"
```

运行代码：

```python
from llm_quiver import LLMQuiver

# 初始化
llm = LLMQuiver(
    config_path="path/to/gpt.toml",
)

# 文本生成模式
prompt_values = ["你是谁啊"]
responses = llm.generate(prompt_values)
#   角色默认是 system
#   ["我是一个由OpenAI开发的人工智能助手，旨在帮助回答问题、提供信息和完成各种任务。你有什么需要帮助的吗？"]

# 对话模式
messages = [[{"role": "user", "content": "你是谁啊"}]]
responses = llm.chat(messages)
#   ["我是一个由OpenAI开发的人工智能助手，旨在帮助回答问题、提供信息和进行对话。你可以问我任何问题，我会尽力帮助你！"]
```

### 2. Toml模板调用模式

首先需要创建 TOML 模板文件,例如 `hello_world.toml`:

```toml
[hello_world_template]
prompt = "你好 {name},请问你是谁?"
```

然后可以这样使用:

```python
from llm_quiver import TomlLLMQuiver

# 初始化时指定模板
llm = TomlLLMQuiver(
    config_path="path/to/gpt.toml",
    toml_prompt_name="hello_world_template",
    toml_template_file="path/to/hello_world.toml"
)

# 传入模板参数
prompt_values = [dict(name="GPT")]
responses = llm.generate(prompt_values)
```

## 配置说明

有两种方式配置 API 密钥等参数:

1. 通过环境变量:

配置加载可以通过传入参数 config_path="path/to/config.toml" 或者设置环境变量 "export LLMQUIVER_CONFIG=path/to/config.toml" 来实现。API_TYPE、API_BASE、API_VERSION、API_KEY、MODEL_NAME 等参数也可以在环境变量中进行设置。

2. 直接传入配置文件路径:

```python
llm = TomlLLMQuiver(
    config_path="path/to/config.toml",
    toml_prompt_name="template_name",
    toml_template_file="path/to/template.toml"
)
```

配置文件示例:
```toml
API_TYPE = "azure_openai"
API_BASE = "https://endpoint.openai.azure.com/"
API_VERSION = "2023-05-15"
API_KEY = "********************************"
MODEL_NAME = "gpt-4o-20240513"
temperature = 0.0
max_tokens = 4096
enable_cache = true
cache_dir = "oai_cache"
```

## 返回值说明

- generate() 和 chat() 方法都返回字符串列表
- 每个元素对应一个输入prompt的响应结果

## 注意事项

1. 使用前需要正确配置 API 密钥
2. 模板文件需要符合 TOML 格式规范
3. 传入的参数需要与模板中的占位符对应