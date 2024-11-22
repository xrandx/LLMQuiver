# LLMQuiver

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/xrandx/LLMQuiver/blob/master/README_CN.md">简体中文</a>
    </p>
</h4>



The auxiliary tool is used to invoke the online LLM service API, supporting local caching, prompt rendering, and configuration management.


## Supported
- Support Openai/Azure LLM service providers (or openai-compatible service providers).
- Support vllm service.
- Local caching (based on sqlite).
- Prompt rendering (based on toml file).
- Configuration management (based on toml file).

## To Be Done
- Multi service provider support.

## Installation

```bash
pip install llm-quiver
```

## Basic Usage

### 1. Direct Call Mode

Assuming you have a configuration file `path/to/gpt.toml`, you need to fill in your own API_KEY, content as follows:

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

Running code:

```python
from llm_quiver import LLMQuiver

# Initialize
llm = LLMQuiver(
    config_path="path/to/gpt.toml",
)

# Text generation mode
prompt_values = ["Who are you?"]
responses = llm.generate(prompt_values)
#   Default role is system
#   ["I am an AI assistant developed by OpenAI, designed to help answer questions, provide information, and complete various tasks. How can I help you?"]

# Chat mode
messages = [[{"role": "user", "content": "Who are you?"}]]
responses = llm.chat(messages)
#   ["I am an AI assistant developed by OpenAI, designed to help answer questions, provide information, and engage in conversations. Feel free to ask me anything!"]
```

### 2. Toml Template Call Mode

First, create a TOML template file, for example `hello_world.toml`:

```toml
[hello_world_template]
prompt = "Hello {name}, who are you?"
```

Then you can use it like this:

```python
from llm_quiver import TomlLLMQuiver

# Specify template during initialization
llm = TomlLLMQuiver(
    config_path="path/to/gpt.toml",
    toml_prompt_name="hello_world_template",
    toml_template_file="path/to/hello_world.toml"
)

# Pass template parameters
prompt_values = [dict(name="GPT")]
responses = llm.generate(prompt_values)
```

## Configuration Guide

There are two ways to configure API keys and other parameters:

1. Through environment variables:

Configuration can be loaded by passing parameter config_path="path/to/config.toml" or setting environment variable "export LLMQUIVER_CONFIG=path/to/config.toml". Parameters like API_TYPE, API_BASE, API_VERSION, API_KEY, MODEL_NAME can also be set in environment variables.

2. Directly passing configuration file path:

```python
llm = TomlLLMQuiver(
    config_path="path/to/config.toml",
    toml_prompt_name="template_name",
    toml_template_file="path/to/template.toml"
)
```

Configuration file example:
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

## Return Value Description

- Both generate() and chat() methods return a list of strings
- Each element corresponds to a response for one input prompt

## Notes

1. API key must be correctly configured before use
2. Template files must comply with TOML format specifications
3. Input parameters must correspond to placeholders in the template