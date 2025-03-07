from typing import List, Dict
from . import io_util
from .wrap_openai import WrapOpenAI
from loguru import logger
from pathlib import Path
import os
from .prompt import prompt_template_parser


class BaseLLMQuiver:
    def __init__(self, config_path: str = None) -> None:
        if config_path:
            self._initialize_by_config(config_path)
        else:
            logger.info("config_path is not set. Try to initialize by environment variables.")
            self._initialize_by_env()

        self.prompt_template = None
        self.prompt_template_type = "basic"

    def read_config(self, config_path):
        config_path = Path(config_path)
        if config_path.exists():
            logger.info(f"LLMQUIVER_CONFIG: {config_path}")
            config = io_util.read_toml(config_path)
        else:
            raise ValueError(f"LLMQUIVER_CONFIG: {config_path} is not found.")
        return config

    def _initialize_by_config(self, config_path: str = None):
        config = self.read_config(config_path)
        self.gen = WrapOpenAI(
            api_type=config["API_TYPE"],
            api_base=config["API_BASE"],
            api_version=config["API_VERSION"],
            api_key=config["API_KEY"],
            modelname=config["MODEL_NAME"],
            temperature=config.get("temperature"),
            top_p=config.get("top_p", None),
            max_tokens=config.get("max_tokens"),
            enable_cache=config.get("enable_cache"),
            cache_dir=config.get("cache_dir"),
            cache_prefix=config.get("cache_prefix", config["MODEL_NAME"]),
            cache_interval=config.get("cache_interval", 0),
        )

    def _initialize_by_env(self):
        config_path = os.environ.get("LLMQUIVER_CONFIG", None)
        if config_path is None:
            raise ValueError(f"LLMQUIVER_CONFIG is invalid: {config_path}.")

        config = self.read_config(config_path)

        keys = ["API_TYPE", "MODEL_NAME", "API_BASE", "API_KEY", "API_VERSION"]
        params = {}
        for key in keys:
            env_value = os.environ.get(key, None)
            config_value = config.get(key, None)
            if env_value and len(env_value) > 0:
                params[key] = env_value
            elif config_value and len(config_value) > 0:
                params[key] = config_value
            else:
                raise ValueError(f"Missing {key} in both the configuration file({config_path}) and environment.")

        self.gen = WrapOpenAI(
            api_type=params["API_TYPE"],
            api_base=params["API_BASE"],
            api_version=params["API_VERSION"],
            api_key=params["API_KEY"],
            modelname=params["MODEL_NAME"],
            temperature=config.get("temperature"),
            top_p=config.get("top_p", None),
            max_tokens=config.get("max_tokens"),
            enable_cache=config.get("enable_cache"),
            cache_dir=config.get("cache_dir"),
            cache_prefix=config.get("cache_prefix", params["MODEL_NAME"]),
            cache_interval=config.get("cache_interval", 0),
        )

    def get_num_tokens_from_string_fn(self):
        if hasattr(self.gen, "num_tokens_from_string"):
            return self.gen.num_tokens_from_string
        else:
            raise NotImplementedError("num_tokens_from_string is not implemented.")

    def render_prompt(self, prompt_value):
        return NotImplemented

    def prepare_prompts(self, prompt_values):
        return NotImplemented

    def generate(
        self, prompt_values: List[str], verbose=False
    ):
        return NotImplemented

    def prepare_messages_list(self, prompt_values: List[Dict]):
        return NotImplemented

    def chat(
        self, messages_list: List[Dict], verbose=False
    ):
        return NotImplemented


class LLMQuiver(BaseLLMQuiver):
    def __init__(self, config_path: str = None) -> None:
        super().__init__(config_path)

    def prepare_prompts(self, prompt_values):
        if not isinstance(prompt_values, list):
            raise TypeError("prompt_values must be a list.")

        if len(prompt_values) == 0:
            return []

        if not isinstance(prompt_values[0], str):
            raise TypeError("LLMQuiver's prompt_values must be a list of str.")

        prompts = prompt_values[:]
        messages_list = [[dict(role="system", content=p)] for p in prompts]
        return messages_list

    def generate(
        self, prompt_values: List[str], verbose=False
    ):
        logger.warning(
            "The generate() method is deprecated and will be removed in a future version. "
            "Please use chat() instead."
        )
        messages_list = self.prepare_prompts(prompt_values)
        return self.gen.chatcomplete(messages_list=messages_list, verbose=verbose)

    def chat(
        self, messages_list: List[List[Dict]], verbose=False
    ):
        return self.gen.chatcomplete(messages_list=messages_list, verbose=verbose)


class TomlLLMQuiver(BaseLLMQuiver):
    def __init__(
        self,
        config_path=None,
        toml_template_file=None,
        toml_prompt_name=None
    ):
        super().__init__(config_path)

        if toml_template_file is None or toml_prompt_name is None:
            raise ValueError("toml_template_file is required for toml template type.")
        if not isinstance(toml_prompt_name, str) or len(toml_prompt_name) == 0:
            raise ValueError("toml_prompt_name is required for toml template type.")

        toml_template_file = Path(toml_template_file)
        if not toml_template_file.exists():
            raise ValueError(f"toml_template_file: {toml_template_file} is not found.")

        logger.info(f"toml_file: {toml_template_file}, prompt_name: {toml_prompt_name}")

        prompt_templ_map = io_util.read_toml(toml_template_file)
        if 'type' not in prompt_templ_map:
            raise ValueError("Can't find the 'type' of the template.")

        if toml_prompt_name not in prompt_templ_map:
            raise ValueError(f"Can't find a template named '{toml_prompt_name}'.")

        self.prompt_template = prompt_templ_map[toml_prompt_name]
        self.prompt_template_type = prompt_templ_map['type']
        if self.prompt_template_type == "basic":
            logger.info("Detect prompt_template is for basic")
            self.prompt_template = prompt_template_parser.PromptTemplateParser(
                template=self.prompt_template)
        elif self.prompt_template_type == "chat":
            logger.info("Detect prompt_template is for chat")
            for msg_id, msg in enumerate(self.prompt_template):
                msg["content"] = prompt_template_parser.PromptTemplateParser(template=msg["content"])
        else:
            raise ValueError(
                f"Unsupported prompt template type: {self.prompt_template_type}. "
                "Supported types are 'basic' and 'chat'."
            )
        logger.debug(f"Initialized prompt template of type '{self.prompt_template_type}'")

    def prepare_prompts(self, prompt_values: List[Dict]):
        if not isinstance(prompt_values, list):
            raise TypeError("prompt_values must be a list.")

        if len(prompt_values) == 0:
            return []

        if not isinstance(prompt_values[0], dict):
            raise TypeError("TomlLLMQuiver's prompt_values must be a list of dict.")

        prompts = [self.prompt_template.format(p) for p in prompt_values]
        messages_list = [[dict(role="system", content=p)] for p in prompts]
        return messages_list

    def prepare_messages_list(self, prompt_values: List[Dict]):
        messages_list = []
        for prompt_value in prompt_values:
            messages = []
            for msg_templ in self.prompt_template:
                messages.append(dict(
                    role=msg_templ['role'],
                    content=msg_templ["content"].format(prompt_value)
                ))
            messages_list.append(messages)
        return messages_list

    def generate(
        self, prompt_values: List[str], verbose=False
    ):
        logger.warning(
            "The generate() method is deprecated and will be removed in a future version. "
            "Please use chat() instead."
        )
        if self.prompt_template_type != "basic":
            raise RuntimeError("Can't enable generate, because the template is not for 'basic'.")
        messages_list = self.prepare_prompts(prompt_values)
        return self.gen.chatcomplete(messages_list=messages_list, verbose=verbose)

    def chat(
        self, prompt_values: List[Dict], verbose=False
    ):
        messages_list = self.prepare_messages_list(prompt_values)
        return self.gen.chatcomplete(messages_list=messages_list, verbose=verbose)
