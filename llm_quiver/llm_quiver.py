from typing import List, Dict
from . import io_util
from .wrap_openai import WrapOpenAI
from loguru import logger
from pathlib import Path
import os
from jinja2 import Template


class LLMQuiver:
    def __init__(self, config_path: str = None) -> None:
        if config_path:
            self._initialize_by_config(config_path)
        else:
            logger.info("config_path is not set. Try to initialize by environment variables.")
            self._initialize_by_env()

        self.prompt_template = None

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

    def prerpare_prompt(self, prompt_values):
        if not isinstance(prompt_values, list):
            raise TypeError("prompt_values must be a list.")

        if len(prompt_values) == 0:
            return []

        if self.__class__.__name__ == "LLMQuiver":
            if not isinstance(prompt_values[0], str):
                raise TypeError("LLMQuiver's prompt_values must be a list of str.")
        elif self.__class__.__name__ == "Jinja2LLMQuiver" or self.__class__.__name__ == "TomlLLMQuiver":
            if not isinstance(prompt_values[0], dict):
                raise TypeError("Jinja2LLMQuiver's or TomlLLMQuiver's prompt_values must be a list of dict.")

        return [self.render_prompt(p) for p in prompt_values]

    def render_prompt(self, prompt_value):
        return prompt_value

    def generate_by_toml_template(
        self, prompts, verbose=False
    ):
        candidates = []
        cand_idx = []
        responses = [None] * len(prompts)
        for i, prompt in enumerate(prompts):
            cand_idx.append(i)
            candidates.append(prompt)

        model_outs = self.gen.chatcomplete(prompts=prompts, verbose=verbose)

        for i, out in zip(cand_idx, model_outs):
            responses[i] = out

        return responses

    def generate(
        self, prompt_values: List[Dict], verbose=False
    ):
        prompts = self.prerpare_prompt(prompt_values)
        return self.gen.chatcomplete(prompts=prompts, verbose=verbose)


class Jinja2LLMQuiver(LLMQuiver):
    def __init__(
        self,
        config_path=None,
        jinja2_template_file=None
    ):
        super().__init__(config_path)

        if jinja2_template_file is None:
            raise ValueError("jinja2_template_file is required for jinja2 template type.")

        jinja2_template_file = Path(jinja2_template_file)
        logger.info(f"jinja2_template_file: {jinja2_template_file}")

        if jinja2_template_file.exists():
            template_content = io_util.read_text(jinja2_template_file)
            self.prompt_template = Template(template_content)
        else:
            raise ValueError(f"jinja2_template_file: {jinja2_template_file} is not found.")

    def render_prompt(self, prompt_value):
        return self.prompt_template.render(prompt_value)


class TomlLLMQuiver(LLMQuiver):
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
        logger.info(f"basic_toml_file: {toml_template_file}, basic_prompt_name: {toml_prompt_name}")

        if not toml_template_file.exists():
            raise ValueError(f"toml_template_file: {toml_template_file} is not found.")

        prompt_templ_map = io_util.read_toml(toml_template_file)
        if toml_prompt_name not in prompt_templ_map:
            raise ValueError(f"Can't find a template named '{toml_prompt_name}'.")

        self.prompt_template = prompt_templ_map[toml_prompt_name]

    def render_prompt(self, prompt_value):
        return self.prompt_template.format_map(prompt_value)
