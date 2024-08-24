from typing import List
from . import io_util
from .wrap_openai import WrapOpenAI
from loguru import logger
from pathlib import Path
import random
import os


class LLMQuiver:
    def __init__(self, prompt_toml_file, prompt_name=None) -> None:
        random.seed(42)
        prompt_toml_file = Path(prompt_toml_file)
        demonstration_path = (
            prompt_toml_file.parent / f"{self.__class__.__name__}_demos.txt"
        )

        logger.info(f"prompt_toml_file: {prompt_toml_file}, prompt_name: {prompt_name}")

        if prompt_toml_file.exists():
            prompt_templ_map = io_util.read_toml(prompt_toml_file)
            if not prompt_name:
                prompt_name = str(self.__class__.__name__)

            if prompt_name not in prompt_templ_map:
                logger.error(f"can't find a template named '{prompt_name}'.")
                prompt_templ_map[prompt_name] = ""
                io_util.write_toml(prompt_templ_map, prompt_toml_file)
            else:
                self.prompt_templ = prompt_templ_map[prompt_name]
        else:
            prompt_templ_map = {prompt_name: ""}
            io_util.write_toml(prompt_templ_map, prompt_toml_file)

        if demonstration_path and demonstration_path.exists():
            self.demonstrations = io_util.read_text_line_by_line(demonstration_path)
        else:
            self.demonstrations = None

        self._initialize()

    def _initialize(self):
        LLMQUIVER_CONFIG = os.environ.get("LLMQUIVER_CONFIG", None)
        if LLMQUIVER_CONFIG is None:
            raise ValueError(f"LLMQUIVER_CONFIG is invalid: {LLMQUIVER_CONFIG}.")
        LLMQUIVER_CONFIG = Path(LLMQUIVER_CONFIG)
        if LLMQUIVER_CONFIG.exists():
            logger.info(f"LLMQUIVER_CONFIG: {LLMQUIVER_CONFIG}")
            config = io_util.read_toml(LLMQUIVER_CONFIG)
        else:
            raise ValueError(f"LLMQUIVER_CONFIG: {LLMQUIVER_CONFIG} is not found.")

        keys = ["API_TYPE", "GPT_MODEL_NAME", "API_BASE", "API_KEY", "API_VERSION"]
        params = {}
        for key in keys:
            env_value = os.environ.get(key, None)
            config_value = config.get(key, None)
            if env_value and len(env_value) > 0:
                params[key] = env_value
            elif config_value and len(config_value) > 0:
                params[key] = config_value
            else:
                raise ValueError(f"Missing {key} in both the configuration file({LLMQUIVER_CONFIG}) and environment.")

        self.gen = WrapOpenAI(
            api_type=params["API_TYPE"],
            api_base=params["API_BASE"],
            api_version=params["API_VERSION"],
            api_key=params["API_KEY"],
            modelname=params["GPT_MODEL_NAME"],
            temperature=config.get("temperature"),
            top_p=config.get("top_p", None),
            max_tokens=config.get("max_tokens"),
            enable_cache=config.get("enable_cache"),
            cache_dir=config.get("cache_dir"),
            cache_prefix=config.get("cache_prefix", params["GPT_MODEL_NAME"])
        )

    def get_num_tokens_from_string_fn(self):
        return self.gen.num_tokens_from_string

    def generate_by_templ(self, templ, templ_vals, input_min_len):
        assert isinstance(templ_vals, list)
        if len(templ_vals) == 0:
            return []
        assert isinstance(templ_vals[0], dict)

        candidates = []
        cand_idx = []
        responses = [None] * len(templ_vals)

        prompts = [templ.format_map(p) for p in templ_vals]

        for i, prompt in enumerate(prompts):
            if len(prompt) < input_min_len:
                continue
            cand_idx.append(i)
            candidates.append(prompt)

        model_outs = self.gen.chatcomplete(prompts=prompts, verbose=True)

        for i, out in zip(cand_idx, model_outs):
            responses[i] = out

        return responses

    def generate_by_templ_and_demos(self, templ, templ_vals, input_min_len, demostr):
        assert isinstance(templ_vals, list)
        if len(templ_vals) == 0:
            return []
        assert isinstance(templ_vals[0], dict)

        candidates = []
        cand_idx = []
        responses = [None] * len(templ_vals)

        prompts = [templ.format_map(p) + "\nhere are some examples: \n" + demostr for p in templ_vals]
        for i, prompt in enumerate(prompts):
            if len(prompt) < input_min_len:
                continue
            cand_idx.append(i)
            candidates.append(prompt)

        model_outs = self.gen.chatcomplete(prompts=prompts, verbose=True)

        for i, out in zip(cand_idx, model_outs):
            responses[i] = out

        return responses

    def generate(
        self, templ_vals: List, input_min_len=20, use_demos_type="default", demo_num=0
    ):
        assert demo_num >= 0
        if demo_num > 0:
            logger.debug(self.demonstrations)
            if self.demonstrations and len(self.demonstrations) != 0:
                used_demos = []
                # random sample
                if use_demos_type == "default" or use_demos_type == "random":
                    used_demos = random.sample(
                        self.demonstrations, min(demo_num, len(self.demonstrations))
                    )
                demostr = "\n\n".join(used_demos) + "\n\n"
                return self.generate_by_templ_and_demos(
                    self.prompt_templ,
                    demostr=demostr,
                    templ_vals=templ_vals,
                    input_min_len=input_min_len,
                )
            else:
                logger.warning("Without domonstration, do zero-shot!")
                return self.generate_by_templ(
                    self.prompt_templ,
                    templ_vals=templ_vals,
                    input_min_len=input_min_len,
                )
        else:
            return self.generate_by_templ(
                self.prompt_templ, templ_vals=templ_vals, input_min_len=input_min_len
            )
