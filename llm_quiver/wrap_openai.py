from openai import BadRequestError, APITimeoutError, RateLimitError
import json
import tiktoken
from openai import AzureOpenAI, OpenAI
import re
import time
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Optional
from .support_api import SupportAPI
from .cache_manager import CacheManager


class WrapOpenAI:
    def __init__(
        self,
        api_type: str = "azure_openai",
        api_base: str = "",
        api_version: str = "2024-02-01",
        api_key: str = "",
        modelname: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 600,
        enable_cache: bool = False,
        cache_dir: Optional[str] = None,
        cache_prefix: Optional[str] = None,
        cache_interval: int = 0
    ):
        try:
            self.api_type = SupportAPI(api_type)
        except KeyError:
            raise ValueError(f"{api_type} is not a valid name for {SupportAPI.__name__}")

        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.modelname = modelname
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.cache_interval = cache_interval

        if self.api_type == SupportAPI.AzureOpenAI:
            self._client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                api_key=self.api_key,
            )
        elif self.api_type in [SupportAPI.OpenAI, SupportAPI.OpenAILike]:
            #   "openai" or "openai_like"
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )

        self._log_format_parameters()
        self._init_cache()
        self.encoding_init_completed = False

    def _log_format_parameters(self):
        """Helper method to format parameters with masked api_key."""
        masked_api_key = '*' * len(self.api_key) if self.api_key else ''
        params = {
            "api_type": self.api_type,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "api_key": masked_api_key,
            "modelname": self.modelname,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "cache_prefix": self.cache_prefix
        }

        # Formatting parameters for printing
        formatted_params = "\n".join(f"{key}: {value}" for key, value in params.items())
        logger.info(f"{formatted_params}")

    def _init_encoding(self):
        if self.api_type in [SupportAPI.AzureOpenAI, SupportAPI.OpenAI]:
            if self.modelname.startswith('gpt-4o'):
                self.enconding_name = "o200k_base"
                self.encoding = tiktoken.get_encoding(self.enconding_name)
            elif self.modelname.startswith('gpt-4') or self.modelname.startswith('gpt-3.5'):
                self.enconding_name = "cl100k_base"
                self.encoding = tiktoken.get_encoding(self.enconding_name)
            else:
                raise ValueError(f"modelname == {self.modelname}")
        else:
            raise ValueError("When `api_type` is not equal to azure_openai or openai, it can't work.")

        self.encoding_init_completed = True

    def _init_cache(self):
        if self.enable_cache:
            if not self.cache_dir:
                raise ValueError("caching is enabled but no cache directory is provided. "
                                 f"cache_dir's value: {self.cache_dir}")

            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            if self.cache_prefix:
                cache_prefix = self.cache_prefix
            else:
                if self.modelname:
                    cache_prefix = self.modelname
                else:
                    cache_prefix = "default"

            cache_path = self.cache_dir / f"{cache_prefix}.cache"
            self.gpt_cache = CacheManager(cache_path, backup_interval=self.cache_interval)

    def infer(self, messages):
        logger.debug(f"messages: {messages}")
        response = self._client.chat.completions.create(
            model=self.modelname,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            messages=messages
        )
        return parse_response(response)

    def complete_with_retry(self, messages, sleep_eps=60, max_retry=3, every_step_sleep=0):
        resp = None

        for _ in range(max_retry):
            try:
                resp = self.infer(messages)
                time.sleep(every_step_sleep)
                break
            except BadRequestError as e:
                logger.warning(f"BadRequestError: {repr(e)}")
                logger.warning(f"{messages}")
                break
            except APITimeoutError as e:
                logger.warning(f"APITimeoutError: {repr(e)}，delay {sleep_eps}s and retry.")
                time.sleep(sleep_eps)
            except RateLimitError as e:
                err_msg = repr(e)
                reg = r"Please retry after (\d+) seconds"
                nums = re.findall(reg, err_msg)
                if len(nums) != 1:
                    time.sleep(sleep_eps)
                    continue
                num = nums[0]
                if not num.isdigit():
                    time.sleep(sleep_eps)
                server_eps = int(num)
                logger.warning(f"{repr(e)}.\nExtraction time delayed, retrying after {server_eps}s.")
                time.sleep(server_eps)

            except Exception as e:
                logger.error(f"{repr(e)}, retrying after a delay of {sleep_eps}s.")
                time.sleep(sleep_eps)

        return resp

    def num_tokens_from_string(self, string: str) -> int:
        if not self.encoding_init_completed:
            self._init_encoding()

        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def chatcomplete(self, messages_list, verbose=False):
        responses = [None] * len(messages_list)

        if verbose:
            messages_list = tqdm(messages_list)

        if self.enable_cache:
            for idx, messages in enumerate(messages_list):
                messages_key = json.dumps(messages)
                response = self.gpt_cache.get_item(messages_key)
                if response is not None and len(response) > 0:
                    responses[idx] = response

        for idx, (messages, cached_response) in enumerate(zip(messages_list, responses)):
            logger.debug(f"## input\n{messages}")
            if cached_response is not None:
                logger.debug(f"## response(cached)\n{cached_response}")
            else:
                response = self.complete_with_retry(messages, sleep_eps=10, max_retry=300, every_step_sleep=2)

                if response is not None:
                    logger.debug(f"## response(new)\n{response}")
                    responses[idx] = response
                    if self.enable_cache:
                        messages_key = json.dumps(messages)
                        self.gpt_cache.set_item(key=messages_key, value=response)
                else:
                    logger.debug("## response(new)\nNone")

        return responses


def parse_response(response):
    """解析API响应"""
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError):
        logger.error("Invalid response format")
        return None
