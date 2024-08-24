from openai import BadRequestError, APITimeoutError, RateLimitError
import tiktoken
from openai import AzureOpenAI, OpenAI
import re
import time
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Optional

from .cache_manager import CacheManager


class WrapOpenAI:
    def __init__(
        self,
        api_type: str = "azure",
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
        cache_prefix: Optional[str] = None
    ):
        self.api_type = api_type
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

        if self.api_type == "azure":
            self._client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                api_key=self.api_key,
            )
        else:
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        self._log_format_parameters()
        self._init_encoding()
        self._init_cache()

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
        if self.modelname.startswith('gpt-4o'):
            self.enconding_name = "o200k_base"
            self.encoding = tiktoken.get_encoding(self.enconding_name)
        elif self.modelname.startswith('gpt-4') or self.modelname.startswith('gpt-3.5'):
            self.enconding_name = "cl100k_base"
            self.encoding = tiktoken.get_encoding(self.enconding_name)
        else:
            raise ValueError(f"modelname == {self.modelname}")

    def _init_cache(self):
        if self.enable_cache:
            if self.cache_dir:
                self.cache_dir = Path(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True, parents=True)
                if self.cache_prefix:
                    cache_prefix = self.cache_prefix
                else:
                    if self.modelname:
                        cache_prefix = "-".join(self.modelname.split("-")[:2])
                    else:
                        cache_prefix = "default"

                cache_path = self.cache_dir / f"{cache_prefix}.cache"
                self.gpt_cache = CacheManager(cache_path)

    def infer(self, prompt):
        return self._client.chat.completions.create(
            model=self.modelname,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": prompt},
            ]
        )

    def try_complete(self, complete_fn, prompt, sleep_eps=60, max_retry=3, every_step_sleep=0):
        resp = None
        if self.enable_cache and self.gpt_cache.exists(prompt):
            return self.gpt_cache.get_item(prompt)

        for _ in range(max_retry):
            try:
                resp = complete_fn(prompt)
                time.sleep(every_step_sleep)
                break
            except BadRequestError as e:
                logger.warning(f"Content violation or other issue, error msg: {repr(e)}")
                logger.warning(f"{prompt}")
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

        if self.enable_cache and resp:
            self.gpt_cache.set_item(key=prompt, value=resp)
            self.gpt_cache.save()
        return resp

    def num_tokens_from_string(self, string: str) -> int:
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def chatcomplete(self, prompts, verbose=False):
        res = []
        if verbose:
            prompts = tqdm(prompts)

        for p in prompts:
            try:
                resp = self.try_complete(self.infer, p, sleep_eps=10, max_retry=300, every_step_sleep=2)
            except KeyboardInterrupt:
                self.gpt_cache.save()
                logger.warning("User interrupted.")
                exit()

            logger.debug(f"## input\n{p}")
            logger.debug(f"## resp\n{resp}")

            if resp is not None \
                    and hasattr(resp, "choices") \
                    and len(resp.choices) > 0 \
                    and hasattr(resp.choices[0], "message") \
                    and hasattr(resp.choices[0].message, "content"):

                response = resp.choices[0].message.content
                logger.debug(f"## output\n{response}")
                res.append(response)
            else:
                res.append(None)
                logger.debug("## output\nNone")

        return res
