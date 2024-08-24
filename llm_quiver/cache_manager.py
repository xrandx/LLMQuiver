from pathlib import Path
import signal
import joblib
from filelock import FileLock, Timeout
from loguru import logger
import time


class CacheManager:
    instances = []

    def __init__(self, cache_path):
        logger.debug(f"cache_path: {cache_path}")
        self.cache_path = Path(cache_path)
        self.lock_path = self.cache_path.with_suffix('.lock')
        self.cache = {}
        self.load()
        CacheManager.instances.append(self)

    def save(self):
        retry_count = 5
        for attempt in range(retry_count):
            try:
                with FileLock(self.lock_path, timeout=10):
                    if self.cache_path.exists():
                        old_cache = joblib.load(self.cache_path)
                        old_length = len(old_cache)
                        new_cache = {**old_cache, **self.cache}
                    else:
                        old_length = 0
                        new_cache = self.cache

                    joblib.dump(new_cache, self.cache_path)
                    logger.info(f"Cache saved to file: '{self.cache_path}', Cache length: {old_length} -> {len(new_cache)}")
                    break
            except Timeout:
                logger.warning(f"Attempt {attempt + 1} to acquire lock timed out. Retrying...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"An error occurred while saving cache: {e}")
                break

    def load(self):
        retry_count = 5
        for attempt in range(retry_count):
            try:
                if self.cache_path.exists():
                    with FileLock(self.lock_path, timeout=10):
                        self.cache = joblib.load(self.cache_path)
                        logger.info(f"Cache loaded from file: {self.cache_path}")
                        break
            except Timeout:
                logger.warning(f"Attempt {attempt + 1} to acquire lock timed out. Retrying...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"An error occurred while loading cache: {e}")
                break

    def get_item(self, key):
        return self.cache.get(key, None)

    def set_item(self, key, value):
        self.cache[key] = value

    def exists(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()

    @classmethod
    def save_all(cls, signal_received, frame):
        logger.warning("Interrupt signal caught: ", signal_received)
        for instance in cls.instances:
            instance.save()
        logger.info("Successfully saved all caches.")
        exit(0)


signal.signal(signal.SIGINT, CacheManager.save_all)
