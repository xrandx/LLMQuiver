from pathlib import Path
import sqlite3
from loguru import logger
import time
import shutil


class CacheManager:
    def __init__(self, cache_path, backup_interval=0):
        logger.info(f"Cache is in: {cache_path}")
        self.cache_path = Path(cache_path)
        is_first_run = not self.cache_path.exists()
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.cursor = self.conn.cursor()
        if is_first_run:
            self.create_table()
        self.last_backup_time = 0
        self.backup_interval = backup_interval  # 1 hour

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS kv_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def count(self):
        """
        Query the current number of records in the kv_cache table.
        """
        self.cursor.execute('SELECT COUNT(*) FROM kv_cache')
        count = self.cursor.fetchone()[0]
        logger.info(f"Now records count: {count}")

    def set_item(self, key, value):
        """
        Insert or update a key-value pair in the kv_cache table.

        If the provided value is None or an empty string, a warning is logged,
        and the operation is aborted. Otherwise, the method attempts to insert
        the key-value pair into the kv_cache table. If the key already exists,
        the value is updated with the new value.

        After a successful insertion or update, the current record count is logged.
        In case of an SQLite error during the operation, an error message is logged,
        and the transaction is rolled back to maintain data integrity.
        Args:
            key (str): The key to insert or update in the cache.
            value (str): The value to associate with the key.

        Raises:
            sqlite3.Error: If there is a database error during the insertion or update.
        """
        if value is None or (isinstance(value, str) and value.strip() == ""):
            logger.warning(f"Value is None or empty for key: {key}")
            return
        try:
            self.cursor.execute('''
                INSERT INTO kv_cache (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            ''', (key, value))
            self.conn.commit()
            self.count()
            self.backup_cache()
        except sqlite3.Error as e:
            logger.error(f"Error inserting key-value: {e}")
            self.conn.rollback()

    def get_item(self, key):
        self.cursor.execute('''
            SELECT value FROM kv_cache WHERE key=?
        ''', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def check_integrity(self):
        try:
            self.cursor.execute("PRAGMA integrity_check")
            result = self.cursor.fetchone()
            return result[0] == "ok"
        except sqlite3.Error as e:
            logger.error(f"Error checking database integrity: {e}")
            return False

    def backup_cache(self):
        current_time = time.time()
        if current_time - self.last_backup_time < self.backup_interval:
            return

        if self.check_integrity():
            backup_path = self.cache_path.with_suffix('.bak')
            try:
                shutil.copy2(self.cache_path, backup_path)
                logger.info(f"Cache backed up to {backup_path}")
                self.last_backup_time = current_time
            except IOError as e:
                logger.error(f"Error backing up cache: {e}")
        else:
            logger.error("Cache integrity check failed. Backup not performed.")

    def delete(self, key):
        self.cursor.execute('''
            DELETE FROM kv_cache WHERE key=?
        ''', (key,))
        self.conn.commit()

    def update(self, key, value):
        self.set(key, value)  # Reuse set method with conflict update logic

    def close(self):
        self.backup_cache()
        self.conn.close()

    def __del__(self):
        self.close()
