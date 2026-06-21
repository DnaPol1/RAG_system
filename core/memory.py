from dataclasses import dataclass, field
from typing import List, Dict
import re
import redis
import json
import os

host = os.getenv("REDIS_HOST", "localhost")

def rough_token_count(text: str) -> int:
    return len(text) // 4

@dataclass()
class Message:
    role: str
    content: str

class SummaryMemory:
    def __init__(self, redis_client, session_id, max_tokens: int = 1500):
        self.redis = redis_client
        self.session_id = session_id
        self.max_tokens = max_tokens

    def _load(self) -> Dict:
        data = self.redis.get(self.session_id)
        if not data:
            return {"summary": "", "messages": ""}
        return data

    def _save(self, data: Dict):
        self.redis.set(self.session_id, data)

    def _current_tokens(self, data: Dict) -> int:
        text = data["summary"] + "\n".join(
            [f"{m['role']}: {m['content']}" for m in data["messages"]]
        )
        return rough_token_count(text)

    def _enforce_token_limit(self, data: Dict):
        """
        Сжатие старых сообщений
        """
        while self._current_tokens() > self.max_tokens:
            if len(data["messages"]) <= 2:
                break
            old_messages = data["messages"][:-2]
            data["messages"] = data["messages"][-2:]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
            if data["summary"]:
                data["summary"] += "\n" + history_text
            else:
                data["summary"] = history_text

    def add(self, role: str, content: str):
        data = self._load()
        data["messages"].append(Message(role, content))
        self._enforce_token_limit()
        self._save(data)

    def build_context(self) -> str:
        data = self._load()
        recent = "\n".join([f"{m['role']}: {m['content']}" for m in data["messages"]])
        return f"""
SUMMARY:
{self.summary}
RECENT:
{recent}
"""

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(host=host, port=6379, db=0, decode_responses=True)

    def get(self, key:str):
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key:str, value, ttl: int = 3600):
        self.client.setex(key, ttl, json.dumps(value, ensure_ascii=False))