from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from core.llm.prompts import SYSTEM_PROMPT

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GIGACHAT_API_KEY")

class GigaChatClient:
    def __init__(self, credentials):
        self.client = GigaChat(
            credentials=API_KEY,
            verify_ssl_certs=False,
            model="GigaChat"
        )

    def generate(self, prompt: str, rag: bool) -> str:
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=SYSTEM_PROMPT
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=prompt
                )
            ]
        )
        try:
            response = self.client.chat(payload)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return str(e)