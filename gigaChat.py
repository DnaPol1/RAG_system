from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

with open("API.txt", "r", encoding="utf-8") as f:
    API_KEY = f.readline().strip()

class GigaChatClient:
    def __init__(self, credentials, model="GigaChat"):
        self.client = GigaChat(
            credentials=API_KEY,
            verify_ssl_certs=False,
            model="GigaChat"
        )

    def generate(self, prompt: str, rag: bool) -> str:
        if rag:
            payload = Chat(
                messages = [
                    Messages(
                        role = MessagesRole.SYSTEM,
                        content =
                        "Tы - юридический помощник"
                        "Отвечай строго на русском языке."
                        "Если в контексте нет ответа - скажи Информации в документах нет"
                        "Запрещено использовать внешние знания"
                    ),
                    Messages(
                        role = MessagesRole.USER,
                        content = prompt
                    )
                ]
            )
        else:
            payload = Chat(
                messages=[
                    Messages(
                        role=MessagesRole.SYSTEM,
                        content=
                        "Ты - эксперт по обработке естественного языка."
                        "Tы - юридический помощник"
                        "Отвечай строго на русском языке."
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