from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

with open(r"C:\Users\Полина\PycharmProjects\RAG_system\llm_gigachat\API.txt",
          "r", encoding="utf-8") as f:
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
        messages=[
          Messages(
            role=MessagesRole.SYSTEM,
            content="Ты - юридический помощник.\n"
                    "Отвечай строго на русском языке.\n"
                    "Не используй слова: стандарт, настоящий стандарт, раздел,"
                    "пункт, подпунткт.\n"
                    "Не ссылайся на номера разделов, пунктов или статей.\n"
                    "Используй приведенный контекст как основной источник информации.\n"
                    "Запрещено использовать markdown разметку.\n"
                    "Если точного ответа нет в контексте, сформулируй наиболее релевантный ответ на "
                    "основе доступной информации.\n"
                    "Не используй внешние знания, не придумывай факты, но допускается обобщение информации"
                    "из контекста.\n"
                    "Формат ответа: четкий, структурированный, по делу.\n"
                    "Если в представленном контексте нет информации для ответа на вопрос - признай это прямо и вежливо перенаправь пользователя в центр"
                    "Мой бизнес.\n"
                    "Если в представленном контексте нет информации для ответа на вопрос, не пиши фразы"
                    "вроде: Как и любая языковая модель, GigaChat не обладает собственным мнением и не транслирует мнение своих разработчиков. Ответ сгенерирован нейросетевой моделью, обученной на открытых данных, в которых может содержаться неточная или ошибочная информация. Во избежание неправильного толкования, разговоры на некоторые темы временно ограничены.\n"
                    "Если информации нет - признай это прямо и вежливо перенаправь пользователя в центр"
                    "Мой бизнес.\n"
          ),
          Messages(
            role=MessagesRole.USER,
            content=prompt
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
