from core.vectorStore import VectorStore
from configs.config import VEC_STORE_PATH

store = VectorStore.load(VEC_STORE_PATH)
store.summary(show_example=False)