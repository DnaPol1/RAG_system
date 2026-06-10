from sentence_transformers import SentenceTransformer, util


class SemanticQueryExpander:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        similarity_threshold: float = 0.7,
        max_expansions: int = 3,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_expansions = max_expansions

    def expand(self, query: str, candidate_queries: list[str]) -> list[str]:
        """
        candidate_queries — корпус возможных формулировок
        (например, все запросы из train/dev)
        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        cand_embs = self.model.encode(candidate_queries, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, cand_embs)[0]

        expansions = []
        for idx in scores.argsort(descending=True):
            if scores[idx] < self.similarity_threshold:
                break
            expansions.append(candidate_queries[int(idx)])
            if len(expansions) >= self.max_expansions:
                break

        return expansions
