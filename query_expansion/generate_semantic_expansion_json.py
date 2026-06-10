import json
from pathlib import Path

from query_expansion.semantic_expander import SemanticQueryExpander


DATASET_PATH = Path("../query-file_dataset.json")
OUTPUT_PATH = Path("../query_expansion/semantic_expansions.json")


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [item["query"] for item in data]

    expander = SemanticQueryExpander(
        similarity_threshold=0.75,
        max_expansions=3,
    )

    expansion_map = {}

    for q in queries:
        expansions = expander.expand(q, queries)
        expansion_map[q] = expansions

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(expansion_map, f, ensure_ascii=False, indent=2)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
