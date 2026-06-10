from setuptools import setup, find_packages

setup(
    name="rag_system",
    version="0.1.0",
    packages=find_packages(),  # автоматически найдет retriever
    install_requires=[
        # здесь можно указать зависимости, но мы их ставим отдельно
    ],
    author="Полина Масалимова",
    description="RAG система для консультирования предпринимателей",
    python_requires=">=3.8",
)