import time
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().embedding_model)


def _get_index(dimension: int):
    """Return the Pinecone index, creating it on first use. Never deletes existing data."""
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name

    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)

    existing = pc.describe_index(name).dimension
    if existing != dimension:
        raise RuntimeError(
            f"Pinecone index '{name}' has dimension {existing}, but the embedding model "
            f"produces {dimension}. Use a different PINECONE_INDEX_NAME or embedding model."
        )
    return pc.Index(name)


@lru_cache
def get_vectorstore() -> PineconeVectorStore:
    embeddings = get_embeddings()
    dimension = len(embeddings.embed_query("dimension probe"))
    return PineconeVectorStore(
        index=_get_index(dimension),
        embedding=embeddings,
        namespace=get_settings().pinecone_namespace,
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": get_settings().top_k})


def add_documents(chunks) -> list[str]:
    return get_vectorstore().add_documents(chunks)
