import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from aws_lambda_powertools import Logger

logger = Logger(service="databricks-cert-rag-indexer")

OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "databricks-cert-guides")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 20
MAX_WORKERS = 5


def get_opensearch_client() -> OpenSearch:
    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    username = os.environ["OPENSEARCH_USERNAME"]
    password = os.environ["OPENSEARCH_PASSWORD"]

    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    logger.info("Connecting to OpenSearch", extra={"host": host, "index": OPENSEARCH_INDEX})

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=(username, password),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def ensure_index_exists(client: OpenSearch):
    if client.indices.exists(index=OPENSEARCH_INDEX):
        logger.info("Index already exists, skipping creation", extra={"index": OPENSEARCH_INDEX})
        return

    logger.info("Creating OpenSearch index", extra={"index": OPENSEARCH_INDEX})

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id":    {"type": "keyword"},
                "exam_name":   {"type": "keyword"},
                "section":     {"type": "keyword"},
                "page_number": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "token_count": {"type": "integer"},
                "s3_key":      {"type": "keyword"},
                "chunk_text":  {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24,
                        },
                    },
                },
            }
        },
    }

    client.indices.create(index=OPENSEARCH_INDEX, body=mapping)
    logger.info("Index created successfully", extra={"index": OPENSEARCH_INDEX})


def embed_batch(batch: list[dict], openai_client: OpenAI, batch_index: int) -> list[dict]:
    texts = [c["chunk_text"] for c in batch]

    logger.info("Embedding batch", extra={
        "batch_index": batch_index,
        "batch_size": len(texts),
    })

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    for i, chunk in enumerate(batch):
        chunk["embedding"] = response.data[i].embedding

    logger.info("Batch embedded successfully", extra={
        "batch_index": batch_index,
        "batch_size": len(batch),
    })

    return batch


def embed_and_index(chunks: list[dict]) -> dict:
    openai_client = get_openai_client()
    os_client = get_opensearch_client()

    ensure_index_exists(os_client)

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    logger.info("Starting embed and index", extra={
        "total_chunks": len(chunks),
        "total_batches": len(batches),
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
    })

    embedded_chunks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(embed_batch, batch, openai_client, i): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                result = future.result()
                embedded_chunks.extend(result)
                logger.info("Batch completed", extra={
                    "batch_index": batch_index,
                    "completed": len(embedded_chunks),
                    "total": len(chunks),
                })
            except Exception as e:
                logger.error("Batch failed", extra={
                    "batch_index": batch_index,
                    "error": str(e),
                })
                raise

    actions = [
        {
            "_index": OPENSEARCH_INDEX,
            "_id": chunk["chunk_id"],
            "_source": {
                "chunk_id":    chunk["chunk_id"],
                "exam_name":   chunk["exam_name"],
                "section":     chunk["section"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "s3_key":      chunk["s3_key"],
                "chunk_text":  chunk["chunk_text"],
                "embedding":   chunk["embedding"],
            },
        }
        for chunk in embedded_chunks
    ]

    logger.info("Bulk indexing to OpenSearch", extra={
        "total_documents": len(actions),
        "index": OPENSEARCH_INDEX,
    })

    success, failed = helpers.bulk(os_client, actions, raise_on_error=False)

    logger.info("Bulk index complete", extra={
        "indexed": success,
        "failed": len(failed),
    })

    if failed:
        logger.error("Some documents failed to index", extra={
            "failed_count": len(failed),
            "sample": json.dumps(failed[:3]),
        })

    return {"indexed": success, "failed": len(failed)}