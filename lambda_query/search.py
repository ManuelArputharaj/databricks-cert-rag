import os
from openai import OpenAI
from opensearchpy import OpenSearch, RequestsHttpConnection
from aws_lambda_powertools import Logger

logger = Logger(service="databricks-cert-rag-query")

OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "databricks-cert-guides")
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def get_opensearch_client() -> OpenSearch:
    endpoint = os.environ["OPENSEARCH_ENDPOINT"]
    username = os.environ["OPENSEARCH_USERNAME"]
    password = os.environ["OPENSEARCH_PASSWORD"]

    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    logger.info("Connecting to OpenSearch", extra={
        "host": host,
        "index": OPENSEARCH_INDEX,
    })

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


def embed_query(query: str) -> list[float]:
    client = get_openai_client()

    logger.info("Embedding query", extra={
        "query": query,
        "model": EMBEDDING_MODEL,
    })

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    logger.info("Query embedded successfully", extra={
        "query": query,
        "embedding_dim": len(response.data[0].embedding),
    })

    return response.data[0].embedding


def search(query: str, exam_name: str = None, top_k: int = TOP_K) -> list[dict]:
    logger.info("Starting search", extra={
        "query": query,
        "exam_filter": exam_name,
        "top_k": top_k,
    })

    query_vector = embed_query(query)
    os_client = get_opensearch_client()

    if exam_name and exam_name.lower() != "all":
        logger.info("Applying exam filter", extra={"exam_name": exam_name})
        knn_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                }
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"exam_name": exam_name}}
                    ],
                }
            },
            "_source": {
                "excludes": ["embedding"]
            },
        }
    else:
        logger.info("No exam filter applied — searching all exams")
        knn_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k,
                    }
                }
            },
            "_source": {
                "excludes": ["embedding"]
            },
        }

    logger.info("Executing KNN search", extra={
        "index": OPENSEARCH_INDEX,
        "top_k": top_k,
    })

    response = os_client.search(index=OPENSEARCH_INDEX, body=knn_query)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        results.append({
            "score":       hit["_score"],
            "chunk_id":    hit["_source"]["chunk_id"],
            "exam_name":   hit["_source"]["exam_name"],
            "section":     hit["_source"]["section"],
            "page_number": hit["_source"]["page_number"],
            "chunk_text":  hit["_source"]["chunk_text"],
            "s3_key":      hit["_source"]["s3_key"],
        })

    top_score = results[0]["score"] if results else 0

    logger.info("Search complete", extra={
        "query": query,
        "results_count": len(results),
        "top_score": round(top_score, 4),
        "top_exam": results[0]["exam_name"] if results else None,
        "top_section": results[0]["section"] if results else None,
    })

    return results