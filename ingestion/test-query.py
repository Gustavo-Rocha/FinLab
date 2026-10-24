import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from Utils.semantic_chunker import SemanticChunker
from Utils.edgar_client import EdgarClient

load_dotenv()

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
EDGAR_EMAIL = "Gustavo oliveira Gustavooliveirarocha@hotmail.com"
MAX_TOKENS = 300

# connect to Qdrant Cloud
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30,
)
# deleta a collection se ela existir
# qdrant.delete_collection(collection_name=COLLECTION_NAME)

# qdrant.create_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config={
#         "dense": models.VectorParams(
#             size=TextEmbedding.get_embedding_size(DENSE_MODEL),
#             distance=models.Distance.COSINE,
#         ),
#         "colbert": models.VectorParams(
#             size=128,  # tamanho de cada embeding individual que o colbert gera
#             distance=models.Distance.COSINE,
#             multivector_config=models.MultiVectorConfig(
#                 comparator=models.MultiVectorComparator.MAX_SIM
#             ),
#         ),
#     },
#     sparse_vectors_config={"sparse": models.SparseVectorParams()},
# )

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

query_text = "what are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {
            "prefetch": [
                {"query": query_dense, "using": "dense", "limit": 10},
                {"query": query_sparse, "using": "sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=query_colbert,
    using="colbert",
    limit=3,
)


MAX_SCORE = max(result.score for result in results.points)

for r in results.points:
    normalized_score = r.score / MAX_SCORE
    print(f"Score: {normalized_score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)
