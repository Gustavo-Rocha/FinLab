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

edgar = EdgarClient(email=EDGAR_EMAIL)
data_10k = edgar.fetch_filing_data("AAPL", "10-K")
text_10k = edgar.get_combined_text(data_10k)

data_10q = edgar.fetch_filing_data("AAPL", "10-Q")
text_10q = edgar.get_combined_text(data_10q)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, text_10k), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})


dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)
points = []

for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={"text": chunk, "metadata": metadata},
    )

    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=5)
