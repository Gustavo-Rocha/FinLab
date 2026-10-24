import os
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "financial"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# connect to Qdrant Cloud
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30,
)
# deleta a collection se ela existir
qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=TextEmbedding.get_embedding_size(DENSE_MODEL),
            distance=models.Distance.COSINE,
        ),
        "colbert": models.VectorParams(
            size=128,  # tamanho de cada embeding individual que o colbert gera
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)
