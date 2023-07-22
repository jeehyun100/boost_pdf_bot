import chromadb
from chromadb.config import Settings

from langchain.embeddings.openai import OpenAIEmbeddings

client = chromadb.Client(Settings(chroma_api_impl="rest",
                                        chroma_server_host="172.19.0.1",
                                        chroma_server_http_port="8000"
                                    ))



OPEN_API_KEY = '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'
emb_fn = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
collection = client.get_or_create_collection(name="my_collection", embedding_function=emb_fn)
#collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
collection.add(
    documents=["doc1", "doc2", "doc3"],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2]],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
    ids=["id4", "id5", "id6"]
)
cnt = collection.count()
print(cnt)

result = collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2]],
    n_results=2,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)

print(result)
