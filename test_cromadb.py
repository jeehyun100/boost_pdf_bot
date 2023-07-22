# create the chroma client
import chromadb
import uuid
from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

OPEN_API_KEY = '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'
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

chunk_size = 1000
overlap = 0

rc_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

client = chromadb.Client(
    Settings(
        chroma_api_impl="rest",
        chroma_server_host="172.19.0.1",
        chroma_server_http_port="8000",
    )
)
client.reset()  # resets the database
collection = client.create_collection("test_samsung")


    




def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".pdf":
            pdf_reader = UnstructuredPDFLoader(os.path.join(pdf_folder_path, file_path))
            pdf_data = pdf_reader.load_and_split(text_splitter=rc_text_splitter)
            
            for modify_doc_meta in pdf_data:
                get_meta = modify_doc_meta.metadata
                get_meta['stock_name'] = '삼성전자'
                get_meta['stock_id'] =  '005930'
                modify_doc_meta.metadata = get_meta

    return pdf_data

def create_retriever(_embeddings, docs, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = Chroma.from_documents(docs, _embeddings)
        except (IndexError, ValueError) as e:
            print(e)
            #st.error(f"Error creating vectorstore: {e}")
            return
    return vectorstore



pdf_folder_path = f'./pdfs/'
files = os.listdir(pdf_folder_path)
    

all_docs =  load_docs(files)
# load it into Chroma
retriever_type = "SIMILARITY SEARCH"
vectordb = create_retriever(embeddings, all_docs, retriever_type)
print("done")

#search_kwargs={"filter":{'$or': [{'source': {'$eq': './SampleDoc/Bikes.pdf'}}, {'source': {'$eq': './SampleDoc/IceCreams.pdf'}}]}}
search_kwargs = {"filter" : {"metadata.stock_name" : "삼성전자"}}
query = "What is the PER of LG전자 in 2021?"
k = 5
result = vectordb.similarity_search_with_score(query, k, where=search_kwargs)

#vectordb.similarity_search_with_score(query, k, filter={"filter" : {"metadata.stock_name" : "삼성전자"}})
#vectordb.similarity_search_with_score(query, k, {"metadata.stock_name" : "삼성전자"})

#What is the PER of LG전자 in 2021?
    
    
    
    # loader = UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn))
    # docs = loader.load()
    

# for doc in docs:
#     collection.add(
#         ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
#     )

# # tell LangChain to use our client and collection name
# db4 = Chroma(client=client, collection_name="my_collection")
# docs = db.similarity_search(query)
# print(docs[0].page_content)