# create the chroma client
import chromadb
import uuid
from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sqlalchemy import create_engine, Column, Integer, String, DateTime, UnicodeText, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from models import Article, db_connect, create_tables
from sqlalchemy import select
from sqlalchemy.orm import Session
import numpy as np
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import langchain
langchain.verbose = True


OPEN_API_KEY = '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'
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

chunk_size = 1000
overlap = 0

rc_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

chroma_setting =  Settings(
        chroma_api_impl="rest",
        chroma_server_host="172.19.0.1",
        chroma_server_http_port="8000",
    )
c_client = chromadb.Client(
   chroma_setting
)
# #client.reset()  # resets the database
# #collection = client.create_collection("test_samsung")
my_collection = "test4"
# collection = c_client_setting.get_or_create_collection(name=my_collection)
# # # # list all collections
# # # list_c = client.list_collections()
# # # # get an existing collection
# # collection2 = client.get_collection(my_collection)
# c_r = collection.count()
chroma_db = Chroma(client_settings=chroma_setting, collection_name=my_collection, embedding_function = embeddings)


# # await Chroma.fromExistingCollection(
#   new OpenAIEmbeddings(),
#   { collectionName: "godel-escher-bach" }
# );




def list_split(lst, n):
    nd_array = np.array(lst)
    return [l_item.tolist() for l_item in np.array_split(lst, n)]


def check_by_id(stock_name):
    engine = db_connect()
    stmt = select(Article).where(Article.stocks.like(f'{stock_name}%'))  
    # User 객체의 인스턴스 안에 있는 각 row 들을 출력
    with Session(engine) as session:
        resultproxy= session.execute(stmt)
        #  for row in session.execute(stmt):
        #      print(row)
 
            
        resultdict = [ rowproxy.Article.__dict__ for rowproxy in resultproxy]
    return resultdict


    
    


def load_docs(files):
    all_text = ""
    all_docs = []
    for file_info in files:
        file_path = file_info['attachment_file']
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".pdf":
            pdf_reader = UnstructuredPDFLoader(os.path.join(pdf_folder_path, file_path))
            pdf_data = pdf_reader.load_and_split(text_splitter=rc_text_splitter)
            
            for modify_doc_meta in pdf_data:
                get_meta = modify_doc_meta.metadata
                get_meta['stock_name'] = file_info['stocks']
                get_meta['stock_id'] =  file_info['created_dt'].strftime('%Y.%m.%d')
                get_meta['doc_id'] =  file_info['id']
                modify_doc_meta.metadata = get_meta
            all_docs.extend(pdf_data)

    return all_docs

def create_retriever(_db, _embeddings, docs, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = _db.from_documents(docs, _embeddings, collection_name = my_collection, client_settings=chroma_setting)
            vectorstore.persist()
        except (IndexError, ValueError) as e:
            print(e)
            #st.error(f"Error creating vectorstore: {e}")
            return
    return vectorstore



pdf_folder_path = f'/shared_data/s_data/paxnet_pdf/full'
files = os.listdir(pdf_folder_path)
    
files_result = check_by_id('삼성전자')
##ddd = row_to_files(fff)

batch_list = list_split(files_result,25)

for batch in batch_list:
    all_docs =  load_docs(batch)
    # load it into Chroma
    retriever_type = "SIMILARITY SEARCH"
    vectordb = create_retriever(chroma_db, embeddings, all_docs, retriever_type)
    print("done")

"""
vectordb = Chroma(collection_name="test_samsung", client_settings=chroma_setting, embedding_function=embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
"""

#search_kwargs={"filter":{'$or': [{'source': {'$eq': './SampleDoc/Bikes.pdf'}}, {'source': {'$eq': './SampleDoc/IceCreams.pdf'}}]}}
#p_search_kwargs = {"k":2, "filter" : {"metadata.stock_name" : "삼성전자"}}
p_search_kwargs = {"filter" : {"stock_name" : "삼성전자"}}
query = "What is the 영업이익 of 삼성전자?"
k = 2
chat_openai = ChatOpenAI(openai_api_key=OPEN_API_KEY, streaming=True, verbose=True, temperature=0)
collection = c_client.get_or_create_collection(name=my_collection)


p_where_kwargs = {"doc_id" : "103008"}
text2 = collection.get(where = p_where_kwargs)
docs_text = text2['documents']
all_text = ' '.join(docs_text)

#retriever = chroma_db.as_retriever(k=2, verbose=True)
retriever = chroma_db.as_retriever(k=2, search_kwargs = p_search_kwargs)

qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
answer = qa.run({"query": query})
# #66f30247-2c6d-476d-a909-201223df74d4
#result = chroma_db.similarity_search_with_score(query, k, where=search_kwargs)
#print(answer)
#    output= qa({"question": data['prompt']}, callbacks=[ConsoleCallbackHandler()])
print(answer)

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