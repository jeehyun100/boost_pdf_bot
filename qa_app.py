import os
import PyPDF2
import random
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from chromadb.config import Settings
import chromadb
from langchain.vectorstores import Chroma
import langchain
from draw_candlestick_complex import get_candlestick_plot
import pandas as pd
langchain.verbose = True
import FinanceDataReader as fdr
df_krx = fdr.StockListing('KRX')




st.set_page_config(page_title="Stock Analyzer",page_icon=':shark:')

@st.cache_data
def load_docs(files):
    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


def conn_chromadb(chroma_setting, my_collection, embeddings):
    chroma_db = Chroma(client_settings=chroma_setting, collection_name=my_collection, embedding_function = embeddings)
    return chroma_db


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list

    st.info("`Generating sample questions ...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


# ...

def main():
    load_dotenv()
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 30%;
        right: 0;
        width: 50%;
        padding: 0px 0px;
        text-align: center;
    ">
        <p>Made by <a href='https://twitter.com/mehmet_ba7'>Mehmet Balioglu</a> Modifed bi JH</p>
    </div>
    """

    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            
           
       
            
          

        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.image("img/logo1.png")


   

    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Stock Analyzer</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.sidebar.title("Menu")
    
        # Sidebar options
    ticker = st.sidebar.selectbox(
        'Ticker to Plot', 
        options = ['TSLA', 'MSFT', 'AAPL']
    )

    days_to_plot = st.sidebar.slider(
        'Days to Plot', 
        min_value = 1,
        max_value = 300,
        value = 120,
    )
    ma1 = st.sidebar.number_input(
        'Moving Average #1 Length',
        value = 10,
        min_value = 1,
        max_value = 120,
        step = 1,    
    )
    ma2 = st.sidebar.number_input(
        'Moving Average #2 Length',
        value = 20,
        min_value = 1,
        max_value = 120,
        step = 1,    
    )
    
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])

    
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"
    #breakpoint()
    st.session_state.openai_api_key = os.environ.get('OPEN_API_KEY')
    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            #warning_text = 'Please enter your OpenAI API key. Get yours from here: [link](https://platform.openai.com/account/api-keys)'
            #warning_html = f'<span>{warning_text}</span>'
            #st.markdown(warning_html, unsafe_allow_html=True)
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)
    
    # Embed using OpenAI embeddings
    # Embed using OpenAI embeddings or HuggingFace embeddings
    if embedding_option == "OpenAI Embeddings":
        embeddings = OpenAIEmbeddings()
    elif embedding_option == "HuggingFace Embeddings(slower)":
        # Replace "bert-base-uncased" with the desired HuggingFace model
        embeddings = HuggingFaceEmbeddings()
    
    
    chroma_setting =  Settings(
        chroma_api_impl="rest",
        chroma_server_host="172.19.0.1",
        chroma_server_http_port="8000",
    )
    my_collection = "test4"

    chroma_db =  conn_chromadb(chroma_setting, my_collection, embeddings)
    
    if chroma_db:
        print("connect db")
        
        
        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        
        p_search_kwargs = {"filter" : {"stock_name" : "삼성전자"}}
        #query = "What is the 영업이익 of 삼성전자?"
        #k = 2

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)

        #retriever = chroma_db.as_retriever(k=2, verbose=True)
        retriever = chroma_db.as_retriever(k=2, search_kwargs = p_search_kwargs)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)
        
        c_client = chromadb.Client(chroma_setting)
        collection = c_client.get_or_create_collection(name=my_collection)

        p_where_kwargs = {"doc_id" : "103008"}
        text2 = collection.get(where = p_where_kwargs)
        docs_text = text2['documents']
        all_text = ' '.join(docs_text)
        
        #loaded_text = chroma_db.get
        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 3  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                all_text, num_eval_questions, 3000)

       # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")
        
        
        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)
            
        # Get the dataframe and add the moving averages
        df = fdr.DataReader(f'{ticker}','2020')

        #df = pd.read_csv(f'{ticker}.csv')
        df[f'{ma1}_ma'] = df['Close'].rolling(ma1).mean()
        df[f'{ma2}_ma'] = df['Close'].rolling(ma2).mean()
        df = df[-days_to_plot:]

        # Display the plotly chart on the dashboard
        st.plotly_chart(
            get_candlestick_plot(df, ma1, ma2, ticker),
            use_container_width = True,
        )
                


if __name__ == "__main__":
    main()
