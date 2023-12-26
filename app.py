import os
import streamlit as st
from documentGPT.cfg import config

from documentGPT.document_ingestion import main
from io import StringIO
from langchain.vectorstores import Chroma
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
os.environ["HUGGINGFACEHUB_API_TOKEN"]= "KEY"

file_input_path = config.source_directory
embedding_func = HuggingFaceEmbeddings(model_name=config.hf_text_embedd_model_name)#,cache_foler = '/mnt/f/conda/name/HUGGING_FACE_CACHE_DO_NOT_DELETE/')

global qachain 
qachain = None


def get_retriever():
    db = Chroma(embedding_function = embedding_func,persist_directory=config.persist_directory)
    retriever = db.as_retriever()
    template="You are a DocumentGPT,Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    prompt=PromptTemplate(template=template,input_variables=['context','question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id=config.hf_model_name),#,cache='/mnt/f/conda/name/HUGGING_FACE_CACHE_DO_NOT_DELETE/'), 
        chain_type="stuff",
        retriever=retriever, 
        return_source_documents=True, 
        chain_type_kwargs={"prompt":prompt})

    return qa_chain


st.set_page_config(layout="wide") 
if os.path.exists(config.source_directory):
    print("Directry Exists")
else:
    os.mkdir(config.source_directory)

def get_document():
    try: 
        total_files = os.listdir(file_input_path)
        if total_files is None:
            return []
        else:
            return total_files
    except Exception as e:
        print("Caught Exception",e)
        return []

docs = get_document()

with st.sidebar:
    st.header("Upload Files.",divider=True)
    # with st.form("uplaod_form"):
    upload_file = st.file_uploader("Upload Document", accept_multiple_files=False,help="Uplaod Either pdf, csv or text files",
                                on_change=None, disabled=False, label_visibility="hidden")
    processButton = st.button("Ingest Files")
        
    if processButton:
        if upload_file is not None:
            docs = get_document()
            if upload_file.name not in docs: 
                with st.status("Processing file.."):
                    with open(os.path.join(file_input_path,upload_file.name),'wb') as file:
                        data=upload_file.read()
                        file.write(data)
                    st.write("File Uploaded Successfully.")
                    st.write("Creating Embeddings.")
                    response = main()

                    qachain = get_retriever()
                    st.write(response)

            else:
                st.write("File ALready Present.")
        else:
            st.write("No input File found.")

    del upload_file
    st.header("Available Files:",divider=True)
    docs = get_document()
    if docs is None:
        st.write("No Document Found. Upload Documents to process.")
    else:   
        for doc in get_document():
            st.write(doc)



try :
    
    qachain = get_retriever()

    with st.form("submit_form"):
        text = st.text_input(label="Enter your query.")
        button=st.form_submit_button("Answer")

    if button:
        st.write(qachain(text))

except ValueError as v:
    print("Exception:",v)
    st.write("Please Uplaod some documents to ask questions.")
except Exception as e:
    print("CAught Exceptiom",e)



