import os
import streamlit as st
from documentGPT.cfg import config

from documentGPT.document_ingestion import main
from io import StringIO
import time


file_input_path = config.source_directory

def get_retriever():
    settings = config.chroma_settings
    


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


# st.title("Document GPT")

with st.form("submit_form"):
    text = st.text_input(label="Enter your query.")
    button=st.form_submit_button("Answer")

if button:
    st.write(text)

