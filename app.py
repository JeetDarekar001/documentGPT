import os
import streamlit as st
from documentGPT.cfg import config

from documentGPT.document_ingestion import main
from io import StringIO
path_prefix = 'documentGPT'


file_input_path = os.path.join(path_prefix,config.source_directory)

st.set_page_config(layout="wide") 
def get_document():
    try: 
        total_files = os.listdir(file_input_path)
        if total_files is None:
            return []
        else:
            return total_files
    except Exception as e:
        print("Caught Exception")
        return []

with st.sidebar:
    st.header("Upload Files to ask questions.",divider=True)
    # with st.form("uplaod_form"):
    upload_file = st.file_uploader("Upload Document", accept_multiple_files=False,help="Uplaod Either pdf, csv or text files",
                                on_change=None, disabled=False, label_visibility="hidden")
    processButton = st.button("Ingest Files")
        
    if  upload_file is not None :
        with open(os.path.join(file_input_path,upload_file.name),'wb') as file:
            data=upload_file.read()
            file.write(data)

    if processButton:
        response = main()

    del upload_file
    st.header("Available Files:",divider=True)
    for doc in get_document():
        st.write(doc)





st.title("Document GPT")

with st.form("submit_form"):
    text = st.text_input(label="Enter your query.")
    button=st.form_submit_button("Answer")

if button:
    st.write(text)

