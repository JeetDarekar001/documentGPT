
from chromadb import Settings
import os
from loguru import logger
class CFG:
    #Base Path
    base_path = "documentGPT"

    # Specifying Input File
    source_directory = os.path.join(base_path,'inputs')
    
    # Specifying Model Type
    model_type = "hf" # openai or hf.
    
    # HuggingFace Arguments
    hf_model_name = "facebook/opt-1.3b"
    hf_text_embedd_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    # Reteriever Engine Arguments
    retriever_search_kwargs=''
    
    # OpenAI Arguments
    open_ai_key = 'afas'
    # To ByPass RateLimit Error
    numberOfTOkensPerMinToEmbedd = 500
    chunk_size = 1000
    chunk_overlap = 200

    # ChromaDB Arguments
    # persist_directory = os.path.join(base_path,'db')

    if model_type == 'openai':
        persist_directory = os.path.join(base_path,'db_open_ai')
    elif model_type == 'hf':
        persist_directory = os.path.join(base_path,'db_hf')


    chroma_settings = Settings(
                                chroma_db_impl="duckdb+parquet",
                                persist_directory=persist_directory,
                                anonymized_telemetry=False,
                            )

config =  CFG()
