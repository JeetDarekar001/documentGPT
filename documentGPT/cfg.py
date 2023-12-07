
from chromadb import Settings
class CFG:
    # Specifying Input File
    source_directory = '../inputs'
    
    # Specifying Model Type
    model_type = "openai" # openai or hf.
    
    # HuggingFace Arguments
    hf_model_name = ""
    hf_text_embedd_model_name = ''

    # Reteriever Engine Arguments
    retriever_search_kwargs=''
    
    # OpenAI Arguments
    open_ai_key = 'afas'
    # To ByPass RateLimit Error
    numberOfTOkensPerMinToEmbedd = 500
    chunk_size = 1000
    chunk_overlap = 200

    # ChromaDB Arguments
    persist_directory = 'db'
    chroma_settings = Settings(
                                chroma_db_impl="duckdb+parquet",
                                persist_directory=persist_directory,
                                anonymized_telemetry=False,
                            )

config =  CFG()
