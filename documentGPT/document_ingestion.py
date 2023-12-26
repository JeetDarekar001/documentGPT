
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, CSVLoader , TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.server import Settings
from langchain.docstore.document import Document

import tiktoken


import time
from typing import List
import math
import os
from .cfg import config
from loguru import logger
import glob



def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier

# Function to calculate openAI pricing
def tokens2price(model, task, tokens):
    models = {
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
        "text-embedding-ada-002": {"embedding": 0.0001},
    }
    price = round_up(models[model][task] / 1000 * tokens, 5)
    return price

def text2tokens(model, text):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]
    results = []
    for files in filtered_files:
        docs = load_document(files)
        results.extend(docs)

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    logger.info(f"Loading documents from '{config.source_directory}'")
    documents = load_documents(config.source_directory, ignored_files)
    if not documents:
        logger.debug("No new documents to load")
        return None
    logger.info(f"Loaded {len(documents)} new documents from {config.source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logger.info(
        f"Split into {len(texts)} chunks of text (max. {config.chunk_size} tokens each)"
    )
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                logger.info("Database Present.")
                return True
    logger.info("Database Not Present.")
    return False

def main():
    if os.path.exists(config.source_directory) and os.listdir(config.source_directory):
        print("int to the web")

        if config.model_type == 'openai':
            pass
        if config.model_type == 'hf':
            embeddings =  HuggingFaceEmbeddings(model_name=config.hf_text_embedd_model_name)
        logger.debug('Created Embeddinfs')
        if does_vectorstore_exist(config.persist_directory):
            logger.info(f"Appending to existing vectorstore at {config.persist_directory}")
            db = Chroma(
                persist_directory=config.persist_directory,
                embedding_function=embeddings,
                client_settings=config.chroma_settings,
            )
            
            collection = db.get()
            texts = process_documents(
                [metadata["source"] for metadata in collection["metadatas"]]
            )
            if texts:
                logger.info(f"Creating embeddings. May take some minutes...")
                # db.add_documents(texts)

                """Checking if length of texts is greater then 500 (1000 per chunks). This step is performed because OpenAI will raise 
                RateLimitError , is number of tokens to embedd cross 150 000 per minute."""
                if config.model_type == "openai":
                    if len(texts) < config.numberOfTOkensPerMinToEmbedd:
                        # Adding Directly as length of texts is not exceeding 500.
                        db.add_documents(texts)
                        db.persist()
                        logger.info("ByPassed config.numberOfTOkensPerMinToEmbedd ")
                    else:
                        """If Length grater then 500 , then we are adding first 500 chunks to db, storing the database
                        then iterating through other"""
                        db.add_documents(texts[:config.numberOfTOkensPerMinToEmbedd])
                        db.persist()
                        logger.debug("Initial Sleeping for 1MIN ,Skipping RateLimitError")
                        time.sleep(60)
                        for i in range(
                            config.numberOfTOkensPerMinToEmbedd,
                            len(texts),
                            config.numberOfTOkensPerMinToEmbedd,
                        ):
                            logger.debug(
                                f"Ingested {i} chunks of total {len(texts)} chunks "
                            )
                            db.add_documents(texts[i : i + config.numberOfTOkensPerMinToEmbedd])
                            db.persist()
                            if i + config.numberOfTOkensPerMinToEmbedd >= len(texts):
                                pass
                            else:
                                logger.debug("Sleeping for 1MIN , Skipping RateLimitError")
                                time.sleep(60)

                    # Calculate cost of embeddiings per embeddings.
                    EMBEDDING_PRICE = 0
                    if config.model_type == "openai":
                        tokens = 0
                        for text in texts:
                            tokens += text2tokens(embeddings.model, text.page_content)
                        logger.info(
                            f"Total Number of tokens obtained from the documents : {tokens}"
                        )

                        EMBEDDING_PRICE = tokens2price(
                            "text-embedding-ada-002", "embedding", tokens
                        )
                        return EMBEDDING_PRICE
                elif config.model_type =='hf':
                    db.add_documents(texts)
                    db.persist()
                    return "Added New File Embeddings."
            else:
                return False
        else:
            logger.info("Creating new vectorstore")
            texts = process_documents()
            logger.info(f"Creating embeddings. May take some minutes...")
            # db = Chroma.from_documents(texts, embeddings, config.persist_directory=config.persist_directory, client_settings=CHROMA_SETTINGS)
            if config.model_type == 'openai':
                if len(texts) < config.numberOfTOkensPerMinToEmbedd:
                    db = Chroma.from_documents(
                                                texts,
                                                embeddings,
                                                persist_directory=config.persist_directory,
                                                client_settings=config.chroma_settings,
                                            )
                    db.persist()
                else:
                    db = Chroma.from_documents(
                                                texts[:config.numberOfTOkensPerMinToEmbedd],
                                                embedding=embeddings,
                                                persist_directory=config.persist_directory,
                                                client_settings=config.chroma_settings,
                                            )
                    db.persist()
                    logger.debug("Waiting for 1 min ,Skipping RateLimitError")
                    time.sleep(60)
                    for i in range( config.numberOfTOkensPerMinToEmbedd,len(texts),config.numberOfTOkensPerMinToEmbedd, ):
                        logger.debug(f"Ingested {i} chunks of total {len(texts)} chunks ")
                        db.add_documents(texts[i : i + config.numberOfTOkensPerMinToEmbedd])
                        db.persist()
                        if i + config.numberOfTOkensPerMinToEmbedd >= len(texts):
                            pass
                        else:
                            logger.debug("Sleeping for 1MIN , Skipping RateLimitError")
                            time.sleep(60)
                    EMBEDDING_PRICE = 0
                    tokens = 0
                    for text in texts:
                        tokens += text2tokens(embeddings.model, text.page_content)
                    logger.info(
                        f"Total Number of tokens obtained from the documents : {tokens}"
                    )
                    EMBEDDING_PRICE = tokens2price(embeddings.model, "embedding", tokens)
                    logger.info(
                        f"Total Cost of embedding {tokens} tokens  : {EMBEDDING_PRICE}"
                    )
                db = None
                logger.info(f"Ingestion complete! You can now  query your documents")

                return EMBEDDING_PRICE
            

            elif config.model_type == 'hf':
            
                db = Chroma.from_documents(
                                                texts,
                                                embeddings,
                                                persist_directory=config.persist_directory,
                                                client_settings=config.chroma_settings,
                                            )
                db.persist()
                return "Embeddings Created Successfully. You are good to go"
            else :
                raise NotImplementedError

if __name__ == '__main__':
    main()