import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import OpenAIEmbeddings

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
"--openai",
"-o",
is_flag=True,
help="Use OpenAI as the embedding model instead (Default is False)",
)

def main(device_type, openai):
    # Take variable source directory because we're in a notebook
    source_directory = input('Source directory (type "default" for default): ')
    if source_directory == "default":
        source_directory = SOURCE_DIRECTORY

    # Load documents and split in chunks
    logging.info(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {source_directory}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    if openai:
        logging.info("Using OpenAI embedding model, your documents are NOT local")
        openai_api_key = input("Please provide a valid OpenAI API key: ")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        logging.info("Using HuggingFaceInstructEmbeddings, your documents are local")
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
