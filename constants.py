import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# for HF models
# model_id = "TheBloke/vicuna-7B-1.1-HF"
# model_basename = None
# model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# model_id = "TheBloke/guanaco-7B-HF"
# model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

# for GPTQ (quantized) models
# model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
# model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
MODEL_ID = "TheBloke/orca_mini_v3_13B-GPTQ"
MODEL_BASENAME = "model.safetensors"
# model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
# ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
# model_id = "TheBloke/wizardLM-7B-GPTQ"
# model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
# model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

# for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
# model_id = "TheBloke/wizard-vicuna-13B-GGML"
# model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# model_basename = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# model_basename = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# model_id = "TheBloke/orca_mini_3B-GGML"
# model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"

# model_id = "TheBloke/Llama-2-7B-Chat-GGML"
# model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"

# Define the acceptable device types
DEVICE_TYPES = [
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
]


# Prompt template
TEMPLATE = \
    """
    You are a helpful systems engineer with 20 years of experience who answers questions
    about systems engineering. You will be provided with relevant information in the form
    of documents. Additional context for the question may be found in the chat history. Your
    task is to write a clear, helpful, detailed, and factually accurate answer to the question
    at the end using only the provided documents and chat history. If the documents do not
    contain the information needed to answer the question, generate an answer regardless, but
    include the string "This answer contains information not present in the source documents."
    at the end.

    Documents: ###
    {context}
    ###

    Chat History: ###
    {history}
    ###

    Question: ###
    {question}
    ###

    Helpful Answer:
    """
