import logging

import click
import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.formatting import formatter

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, DEVICE_TYPES


def load_model(device_type, model_id, max_length, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# chose device typ to run on as well as to show source documents.
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
# @click.option(
#     "--show_sources",
#     "-s",
#     is_flag=True,
#     help="Show sources along with answers (Default is False)",
# )
@click.option(
    "--openai",
    "-o",
    is_flag=True,
    help="Use OpenAI as the llm model instead (Default is False)",
)
@click.option(
    "--max_length",
    default = 2048
)
# def main(device_type, show_sources):
def main(device_type, max_length, openai):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by ingest.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retrieval chain.
    5. Question answers.
    """

    # Take parameters in as inputs (command line doesn't work great with notebooks)
    show_sources = ""
    while (show_sources not in [True, False]):
        res = input ("Show sources (y/n): ")
        if res == 'y':
            show_sources = True
        elif res == 'n':
            show_sources = False

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
    model_id = "TheBloke/orca_mini_v3_13B-GPTQ"
    model_basename = "gptq_model-4bit-128g.safetensors"
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

    if openai:
        logging.info("Using OpenAI models, your queries are NOT local")
        openai_api_key = input("Please provide a valid OpenAI API key: ")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        llm = ChatOpenAI(openai_api_key=openai_api_key)
    else:
        if (input(f"Use default model id and basename\n{model_id}\n{model_basename}\n(y/n)? ") not in ["Y", "y"]):
            model_id = input("Model ID: ")
            model_basename = input("Model basename: ")
        if (model_basename == "None"):
            model_basename = None

        logging.info(f"Running on: {device_type}")
        logging.info(f"Display Source Documents set to: {show_sources}")
        logging.info(f"Model ID: {model_id}")
        logging.info(f"Model basename: {model_basename}")

        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )

        # load the LLM for generating Natural Language responses
        llm = load_model(device_type, model_id=model_id, max_length=max_length, model_basename=model_basename)

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    template = \
        """
        Use the following context and conversation history to answer the question at the end. If you don't know the answer,\
        just say that you don't know, don't try to make up an answer.
        Context: {context}
        History: {history}
        Question: {question}
        Helpful Answer:
        """

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    # # Remove old chat history to stay within context window
    # def truncate_prompt(history, context, question):
    #     template = \
    #     """
    #     Use the following context and conversation history to answer the question at the end. If you don't know the answer,\
    #     just say that you don't know, don't try to make up an answer.
    #     Context: {context}
    #     History: {history}
    #     Question: {question}
    #     Helpful Answer:
    #     """
    #     prompt = formatter.format(template, context, history, question)
    #     extra = len(prompt) - max_length
    #     if extra > 0:
    #         prompt = formatter.format(template, context, history[extra:], question)
    #     return prompt
    
    # truncate_prompt.input_variables=["history", "context", "question"]

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("\n----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
