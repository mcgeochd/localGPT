'''
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
'''

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from constants import TEMPLATE

# this is specific to Llama-2. 

def get_prompt_template(system_prompt=TEMPLATE, promptTemplate_type=None, history=False):

    if promptTemplate_type=="llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Documents: ###
            {context}
            ###

            Chat History: ###
            {history}
            ###

            Question: ###
            {question}
            ###"""

            prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Documents: ###
            {context}
            ###

            Chat History: ###
            
            ###

            Question: ###
            {question}
            ###

            Helpful Answer:"""

            prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    else:
        # change this based on the model you have selected. 
        if history:
            prompt_template = system_prompt + """
    
            Documents: ###
            {context}
            ###

            Chat History: ###
            {history}
            ###

            Question: ###
            {question}
            ###

            Helpful Answer:"""
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = system_prompt + """
            
            Documents: ###
            {context}
            ###

            Chat History: ###

            ###

            Question: ###
            {question}
            ###

            Helpful Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory, 