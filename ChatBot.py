from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI
import sys
import os
from colorama import Fore, Style

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 500
    # set number of output tokens
    num_outputs = 4096
    # set maximum chunk overlap
    max_chunk_overlap = 1.0
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    
    llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo",
        max_tokens=num_outputs,
        frequency_penalty=0,
        presence_penalty=0.6,
        model_kwargs={
            "top_p": 1,
        }
    )
)
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = VectorStoreIndex.from_documents(
     documents, service_context=service_context
    )

    index.storage_context('index.json')


    return index

def ask_ai():
    index = VectorStoreIndex.load_from_disk('index.json')
    print(
            Fore.GREEN
            + Style.BRIGHT
            + "Đây là ChatBot là có vấn cho Viện quốc tế của đại học Hutech.\n"
            + Fore.GREEN + Style.BRIGHT + "ChatBot: Tôi có thể giúp gì được cho bạn?"
        )
    while True: 
        query =  input(Fore.WHITE + Style.BRIGHT +"User: ")
        response = index.query(query)
        print(Fore.YELLOW + Style.BRIGHT + Style.NORMAL +"Chatbot: " + response.response)


os.environ["OPENAI_API_KEY"] = "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"

construct_index("data")
ask_ai()