from langchain_chroma import Chroma
from init import init
from langchain_openai import OpenAIEmbeddings

init()


def create_chroma_vector(storage_path):
    # Initialize the Chroma vectorstore
    return Chroma(persist_directory=storage_path,
                  embedding_function=OpenAIEmbeddings())
