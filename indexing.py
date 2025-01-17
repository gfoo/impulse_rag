from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
from init import init
from vectorstore import create_chroma_vector

if __name__ == "__main__":

    # Check if arguments are passed
    if len(sys.argv) < 3:
        print("Please provide a pdf folder path and a storage path as arguments")
        print("Example: python ", sys.argv[0], " ./pdfs ./chroma_storage")
        sys.exit(1)

    folder_path = sys.argv[1]
    storage_path = sys.argv[2]
    print("Folder path=", folder_path)
    print("Storage path=", storage_path)

    init()

    # splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,  # 300 tokens per chunk
        chunk_overlap=50)

    all_splits = []
    # load, split and index all pdfs from folder ./pdfs
    for pdf in Path(folder_path).rglob("*.pdf"):

        # Load
        loader = PyPDFLoader(pdf)
        pages = loader.load()

        # Make splits
        splits = text_splitter.split_documents(pages)
        all_splits.extend(splits)
        print("    + DOC: ", pdf, "nb pages=",
              len(pages), "splits=", len(splits))

    # Initialize the Chroma vectorstore
    vectorstore = create_chroma_vector(storage_path)

    # Index and store the documents in the Chroma vectorstore
    print("Indexing (***CALL TO OPENAI API FOR EMBEDDINGS***) nb splits=", len(all_splits))
    vectorstore.add_documents(documents=all_splits)

    print("Documents have been added and persisted to the vectorstore at", storage_path)
