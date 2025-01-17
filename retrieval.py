import sys
from init import init
from vectorstore import create_chroma_vector

init()


def retrieve(question, num_docs, vectorstore):
    # Retrieve the most similar documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs})
    return retriever.invoke(question)


if __name__ == "__main__":

    # Check if arguments are passed
    if len(sys.argv) < 2:
        print("Please provide a storage path as arguments")
        print("Example: python ", sys.argv[0], " ./chroma_storage")
        sys.exit(1)

    storage_path = sys.argv[1]
    print("Storage path=", storage_path)

    # Load the vectorstore from the persisted directory
    vectorstore = create_chroma_vector(storage_path)

    # Get the number of documents in the vectorstore
    num_documents = vectorstore._collection.count()
    print("Loaded Chroma vectorstore with", num_documents, "documents")
    print("For any query ***CALL TO OPENAI API FOR EMBEDDINGS***")

    while True:
        # Query the vectorstore from user input
        question = input("Enter a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            print("Exiting...")
            break

        # Ask for the number of documents to retrieve
        try:
            num_docs = int(
                input("Enter the number of documents to retrieve: "))
        except ValueError:
            print("Invalid number. Please enter an integer.")
            continue

        docs_retrieved = retrieve(question, num_docs, vectorstore)

        print(f"{num_docs} most similar documents:")
        print()
        print(" ***** ")
        print()
        # pretty print of doc_retrieved
        for doc in docs_retrieved:
            print(f" + Source: {doc.metadata.get('source', 'Unknown')}")
            print()
            print(doc.page_content)
            print()
            print("----")
            print()
        print()
        print(" ***** ")
