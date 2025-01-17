from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retrieval import retrieve
import sys
from init import init
from vectorstore import create_chroma_vector

if __name__ == "__main__":

    # Check if arguments are passed
    if len(sys.argv) < 2:
        print("Please provide a storage path as arguments")
        print("Example: python ", sys.argv[0], " ./chroma_storage")
        sys.exit(1)

    storage_path = sys.argv[1]
    print("Storage path=", storage_path)

    init()

    # vectore database
    vectorstore = create_chroma_vector(storage_path)

    # Get the number of documents in the vectorstore
    num_documents = vectorstore._collection.count()
    print("Loaded Chroma vectorstore with", num_documents, "documents")
    print("For any query ***CALL TO OPENAI API FOR EMBEDDINGS***")

    # Prompt
    template = """Répond à la question en tenant compte uniquement du contexte suivant :
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    print(" ***** ")
    print()

    while True:
        # Query the vectorstore from user input
        question = input(
            "Enter a question in french (or type 'quit' to exit): ")
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

        # Chain
        chain = prompt | llm

        # Run
        answer = chain.invoke(
            {"context": docs_retrieved, "question": question})

        print("Answer:", answer.content)
        # pretty print of doc_retrieved
        for doc in docs_retrieved:
            print(f" + {doc.metadata.get('source', 'Unknown')}")
        print()
        print(" ***** ")
        print()
