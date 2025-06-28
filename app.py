from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from rag.chain import create_rag_chain
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = create_rag_chain(llm)

    while True:
        query = input("\nUser: ")
        result = qa_chain.run(query)
        print(f"Assistant: {result}")
