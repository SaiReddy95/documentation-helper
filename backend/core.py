import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_type="azure",
        chunk_size = 1,
    )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = AzureChatOpenAI(
        deployment_name="gpt-4",
        openai_api_type="azure",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE'), 
        openai_api_version="2023-03-15-preview",
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="tell me about some of the activities I like to do at COPA?"))
