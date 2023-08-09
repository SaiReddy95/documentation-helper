import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from collections import namedtuple
from dotenv import load_dotenv
import pinecone
import PyPDF2

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "langchain-doc-index"
# Define the MockDocument structure outside the function for clarity
MockDocument = namedtuple("MockDocument", ["page_content", "metadata"])



def ingest_docs():
    with open('/Users/saimethukupally/Documents/GitHub/documentation-helper/langchain-docs/Journaling - 2021 to 2023.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    #loader = ReadTheDocsLoader(path = "/Users/saimethukupally/Documents/GitHub/documentation-helper/langchain-docs")
    raw_documents = text
    print(f"loaded {len(raw_documents)} documents")
    doc = MockDocument(page_content=text, metadata={})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents([doc])
    print(f"Splitted into {len(documents)} chunks")
    # for doc in documents:
    #     new_url = doc.metadata["source"]
    #     new_url = new_url.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_type="azure",
        chunk_size = 1,
    )
    # embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()
