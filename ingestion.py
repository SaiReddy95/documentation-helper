import os

from langchain.document_loaders import ReadTheDocsLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone
# import pinecone

# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
# )
# INDEX_NAME = "langchain-doc-index"
import PyPDF2



def ingest_docs():
    with open('/Users/saimethukupally/Documents/GitHub/documentation-helper/langchain-docs/Journaling - 2021 to 2023.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    #loader = ReadTheDocsLoader(path = "/Users/saimethukupally/Documents/GitHub/documentation-helper/langchain-docs")
    raw_documents = reader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    # for doc in documents:
    #     new_url = doc.metadata["source"]
    #     new_url = new_url.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    # embeddings = OpenAIEmbeddings()
    # print(f"Going to add {len(documents)} to Pinecone")
    # Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    # print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()
