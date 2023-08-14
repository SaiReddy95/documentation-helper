import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

load_dotenv()

chat = AzureChatOpenAI(
    # deployment_name="gpt-4",
    openai_api_type="azure",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_base=os.getenv('OPENAI_API_BASE'), 
    openai_api_version="2023-03-15-preview",
    verbose=True,
    temperature=0,)

def adjusted_test_configuration():
    try:
        query = "Hello, how are you?"
        response = chat([HumanMessage(content="Translate this sentence from English to French: I love programming.")])
        print("Response from OpenAI:", response.dict()['content'])
    except Exception as e:
        print("An error occurred:", str(e))



if __name__ == "__main__":
    adjusted_test_configuration()



# chat({
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello, how are you?"}
#     ]
# })