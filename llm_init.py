# from langchain.llms import OpenAI
from langchain.chat_models import init_chat_model
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
import getpass


def init_llm_and_embeddings():

    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")



    # embeddings = OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    # llm = OpenAI()
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    return llm, embeddings