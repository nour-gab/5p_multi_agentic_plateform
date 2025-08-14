# PortalyzeBot.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle

load_dotenv()

class PortalyzeBot:
    def __init__(self, merged_report_path="data/report/merged_report.txt"):
        self.merged_report_path = merged_report_path
        self.index_path = "outputs/portalyze_index.pkl"
        self.qa_chain = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                vectorstore = pickle.load(f)
        else:
            with open(self.merged_report_path, "r", encoding="utf-8") as f:
                text = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.create_documents([text])
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(docs, embeddings)
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(vectorstore, f)

        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever())

    def ask(self, question: str, chat_history=None):
        if chat_history is None:
            chat_history = []
        response = self.qa_chain({"question": question, "chat_history": chat_history})
        return response["answer"]
