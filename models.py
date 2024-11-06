import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from abc import ABC, abstractmethod
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RetrievalChain 추상 클래스 정의
class RetrievalChain(ABC):
    def __init__(self, source_uri):
        self.source_uri = source_uri
        self.k = 10  # 검색할 문서 수

    @abstractmethod
    def load_documents(self, source_uris):
        pass

    @abstractmethod
    def create_text_splitter(self):
        pass

    def split_documents(self, docs, text_splitter):
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        embedding = OpenAIEmbeddings()
        return embedding

    def create_vectorstore(self, split_docs):
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    @staticmethod
    def format_docs(docs):
        return "\n".join(
            [
                f"<document><content>{doc.page_content}</content><source>{doc.metadata.get('source', '정보 없음')}</source><page>{int(doc.metadata.get('page', 0)) + 1}</page></document>"
                for doc in docs
            ]
        )

    def create_chain(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        self.model = self.create_model()
        self.chain = self.model
        return self

# JSONRetrievalChain 클래스 정의
class JSONRetrievalChain(RetrievalChain):
    def __init__(self, source_uri):
        super().__init__(source_uri)

    def load_documents(self, source_uris):
        data_dir = "./data/mock.json"
        import json
        split_docs = []

        with open(data_dir, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                if isinstance(json_data, dict) and 'data' in json_data:
                    for index, item in enumerate(json_data['data']):
                        if isinstance(item, dict):
                            doc = Document(
                                page_content=item.get('description') or item.get('name') or '',
                                metadata={
                                    'id': item.get('id'),
                                    'category': item.get('category'),
                                    'price': item.get('price'),
                                    'image': item.get('image'),
                                    'img_url': item.get('img_url'),
                                    'info_url': item.get('info_url'),
                                    'name': item.get('name'),
                                }
                            )
                            split_docs.append(doc)
                        else:
                            print("JSON 데이터가 예상된 형식이 아닙니다.")
                    print(f"전체 {len(json_data['data'])}개 중 {len(split_docs)}개의 문서를 로드했습니다.")
            except json.JSONDecodeError as e:
                print(f"JSON 파일 파싱 중 오류가 발생했습니다: {e}")
        return split_docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
