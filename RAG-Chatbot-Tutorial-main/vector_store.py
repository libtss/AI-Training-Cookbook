import os
#The API details is stored at .env
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY_embedding')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT_embedding')


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS


def process_pdf(pdf_file):
    #Load PDF document
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #Split the text in chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    #input your openai_api details
    embeddings = AzureOpenAIEmbeddings(openai_api_version="openai_api_version",azure_deployment= "azure_deployment",chunk_size=1 )
    #FAISS Vector Database
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
  
    return VectorStore

pdf_path = "faq_library.pdf"
Vector_store = process_pdf(pdf_path)
