import os
import glob
from huggingface_hub import delete_collection
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv(override=True)

db_name = "vector_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def fetch_documents():
  folders = glob.glob('../*/')
  folders = [f for f in folders if f not in ['..\\implementation\\','..\\vector_db\\']]
  documents=[]
  print(folders)
  for folder in folders:
    doc_type = folder.replace(".",'').replace('//','').replace('\\','')
    print(folder)
    loader = DirectoryLoader(folder,glob='**/*.md',loader_cls=TextLoader,loader_kwargs={'encoding':'utf-8'})

    folder_docs = loader.load()
    for docs in folder_docs:
      docs.metadata['doc_type'] = doc_type
      documents.append(docs)
  return documents

def create_chunks(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
  chunks = text_splitter.split_documents(documents)
  return chunks

def create_embeddings(chunks):
  if os.path.exists(db_name):
    Chroma(persist_directory=db_name,embedding_function=embeddings).delete_collection
  vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory=db_name)
  collection = vectorstore._collection
  count = collection.count()
  sample_embedding = collection.get(include=['embeddings'])['embeddings'][0]
  dimensions = len(sample_embedding)
  print(f"There are {count} vecotors with {dimensions} simensions in the vecotr store")
  return vectorstore


if __name__ == '__main__':
  documents = fetch_documents()
  chunks = create_chunks(documents)
  create_embeddings(chunks)
  print("Ingestion Complete")
