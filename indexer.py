# Phase One: Storing vectors into vector DB
# PDF load karne ka index.js file

import os
import asyncio
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

async def indexDocument():
    
    const_PDF_PATH = './dsa.pdf'
    pdfLoader = PyPDFLoader(const_PDF_PATH)
    rawDocs = pdfLoader.load()
    print("PDF loaded")
    print(len(rawDocs))

#     # Chunking karo

    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunkedDocs = textSplitter.split_documents(rawDocs)
    print("Chunking Completed")
    print(len(chunkedDocs))
    # print("-----------start-------------------")
    # print(chunkedDocs)
    # print("---------------end----------------")

#     # vector Embedding model
    
    # --- FIX IS HERE: Added 'transport="rest"' ---
    # This prevents it from looking for Google Cloud Credentials
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model='models/text-embedding-004',
        transport="rest" 
    )

    print("Embedding model configured")

#     #   Database ko bhi configure
#     #  Initialize Pinecone Client

    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pineconeIndex = os.getenv("PINECONE_INDEX_NAME")
    print("Pinecone configured")

#     # langchain (chunking,embedding,database)

    print("Uploading to Pinecone...")
    await PineconeVectorStore.afrom_documents(
        documents=chunkedDocs,
        embedding=embeddings,
        index_name=pineconeIndex
    )

    print("Data Stored succesfully")

if __name__ == "__main__":
    asyncio.run(indexDocument())