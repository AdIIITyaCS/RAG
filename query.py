import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from google import genai
from google.genai import types

load_dotenv()

ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
History = []

async def transformQuery(question):

    # 1. Add current question to history temporarily
    History.append({
        'role': 'user',
        'parts': [{'text': question}]
    })  

    # 2. Call Gemini to rephrase the question
    response = ai.models.generate_content(
        model="gemini-2.0-flash",
        contents=History,
        config=types.GenerateContentConfig(
            system_instruction="""You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
            Only output the rewritten question and nothing else.
            """,
        )
    )
    
    # 3. Remove the raw question from history
    History.pop()
    
    
    # 4. Return the rephrased text
    return response.text


async def chatting(question):
    # --- FIX 1: Await the async function ---
    queries = await transformQuery(question)
    
    # # covert this question into vector
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model='models/text-embedding-004',
        transport="rest"
    )
 
    # Now 'queries' is a real string, so this will work
    queryVector = embeddings.embed_query(queries)
    # # query vector

    # # make connection with pinecone
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pineconeIndex = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

    searchResults = pineconeIndex.query(
        top_k=10,
        vector=queryVector,
        include_metadata=True,
    )

    #   top 10 documents: 10 metadata text part 10 document
    texts = []
    for match in searchResults["matches"]:
        text = match["metadata"]["text"]   # sirf text nikaala
        texts.append(text)                 # list me daal diya
 
    context = "\n\n---\n\n".join(texts)    # beech me separator laga kar join kiya

    # create the context for the LLM

    # # Gemini
    History.append({
        'role':'user',
        'parts':[{'text':queries}]
    })  

    response = ai.models.generate_content(
        model="gemini-1.5-flash",
        contents=History,
        config=types.GenerateContentConfig(
            system_instruction=f"""You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document."
    Keep your answers clear, concise, and educational.
      
      Context: {context}
      """,
        ),
    )

    History.append({
        'role':'model',
        'parts':[{'text':response.text}]
    })

    print("\n")
    print(response.text)
    print('\n')

def main():
    while True:
        userProblem = input("Ask me anything (or type 'quit')--> ")
        if userProblem.lower() in ["quit", "exit"]:
            break
        
        # --- FIX 2: Use asyncio.run to execute the async chatting function ---
        asyncio.run(chatting(userProblem))

if __name__ == "__main__":
    main()