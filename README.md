Description This project builds a Retrieval Augmented Generation (RAG) system, allowing you to "chat" with your own private PDF documents. Unlike standard chatbots that only know what they were trained on, this system can read a specific file (like a textbook or HR policy), understand it, and answer questions based strictly on that content. It consists of two parts: the Indexer (which loads and memorizes the PDF) and the Query Engine (which searches the memory and generates answers).

Key Concept

RAG (Retrieval Augmented Generation): The core technique used here. Instead of sending the entire PDF to the AI (which would be too large and expensive), we only "retrieve" the specific paragraph relevant to the user's question and "augment" the AI's prompt with that paragraph.

Vector Embeddings: To make the PDF searchable by meaning (not just keywords), we use GoogleGenerativeAIEmbeddings to convert text into lists of numbers (Vectors). "King" and "Queen" will have similar numbers, allowing the AI to understand concepts.

Pinecone Vector Database: A specialized database designed to store these vectors. The indexer.py script uploads your PDF's "mathematical meaning" here, and query.py searches this database to find the right answers instantly.
