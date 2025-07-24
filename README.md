# Building the Naive RAG

This project implements a Retrieval-Augmented Generation (RAG) system for movie scripts. The system allows users to search for movie scripts, ask questions about them, and receive detailed answers based on the content of the scripts.

## Project Overview

The project is structured in stages, each building upon the previous one to create a complete RAG pipeline:

### Stage 1: Loading Data (stage1_load.py)

This stage handles the initial data loading process:
- Fetches a list of available movie scripts from the Internet Movie Script Database (IMSDb)
- Allows the user to select a movie from the list
- Loads the full script for the selected movie using IMSDbLoader
- Displays the loaded script

### Stage 2: Text Splitting (stage2_split.py)

This stage processes the loaded script by splitting it into manageable chunks:
- Loads the movie script (same as Stage 1)
- Cleans the script text using regex
- Splits the script into chunks using RecursiveCharacterTextSplitter
  - Uses a chunk size of 500 characters
  - Uses a chunk overlap of 10 characters
  - Uses "INT." as a separator to identify scene boundaries
- Displays information about the number of scenes found and their content

### Stage 3: Embedding (stage3_embed.py)

This stage converts the text chunks into vector representations:
- Loads and splits the movie script (same as Stages 1-2)
- Uses HuggingFaceEmbeddings with the "sentence-transformers/all-MiniLM-L6-v2" model
- Sets up a connection to a Qdrant vector database
- Creates a collection in Qdrant for the movie script
- Adds metadata to each document chunk (movie title, scene number, script URL, chunk length)
- Generates stable UUIDs for each document
- Stores the documents with their embeddings in the vector database

### Stage 4: Retrieval (stage4_retriever.py)

This stage implements the retrieval component of the RAG system:
- Loads, splits, and embeds movie scripts (same as Stages 1-3)
- Takes a user query about the movie
- Uses ChatGroq LLM with the "llama-3.3-70b-versatile" model to rewrite the query for better retrieval
- Initializes a retriever from the vector store
- Retrieves the top 5 most relevant scenes based on the rewritten query
- Displays the retrieved scenes

### Stage 5: Generation (stage5_generate.py)

This stage completes the RAG pipeline by generating answers to user queries:
- Includes all functionality from previous stages
- Creates a new prompt template for the generation step
- Uses the retrieved scenes as context for the LLM
- Generates a comprehensive answer to the user's query
- Displays the final response

## Requirements

The project requires the following dependencies:
- langchain-community: For document loading and processing
- langchain-core: Core components for the LangChain framework
- langchain-openai: OpenAI integration for LangChain
- langchain-groq: Groq integration for LangChain
- langchain-qdrant: Qdrant vector store integration for LangChain
- langchain-huggingface: Hugging Face integration for LangChain
- qdrant-client: Client for the Qdrant vector database
- beautifulsoup4: For web scraping
- requests: For HTTP requests

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements_project.txt
   ```

2. Set up a Qdrant vector database (locally or using Docker)

3. Create a `.env` file with your API keys if needed

## Usage

Run each stage file to see the progression of the RAG system:

1. Start with stage1_load.py to load a movie script
2. Move to stage2_split.py to see how the script is split into chunks
3. Continue with stage3_embed.py to embed the chunks into a vector database
4. Use stage4_retriever.py to retrieve relevant scenes based on a query
5. Finally, use stage5_generate.py to generate answers to questions about the movie

Each stage builds upon the previous ones, demonstrating the complete RAG pipeline from data loading to answer generation.