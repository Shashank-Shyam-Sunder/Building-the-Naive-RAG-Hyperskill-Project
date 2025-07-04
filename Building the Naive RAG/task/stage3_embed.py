from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import IMSDbLoader
import requests
from bs4 import BeautifulSoup
import re
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import uuid

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# test_vector = embeddings.embed_query("Test text")
# print(len(test_vector))
# %%
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get("https://imsdb.com/all-scripts.html", headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

titles = []
for link in soup.select('p a[href^="/Movie Scripts/"]'):
    titles.append(link.get_text(strip=True))

# if titles:
#     print("Movies titles loaded successfully!")
# else:
#     print("No movies found.")
# for i, t in enumerate(titles):
#     print(f'{i + 1}. {t}')

# Step 3. User searches for a movie
# user_input = input("\nEnter movie title exactly as shown above: ").strip()
user_input = input("\n").strip()
# user_input = "Mission Impossible"
if user_input in titles:
    # Step 4. Construct full script URL
    formatted_title = user_input.replace(" ", "-")
    script_url = f"https://imsdb.com/scripts/{formatted_title}.html"

    # Step 5. Load movie script using IMSDbLoader
    script_loader = IMSDbLoader(script_url)
    script_data = script_loader.load()
    script_text = script_data[0].page_content
    script_text_clean = re.sub(r'\s+', ' ', script_text).strip()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10, separators=["INT.","EXT."])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10, separators=["INT."])
    script_chunks = text_splitter.create_documents([script_text_clean])
    print(f"Loaded script for {user_input} from {script_url}.\n")
    print(f"Found {len(script_chunks)} scenes in the script for {user_input}.")
    # for _ in range(len(script_chunks[:10])):
    # for _ in range(len(script_chunks)):
    #     chunk = script_chunks[_]
    #     print(f"Scene {_ + 1}: {chunk.page_content}")

    ### Creating a vector database for the movie scripts
    # Step 5: Embedding the scripts to Qdrant vector store by langchain_qdrant module
    # %%
    # Qdrant connection (local Docker)
    client = QdrantClient(host="localhost", port=6333)

    # Define collection name and vector settings
    COLLECTION_NAME = formatted_title
    VECTOR_SIZE = 384  # Adjust if your embedding size is different

    # Create a collection only if it doesn't already exist
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    # print(f"Collection '{COLLECTION_NAME}' is ready.")
    documents = []

    for i, chunk in enumerate(script_chunks):
        doc = Document(
            page_content=chunk.page_content,
            metadata={
                "movie_title": user_input,
                "scene_number": i + 1,
                "script_url": script_url,
                "chunk_length": len(chunk.page_content)  # Length of the chunk text
            }
        )
        documents.append(doc)

    # Generate stable UUIDs based on movie title + scene number
    uuids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_input}_{i + 1}")) for i in range(len(documents))]

    # Add documents with stable IDs to the vector store
    vector_store.add_documents(documents=documents, ids=uuids)
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    point_count = collection_info.points_count

    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        point_count = collection_info.points_count

        if point_count > 0:
            # print(f"✅Embedded script for '{user_input}' with {point_count} scene chunks in collection '{COLLECTION_NAME}'.")
            print(f"Embedded script for {user_input}.")
        else:
            print(f"Collection '{COLLECTION_NAME}' exists but has no embedded vectors. Possible reasons:")
            print(f"- No script chunks were generated for {user_input}")
            print(f"- Embedding insertion failed in vector_store.add_documents()")
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' does not exist or could not be retrieved. Error: {e}")


else:
    print(f"Script for {user_input} wasn't found in the list of movie scripts.")
# %%