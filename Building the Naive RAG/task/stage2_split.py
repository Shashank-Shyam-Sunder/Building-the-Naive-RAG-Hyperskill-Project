
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import IMSDbLoader
import requests
from bs4 import BeautifulSoup
import re

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
    for _ in range(len(script_chunks)):
        chunk = script_chunks[_]
        print(f"Scene {_ + 1}: {chunk.page_content}")
else:
    print(f"Script for {user_input} wasn't found in the list of movie scripts.")
# %%