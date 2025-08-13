import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import IMSDbLoader

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get("https://imsdb.com/all-scripts.html", headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

titles = []
for link in soup.select('p a[href^="/Movie Scripts/"]'):
    titles.append(link.get_text(strip=True))

print("Movies loaded:")
for i, t in enumerate(titles):
    print(f'{i + 1}. {t}')

# Step 3. User searches for a movie
# user_input = input("\nEnter movie title exactly as shown above: ").strip()
user_input = input("\n").strip()

if user_input in titles:
    # Step 4. Construct full script URL
    formatted_title = user_input.replace(" ", "-")
    script_url = f"https://imsdb.com/scripts/{formatted_title}.html"

    # Step 5. Load movie script using IMSDbLoader
    script_loader = IMSDbLoader(script_url)
    script_data = script_loader.load()

    # Step 6. Display results
    print(f"\nLoaded script for {user_input} from {script_url}.\n")
    print("Full movie script:\n")
    print(script_data[0].page_content)
else:
    print(f"Script for {user_input} wasn't found in the list of movie scripts.")
