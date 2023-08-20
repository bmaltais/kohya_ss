import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from html2text import html2text

# Specify the URL of the webpage you want to scrape
url = "https://hoshikat-hatenablog-com.translate.goog/entry/2023/05/26/223229?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en-US&_x_tr_pto=wapp"

# Send HTTP request to the specified URL and save the response from server in a response object called r
r = requests.get(url)

# Create a BeautifulSoup object and specify the parser
soup = BeautifulSoup(r.text, 'html.parser')

# Find all image tags
images = soup.find_all('img')

for image in images:
    # Get the image source
    image_url = urljoin(url, image['src'])
    
    # Get the image response
    image_response = requests.get(image_url, stream=True)
    
    # Get the image name by splitting the url at / and taking the last string, and add it to the desired path
    image_name = os.path.join("./logs", image_url.split("/")[-1])
    
    # Open the image file in write binary mode
    with open(image_name, 'wb') as file:
        # Write the image data to the file
        file.write(image_response.content)

# Convert the HTML content to markdown
markdown_content = html2text(r.text)

# Save the markdown content to a file
with open("converted_markdown.md", "w", encoding="utf8") as file:
    file.write(markdown_content)
