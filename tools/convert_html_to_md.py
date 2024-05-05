import argparse
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from html2text import html2text
from pathlib import Path

def is_writable_path(target_path):
    """
    Check if a path is writable.
    """
    path = Path(os.path.dirname(target_path))
    if path.is_dir():
        if os.access(path, os.W_OK):
            return target_path
        else:
            raise argparse.ArgumentTypeError(f"Directory '{path}' is not writable.")
    else:
        raise argparse.ArgumentTypeError(f"Directory '{path}' does not exist.")

def main(url, markdown_path):
    # Create a session object
    with requests.Session() as session:
        # Send HTTP request to the specified URL
        response = session.get(url)
        response.raise_for_status()  # Check for HTTP issues

        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ensure the directory for saving images exists
        os.makedirs("./logs", exist_ok=True)

        # Find all image tags and save images
        for image in soup.find_all('img'):
            image_url = urljoin(url, image['src'])
            try:
                image_response = session.get(image_url, stream=True)
                image_response.raise_for_status()
                image_name = os.path.join("./logs", os.path.basename(image_url))
                with open(image_name, 'wb') as file:
                    file.write(image_response.content)
            except requests.RequestException as e:
                print(f"Failed to download {image_url}: {e}")

        # Convert the HTML content to markdown
        markdown_content = html2text(response.text)

        # Save the markdown content to a file
        try:
            with open(markdown_path, "w", encoding="utf8") as file:
                file.write(markdown_content)
            print(f"Markdown content successfully written to {markdown_path}")
        except Exception as e:
            print(f"Failed to write markdown to {markdown_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HTML to Markdown")
    parser.add_argument("url", help="The URL of the webpage to convert")
    parser.add_argument("markdown_path", help="The path to save the converted markdown file", type=is_writable_path)
    args = parser.parse_args()

    main(args.url, args.markdown_path)
