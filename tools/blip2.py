from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

def load_model():
    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the BLIP2 processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Initialize the BLIP2 model
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    
    # Move the model to the specified device
    model.to(device)
    
    return processor, model, device


def generate_caption(urls, prompt, processor, model, device):
    """
    Fetches and processes each image in urls, generates captions based on the image, and prints the generated captions.
    
    Parameters:
    - urls: A list of URLs pointing to the images to be captionned.
    - prompt: A text prompt to be used along with the images for caption generation.
    - processor: The preprocessor for the BLIP2 model.
    - model: The BLIP2 model to be used for generating captions.
    """
    for url in urls:
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=40, min_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
        
urls = ["http://images.cocodataset.org/test-stuff2017/000000000311.jpg", "http://images.cocodataset.org/test-stuff2017/000000000300.jpg"]
prompt = ""

processor, model, device = load_model()
generate_caption(urls=urls, prompt=prompt, processor=processor, model=model, device=device)        
