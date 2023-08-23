import requests, torch, sys, os
import argparse

from importlib import reload
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm

import caption_processor

model = None
processor = None
device = None

def load_model(model_name="Salesforce/blip2-opt-2.7b"):
  global model, processor, device

  print("Loading Model")
  processor = AutoProcessor.from_pretrained(model_name)
  model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

  if torch.cuda.is_available():
    print("CUDA available, using GPU")
    device = "cuda"
  else:
    print("CUDA not available, using CPU")
    device = "cpu"

  print("Moving model to device")
  model.to(device)

def main(path):
  # reloading caption_processor to enable us to change its values in between executions
  # without having to reload the model, which can take very long
  # probably cleaner to do this with a config file and just reload that
  # but this works for now
  reload(caption_processor)
  prompt_file_dict = {}

  # list all sub dirs in path
  sub_dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

  print("Reading prompts from sub dirs and finding image files")
  for prompt in sub_dirs:
    prompt_file_dict[prompt] = [file for file in os.listdir(os.path.join(path, prompt)) if file.endswith((".jpg", ".png", ".jpeg", ".webp"))]

  for prompt, file_list in prompt_file_dict.items():
    print(f"Found {str(len(file_list))} files for prompt \"{prompt}\"")

  for prompt, file_list in prompt_file_dict.items():
    total = len(file_list)

    for file in tqdm(file_list):
      # read image
      image = Image.open(os.path.join(path, prompt, file))

      caption = ""
      # generate caption
      try:
        caption = caption_processor.CaptionProcessor(model, processor, device).caption_me(prompt, image)
      except:
        print("Error creating caption for file: " + file)

      # save caption to file
      # file without extension
      with open(os.path.join(path, prompt, os.path.splitext(file)[0] + ".txt"), "w", encoding="utf-8") as f:
        f.write(caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter the path to the file")
    parser.add_argument("path", type=str, nargs='?', default="", help="Path to the file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()
    interactive = args.interactive

    load_model(model_name="Salesforce/blip2-opt-2.7b")

    if interactive:
        while True:
            path = input("Enter path: ")
            main(path)
            continue_prompt = input("Continue? (y/n): ")
            if continue_prompt.lower() != 'y':
                break
    else:
        path = args.path
        search_subdirectories = False
        main(path)
