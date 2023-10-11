import torch
import re

class CaptionProcessor:
  def __init__(self, model, processor, device):
    self.model = model
    self.processor = processor
    self.device = device

  def gen(self, inputs, max_length=10, min_length=0, top_k=30, top_p=0.92, num_beams=4):
    return self.model.generate(
      **inputs,
      # max_new_tokens=25,      # Number of tokens to generate
      max_length=max_length,    # Maximum length of the sequence to be generated, mutually exclusive with max_new_tokens
      num_beams=num_beams,      # Number of beams to use for beam search
      num_return_sequences=1,   # Number of captions to generate
      early_stopping=True,      # Stop when no new tokens are generated
      repetition_penalty=1.5,   # Penalize repeated words
      no_repeat_ngram_size=2,   # Number of words that can be repeated
      # do_sample=True,         # Introduce randomness to captions
      # temperature=0.9,        # Measure of randomness 0-1, 0 means no randomness
      top_k=top_k,              # Number of highest probability tokens to keep, 0 means no filtering
      top_p=top_p,              # Probability threshold, 0 means no filtering
      min_length=min_length,    # Minimum length of the sequence to be generated
    )

  def process(self, prompt, image):
    return self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

  def caption_from(self, generated):
    caption_list = self.processor.batch_decode(generated, skip_special_tokens=True)
    caption_list = [caption.strip() for caption in caption_list]
    return caption_list if len(caption_list) > 1 else caption_list[0]

  def sanitise_caption(self, caption):
    return caption.replace(" - ", "-")

  # TODO this needs some more work
  def sanitise_prompt_shard(self, prompt):
    # Remove everything after "Answer:"
    prompt = prompt.split("Answer:")[0].strip()
    
    # Define a pattern for multiple replacements
    replacements = [
        (r", a point and shoot(?: camera)?", ""), # Matches ", a point and shoot" with optional " camera"
        (r"it is a ", ""),
        (r"it is ", ""),
        (r"hair hair", "hair"),
        (r"wearing nothing", "nude"),
        (r"She's ", ""),
        (r"She is ", "")
    ]
    
    # Apply the replacements using regex
    for pattern, replacement in replacements:
        prompt = re.sub(pattern, replacement, prompt)
    
    return prompt

  def ask(self, question, image):
    return self.sanitise_prompt_shard(self.caption_from(self.gen(self.process(f"Question: {question} Answer:", image))))

  def caption_me(self, initial_prompt, image):
    prompt = ""
    
    try:
      # [STYLE OF PHOTO] photo of a [SUBJECT], [IMPORTANT FEATURE], [MORE DETAILS], [POSE OR ACTION], [FRAMING], [SETTING/BACKGROUND], [LIGHTING], [CAMERA ANGLE], [CAMERA PROPERTIES],in style of [PHOTOGRAPHER]
      # print("\n")
      hair_color = self.ask("What is her hair color?", image)
      hair_length = self.ask("What is her hair length?", image)
      p_hair = f"{hair_color} {hair_length} hair"
      # print(p_hair)

      p_style = self.ask("Between the choices selfie, mirror selfie, candid, professional portrait what is the style of the photo?", image)
      # print(p_style)

      p_clothing = self.ask("What is she wearing if anything?", image)
      # print(p_clothing)

      p_action = self.ask("What is she doing? Could be something like standing, stretching, walking, squatting, etc", image)
      # print(p_action)

      p_framing = self.ask("Between the choices close up, upper body shot, full body shot what is the framing of the photo?", image)
      # print(p_framing)

      p_setting = self.ask("Where is she? Be descriptive and detailed", image)
      # print(p_setting)

      p_lighting = self.ask("What is the scene lighting like? For example: soft lighting, studio lighting, natural lighting", image)
      # print(p_lighting)

      p_angle = self.ask("What angle is the picture taken from? Be succinct, like: from the side, from below, from front", image)
      # print(p_angle)

      p_camera = self.ask("What kind of camera could this picture have been taken with? Be specific and guess a brand with specific camera type", image)
      # print(p_camera)

      # prompt = self.sanitise_caption(f"{p_style}, {initial_prompt} with {p_hair}, wearing {p_clothing}, {p_action}, {p_framing}, {p_setting}, {p_lighting}, {p_angle}, {p_camera}")
      prompt = self.sanitise_caption(f"{p_style}, with {p_hair}, wearing {p_clothing}, {p_action}, {p_framing}, {p_setting}, {p_lighting}, {p_angle}, {p_camera}")

      return prompt
    except Exception as e:
      print(e)

    return prompt