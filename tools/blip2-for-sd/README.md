# blip2-for-sd

source: https://github.com/Talmendo/blip2-for-sd

Simple script to make BLIP2 output image description in a format suitable for Stable Diffusion.

Format followed is roughly
`[STYLE OF PHOTO] photo of a [SUBJECT], [IMPORTANT FEATURE], [MORE DETAILS], [POSE OR ACTION], [FRAMING], [SETTING/BACKGROUND], [LIGHTING], [CAMERA ANGLE], [CAMERA PROPERTIES],in style of [PHOTOGRAPHER]`

## Usage
- Install dependencies according to requirements.txt

- run main.py
`python main.py`

The default model will be loaded automatically from huggingface.
You will be presented with an input to specify the folder to process after the model is loaded.

<img width="854" alt="Screenshot 2023-08-04 102650" src="https://github.com/Talmendo/blip2-for-sd/assets/141401796/fa40cae5-90a4-4dd5-be1d-fc0e8312251a">


- The image or source folder should have the following structure:

![Screenshot 2023-08-04 102544](https://github.com/Talmendo/blip2-for-sd/assets/141401796/eea9c2b0-e96a-40e4-8a6d-32dd7aa3e802)


Each folder represents a base prompt to be used for every image inside.

- You can adjust BLIP2 settings in `caption_processor.py` inbetween runs, without having to stop the script. Just update it before inputting the new source folder.

## Models
Default model is `Salesforce/blip2-opt-2.7b`, works quite well and doesn't require much VRAM.
Also tested with `Salesforce/blip2-opt-6.7b-coco` which seems to gives better results at the cost of much more VRAM and a large download (~30GB).