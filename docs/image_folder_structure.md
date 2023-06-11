# Drambootd, Lora and TI image folder structure

To ensure successful training with Kohya, it is crucial to follow a specific folder structure that provides the necessary image repeats. Please adhere to the following structure precisely:

Folder Structure Example:

```txt
c:
|
├──images
|   |
|   ├── 30_cat
|   |   |
|   |   ├── image1.jpg
|   |   ├── image1.txt
|   |   ├── image2.png
|   |   └── image2.txt
|   |
|   ├── 30_dog
|   |   |
|   |   ├── image1.jpg
|   |   ├── image1.txt
|   |   ├── image2.png
|   |   └── image2.txt
|   |
|   └── 40_black mamba
|       |
|       ├── image1.jpg
|       ├── image1.txt
|       ├── image2.png
|       └── image2.txt
|
├──regularization
|   |
|   ├── 1_cat
|   |   |
|   |   ├── reg1.jpg
|   |   ├── reg2.jpg
|   |
|   ├── 1_dog
|   |   |
|   |   ├── reg1.jpg
|   |   ├── reg2.jpg
|   |
|   └── 1_black mamba
|       |
|       ├── reg1.jpg
|       ├── reg2.jpg

```

Please note the following important information regarding file extensions and their impact on concept names during model training:

If a file with a .txt or .caption extension and the same name as an image is present in the image subfolder, it will take precedence over the concept name during the model training process.
For example, if there is an image file named image1.jpg in the 30_cat subfolder, and there is a corresponding text file named image1.txt or image1.caption in the same subfolder, the concept name used during training will be determined by the content of that text file rather than the subfolder name.

Ensure that the content of such text files accurately reflects the desired concept name or any relevant caption information associated with the corresponding image.

By considering this information and maintaining the proper folder structure, including any necessary text or caption files, you can ensure a smooth and effective training process with Kohya.