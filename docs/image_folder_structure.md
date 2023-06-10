# Drambootd, Lora and TI image folder structure

To ensure successful training, Kohya utilizes a specific folder structure that provides the necessary repeat value for the images. It is essential to follow this structure precisely.

For instance, let's consider training two concepts, each with 30 repeats. We will refer to the first concept as cat and the second concept as dog. While you need at least one concept for training, you can include as many as desired.

The folder structure for these concepts should be as follows:

```txt
images
|
├── 30_cat
|   |
|   ├── image1.jpg
|   ├── image1.txt
|   ├── image2.png
|   └── image2.txt
|
└── 30_dog
    |
    ├── image1.jpg
    ├── image1.txt
    ├── image2.png
    └── image2.txt
```

Please note that the same folder structure is required for regularization images as well.

By adhering to this folder structure, you ensure that the images are organized appropriately and that the training process proceeds smoothly.