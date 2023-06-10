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
|   └── 30_dog
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
|   └── 1_dog
|       |
|       ├── reg1.jpg
|       ├── reg2.jpg
```

Ensure that you maintain the same structure for the regularization images.

In the Kohya_ss GUI, follow these steps:

1. Enter the path to the images folder in the Image folder field.
2. Enter the path to the regularisation folder in the Regularisation folder field.

By adhering to this folder structure, you can ensure a smooth and effective training process with Kohya.