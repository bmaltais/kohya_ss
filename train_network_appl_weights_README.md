# Exploring Layer-Specific Application Rates for LoRA
## Introduction
Added a tool, train_network_appl_weights.py, for exploring layer-specific application rates. Currently, it supports SDXL only.

## Concept
The process involves running the standard training process with varying layer-specific application rates on trained networks like LoRA. The goal is to explore which rates produce images closest to the training data.

## Penalty for Total Application Rates
It's possible to use the total of the layer-specific application rates as a penalty, aiming to reproduce images while minimizing the impact of less significant layers.

## Multi-Network Exploration
The exploration can be conducted on multiple networks and requires at least one piece of training data.

Note: The effectiveness with a specific number of images has not been confirmed, but it has been tested with approximately 50 images. The training data does not necessarily have to be from LoRA's training phase, although this has not been confirmed.

## Command Line Options
The command line options are almost identical to those for `sdxl_train_network.py`, with the following additions and extensions:

- `--application_loss_weight`: Weight of the layer-specific application rate when added to the loss. Default is 0.0001. Increasing this value trains the model to minimize the application rates. Setting it to 0 allows free exploration of the application rates that yield the highest fidelity.
- `--network_module`: Allows specifying multiple modules for exploration, e.g., `--network_module networks.lora networks.lora`.
- `--network_weights`: Allows specifying weights for multiple networks to be explored, e.g., `--network_weights model1.safetensors model2.safetensors`.

## Parameters
The number of parameters for layer-specific application rates is 20, including BASE, IN00-08, MID, OUT00-08. BASE is applied to the Text Encoder (Note: LoRA's operation on the Text Encoder has not been confirmed).

Although the parameters are saved to a file, it's recommended to copy and save the values displayed on the screen.

## Remarks
Confirmed to work with the AdamW optimizer and a learning rate of 1e-1. The learning rate can be set quite high. With this setting, reasonable results can be obtained in about 1/20 to 1/10 the epochs used during LoRA training.
Increasing `application_loss_weight` above 0.0001 significantly reduces the total application rate, meaning LoRA is applied less. Adjust as needed.
Using negative values for the application rate can lead to minimizing the total by excessively reducing less influential layers' application rates. Negative values are weighted ten times (e.g., -0.01 is almost the same penalty as 0.1). Modify the source code to change the weighting.

## Potential Uses
Beyond reducing unnecessary layers' application rates, potential uses include:

- Searching for LoRA application rates to maintain a character while changing their pose based on a reference image.
- Exploring application rates for LoRA to maintain a character's style while altering the artistic style of the image.
- Exploring necessary layers to reproduce a character's attributes using an image in a different style as training data.
- Applying numerous LoRAs to an ideal image as training data and searching for the application rates that achieve the highest fidelity (though more LoRAs will slow down the training).