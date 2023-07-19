# Explaining LoRA Learning Settings Using Kohya_ss for Stable Diffusion Understanding by Anyone

To understand the meaning of each setting in kohya_ss, it is necessary to know how LoRA performs additional learning.

We will also explain what the "model," which is the target of additional learning, is.

## What is a "model"

Stable Diffusion loads and uses modules called " models ". A model is, so to speak, a "brain", and its true identity is " weight information of a neural network ".

A neural network is made up of many " neurons ", and the clusters of neurons form many layers of " layers ". Neurons in one layer are connected to neurons in another layer by lines, and the strength of the connection is " weight ". It is this "weight" that holds a huge amount of picture information.

### LoRA adds a small neural net

LoRA is a kind of "additional learning", but additional learning is to upgrade the neural network.

An additional learning method called "DreamBooth" uses this method.

With this method, if you want to publish the additional training data, you need to distribute the whole model that has been updated with additional training.

Models are typically 2G to 5G bytes in size, making them difficult to distribute.

In contrast, LoRA learning leaves the model alone and creates a new “small neural net ” for each position you want to learn. Additional training is done on this small neural net .

When you want to distribute LoRA, you only need to distribute this small neural network , so the data size is small.

### Structure of a small neural net

LoRA's small neural net consists of three layers. The number of neurons in the "input layer" on the left and the "output layer" on the right is the same as the number of neurons in the "input layer" and "output layer" of the target neural network . The number of neurons in the middle layer (middle layer) is called the "rank number" (or the number of dimensions), and this number can be freely determined when learning.

### LoRA Learning Target 1: U-Net

U-Net is divided into "Down" (left half), "Mid" (bottom) and "Up" (right half).

And it consists of 25 blocks in total: Down12 block, Mid1 block, and Up12 block. The neural net added here is simply called "UNet" in Kohya_ss.

### RoLA Learning Object 2: Text Encoder

This isn't the only time LoRA adds neural nets .

The block called "Cross Attention" in the figure above receives text information from a module called "Text Encoder ". This "text encoder " has the role of converting the prompt, which is text data, into a string of numbers (vector).

There is only one text encoder , which is shared by all Attention Blocks in U-Net. This text encoder is originally treated as a "finished product" within Stable Diffusion and is not subject to model learning, but it is also subject to additional learning by LoRA.

The LoRA updated text encoder is used in all Attention blocks, so any neural nets added here will have a huge impact on the final image.

The neural network added here is called "Text Encoder" in Kohya_ss.

## Basic training parameters
### LoRA type

Specifies the type of LoRA learning. The LoRA explained above is the "standard" type. "DyLoRA" learns multiple ranks below the specified rank at the same time, so it is convenient when you want to select the optimum rank. LoHa is highly efficient LoRA, and LoCon extends learning to U-Net's Res block.

There is no problem with the Standard type at first. If you are having trouble learning, try another type.

### LoRA network weights

If you want to use the already learned LoRA file for additional learning, specify the LoRA file here.

The LoRA specified here will be read at the start of learning, and learning will start from this LoRA state. LoRA after learning is saved as another file, so the LoRA file specified here will not be overwritten.

### DIM from weights

This is an option only when doing additional training with LoRA network weights.

As shown in the figure above, LoRA adds a small neural network , but the number of neurons (number of ranks) in the middle layer can be freely set with Network Rank (described later).

However, turning this option on will set the number of ranks of the created LoRA to the same number of ranks as the LoRA specified in LoRA network weights. When this is turned on, the specification of Network Rank is ignored.

For example, when the number of LoRA ranks used for additional learning is 32, the number of LoRA ranks to be created will also be set to 32.

Default is off.

### Train batch size

Specify a batch size. A batch is "the number of images to read at once". A batch size of 2 will train two images at a time simultaneously. If multiple different pictures are learned at the same time, the tuning accuracy for each picture will drop, but since it will be learning that comprehensively captures the characteristics of multiple pictures, the final result may instead be better.

(If you tune too much to a specific picture, it will become LoRA that is not applicable.)

Since multiple pictures are learned at once, the higher the batch size, the shorter the learning time. However, the tuning accuracy decreases and the number of weight changes decreases, so there is a possibility that the learning may be insufficient in some cases.

(There is also a report that when increasing the batch size, it is better to increase the learning rate (described later). For example, if the batch size is 2, the learning rate should be doubled.)

Also, the higher the batch size, the more memory is consumed. Let's decide according to the size of VRAM of your PC.

With 6GB of VRAM, a batch size of 2 would be barely possible.

Default is 1.

*Since all the images read at the same time for each batch must be the same size, if the sizes of the training images are different, the number of images that are processed simultaneously may be less than the number of batches specified here.

### Epoch

One epoch is "one set of learning".

For example, let's say you want to learn by reading 50 images each 10 times. In this case, 1 epoch is 50x10 = 500 trainings. If it is 2 epochs, this will be repeated twice, so it will be 500x2 = 1000 times of learning.

After training for the specified number of epochs, a LoRA file will be created and saved to the specified location.

For LoRA, 2-3 epochs of learning is sufficient.

### Save every N epochs

You can save the progress as a LoRA file for each epoch number specified here.

For example, if you specify 10 in "Epoch" and specify 2 in "Save every N epochs", the LoRA file will be saved in the specified folder every 2 epochs (at the end of 2, 4, 6, 8 epochs).

If you don't need to create an intermediate LoRA, set the value here to the same value as "Epoch".

### Caption Extension

If you have prepared a caption file for each image, specify the extension of the caption file here.

If this is blank, the extension will be ".caption". If the extension of the caption file is ".txt", specify ".txt" here.

If you don't have a caption file, you can ignore it.

### Mixed precision

Specifies the type of mixed precision for the weight data during training.

The weight data is originally in 32-bit units (when no is selected), but if necessary, learning by mixing 16-bit unit data will lead to considerable memory savings and speedup. fp16 is a data format with half the precision , and bf16 is a data format devised to handle the same numerical width as 32-bit data .

You can get LoRA with a sufficiently high accuracy at fp16.

### Save precision


Specifies the type of weight data to save in the LoRA file.

float is 32-bit, fp16 and bf16 are 16-bit units. The two below have smaller file sizes.

The default is fp16.

### Number of CPU threads per core

The number of threads per CPU core during training. Basically, the higher the number, the higher the efficiency, but it is necessary to adjust the settings according to the specifications.

Default is 2.

### Seeds
During learning, there are a number of random processes such as ``in what order to read the images'' and ``how much noise to put on the training images (details omitted)''.

Seed is like an ID for determining the random processing procedure, and if the same Seed is specified, the same random procedure will be used each time, making it easier to reproduce the learning results.

However, there are random processes that do not use this seed (such as randomly cropping images), so specifying the same seed does not always give the same learning results.

Default is blank. If not specified, Seed will be set appropriately when training is executed.

If you want to reproduce the result as much as possible, there is no loss by setting a number (such as 1234) appropriately.

### Cache latents

The training image is read into VRAM, "compressed" to a state called Latent before entering U-Net, and is trained in VRAM in this state. Normally, images are "compressed" each time they are loaded, but you can specify that "compressed" images are kept in main memory by checking Cache latents.

Keeping it in the main memory saves VRAM space and speeds up, but you can't process the image before "compression", so you can't use augmentation (described later) other than flip_aug. Also, random crop (described later), which crops the image in a random range each time, cannot be used.

Default is on.

### Cache latents to disk

Similar to the Cache latents option, but checking this allows you to specify that compressed image data be saved to disk as temporary files.

This temporary file can be reused even after restarting kohya_ss, so if you want to do LoRA learning with the same data many times, turning on this option will increase learning efficiency.

However, if you turn this on, you will not be able to use augmentation and random crop other than flip_aug.

Default is off.

### Learning rate:

Specify the learning rate. " Learning" is to change the thickness (weight) of the wiring in the neural network so that a picture that looks exactly like the given picture can be made, but every time a picture is given, the wiring is changed. If you tune too much only to the given picture, you will not be able to draw other pictures at all.

To avoid this, we change the weights slightly each time to incorporate a little bit more of the given picture. The "learning rate" determines the amount of this "just a little".

The default value is 0.0001.

### LR Scheduler:

You can change the learning rate in the middle of learning. A scheduler is a setting for how to change the learning rate.

adafactor: Select this to set the optimizer (described later) to Adafactor . Learn while automatically adjusting the learning rate according to the situation to save VRAM
constant: the learning rate does not change from beginning to end
constant_with_warmup: Start with a learning rate of 0 and gradually increase it toward the set value of Learning rate during warm-up, and use the set value of Learning rate during main learning.
cosine : Gradually decrease the learning rate toward 0 while drawing a wave (cosine curve)
cosine _with_restarts: repeat cosine many times (see also description of LR number of cycles)
linear: Start at the Learning rate setting and decrease linearly towards 0
polynomial: Same behavior as linear, but a bit more complicated to reduce (see also LR power description)
Set to constant if you want the learning rate to be fixed at the Learning rate setting.

Default is cosine

### LR warmup

If you have selected constant_with_warmup in the scheduler, set here how many times to warm up.

The number specified here is a percentage of the total number of steps.

For example, if you train 50 images 10 times with a batch size of 1 and do this for 2 epochs, the total number of steps is 50x10x2=1000. If you set LR warmup to 10, the first 10% of the 1000 total steps, or 100 steps, will be the warmup.

You can ignore this if your scheduler is not constant_with_warmup.

Default is 10.

### Optimizer

The optimizer is a setting for "how to update the neural net weights during training ". Various methods have been proposed for smart learning, but the most commonly used in LoRA learning is ``AdamW'' (32-bit) or ``AdamW8bit''. AdamW8bit uses less VRAM and has enough accuracy, so if you get lost, use this.

In addition, "Adafactor", which adjusts the learning rate appropriately according to the progress of learning while incorporating Adam's method, is also often used (Learning rate setting is ignored when using Adafactor).

``DAdapt'' is an optimizer that adjusts the learning rate , and ``Lion'' is a relatively new optimizer , but it has not been fully verified yet. There is a report that "SGDNesterov" has good learning accuracy but slows down.

The default is AdamW8bit. There is no problem basically as it is.

### Optimizer extra arguments

If you want more granularity for a given optimizer , write the command here.

You can usually leave this field blank.

### Text Encoder learning rate
Sets the learning rate for the text encoder . As I wrote earlier, the effect of additional training on text encoders affects the entire U-Net.

Therefore, it is usually set lower than the learning rate (Unet learning rate) for each block of U-Net.

The default value is 0.00005(5e-5).

If you specify a number here, it takes precedence over the Learning rate value.

### Unet learning rate

Sets the learning rate for U-Net. This is the learning rate when performing additional learning on each attention block (and other blocks depending on the setting) in U-Net.

The default value is 0.0001.

If you specify a number here, it takes precedence over the Learning rate value.

### Network Rank (Dimension)

Specifies the number of neurons in the hidden layer of the "additional small neural net " described earlier in the article (see the figure above for details).

The larger the number of neurons , the more learning information can be stored, but the possibility of learning unnecessary information other than the learning target increases, and the LoRA file size also increases.

Generally, it is often set to a maximum of about 128, but there are reports that 32 is sufficient.

When making LoRA on a trial basis, it may be better to start from around 2 to 8.

Default is 8.

### Network alpha

This was introduced as a convenience measure to prevent weights from being rounded to 0 when saving LoRA.

Due to the structure of LoRA, the weight value of the neural network tends to be small, and if it becomes too small, it may become indistinguishable from zero (that is, the same as not learning anything). Therefore, a technique was proposed in which the actual (stored) weight value is kept large, but the weight is always weakened at a constant rate during learning to make the weight value appear smaller. Network alpha determines this "weight weakening rate".

The smaller the Network alpha value, the larger the stored LoRA neural net weights.

How much the weight weakens when used (usage strength) is calculated by "Network_Alpha/Network_Rank" (roughly a value between 0 and 1) and is closely related to the Network Rank number.

If the accuracy of LoRA after learning is not good enough, the weight data may be too small and collapsed to 0. In such a case, try lowering the Network Alpha value (=increasing the save weight value).

The default is 1 (that is, maximize the stored weight value).

If Network Alpha and Network Rank have the same value, the effect will be turned off.

*Network Alpha value must not exceed Network Rank value. It is possible to specify a higher number, but there is a high probability that it will result in an unintended LoRA.

Also, when setting the Network Alpha, you should consider the effect on the learning rate.

For example, with an Alpha of 16 and a Rank of 32, the strength of the weight used is 16/32 = 0.5, meaning that the learning rate is only half as powerful as the Learning Rate setting.

If Alpha and Rank are the same number, the strength used will be 1 and will have no effect on the learning rate.

### Max resolution

Specify the maximum resolution of training images in the order of "width, height". If the training images exceed the resolution specified here, they will be scaled down to this resolution.

The default is "512,512". Many models use images of this size, so it is safe to use images of this size when learning LoRA.

### Stop text encoder training

You can stop learning the text encoder in the middle. As I wrote above, updating the text encoder has a big impact on the whole, so it is easy to fall into overfitting (tuning too much to the training image and other images can not be drawn), and it is also overfitting to stop learning at a moderate point is one way to prevent

The number specified here is a percentage of the total training step. Once learning reaches this percentage, the text encoder stops learning.

For example, if the total number of steps is 1000 and you specify 80 here, the text encoder will finish training when the learning progress is 80%, i.e. 1000x0.8=800 steps.

Training of U-Net continues with 200 remaining steps.

If this is 0, the text encoder training will not stop until the end.

### Enable buckets

" bucket " is a "bucket" (container) as the name suggests. The training images used in LoRA do not have to be of the same size, but images of different sizes cannot be trained at the same time. Therefore, it is necessary to sort the images into "buckets" according to their size before training. Put similar sized images in the same bucket and different sized images in different buckets.

Default is on.

If your training images are all the same size, you can turn this option off, but leaving it on has no effect.

*If you turn off Enable buckets when the size of the training images is not unified, the training images will be enlarged or reduced to have the same size.

Enlargement and reduction are performed while maintaining the aspect ratio of the image. If the aspect ratio is not the same as the standard size, the vertical or horizontal size of the image after scaling may exceed the standard size. For example, if the base size is 512x512 ( 1 aspect ratio ) and the image size is 1536x1024 ( 1.5 aspect ratio ), the image will be scaled down to 768x512 ( 1.5 aspect ratio remains).

## Advanced Configuration
After this are the options in the Advanced Configuration section.

### Weights, Blocks, Conv

These are the "learning weight" and "rank" settings for each block in U-Net. Selecting each tab will bring up the corresponding configuration screen.

*These settings are for advanced users. If you have no preference, you can leave all fields blank.


#### Weights: Down LR weights/Mid LR weights/Up LR weights
As you can see from the U-Net structure diagram, U-Net consists of 12 IN blocks, 1 MID block, and 12 OUT blocks, a total of 25 blocks.

If you want different learning rate weights for each block, you can set them here individually.

The weight here is the "strength of learning" represented by a numerical value of 0 to 1. If it is 0, it is "not learning at all", and if it is 1, it is "learning at the learning rate set in Learning rate". can vary the intensity of learning.

A weight of 0.5 means half the learning rate.

"Down LR weights" specify the weights for each of the 12 IN blocks.

"Mid LR weights" specifies the weights of the MID block.

"Up LR weights" specify the weight of each of the 12 OUT blocks.

 

#### Weights: Blocks LR zero threshold
I explained that "LoRA adds neural nets ", but it doesn't make sense to add neural nets with too small weights (i.e. barely learned). Therefore, you can set "Do not add neural nets to blocks with too small weights ".

Blocks that do not exceed the weight value set here will not be added to the neural net . For example, if you specify 0.1 here, the neural net will not be added to blocks with weights less than or equal to 0.1 (note that exclusions also include the specified value!).

The default is blank, which is 0 (do nothing).

 

#### Blocks: Block dims, Block alphas
Here you can set different rank (dim) and alpha values ​​for each of the 25 blocks IN0~11, MID, OUT0~11.

See Network Rank, Network alpha for rank and alpha values.

Blocks with higher rank are expected to hold more information.

You must always specify 25 numbers for this parameter value, but since LoRA targets attention blocks, IN0, IN3, IN6, IN9, IN10, IN11, IN11, OUT0, and IN1 do not have attention blocks. , IN2 settings (1st, 4th, 7th, 11th, 12th, 14th, 15th, 16th digits) are ignored during learning.

*This is a setting for advanced users. If you don't care, you can leave it blank. If not specified here, "Network Rank(Dimension)" value and "Network Alpha" value will be applied to all blocks.

 

#### Conv: Conv dims, Conv, alphas
The attention block that LoRA learns from has a neural network called "Conv ", which is also updated by additional learning (see the diagram of the attention layer structure at the top of the article). This is a process called "convolution", and the size of the "filter" used there is 1x1 square.

Read this article about convolutions .

On the other hand, some of the blocks other than Attention (Res, Down blocks) and some of the Attention blocks in OUT are convoluted using a 3x3 square filter. Originally, that is not the learning target of LoRA, but by specifying it with this parameter, the 3x3 convolution of the Res block can also be the learning target.

Since there are more learning targets, there is a possibility that more precise LoRA learning can be performed.

The setting method is the same as "Blocks: Blocks dims, Blocks alphas".

A 3x3 conv exists on all 25 layers.

*This is a setting for advanced users. If you don't care, you can leave it blank.

### No token padding
Captions attached to training images are processed every 75 tokens tokens " can basically be regarded as "words").

If the caption length is less than 75 tokens align to 75 tokens This is called "padding".

Here you can specify not to pad tokens

Default is off. You can basically leave it off.

 

### Gradient accumulation steps
Changing the weights (that is, "learning") is usually done for each batch read, but it is also possible to do multiple batches of training at once. This option specifies how many batches to learn at once.

This has a similar effect (not the "same effect"!) as increasing the number of batches.

For example, if the batch size is 4, the number of images read simultaneously in one batch is 4. In other words, one learning is performed every four readings. If we set the Gradient accumulation steps to 2, training will be performed once every 2 batches, resulting in 1 learning per 8 reads. This works similarly (but not the same!) as batch number 8.

If you increase this value, the number of times of learning will decrease, so the processing will be faster, but it will consume more memory.

Default is 1.

 

### Weighted captions
Currently, the most popular Stable Diffusion usage environment is "Stable Diffusion WebUI", which has a unique prompt description method. For example, if you want to emphasize "Black" very strongly when specifying " black cat " at the prompt, put the word you want to emphasize in parentheses like "(black:1.2) cat" and put ": number" after the word , Words are emphasized by multiples of that number.

This option allows this notation to be used in the training image captions as well.

If you want to write complex captions, it's a good idea to give it a try.

Default is off.

 

### Prior loss weight
The prior loss weight determines how much importance is given to the " regularization images" (see the description of the Regularization folder above for details) during training .

If this value is low, the regularization images are considered less important, and LoRA is generated that is more characteristic of the training images.

This setting has no meaning if you are not using a regularized image.

This is a value between 0 and 1, and defaults to 1 ( also respects regularized images).

### LR number of cycles
If you select " Cosine with restart" or "Polynomial" for the scheduler, this option specifies how many cycles the scheduler runs during training.

If the number of this option is 2 or greater, the scheduler will run multiple times during a single training run.

In both Cosine with restart and Polynomial, the learning rate gradually decreases to 0 as learning progresses, but if the number of cycles is 2 or more, the learning rate is reset and restarted when the learning rate reaches 0.

The figure below (source) is an example of the change in learning rate for Cosine with restart (purple) and Polynomial (light green).

The purple example has the number of cycles set to 4. The light green example has a cycle number of 1.

Since the specified number of cycles is executed within the determined learning step, the more the number of cycles increases, the more the learning rate changes.

Default is blank, leaving blank equals 1.



Example of learning rate movement
Cosine with restart "LR number of cycle = 4" (purple)
Polynomial "LR power = 2" (light green)
 

### LR power
This is an option when the scheduler is set to Polynomial. The higher this number, the steeper the initial learning rate drops. (The slope of the light green line in the image above becomes steeper).

When power is 1, it has the same shape as the linear scheduler.

If the number is too large, the learning rate will stick close to 0, resulting in insufficient learning, so be careful.

Defaults to blank, leaving blank equals 1 (that is, the same as the linear scheduler).

 

### Additional parameters
If you want to tweak learning setting parameters that are not displayed in the kohya_ss GUI , enter them here as commands.

You can usually leave this field blank.

 

### Save every N steps
A LoRA file is created and saved each time the number of steps specified here is completed.

For example, when the total number of learning steps is 1000, if you specify 200 here, LoRA files will be saved at the end of 200, 400, 600, and 800 steps.

See also "Save every N epochs" for saving intermediate LoRA.

Default is 0 (do not save intermediate LoRA).

 

### Save last N steps
This is an option when Save every N steps is specified to save LoRA during learning.

If you want to keep only recent LoRA files and discard old LoRA files, you can set "how many recent steps of LoRA files to keep" here.

For example, if the total number of training steps is 600 and the Save every N steps option is specified to save every 100 steps. Then LoRA files will be saved at the 100th, 200th, 300th, 400th, and 500th steps, but if Save every N steps is set to 300, only the last 300 steps of LoRA files will be saved. In other words, at the 500th step, LoRA older than the 200th (=500-300) step (that is, LoRA at the 100th step) is deleted.

Default is 0.

 

### Keep n tokens
If your training images have captions, you can randomly shuffle the comma-separated words in the captions (see Shuffle caption option for details). However, if you have words that you want to keep at the beginning, you can use this option to specify "Keep the first 0 words at the beginning".

The number of first words specified here will always be fixed at the beginning.

Default is 0. This option does nothing if the shuffle caption option is off.

* A "word" here is a piece of text separated by commas. No matter how many words the delimited text contains, it counts as "one word".

In the case of " black cat , eating, sitting", " black cat " is one word.

### Clip skip
The text encoder uses a mechanism called "CLIP", which is made up of 12 similar layers.

Texts ( tokens ) are originally converted to numeric sequences (vectors) through these 12 layers, and the vectors coming out of the last layer are sent to the U-Net Attention block.

However, the model developed independently by the service "Novel AI", commonly known as "Novel AI model", adopted a unique specification that uses the vector output by the second to last layer instead of the last layer. The same is true for models derived from Novel AI models. Therefore, it is necessary to specify "Which layer of CLIP is the vector from which the base model used for learning is used?"

"Clip skip" specifies the layer number of this "Xth from the end".

Setting this to 2 sends the penultimate layer's output vector to the Attention block. If 1, the output vector of the last layer is used.

If the base model is a Novel AI model (or a mix of them), 2 should be fine. In other cases, 1 is fine.

 

### Max Token Length


Specifies the length of the maximum token included in the caption .

The "tokens" here are not the number of words, but the number of tokens Note that commas also count as one token.

It's unlikely that you'll use more than 75 tokens in your caption, but if you find your caption to be too long, specify a higher number here.

 

### Full fp16 training (experimental)
When the option "Mixed precision" described above is turned on (fp16 or bf16), a mixture of 32-bit and 16-bit data is used during training, but when this option is turned on, all weight data is 16-bit (fp16 format). Although it saves memory, the accuracy of some data is halved, so there is a possibility that the learning accuracy will also drop.

Default is off. You should leave it off unless you really want to save memory.

 

### Gradient checkpointing
Normally, during training, we modify and update the weights of a large number of neural nets all at once each time an image is loaded. By fixing this "gradually" rather than "all at once," you can save memory by reducing computation.

This option specifies that the weight calculation should be done incrementally. Turning this on or off will have no effect on LoRA's learning results.

Default is off.

 

### Shuffle caption
If the training images have captions, most of the captions are written in the form of words separated by commas, such as " black cat , eating, sitting". The Shuffle caption option randomly changes the order of these comma-separated words each time.

Words in captions are generally given more weight the closer they are to the beginning. Therefore, if the word order is fixed, backward words may not be learned well, and forward words may have unintended associations with training images. It is hoped that this bias can be corrected by reordering the words each time the image is loaded.

This option has no meaning if the caption is written in sentences instead of comma separated.

Default is off.

* A "word" here is a piece of text separated by commas. No matter how many words the delimited text contains, it counts as "one word".

In the case of " black cat , eating, sitting", " black cat " is one word.

 

### Persistent data loaders
The data required for training is discarded and reloaded after each epoch. This is an option to keep it instead of throwing it away. Turning this option on speeds up the start of training for new epochs, but uses more memory to hold the data.

Default is off.

 

### Memory efficient attention
If this is checked, VRAM usage is suppressed and attention block processing is performed. It's slower than the next option "xformers". Turn it on if you don't have enough VRAM.

Default is off.

### Use xformers
Using a Python library called "xformers" will trade attention blocking for less VRAM usage at the cost of some speed. Turn it on if you don't have enough VRAM.

Default is on.

 

### Color augmentation
"augmentation" means "padded image". By slightly processing the training images each time, we artificially increase the number of types of training images.

When Color Augmentation is turned on, the Hue of the image is changed randomly each time. LoRA learned from this is expected to have a slight range in color tone.

Not available if the Cache latents option is on.

Default is off.

 

### Flip augmentation
If this option is turned on, the image will be horizontally flipped randomly. It can learn left and right angles, which is useful when you want to learn symmetrical people and objects .

Default is off.

 

### Min SNR gamma
In LoRA learning, learning is performed by putting noise of various strengths on the training image (details about this are omitted), but depending on the difference in strength of the noise on which it is placed, learning will be stable by moving closer to or farther from the learning target. not, and the Min SNR gamma was introduced to compensate for that. Especially when learning images with little noise on them, it may deviate greatly from the target, so try to suppress this jump.

I won't go into details because it's confusing, but you can set this value from 0 to 20, and the default is 0.

According to the paper that proposed this method, the optimal value is 5.

I don't know how effective it is, but if you're unsatisfied with the learning results, try different values.

 

### Don't upscale bucket resolution
The Bucket size defaults to 256-1024 pixels (or a maximum resolution if specified with the Max resolution option, which takes precedence). Images that fall outside this size range, either vertically or horizontally, will be scaled (preserving the aspect ratio ) to fit within the specified range.

However, when this option is turned on, the bucket size range setting is ignored and the buckets are automatically prepared according to the size of the training images, so all training images are loaded unscaled. . However, even at this time, some parts of the image may be cropped to fit the Bucket resolution steps (described later).

Default is on.

 

### Bucket resolution steps
If using buckets , specify the resolution interval for each bucket here.

For example, if you specify 64 here, each training image will be sorted into separate buckets by 64 pixels according to their size. This sorting is done for each vertical and horizontal.

If the image size does not fit the specified size of the bucket, the protruding part will be cut off.

For example, if the maximum resolution is 512 pixels and the bucket step size is every 64 pixels , then the buckets will be 512, 448, 384... but a 500 pixel image will be put into a 448 pixel bucket, with an extra 52 pixels are clipped.

Default is 64 pixels .

* If this number is too small, the buckets will be divided too finely, and in the worst case, it will be like "one bucket for each image".

Note that we always load images from the same bucket for each batch, so having too few images in a bucket will unintentionally reduce the number of batches.

 

### Random crop instead of center crop
As mentioned above, half-sized images are sorted into buckets and then partly cropped to align the size, but usually it is cropped so as to keep the center of the image.

When this option is on, it randomly determines which part of the picture is cut. Turn on this option if you want to extend the learning range beyond the center of the image.

*This option cannot be used when the cache latents option is on.

 

### Noise offset type
This is an option to specify which method to use when adding additional noise to training images. At the time of learning, we always add noise to the image (details are omitted here), but it is preferable that this noise is "hard to predict" noise, so adding more noise makes it more "predictable". "hard" noise.

Default is Original. Multires adds noise in a slightly more complicated way.

 

#### Noise offset
This is an option when "Original" is selected for Noise offset type. If you enter a value greater than 0 here, additional noise will be added. Values ​​range from 0 to 1, where 0 adds no noise at all. A value of 1 adds strong noise.

It has been reported that adding about 0.1 noise makes LoRA's colors more vivid (brighter and darker). Default is 0.

 

####A daptive noise scale
Used in combination with the Noise offset option. Specifying a number here will further adjust the amount of additional noise specified by Noise offset to be amplified or attenuated. The amount of amplification (or attenuation) is automatically adjusted depending on how noisy the image is currently. Values ​​range from -1 to 1, with positive values ​​increasing the amount of added noise and negative values ​​decreasing the amount of added noise.

Default is 0.

 

#### Multires noise iterations
This is an option when "Multires" is selected for Noise offset type. If you enter a value greater than 0 here, additional noise will be added.

Multires creates noise of various resolutions and adds them together to create the final additive noise. Here you specify how many "various resolutions" to create.

Default is 0, when 0 there is no additional noise. It is recommended to set it to 6 if you want to use it.

 

#### Multires noise discount
Pair with the Multires noise iterations option. It is a numerical value for weakening the noise amount of each resolution to some extent. A value between 0 and 1, the lower the number, the weaker the noise. By the way, the amount of attenuation differs depending on the resolution, and noise with low resolution is attenuated a lot.

Default is 0, if 0 it will be set to 0.3 when used. 0.8 is usually recommended. If the number of training images is relatively small, it seems to be good to lower it to about 0.3.

 

### Dropout caption every n epochs
Normally, images and captions are trained in pairs, but it is possible to train only "images without captions" without using captions for each specific epoch.

This option allows you to specify "Don't use captions every 0 epochs ( Dropout )".

For example, if you specify 2 here, image learning without captions will be performed every 2 epochs (2nd epoch, 4th epoch, 6th epoch...).

When learning images without captions, LoRA is expected to learn more comprehensive image features. It can also be expected to have the effect of not associating too many image features with specific words. However, if you don't use too many captions, the LoRA may become a LoRA without prompts, so be careful.

The default is 0, which means no caption dropout .

### Rate of caption dropout
It is similar to Dropout caption every n epochs above, but you can learn as "images without captions" without using captions for a certain percentage of the entire learning process.

Here you can set the percentage of images without captions. 0 is the setting for "always use captions during learning", and 1 is the setting for "never use captions during learning".

It is random which images are learned as "images without captions".

For example, if 20 images are read 50 times each and LoRA learning is performed for only 1 epoch, the total number of image learning is 20 images x 50 times x 1 epoch = 1000 times. At this time, if the Rate of caption dropout is set to 0.1, 1000 times x 0.1 = 100 times will be learned as "images without captions".

Default is 0, which trains all images with captions.

 

### VAE batch size
If you turn on the Cache latents option, you can keep the "compressed" image data in the main memory. size. Since the number of images specified by batch size is learned at once, it is normal to match the VAE batch size with this.

Default is 0, in which case it is set to the same number as Batch size.

 

### Save training state
LoRA will take a long time to train if there are many training images, number of iterations, and number of epochs.

If you turn on this option, you can interrupt the study in the middle and resume the study from where you left off at a later date.

Intermediate learning data is saved in a folder called "last-state".

 

### Resume from saved training state
Specify the location of the "last-state" folder here if you want to resume learning that has been interrupted.

In order to resume learning, the intermediate progress data of learning must be saved.

 

### Max train epoch
Specify the maximum number of epochs for training. It is basic to specify the number of epochs with the Epoch option, but learning will always end when the number of epochs specified here is reached.

Default is blank. You can leave this field blank.

 

### Max num workers for DataLoader
This option specifies the number of CPU processes to use when reading data for training. Increasing this number will enable subprocesses and increase the speed of reading data, but increasing the number too much may actually result in inefficiency.

Note that no matter how large the number is specified, it will not exceed the number of concurrently executing threads of the CPU used.

The default is 0, which loads data only in the CPU's main process.

 

### WANDB API Key
There is a machine learning service called " WandB " (Weights&Biases) . This is a service that displays the progress of learning in graphs to find the optimal settings, records and shares learning logs online, and kohya_ss can now use this service.

However, you will need an account for this service. After creating an account, you can get an " API key" from https://app.wandb.ai/authorize . If you enter the acquired API key here, you will be automatically logged in when learning and you will be able to link with WandB services.

I won't go into details about WandB, but if you want to become a "LoRA craftsman", give it a try.

 

### WANDB Logging
Here you can specify whether or not to record learning progress logs using the WandB service.

The default is off, and when off, it logs in the form of a tool called 'tensorboard'.

## Sample images config
If you want to check what image generation with LoRA looks like while learning, enter the image generation prompt here.

However, since LoRA has a relatively short learning time, there may not be much need for image generation tests.

 

### Sample every n steps
Specify at what step you want to generate an image during learning. For example, specifying 100 will generate an image every 100 steps.

Default is 0, if 0 no image is generated.

 

### Sample every n epochs
Specifies the number of epochs to generate images during training. For example, 2 will generate an image every 2 epochs.

Default is 0, if 0 no image is generated.

 

### Sample sampler
Specifies the sampler to use for image generation . Many of the samplers specified here are the same as the samplers provided in the Stable Diffusion Web UI , so please refer to the web UI explanation site for details.

The default is euler_a.

 

### Sample prompts
Enter the prompt here.

However, you can enter other settings here than just prompts. If you want to enter other settings, specify the setting by combining two minus letters and alphabets like "--n". For example, if you want to put "white, dog" in the negative prompt, write "--n white, dog".

Here are some commonly used settings:

--n: negative prompt

--w: image width

--h: image height

--d: Seeds

--l: CFG Scale

--s: number of steps

Default is blank. When the field is blank, the description example is displayed in faint color, so please refer to it.

 