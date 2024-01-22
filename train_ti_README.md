# About learning Textual Inversion

[Textual Inversion](https://textual-inversion.github.io/). I heavily referenced https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion for the implementation.

The trained model can be used as is on the Web UI.

In addition, it is probably compatible with SD2.x, but it has not been tested at this time.

## Learning method

Use ``train_textual_inversion.py``.

Data preparation is exactly the same as ``train_network.py``, so please refer to [their document](./train_network_README-en.md).

## options

Below is an example command line (DreamBooth technique).

```shell
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py
     --pretrained_model_name_or_path=..\models\model.ckpt
     --train_data_dir=..\data\db\char1 --output_dir=..\ti_train1
     --resolution=448,640 --train_batch_size=1 --learning_rate=1e-4
     --max_train_steps=400 --use_8bit_adam --xformers --mixed_precision=fp16
     --save_every_n_epochs=1 --save_model_as=safetensors --clip_skip=2 --seed=42 --color_aug
     --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

``--token_string`` specifies the token string for learning. __The learning prompt should contain this string (eg ``mychar4 1girl`` if token_string is mychar4)__. This string part of the prompt is replaced with a new token for Textual Inversion and learned.

``--debug_dataset`` will display the token id after substitution, so you can check if the token string after ``49408`` exists as shown below. I can confirm.

```python
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
          49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```

Words that the tokenizer already has (common words) cannot be used.

In ``--init_word``, specify the string of the copy source token when initializing embeddings. It seems to be a good idea to choose something that has a similar concept to what you want to learn. You cannot specify a character string that becomes two or more tokens.

``--num_vectors_per_token`` specifies how many tokens to use for this training. The higher the number, the more expressive it is, but it consumes more tokens. For example, if num_vectors_per_token=8, then the specified token string will consume 8 tokens (out of the 77 token limit for a typical prompt).

In addition, the following options can be specified.

* --weights
   * Load learned embeddings before learning and learn additionally from there.
* --use_object_template
   * Learn with default object template strings (such as ``a photo of a {}``) instead of captions. It will be the same as the official implementation. Captions are ignored.
* --use_style_template
   * Learn with default style template strings instead of captions (such as ``a painting in the style of {}``). It will be the same as the official implementation. Captions are ignored.

## Generate with the image generation script in this repository

In gen_img_diffusers.py, specify the learned embeddings file with the ``--textual_inversion_embeddings`` option. Using the filename (without the extension) of the embeddings file at the prompt will apply the embeddings.