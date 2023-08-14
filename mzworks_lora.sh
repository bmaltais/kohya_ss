#!/bin/zsh

UNAME_OS=$(uname -s)
UNAME_HW=$(uname -m)

# ---- Env check
if [[ ${UNAME_OS} != "Darwin" ]]; then
    echo "This OS is ${UNAME_OS}. This program is for Darwin only."
    exit
fi

if [[ ${UNAME_HW} != "arm64" ]]; then
    echo "This HW is ${UNAME_HW}. This program is for arm64 only."
    exit
fi

# ----
PRETRAINED_MODEL_NAME="emilianJR/AnyLORA"
OUTPUT_NAME="mizunagiworks_lora"
# ex) ln -s {{ /path/to }} ./dataset/train
DATADIR_TRAIN="./dataset/train"
# ex) ln -s {{ /path/to }} ./dataset/reg
DATADIR_REG="./dataset/reg"

TRAIN_EPOCHS=10


function mode_train {
    accelerate launch --num_cpu_threads_per_process 4 train_network.py \
        --pretrained_model_name_or_path ${PRETRAINED_MODEL_NAME} \
        --dataset_config "./mzworks_lora.toml" \
        --output_dir "./mzworks/model" \
        --output_name ${OUTPUT_NAME} \
        --save_model_as safetensors \
        --prior_loss_weight 1.0 \
        --max_train_epochs ${TRAIN_EPOCHS} \
        --learning_rate 1e-4 \
        --mixed_precision "no" \
        --gradient_checkpointing \
        --save_every_n_epochs 1 \
        --sample_every_n_steps 100 \
        --sample_prompts "./mzworks_sample_prompts.txt" \
        --network_module networks.lora \
        --save_precision "float"
}


function mode_generate {
    python gen_img_diffusers.py \
        --ckpt ${PRETRAINED_MODEL_NAME} \
        --n_iter 1 --scale 7.5 --steps 30 \
        --outdir ./mzworks/${OUTPUT_NAME}/image --W 512 --H 512 --sampler k_euler_a \
        --network_module networks.lora \
        --network_weights ./mzworks/model/${OUTPUT_NAME}.safetensors \
        --network_mul 1.0 --max_embeddings_multiples 3 --clip_skip 1 \
        --batch_size 1 --images_per_prompt 1 --interactive
}

function mode_tag {
    python finetune/tag_images_by_wd14_tagger.py \
        --batch_size 8 \
        --caption_extension ".caption" \
        --remove_underscore \
        ${DATADIR_TRAIN}
}


if [[ ${1} == "train" || ${1} == "t" ]]; then
    mode_train;
elif [[ ${1} == "generate" || ${1} == "g" ]]; then
    mode_generate;
elif [[ ${1} == "tag" ]]; then
    mode_tag
else
    echo "usage: ${0} [arg]"
    echo "train or t"
    echo "        Train"
    echo "generate or g"
    echo "        Image generate"
    echo "tag"
    echo "        Tag generate"
    echo ""
fi
