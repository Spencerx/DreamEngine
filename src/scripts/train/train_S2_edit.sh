export DATASET_CONFIG="./QwenVL-GEN/qwen2vl-sd3/dataset_config/s2_ultraedit.json"


export OUTPUT_DIR="./QwenVL-GEN/diffusers/examples/dreambooth/ckpts/s1-joint-align-test"
export WANDB_API_KEY=d2370850d4e373a071a8597e99e898e153387e70

# disable wandb
export WANDB_MODE="dryrun"

export VALIDATION_IMAGES="./datasets/JourneyDB/data/train/imgs/003/0a3b185f-b0e4-4650-90cd-91909c523b49.jpg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/1665_Girl_with_a_Pearl_Earring.jpg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/husky_grass.jpeg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/snow_house.png"

export VALIDATION_EDIT_IMAGES="./datasets/JourneyDB/data/train/imgs/003/0a3b185f-b0e4-4650-90cd-91909c523b49.jpg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/1665_Girl_with_a_Pearl_Earring.jpg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/husky_grass.jpeg,\
./QwenVL-GEN/diffusers/examples/dreambooth/test_image/snow_house.png"




accelerate launch --config_file ./QwenVL-GEN/diffusers/examples/dreambooth/default_config_1gpu.yaml \
  train.py \
  --output_dir=$OUTPUT_DIR \
  --pretrained_diffusion_ckpt="./models/stable-diffusion-3-medium-diffusers" \
  --pretrained_lmm_ckpt="./models/Qwen2-VL-7B-Instruct" \
  --resume_from_checkpoint="./QwenVL-GEN/qwen2vl-sd3/s1-joint-vision-language-alignment-jdbcoco2.6M-bsz256-lr5e-5-freezelmm/checkpoint-40000/transformer" \
  --use_lmm_attention_lora --use_dit_attention_lora  \
  --unfreeze_adapter \
  --lora_rank=32 \
  --dataset_config=$DATASET_CONFIG \
  --gradient_accumulation_steps=2 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --max_grad_norm=1.0 \
  --mixed_precision="bf16" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=5000 \
  --max_train_steps=50000 \
  --validation_images=$VALIDATION_IMAGES \
  --validation_prompts="a lovely dog|a black and white flower, in the rain|a small  Robot looking at a small beautiful flower in the ground|a beautiful girl smiling and holding a flower" \
  --validation_steps=10 \
  --seed="0" \
  --checkpointing_steps=100 \
  --lmm_output_layer_index=-1 \
  --validation_edit_images=$VALIDATION_EDIT_IMAGES \
  --validation_edit_prompts="add a white hat to the woman|make the girl eatting a banana|change the dog to a cat|remove the snow"


  
# python ./gpx.py --size 60000 --gpus 8 --interval 0.05