export DATASET_CONFIG="./dataset_config/ti2i_jdb1M.json"
export OUTPUT_DIR="./diffusers/examples/dreambooth/ckpts/s1-joint-align-test"

# disable wandb
export WANDB_MODE="dryrun"

export VALIDATION_IMAGES="./datasets/JourneyDB/data/train/imgs/003/0a3b185f-b0e4-4650-90cd-91909c523b49.jpg,\
./diffusers/examples/dreambooth/test_image/1665_Girl_with_a_Pearl_Earring.jpg,\
./diffusers/examples/dreambooth/test_image/husky_grass.jpeg,\
./diffusers/examples/dreambooth/test_image/snow_house.png"





accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --zero_stage 2 \
  train.py \
  --output_dir=$OUTPUT_DIR \
  --pretrained_diffusion_ckpt="./models/stable-diffusion-3.5-large" \
  --pretrained_lmm_ckpt="./models/Qwen2-VL-2B-Instruct" \
  --unfreeze_adapter \
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
  --validation_steps=20 \
  --seed="0" \
  --checkpointing_steps=20 \
  --lmm_output_layer_index=-1 \
  --structure="direct" \
  --cfg_ratio=0.1 \
  --random_vit_skip


