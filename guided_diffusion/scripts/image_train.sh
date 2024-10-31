MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 48 --save_cpt_dir ./checkpoints/isic2018_64dm"

export NCCL_P2P_DISABLE=1
mpiexec -n 4 python scripts/image_train.py --data_dir /home/slidm/DDPM/guided-diffusion/datasets/ISIC_2018 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS