TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 64 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --save_cpt_dir ./checkpoints/isic2018_64classifier"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

export NCCL_P2P_DISABLE=1
#mpiexec -n 2 
CUDA_VISIBLE_DEVICES=4 python scripts/classifier_train.py --data_dir /home/slidm/DDPM/guided-diffusion/datasets/ISIC_2018 $TRAIN_FLAGS $CLASSIFIER_FLAGS