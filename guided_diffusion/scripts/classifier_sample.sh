MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --class_cond True "
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 "
SAMPLE_FLAGS="--batch_size 50 --num_samples 8000 --timestep_respacing ddim25 --use_ddim True --class_num 7 --save_img_dir ./results/isic2018_64cs_050000iter"

#mpiexec -n N 
CUDA_VISIBLE_DEVICES=4 python scripts/classifier_sample.py \
    --model_path ./checkpoints/isic2018_64dm/model050000.pt \
    --classifier_path ./checkpoints/isic2018_64classifier/model200000.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS