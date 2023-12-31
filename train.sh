export CUDA_VISIBLE_DEVICES=0
python scripts/segmentation_train.py --data_name BRATS \
--data_dir data/brats/MICCAI_BraTS2020_TrainingData --out_dir output/brats/base \
--image_size 256 --num_channels 128 --class_cond False \
--num_res_blocks 2 --num_heads 1 --learn_sigma True \
--use_scale_shift_norm False --attention_resolutions 16 \
--diffusion_steps 1000 --noise_schedule linear \
--rescale_learned_sigmas False --rescale_timesteps False \
--lr 1e-4 --batch_size 4 --version 1