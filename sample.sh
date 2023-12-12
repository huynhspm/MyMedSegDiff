export CUDA_VISIBLE_DEVICES=2
# python scripts/segmentation_sample.py --data_name ISIC \
# --data_dir data/isic --out_dir output/base/sample \
# --model_path output/base/emasavedmodel_0.9999_160000.pt \
# --image_size 256 --num_channels 128 --class_cond False \
# --num_res_blocks 2 --num_heads 1 --learn_sigma True \
# --use_scale_shift_norm False --attention_resolutions 16 \
# --diffusion_steps 1000 --noise_schedule linear \
# --rescale_learned_sigmas False --rescale_timesteps False \
# --num_ensemble 5 --version 1

python scripts/segmentation_sample.py --data_name BRATS \
--data_dir data/brats/MICCAI_BraTS2020_ValidationData --out_dir output/brats/sample \
--model_path output/brats/emasavedmodel_0.9999.pt \
--image_size 256 --num_channels 128 --class_cond False \
--num_res_blocks 2 --num_heads 1 --learn_sigma True \
--use_scale_shift_norm False --attention_resolutions 16 \
--diffusion_steps 1000 --noise_schedule linear \
--rescale_learned_sigmas False --rescale_timesteps False \
--num_ensemble 5 --version 1 --debug True