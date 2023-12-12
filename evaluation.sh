export CUDA_VISIBLE_DEVICES=0
python scripts/segmentation_env.py --inp_pth output/isic/classifier-free/sample/eval \
--out_pth data/isic/ISBI2016_ISIC_Part1_Test_GroundTruth