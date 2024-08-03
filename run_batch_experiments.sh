# Run experiments simultaneously with 3 GPUs
# If you only have one GPU available, you may delete the ""&"" sign at the end of each line

CUDA_VISIBLE_DEVICES=0 python3 main.py --data_root_dir FEATURE_DIR --split_dir tcga_ucec --model_type mome --apply_sig --n_bottlenecks 2 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --data_root_dir FEATURE_DIR --split_dir tcga_blca --model_type mome --apply_sig --n_bottlenecks 2 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --data_root_dir FEATURE_DIR --split_dir tcga_luad --model_type mome --apply_sig --n_bottlenecks 2 &