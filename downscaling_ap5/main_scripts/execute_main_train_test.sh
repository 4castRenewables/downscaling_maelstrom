#!/bin/sh
python3 -u main_train.py -in "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/preprocessed_tier2/test_subset" -out "../trained_models"  -model "wgan" -dataset "tier2" -exp_name "wgan_benchmark_jube_$jube_benchmark_id" -conf_md "../config/config_wgan_benchmark.json" -conf_ds "../config/config_ds_tier2.json" -js_norm "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_benchmark_dataset/preprocessed_tier2/test_subset/norm.json" -id 89

