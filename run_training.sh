#!/bin/bash


screen -S model0 -dm bash -c 'docker run --runtime=nvidia --gpus device=0 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 0 > model0_training.log 2>&1'

screen -S model1 -dm bash -c 'docker run --runtime=nvidia --gpus device=1 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 1 > model1_training.log 2>&1'

screen -S model2 -dm bash -c 'docker run --runtime=nvidia --gpus device=2 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 2 > model2_training.log 2>&1'

screen -S model3 -dm bash -c 'docker run --runtime=nvidia --gpus device=3 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 3 > model3_training.log 2>&1'

screen -S model4 -dm bash -c 'docker run --runtime=nvidia --gpus device=4 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 4 > model4_training.log 2>&1'

screen -S model5 -dm bash -c 'docker run --runtime=nvidia --gpus device=5 --shm-size=32g --rm -v /data/genmedbfx/skinstager_training:/workspace -v /data/genmedbfx/skinstager_training/model_cache/torch:/root/.cache/torch -v /data/genmedbfx/skinstager_training/model_cache/huggingface:/root/.cache/huggingface -w /workspace skin-seg-train:cuda121 python MLpipeline_multi.py 5 > model5_training.log 2>&1'