cd unet_pipeline
python Train.py ./experiments/seunet1024fold10albJW/01_train_config_part0.yaml
python Train.py ./experiments/seunet1024fold10albJW/02_train_config_part1.yaml
python Train.py ./experiments/seunet1024fold10albJW/03_train_config_part2.yaml
python Inference.py ./experiments/seunet1024fold10albJW/04_inference_config.yaml
python TripletSubmit.py ./experiments/seunet1024fold10albJW/05_submit.yaml

