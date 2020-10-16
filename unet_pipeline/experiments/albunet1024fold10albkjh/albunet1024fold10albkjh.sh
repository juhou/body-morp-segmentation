cd unet_pipeline
python Train.py ./experiments/albunet1024fold10albkjh/01_train_config_part0.yaml
python Train.py ./experiments/albunet1024fold10albkjh/02_train_config_part1.yaml
python Inference.py ./experiments/albunet1024fold10albkjh/04_inference_config.yaml
python TripletSubmit.py ./experiments/albunet1024fold10albkjh/05_submit.yaml






