cd unet_pipeline
python Train.py ./experiments/albunet1024/01_train_config_part0.yaml
python Train.py ./experiments/albunet1024/02_train_config_part1.yaml
python Inference.py ./experiments/albunet1024/04_inference_config.yaml
python TripletSubmit.py ./experiments/albunet1024/05_submit.yaml

