cd unet_pipeline
python Train.py ./experiments/albunet512/01_train_config_part0.yaml
python Train.py ./experiments/albunet512/02_train_config_part1.yaml
python Train.py ./experiments/albunet512/03_train_config_part2.yaml
python Inference.py ./experiments/albunet512/04_inference_config.yaml
python TripletSubmit.py ./experiments/albunet512/05_submit.yaml

