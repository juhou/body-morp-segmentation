cd unet_pipeline
python Train.py ./experiments/albunet1024fold10/01_train_config_part0.yaml
python Train.py ./experiments/albunet1024fold10/02_train_config_part1.yaml
python Train.py ./experiments/albunet1024fold10/03_train_config_part2.yaml
python Inference.py ./experiments/albunet1024fold10/04_inference_config.yaml
python TripletSubmit.py ./experiments/albunet1024fold10/05_submit.yaml




python Train.py ./experiments/albunet1024fold10/01_train_config_part0_2070.yaml
python Train.py ./experiments/albunet1024fold10/02_train_config_part1_2070.yaml
python Train.py ./experiments/albunet1024fold10/03_train_config_part2_2070.yaml

