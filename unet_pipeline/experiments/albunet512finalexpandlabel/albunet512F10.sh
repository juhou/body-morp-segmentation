cd unet_pipeline
python Train.py ./experiments/albunet512finalexpandlabel/01_train_config_part0.yaml
python Train.py ./experiments/albunet512finalexpandlabel/02_train_config_part1.yaml
python Inference.py ./experiments/albunet512finalexpandlabel/03_inference_config.yaml
python TripletSubmit.py ./experiments/albunet512finalexpandlabel/04_submit.yaml

