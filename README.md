
# body-morp-segmentation

Please see the attached pdf file named '파이프라인 설명서.pdf'

## Model Zoo
- AlbuNet (resnet34) from [\[ternausnets\]](https://github.com/ternaus/TernausNet)

## Original source code

https://github.com/sneddy/pneumothorax-segmentation


## Main Features

### Combo loss
Used \[[combo loss\]](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/master/selim_sef/training/losses.py) combinations of BCE, dice and focal. In the best experiments the weights of (BCE, dice, focal), that I used were:
- (1,1,1) for albunet


### Checkpoints averaging
Top3 checkpoints averaging from each fold from each pipeline on inference


### Horizontal flip TTA

## File structure
    ├── unet_pipeline
    │   ├── experiments
    │   │   ├── some_experiment
    │   │   │   ├── train_config.yaml
    │   │   │   ├── inference_config.yaml
    │   │   │   ├── submit_config.yaml
    │   │   │   ├── checkpoints
    │   │   │   │   ├── fold_i
    │   │   │   │   │   ├──topk_checkpoint_from_fold_i_epoch_k.pth 
    │   │   │   │   │   ├──summary.csv
    │   │   │   │   ├──best_checkpoint_from_fold_i.pth
    │   │   │   ├── log
    ├── input                
    │   ├── train
    │   ├── test   
    │   └── new_train_rle.csv
    └── requirements.txt

## Install
```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
kaggle competitions download -c body-~~~
```

## Pipeline launch example
Training:
```bash
cd unet_pipeline
python Train.py experiments/albunet512/train_config_part0.yaml
python Train.py experiments/albunet512/train_config_part1.yaml
```
As an output, we get a checkpoints in corresponding folder.


Inference:
```bash
cd unet_pipeline
python Inference.py experiments/albunet512/inference_config.yaml
```
As an output, we get a pickle-file with mapping the file name into a mask with pneumothorax probabilities.

Submit:
```bash
cd unet_pipeline
python TripletSubmit.py experiments/albunet512/submit.yaml
```
As an output, we get submission file with rle.
