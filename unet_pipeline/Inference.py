import argparse
from tqdm import tqdm
import os
import importlib
from pathlib import Path
import pickle
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
import albumentations as albu 
import torch
import ttach as tta
from BodyMorpDataset import BodyMorpDataset
from utils.helpers import load_yaml, init_seed, init_logger


def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('cfg', type=str, help='experiment name')
    return parser.parse_args()

def build_checkpoints_list(cfg):
    pipeline_path = Path(cfg['CHECKPOINTS']['PIPELINE_PATH'])
    pipeline_name = cfg['CHECKPOINTS']['PIPELINE_NAME']

    checkpoints_list = []
    if cfg.get('SUBMIT_BEST', False):
        best_checkpoints_folder = Path(
            pipeline_path, 
            cfg['CHECKPOINTS']['BEST_FOLDER']
        )

        usefolds = cfg['USEFOLDS']
        for fold_id in usefolds:
            filename = '{}_fold{}.pth'.format(pipeline_name, fold_id)
            checkpoints_list.append(Path(best_checkpoints_folder, filename))
    else:
        folds_dict = cfg['SELECTED_CHECKPOINTS']
        for folder_name, epoch_list in folds_dict.items():
            checkpoint_folder = Path(
                pipeline_path,
                cfg['CHECKPOINTS']['FULL_FOLDER'],
                folder_name,
            )
            for epoch in epoch_list:
                checkpoint_path = Path(
                    checkpoint_folder,
                    '{}_{}_epoch{}.pth'.format(pipeline_name, folder_name, epoch)
                )
                checkpoints_list.append(checkpoint_path)
    return checkpoints_list


def inference_image(model, images, device):
    images = images.to(device)
    predicted = model(images)
    masks = torch.sigmoid(predicted) 
    masks = masks.squeeze(1).cpu().detach().numpy()
    return masks


# def inference_model(model, loader, device, use_flip):
#     mask_dict = {}
#     for image_ids, images in tqdm(loader):
#         masks = inference_image(model, images, device)
#         if use_flip:
#             flipped_imgs = torch.flip(images, dims=(3,))
#             flipped_masks = inference_image(model, flipped_imgs, device)
#             flipped_masks = np.flip(flipped_masks, axis=3) # 종욱: 2class일땐 axis=2
#             masks = (masks + flipped_masks) / 2
#         for name, mask in zip(image_ids, masks):
#             mask_dict[name] = mask.astype(np.float32)
#     return mask_dict


def inference_model(model, loader, device, use_flip):
    mask_dict = {}

    for image_ids, images in tqdm(loader):
        masks = inference_image(model, images, device)
        if use_flip:
            #images[1,1,:,:].numpy()
            for target_s in [512, 640, 768, 896]:
                images_ = torch.tensor(images)
                pad_size = int((1024 - 640)/2)
                images_ = F.interpolate(images_, size=target_s)
                images_ = F.pad(images_, (pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
                tta_mask = inference_image(model, images_, device)
                tta_mask = torch.from_numpy(tta_mask)
                tta_mask = F.pad(tta_mask, (-pad_size,-pad_size,-pad_size,-pad_size), mode='constant', value=0)
                tta_mask = F.interpolate(tta_mask, size=1024)
                masks += tta_mask.numpy()
            masks = masks / 5

            # planes = [2, 3]
            # tta_mask = inference_image(model, images.rot90(1, planes), device)
            # tta_mask = np.rot90(tta_mask, 3, planes)
            # masks += tta_mask
            # tta_mask = inference_image(model, images.rot90(2, planes), device)
            # tta_mask = np.rot90(tta_mask, 2, planes)
            # masks += tta_mask
            # tta_mask = inference_image(model, images.rot90(3, planes), device)
            # tta_mask = np.rot90(tta_mask, 1, planes)
            # masks += tta_mask
            # tta_mask = inference_image(model, images.flip(planes[0]), device)
            # tta_mask = np.flip(tta_mask, planes[0])
            # masks += tta_mask
            # tta_mask = inference_image(model, images.rot90(1, planes).flip(planes[0]), device)
            # tta_mask = np.rot90(np.flip(tta_mask, planes[0]), 3, planes)
            # masks += tta_mask
            # tta_mask = inference_image(model, images.rot90(2, planes).flip(planes[0]), device)
            # tta_mask = np.rot90(np.flip(tta_mask, planes[0]), 2, planes)
            # masks += tta_mask
            # tta_mask = inference_image(model, images.flip(planes[0]).rot90(1, planes), device)
            # tta_mask = np.flip(np.rot90(tta_mask, 3, planes), planes[0])
            # masks += tta_mask
            # masks = masks/8
        for name, mask in zip(image_ids, masks):
            mask_dict[name] = mask.astype(np.float32)
    return mask_dict


def main():
    args = argparser()
    config_path = Path(args.cfg.strip("/"))
    experiment_folder = config_path.parents[0]
    inference_config = load_yaml(config_path)
    print(inference_config)
    
    batch_size = inference_config['BATCH_SIZE']
    device = inference_config['DEVICE']
    
    module = importlib.import_module(inference_config['MODEL']['PY'])
    model_class = getattr(module, inference_config['MODEL']['CLASS'])
    model = model_class(**inference_config['MODEL'].get('ARGS', None)).to(device)
    model.eval()

    num_workers = inference_config['NUM_WORKERS']
    transform = albu.load(inference_config['TEST_TRANSFORMS']) 
    dataset_folder = inference_config['DATA_DIRECTORY'] 
    dataset = BodyMorpDataset(
        data_folder=dataset_folder, mode='test', 
        transform=transform,
    )
    dataloader =  DataLoader(
        dataset=dataset, batch_size=batch_size, 
        num_workers=num_workers, shuffle=False
    )

    use_flip = inference_config['FLIP']
    checkpoints_list = build_checkpoints_list(inference_config)
  
    mask_dict = defaultdict(int)
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        print(checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

        model.eval()
        current_mask_dict = inference_model(model, dataloader, device, use_flip)
        for name, mask in current_mask_dict.items():
            mask_dict[name] = (mask_dict[name] * pred_idx + mask) / (pred_idx + 1)

    if 'RESULT_FOLDER' in inference_config:
        result_path = Path(inference_config['RESULT_FOLDER'], inference_config['RESULT'])
    else:
        result_path = Path(experiment_folder, inference_config['RESULT'])

    with open(result_path, 'wb') as handle:
        pickle.dump(mask_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()