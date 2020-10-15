import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
import os

import cv2

import numpy as np
import pandas as pd
from collections import defaultdict

from utils.mask_functions import mask2rle, rle_encode
from utils.helpers import load_yaml

def argparser():
    parser = argparse.ArgumentParser(description='Body Morp pipeline')
    parser.add_argument('cfg', type=str, help='experiment name')
    return parser.parse_args()

def extract_largest(mask, n_objects):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours[:n_objects],
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def apply_thresholds(mask, n_objects, area_threshold, top_score_threshold, 
                     bottom_score_threshold, leak_score_threshold, use_contours, min_contour_area, name_i):
    # if n_objects == 1:
    #     crazy_mask = (mask > top_score_threshold).astype(np.uint8)
    #     if crazy_mask.sum() < area_threshold:
    #         return -1
    #     mask = (mask > bottom_score_threshold).astype(np.uint8)
    # else:
    #
    mask = mask.astype(np.uint8)

    if min_contour_area > 0:
        choosen = remove_smallest(mask, min_contour_area)
    elif use_contours:
        choosen = extract_largest(mask, n_objects)
    else:
        choosen = mask * 255

    if mask.shape[0] == 512:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(512, 512),
            interpolation=cv2.INTER_LINEAR
        )

    reshaped_mask = (reshaped_mask > 63).astype(int) * 255
    # cv2.imwrite(name_i + '_.png', reshaped_mask)
    return rle_encode(reshaped_mask)

def build_rle_dict(mask_dict, n_objects_dict,  
                   area_threshold, top_score_threshold,
                   bottom_score_threshold,
                   leak_score_threshold, 
                   use_contours, min_contour_area, sub_img_path):
    rle_dict = {}

    for name, mask in tqdm(mask_dict.items()):
        # 물체 개수를 판단 (채널이 다르므로 늘 1개)
        # TODO: class 개수 4개로 임의 설정
        num_class = 4

        max_mask = (mask.max(axis=0,keepdims=1) == mask) * 1.0
        mask_123 = np.zeros([max_mask.shape[1], max_mask.shape[2]])
        max_masked = mask * max_mask

        # 첫번째 mask는 필요없음
        for i in range(1, num_class):
            # 1,2,3번째 마스크 : 예측값은 1롤만 한다
            mask_i = max_mask[i,:,:]


            # 데이터 이름에 _1,_2,_3 붙이기
            name_i = name + f'_{i}'
            n_objects = n_objects_dict.get(name_i, 0)

            if n_objects == 0:
                # 물체가 없는경우 rle는 공백
                rle_dict[name_i] = ''
            else:
                # 물체가 있는 경우에만 처리를 함
                # 마스크는 0아니면 1
                rle_dict[name_i] = apply_thresholds(
                    mask_i, n_objects,
                    area_threshold, top_score_threshold,
                    bottom_score_threshold,
                    leak_score_threshold,
                    use_contours, min_contour_area, sub_img_path + name_i
                )
            mask_123 = mask_123 + max_mask[i,:,:]*i

        rgb_image = np.transpose(max_masked[1:4,:,:]*255, (1, 2, 0))
        
        # rgb 이미지에는 probability가 반영되어있음
        cv2.imwrite(sub_img_path + name + '_rgb.png', rgb_image)
        cv2.imwrite(sub_img_path + name + '.png', mask_123)


    return rle_dict

def buid_submission(rle_dict, sample_sub):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = sample_sub.columns
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub

def load_mask_dict(cfg):
    reshape_mode = cfg.get('RESHAPE_MODE', False)
    if 'MASK_DICT' in cfg:
        result_path = Path(cfg['MASK_DICT'])
        with open(result_path, 'rb') as handle:
            mask_dict = pickle.load(handle)
        return mask_dict
    if 'RESULT_WEIGHTS' in cfg:
        result_weights = cfg['RESULT_WEIGHTS']
        mask_dict = defaultdict(int)
        for result_path, weight in result_weights.items():
            print(result_path, weight)
            with open(Path(result_path), 'rb') as handle:
                current_mask_dict = pickle.load(handle)
                for name, mask in current_mask_dict.items():
                    if reshape_mode and mask.shape[1] != 512:
                        reshaped_mask = np.zeros([mask.shape[0],512,512])
                        for c in range(mask.shape[0]):
                            reshaped_mask[c,:,:] = cv2.resize(mask[c,:,:], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                        mask = reshaped_mask
                    #crazy_mask = (mask > 0.75).astype(np.uint8)
                    #if crazy_mask.sum() < 1000:
                    #  mask = np.zeros_like(mask)
                    mask_dict[name] = mask_dict[name] + mask * weight
        return mask_dict


def main():
    args = argparser()
    # config_path = 'experiments/albunet_public/05_submit.yaml' #
    config_path = Path(args.cfg.strip("/"))
    sub_config = load_yaml(config_path)
    print(sub_config)
    
    sample_sub = pd.read_csv(sub_config['SAMPLE_SUB'])
    n_objects_dict = sample_sub.ImageId.value_counts().to_dict()
    
    print('start loading mask results....')
    mask_dict = load_mask_dict(sub_config)
    
    use_contours = sub_config['USECONTOURS']
    min_contour_area = sub_config.get('MIN_CONTOUR_AREA', 0)

    area_threshold = sub_config['AREA_THRESHOLD']
    top_score_threshold = sub_config['TOP_SCORE_THRESHOLD']
    bottom_score_threshold = sub_config['BOTTOM_SCORE_THRESHOLD']
    if sub_config['USELEAK']:
        leak_score_threshold = sub_config['LEAK_SCORE_THRESHOLD']
    else:
        leak_score_threshold = bottom_score_threshold

    sub_file = Path(sub_config['SUB_FILE'])
    sub_img_path = '../subs/' + sub_file.parts[-1][:-4] + '/'
    if not os.path.exists(sub_img_path):
        os.mkdir(sub_img_path)

    rle_dict = build_rle_dict(
        mask_dict, n_objects_dict, area_threshold,
        top_score_threshold, bottom_score_threshold,
        leak_score_threshold, use_contours, min_contour_area, sub_img_path
    )
    sub = buid_submission(rle_dict, sample_sub)
    print((sub.EncodedPixels != -1).sum())
    print(sub.head())


    sub.to_csv(sub_file, index=False)

if __name__ == "__main__":
    main()
