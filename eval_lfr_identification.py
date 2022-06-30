import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from evaluation_util import *
sys.path.insert(0, './datasets/LFR dataset')
from eval_lfr_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do LFR test')
    # general
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/LFR dataset/imgs', type=str, help='path to load images')
    parser.add_argument('--metadata-path', default='datasets/LFR dataset/metadata', type=str, help='path to metada files')
    parser.add_argument('--result-dir', default='datasets/LFR dataset/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--pose', default='right', type=str, help='')
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size
    
    pose = args.pose

    path_probe = metadata_path + "/probe_set_{}.txt".format(pose)
    path_gallery = metadata_path + "/gallery_set_front2.txt"

    probe_set, gallery_set = load_probe_gallery_set(path_probe, path_gallery, delim=",")
    all_img_id_used = pd.concat([probe_set, gallery_set])

    img_features = get_image_feature(img_path, all_img_id_used["img_name"].values, None, model_path, batch_size, already_align=True)
    print("images features loaded")

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    map_names_to_id = get_map_names_to_id(all_img_id_used["img_name"].values)
    probe_ids = convert_names_to_id(probe_set["img_name"].values, map_names_to_id)
    gallery_ids = convert_names_to_id(gallery_set["img_name"].values, map_names_to_id)

    probe_features = img_features[probe_ids]
    gallery_features = img_features[gallery_ids]

    mask = compute_mask_lfr(probe_set["img_id"], gallery_set["img_id"])

    print("Start evaluation")
    evaluation(probe_features, gallery_features, mask, is_cmc=True)
    print("End evaluation")