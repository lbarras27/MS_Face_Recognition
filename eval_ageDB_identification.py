import eval_template

import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do AgeDB test')
    # general
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/AgeDB/imgs', type=str, help='path to img directory')
    parser.add_argument('--metadata-path', default='datasets/AgeDB/metadata', type=str, help='path to metadata files')
    parser.add_argument('--result-dir', default='datasets/AgeDB/results', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--gap', default=5, type=int, help='')
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size
    
    gap = args.gap

    path_probe = metadata_path + "/probe_set_gap_{0}.txt".format(gap)
    path_gallery = metadata_path +  "/gallery_set_gap_{0}.txt".format(gap)
    
    landmarks = eval_template.get_landmarks_vmer(metadata_path + "/landmarks.txt")

    probe_set, gallery_set = eval_template.load_probe_gallery_set(path_probe, path_gallery)
    all_img_id_used = pd.concat([probe_set, gallery_set])

    img_features = eval_template.get_image_feature(img_path, all_img_id_used["img_name"].values, landmarks, model_path, batch_size)

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    map_names_to_id = eval_template.get_map_names_to_id(all_img_id_used["img_name"].values)
    probe_ids = eval_template.convert_names_to_id(probe_set["img_name"].values, map_names_to_id)
    gallery_ids = eval_template.convert_names_to_id(gallery_set["img_name"].values, map_names_to_id)
    
    probe_features = img_features[probe_ids]
    gallery_features = img_features[gallery_ids]
    
    mask = eval_template.compute_mask_agedb(probe_set["img_id"], gallery_set["img_id"])

    print("Start evaluation")
    eval_template.evaluation(probe_features, gallery_features, mask)
    print("End evaluation")