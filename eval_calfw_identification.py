import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from evaluation_util import *
sys.path.insert(0, './datasets/calfw')
from eval_calfw_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do CALFW test')
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/calfw/imgs', type=str, help='path to load images')
    parser.add_argument('--metadata-path', default='datasets/calfw/metadata', type=str, help='path to metada files')
    parser.add_argument('--gallery-set', default='', type=str, help='name of gallery set file')
    parser.add_argument('--probe-set', default='', type=str, help='name of probe set file')
    parser.add_argument('--result-dir', default='datasets/calfw/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
   
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    
    gallery_set_filename = args.gallery_set
    if gallery_set_filename == '':
        gallery_set_filename = "gallery_set.txt"
        
    probe_set_filename = args.probe_set
    if probe_set_filename == '':
        probe_set_filename = "probe_set.txt"
        
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size

    path_probe = metadata_path + "/" + probe_set_filename
    path_gallery = metadata_path + "/" + gallery_set_filename

    df_names_ids = get_names_ids_calfw(metadata_path + "/all_names_id.csv")
    landmarks = get_landmarks_calfw(metadata_path)

    # align the images with the help of landmarks points and then compute features vectors
    img_features = get_image_feature(img_path, df_names_ids["img_name"].values, landmarks, model_path, batch_size)

    # if use_flip_test equal True, sum the features vector of the image with the features vector of the flipped image and normalize.
    # else use only the features vector of the image.
    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    probe_set, gallery_set = load_probe_gallery_set(path_probe, path_gallery)

    # build a map that map image names to unique integer id and apply it on gallery_set and probe_set images
    map_names_to_id = get_map_names_to_id(df_names_ids["img_name"].values)
    probe_ids = convert_names_to_id(probe_set["img_name"].values, map_names_to_id)
    gallery_ids = convert_names_to_id(gallery_set["img_name"].values, map_names_to_id)
    
    probe_features = img_features[probe_ids]
    gallery_features = img_features[gallery_ids]
    
    # map the identities present in the gallery set to the identities present to the probe set
    mask = compute_mask_calfw(probe_set["img_id"], gallery_set["img_id"])

    print("Start evaluation (identification protocol)")
    evaluation(probe_features, gallery_features, mask)
    print("End evaluation")