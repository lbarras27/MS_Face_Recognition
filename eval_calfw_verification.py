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
    parser.add_argument('--pair-name', default='', type=str, help='name of the pair list file')
    parser.add_argument('--result-dir', default='datasets/calfw/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
   
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    pair_filename = args.pair_name
    if pair_filename == '':
        pair_filename = "pairs_CALFW.txt"
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size

    img_names = get_img_names_calfw(img_path)
    landmarks = get_landmarks_calfw(metadata_path)

    # align the images with the help of landmarks points and then compute features vectors
    img_features = get_image_feature(img_path, img_names, landmarks, model_path, batch_size)
    
    # if use_flip_test equal True, sum the features vector of the image with the features vector of the flipped image and normalize.
    # else use only the features vector of the image.
    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    df_pair_label = load_pair_list_label_calfw(metadata_path + "/" + pair_filename)

    # compute scores for verification protocol for each image features and store the results
    scores = verification(img_features, img_names, df_pair_label["img_name1"].values, df_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_calfw.csv", scores, delimiter=",")

    labels = df_pair_label["label"].values
    labels[labels > 0] = 1
    labels[labels == 0] = 0
    print_roc(scores, labels, "CALFW", "verification", save_path=result_dir)
    
    print(scores.min(), scores.max())
    compute_accuracy_with_best_threshold(scores, labels, 10)