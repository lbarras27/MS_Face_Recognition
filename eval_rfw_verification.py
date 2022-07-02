import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from evaluation_util import *
sys.path.insert(0, './datasets/RFW')
from eval_rfw_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do RFW test')
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/RFW', type=str, help='path to load images')
    parser.add_argument('--metadata-path', default='datasets/RFW/test/txts', type=str, help='path to metada files')
    parser.add_argument('--result-dir', default='datasets/RFW/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--ethnicity', default='Asian', type=str, help='')
    args = parser.parse_args()

    ethnicity = args.ethnicity
    
    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size
    

    img_names, landmarks = get_img_name_landmarks_rfw(metadata_path + "/{0}/{1}_lmk.txt".format(ethnicity, ethnicity))

    # align the images with the help of landmarks points and then compute features vectors
    img_features = get_image_feature(img_path, img_names, landmarks, model_path, batch_size)

    # if use_flip_test equal True, sum the features vector of the image with the features vector of the flipped image and normalize.
    # else use only the features vector of the image.
    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    df_pair_label = load_pair_list_label_rfw(metadata_path, ethnicity)

    # compute scores for verification protocol for each image features and store the results
    scores = verification(img_features, img_names, df_pair_label["img_name1"].values, df_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_rfw_{0}.csv".format(ethnicity), scores, delimiter=",")

    labels = df_pair_label["label"].values
    labels[labels > 0] = 1
    labels[labels == 0] = 0
    print_roc(scores, labels, "RWF", ethnicity, save_path=result_dir)

    print(scores.min(), scores.max())
    compute_accuracy_with_best_threshold(scores, labels, 10)