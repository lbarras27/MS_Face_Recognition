import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from evaluation_util import *
sys.path.insert(0, './datasets/VMER')
from eval_vmer_util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do VMER test')
    # general
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/VMER/imgs', type=str, help='path to load images')
    parser.add_argument('--metadata-path', default='datasets/VMER/metadata', type=str, help='path to metada files')
    parser.add_argument('--result-dir', default='datasets/VMER/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--ethnicity', default='caucasian', type=str, help='')
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size
    
    ethnicity = args.ethnicity

    path_to_pairs_list = metadata_path + "/{0}_pair_list3.txt".format(ethnicity)

    img_name_pair_label = load_pair_list_label_vmer(path_to_pairs_list)
    all_img_names = pd.concat([img_name_pair_label["img_name1"], img_name_pair_label["img_name2"]])
    img_names = all_img_names.unique()
    
    landmarks = get_landmarks_vmer(metadata_path + "/landmarks.txt")

    img_features = get_image_feature(img_path, img_names, landmarks, model_path, batch_size)

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]


    scores = verification(img_features, img_names, img_name_pair_label["img_name1"].values, img_name_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_vmer_{0}.csv".format(ethnicity), scores, delimiter=",")

    labels = img_name_pair_label["label"].values

    print_roc(scores, labels, "VMER", ethnicity)
    
    print(scores.min(), scores.max())
    compute_accuracy_with_best_threshold(scores, labels, 10)