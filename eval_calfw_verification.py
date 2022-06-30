import eval_template

import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do CALFW test')
    # general
    parser.add_argument('--model-prefix', default='model/backbone.pth', help='path to load model.')
    parser.add_argument('--image-path', default='datasets/calfw/imgs', type=str, help='path to load images')
    parser.add_argument('--metadata-path', default='datasets/calfw/metadata', type=str, help='path to metada files')
    parser.add_argument('--result-dir', default='datasets/calfw/results/', type=str, help='path to save results')
    parser.add_argument('--batch-size', default=32, type=int, help='')
   
    args = parser.parse_args()

    model_path = args.model_prefix
    img_path = args.image_path
    metadata_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size

    img_names = eval_template.get_img_names_calfw(img_path)
    landmarks = eval_template.get_landmarks_calfw(metadata_path)

    img_features = eval_template.get_image_feature(img_path, img_names, landmarks, model_path, batch_size)

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    df_pair_label = eval_template.load_pair_list_label_calfw(metadata_path + "/pairs_CALFW.txt")

    scores = eval_template.verification(img_features, img_names, df_pair_label["img_name1"].values, df_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_calfw.csv", scores, delimiter=",")

    labels = df_pair_label["label"].values
    labels[labels > 0] = 1
    labels[labels == 0] = 0
    eval_template.print_roc(scores, labels, "CALFW", "verification")
    
    print(scores.min(), scores.max())
    eval_template.compute_accuracy_with_best_threshold(scores, labels, 10)