import eval_template

import sklearn
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
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
    metada_path = args.metadata_path
    result_dir = args.result_dir
    gpu_id = None
    use_flip_test = True
    batch_size = args.batch_size
    
    pose = args.pose

    path_to_pairs_list = metada_path + "/lfr_front_{}_pairs_list.txt".format(pose)

    img_names = eval_template.get_img_names_lfr(path_to_pairs_list)

    img_features = eval_template.get_image_feature(img_path, img_names, None, model_path, batch_size, already_align=True)

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]

    df_pair_label = eval_template.load_pair_list_label_lfr(path_to_pairs_list)

    scores = eval_template.verification(img_features, img_names, df_pair_label["img_name1"].values, df_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_lfr_front_{0}.csv".format(pose), scores, delimiter=",")

    labels = df_pair_label["label"].values

    eval_template.print_roc(scores, labels, "LFR", "front-{}".format(pose))
    
    print(scores.min(), scores.max())
    eval_template.compute_accuracy_with_best_threshold(scores, labels, 10)