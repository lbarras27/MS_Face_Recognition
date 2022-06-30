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
    parser.add_argument('--image-path', default='datasets/AgeDB/imgs', type=str, help='path to images')
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

    img_name_pair_label = eval_template.load_pair_list_label_agedb(metadata_path + "/pairs_list_gap_{0}.txt".format(gap))
    all_img_names = pd.concat([img_name_pair_label["img_name1"], img_name_pair_label["img_name2"]])
    img_names = all_img_names.unique()
    
    landmarks = eval_template.get_landmarks_vmer(metadata_path + "/landmarks.txt")

    img_features = eval_template.get_image_feature(img_path, img_names, landmarks, model_path, batch_size)

    if use_flip_test:
        img_features = img_features[:, 0:img_features.shape[1] // 2] + img_features[:, img_features.shape[1] // 2:]
        img_features = sklearn.preprocessing.normalize(img_features)
    else:
        img_features = img_features[:, 0:img_features.shape[1] // 2]


    scores = eval_template.verification(img_features, img_names, img_name_pair_label["img_name1"].values, img_name_pair_label["img_name2"].values)
    np.savetxt(result_dir + "/scores_agedb_gap_{0}.csv".format(gap), scores, delimiter=",")

    labels = img_name_pair_label["label"].values
    
    eval_template.print_roc(scores, labels, "AgeDB", "verification_gap_{0}".format(gap), save_path=result_dir)

    print(scores.min(), scores.max())
    eval_template.compute_accuracy_with_best_threshold(scores, labels, 10)