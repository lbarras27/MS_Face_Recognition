import pandas as pd
import numpy as np
import os

def load_pair_list_label_rfw(path_file, ethnicity):
    '''
    Load file containing pairs for rfw dataset
    
        Parameters: 
            path_file (string): path to pairs list file
            ethnicity (string): the ethnicity (African, Asian, Caucasian, Indian)

        Returns: 
            pandas dataframe containing the image names and the corresponding label
    '''
    path_to_file = os.path.join(path_file, "{0}/{1}_pairs.txt".format(ethnicity, ethnicity))
    

    pairs = []
    with open(path_to_file) as f:
        lines = f.readlines()

    for line in lines:
        pairs.append(line.split())

    path = "/test/data/{0}".format(ethnicity)
    pairs_names_label = []
    for p in pairs:
        if len(p) == 3:
            path_pair = path + "/" + p[0] + "/"
            pair1 = path_pair + p[0] + "_{0:04d}".format(int(p[1])) + ".jpg"
            pair2 = path_pair + p[0] + "_{0:04d}".format(int(p[2])) + ".jpg"
            pairs_names_label.append([pair1, pair2, 1])
        else:
            path_pair1 = path + "/" + p[0] + "/"
            pair1 = path_pair1 + p[0] + "_{0:04d}".format(int(p[1])) + ".jpg"
            path_pair2 = path + "/" + p[2] + "/"
            pair2 = path_pair2 + p[2] + "_{0:04d}".format(int(p[3])) + ".jpg"
            pairs_names_label.append([pair1, pair2, 0])
    
    return pd.DataFrame(pairs_names_label, columns=["img_name1", "img_name2", "label"])
    
def get_img_name_landmarks_rfw(path):
    '''
    Get the 5 landmarks points to the rfw dataset with their corresponding image_name
    
        Parameters: 
            path (string): path to the landmark file

        Returns:
            img_names (list of string): names of the images
            landmarks (dict): that contains the 5 landmarks points and where the keys are the image names
    '''
    #path_landmark = "txts/African/African_lmk.txt"
    img_names_landmarks = []
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        img_names_landmarks.append(line.split())
        
    landmarks = {}
    for img_name_land in img_names_landmarks:
        landmark = np.array([float(land_float) for land_float in img_name_land[2:]]).reshape((5, 2))
        landmarks[img_name_land[0]] = landmark
    
    img_names = [col[0] for col in img_names_landmarks]

    return img_names, landmarks
    
def compute_mask_rfw(probe_ids, gallery_ids):
    '''
    Compute the mask for rfw (map the same identity from the probe to gallery set)
    
        Parameters: 
            probe_ids (list of int): the probe set ids
            gallery_ids (list of int): the gallery set ids (not used for this dataset)

        Returns: 
            mask (list of int): the gallery ids corresponding to the same identity in the probe set
    '''
    mask = [i for i in range(len(probe_ids))]
    return mask