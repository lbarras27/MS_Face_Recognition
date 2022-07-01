import pandas as pd
import numpy as np
import os

def load_pair_list_label_calfw(path_file):
    '''
    Load file containing pairs for calfw dataset

        Parameters: 
            path_file (string): path to pairs list file

        Returns: 
            pandas dataframe containing the image names and the corresponding label
    '''
    df = pd.read_csv(path_file, sep=" ", header=None, names=["img", "label"])
    img_name1 = df[::2]["img"].values
    img_name2 = df[1::2]["img"].values
    label = df[::2]["label"].values
    img_name_pair_label = pd.DataFrame({"img_name1": img_name1, "img_name2": img_name2, "label": label})
    
    return img_name_pair_label
    
def get_img_names_calfw(path):
    '''
    Get all image names from the calfw dataset
    
        Parameters: 
            path (string): path to the directory containing all images

        Returns: 
            img_names (list of string): all the image names of calfw dataset
    '''
    img_names = os.listdir(path)
    return img_names
    
def get_names_ids_calfw(path):
    '''
    Get all image names with its corresponding id for calfw dataset
    
        Parameters: 
            path (string): path to the file containing all images with their corresponding ids

        Returns: 
            df_names_ids (pandas dataframe): all the image names with their corresponding ids of calfw dataset
    '''
    df_names_ids = pd.read_csv(path, sep=" ", header=None, names=["img_name", "img_id"])
    return df_names_ids
    
def get_landmarks_calfw(path):
    '''
    Get the 5 landmarks points to the calfw dataset with their corresponding image_name
    
        Parameters: 
            path (string): path to the calfw directory 

        Returns: 
            all_landmarks (dict): that contains the 5 landmarks points and where the keys are the image names
    '''
    path = os.path.join(path, "CA_landmarks")
    landmarks_files = os.listdir(path)
    all_landmarks = {}
    for landmarks_file in landmarks_files:
        with open(os.path.join(path, landmarks_file)) as f:
            landmarks_lines = f.readlines()
        landmarks = np.zeros((5, 2))
        for i, line in enumerate(landmarks_lines):
            lx, ly = line.strip().split()
            landmarks[i, 0] = lx
            landmarks[i, 1] = ly
        img_name = landmarks_file.split("_5loc")[0] + ".jpg"
        all_landmarks[img_name] = landmarks

    return all_landmarks

def compute_mask_calfw(probe_ids, gallery_ids):
    '''
    Compute the mask for calfw (map the same identity from the probe to gallery set)
    
        Parameters: 
            probe_ids (list of int): the probe set ids
            gallery_ids (list of int): the gallery set ids (not used for this dataset)

        Returns: 
            mask (list of int): the gallery ids corresponding to the same identity in the probe set
    '''
    mask = [i for i in range(len(probe_ids))]
    return mask