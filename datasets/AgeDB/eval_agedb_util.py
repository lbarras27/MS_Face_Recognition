import pandas as pd
import os
import numpy as np

def load_pair_list_label_agedb(path_file):
    '''
    Load file containing pairs for AgeDB dataset
    
        Parameters: 
            path_file (string): path to pairs list file

        Returns: 
            pandas dataframe containing the image names and the corresponding label
    '''
    img_name_pair_label = pd.read_csv(path_file, sep=" ", header=None, names=["img_name1", "img_name2", "label"])
    return img_name_pair_label
    
def get_img_names_agedb(path):
    '''
    Get all image names from the AgeDB dataset
    
        Parameters: 
            path (string): path to the directory containing all images

        Returns: 
            img_names (list of string): all the image names of calfw dataset
    '''
    img_names = os.listdir(path)
    return img_names
    
def compute_mask_agedb(probe_ids, gallery_ids):
    '''
    Compute the mask for AgeDB (map the same identity from the probe to gallery set)
    
        Parameters: 
            probe_ids (list of int): the probe set ids
            gallery_ids (list of int): the gallery set ids (not used for this dataset)

        Returns: 
            mask (list of int): the gallery ids corresponding to the same identity in the probe set
    '''
    mask = [i for i in range(len(probe_ids))]
    return mask
    
def get_landmarks_agedb(path):
    '''
    Get the 5 landmarks points to the agedb dataset with their corresponding image_name
    
        Parameters: 
            path (string): path to the landmarks file

        Returns: 
            all_landmarks (dict): that contains the 5 landmarks points and where the keys are the image names
    '''
    all_landmarks = {}
    with open(path) as f:
        lines = f.readlines()
        
    for line in lines:
        name_landmarks = line.split()
        all_landmarks[name_landmarks[0]] = np.array(name_landmarks[1:], dtype=float).reshape((5, 2))
        
    return all_landmarks