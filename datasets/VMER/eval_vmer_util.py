import pandas as pd
import numpy as np

def load_pair_list_label_vmer(path_file):
    '''
    Load file containing pairs for vmer dataset
    
        Parameters: 
            path_file (string): path to pairs list file

        Returns: 
            pandas dataframe containing the image names and the corresponding label
    '''
    img_name_pair_label = pd.read_csv(path_file, sep=" ", header=None, names=["img_name1", "img_name2", "label"])
    return img_name_pair_label
    
def get_landmarks_vmer(path):
    '''
    Get the 5 landmarks points to the vmer dataset with their corresponding image_name
    
        Parameters: 
            path (string): path to the vmer landmarks file

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
    
def compute_mask_vmer(probe_ids, gallery_ids):
    '''
    Compute the mask for vmer (map the same identity from the probe to gallery set)
    
        Parameters: 
            probe_ids (list of int): the probe set ids
            gallery_ids (list of int): the gallery set ids

        Returns: 
            mask (list of int): the gallery ids corresponding to the same identity in the probe set
    '''
    mask = []
    for query_id in probe_ids:
        pos = [i for i, x in enumerate(gallery_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask