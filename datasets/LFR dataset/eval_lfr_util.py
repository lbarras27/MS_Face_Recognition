import pandas as pd
import numpy as np

def load_pair_list_label_lfr(path_file):
    '''
    Load file containing pairs for lfr dataset
    
        Parameters: 
            path_file (string): path to pairs list file

        Returns: 
            pandas dataframe containing the image names and the corresponding label
    '''
    img_name_pair_label = pd.read_csv(path_file, sep=",", header=None, names=["img_name1", "img_name2", "label"])
    return img_name_pair_label
    
def get_img_names_lfr(path_to_pair_file):
    '''
    Get all image names used in pairs file for lfr dataset
    
        Parameters: 
            path_to_pair_file (string): path to the file containing pairs

        Returns: 
            unique_img_names (list of string): all the image names used in pairs file for lfr dataset
    '''
    img_name_pair_label = pd.read_csv(path_to_pair_file, sep=",", header=None, names=["img_name1", "img_name2", "label"])
    all_img_names = pd.concat([img_name_pair_label["img_name1"], img_name_pair_label["img_name2"]])
    unique_img_names = all_img_names.unique()
    return unique_img_names
    
def compute_mask_lfr(probe_ids, gallery_ids):
    '''
    Compute the mask for lfr (map the same identity from the probe to gallery set)
    
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