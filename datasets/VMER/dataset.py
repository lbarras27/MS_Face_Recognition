import os, sys
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image
import cv2

def get_identities_imgs_name(path, ethnicity_ids):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values. 
    
        Parameters: 
            path (string): path to the folder that contains all the images
            ethnicity_ids (list): a list that contains people id to use

        Returns: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value
    '''
    
    identities = ethnicity_ids
    identities_imgs_name = {}
    for identity in identities:
        imgs_name = []
        for img_name in os.listdir(os.path.join(path, identity)):
            imgs_name.append(img_name)
        identities_imgs_name[identity] = imgs_name
    
    return identities_imgs_name
    
# same that the previous one but use only identities that have landmarks points.
def get_identities_imgs_name2(path, ethnicity_ids):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values. 
    
        Parameters: 
            path (string): path to the folder that contains all the images
            ethnicity_ids (list): a list that contains people id to use

        Returns: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value
    '''
    
    with open(os.path.join(path, "landmarks.txt"), "r") as f:
        lines = f.readlines()

    img_names = []
    for line in lines:
        img_names.append(line.split()[0])
        
    identities = {}
    for img_name in img_names:
        ident_name = img_name.split("/")
        if ident_name[0] in identities:
            identities[ident_name[0]].append(ident_name[1])
        else:
            identities[ident_name[0]] = [ident_name[1]]
    
    if ethnicity_ids is not None:
        identities_imgs_name = {}
        for id, list_names in identities.items():
            if id in ethnicity_ids:
                identities_imgs_name[id] = list_names
    else:
        identities_imgs_name = identities
            
    return identities_imgs_name

    
    
def generate_pairs(identities_imgs_name, n=1):
    '''
    Generate randomly n positive pairs and n negative pairs for each identities.
    
        Parameters: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value
            n (int): number of positive and negative pairs to generate by identity

        Returns: 
            pairs_same (np.array of size n): all the positives pairs
            pairs_diff (np.array of size n): all the negatives pairs
    '''
    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue
        
        pair_same = np.array(v)[np.random.choice(len(v), n*2, replace=False)]
        
        for i in range(0, len(pair_same), 2):
            pairs_same.append(["{0}/{1}".format(k, pair_same[i]), "{0}/{1}".format(k, pair_same[i+1])])

        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
            
        other_ids = np.random.choice(id_list, n, replace=True)
        
        for i in range(0, len(other_ids)):
            other_id = other_ids[i]
            if len(identities_imgs_name[other_id]) == 0:
                continue
            first_elem = pair_same[i]
            second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
            pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)
    
    return pairs_same, pairs_diff
    

def write_pairs(identities_pairs, identities_pairs_not_same, output="out.txt"):
    '''
    Write the positives and negatives pairs in the file output with their corresponding label (1 positive, 0 negative).
    
        Parameters: 
            identities_pairs (np.array of size n): the positives pairs
            identities_pairs_not_same (np.array of size n): the negatives pairs
            output (string): name of the output file
    '''
    with open(output, "w") as f:
        for i in range(len(identities_pairs)):
            f.write(identities_pairs[i][0] + " " + identities_pairs[i][1] + " 1")
            f.write('\n')
        
        for i in range(len(identities_pairs_not_same)):
            f.write(identities_pairs_not_same[i][0] + " " + identities_pairs_not_same[i][1] + " 0")
            f.write('\n')
            
            
def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", ethnicity="african"):
    '''
    Generate the probe set and the gallery set files with images of only people in the specified ethnicity.  
    
        Parameters: 
            path (string): path to the file that contains all the image names (file generated by the previous method)
            output_gallery (string): name of the output gallery set file
            output_probe (string): name of the output probe set file
            ethnicity (string): ethnicity in the gallery and probe set (african, caucasian, asian, indian)
            

    '''
    
    df = pd.read_xml("metadata/finalTest.xml", parser="etree")
    if ethnicity == "african":
        ethnicity_ids = df[df["ethnicity"] == 1]["id"].values.tolist()
    elif ethnicity == "asian":
        ethnicity_ids = df[df["ethnicity"] == 2]["id"].values.tolist()
    elif ethnicity == "indian":
        ethnicity_ids = df[df["ethnicity"] == 4]["id"].values.tolist()
    elif ethnicity == "caucasian":
        ethnicity_ids = df[df["ethnicity"] == 3]["id"].values.tolist()
    
    all_identities_img_names = get_identities_imgs_name2(path=path, ethnicity_ids=None)
    identities_img_names = get_identities_imgs_name2(path=path, ethnicity_ids=ethnicity_ids)

    probe_set = []
    gallery_set = []
    ids_gallery = []
    for id, img_names in all_identities_img_names.items():
        gallery_set.append(id + "/" + np.array(img_names)[np.random.choice(len(img_names), 1)[0]])
        ids_gallery.append(id)
    
    ids_probe = []
    for id, img_names in identities_img_names.items():
        imgs_sel = np.array(img_names)[np.random.choice(len(img_names), 50, replace=False)]
        for i in range(len(imgs_sel)):
            probe_set.append(id + "/" + imgs_sel[i])
            ids_probe.append(id)
    
    map_id = {}
    for i, id in enumerate(ids_gallery):
        map_id[id] = i
    
    ids_gallery = [map_id[id] for id in ids_gallery]
    ids_probe = [map_id[id] for id in ids_probe]
    
    with open(output_gallery, "w") as f:
        for i in range(len(ids_gallery)):
            f.write("{0} {1}".format(gallery_set[i], ids_gallery[i]))
            f.write("\n")

    with open(output_probe, "w") as f:
        for i in range(len(ids_probe)):
            f.write("{0} {1}".format(probe_set[i], ids_probe[i]))
            f.write("\n")
            
    
def extract_landmarks_points(img_path, output="metadata/landmarks.txt", device="cuda:0"):
    '''
    Extract landmarks points with MTCNN and save in output.
    
        Parameters: 
            path_img (string): path to the images
            output (string): outpout file that will contains landmarks points
            device (string): device to use in the model (cpu, cuda)

    '''
    mtcnn = MTCNN(keep_all=True, device=device)
    all_imgs_names = []
    for identity in os.listdir(img_path):
        imgs_identity = os.listdir(img_path + "/" + identity)
        imgs_identity = [identity+"/"+img for img in imgs_identity]
        all_imgs_names += imgs_identity
        
    with open(output, "w") as f:
        for i in range(len(all_imgs_names)):
            imgs = Image.open(img_path + "/" + all_imgs_names[i])
            boxes, probs, landmarks = mtcnn.detect(imgs, landmarks=True)
            
            if landmarks is not None and len(landmarks) == 1:
                f.write(all_imgs_names[i]+" ")
                f.write(" ".join(landmarks.flatten().astype(str)))
                f.write("\n")
            
