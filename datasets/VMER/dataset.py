import os, sys
import numpy as np
import pandas as pd

def get_identities_imgs_name(path="../test", ethnicity_ids=None):
    
    identities = ethnicity_ids
    identities_imgs_name = {}
    for identity in identities:
        imgs_name = []
        for img_name in os.listdir(os.path.join(path, identity)):
            imgs_name.append(img_name)
        identities_imgs_name[identity] = imgs_name
    
    return identities_imgs_name
    
def get_identities_imgs_name2(path=".", ethnicity_ids=None):
    
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

    
 
def generate_pairs(identities_imgs_name):

    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue
        
        if len(v) >= 4:
            pair_same = np.array(v)[np.random.choice(len(v), 4, replace=False)]
            pairs_same.append(["{0}/{1}".format(k, pair_same[0]), "{0}/{1}".format(k, pair_same[1])])
            pairs_same.append(["{0}/{1}".format(k, pair_same[2]), "{0}/{1}".format(k, pair_same[3])])
        else:
            pair_same = np.array(v)[np.random.choice(len(v), 2, replace=False)]
            pairs_same.append(["{0}/{1}".format(k, pair_same[0]), "{0}/{1}".format(k, pair_same[1])])
        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)

        if len(v) >= 4:
            other_ids = np.random.choice(id_list, 2, replace=False)
            other_id = other_ids[0]
            if len(identities_imgs_name[other_id]) == 0:
                continue
            first_elem = pair_same[0]
            second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
            pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])

            other_id = other_ids[1]
            if len(identities_imgs_name[other_id]) == 0:
                continue
            first_elem = pair_same[2]
            second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
            pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])
        else:
            other_id = np.random.choice(id_list, 1)[0]
            if len(identities_imgs_name[other_id]) == 0:
                continue
            first_elem = pair_same[0]
            second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
            pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)
    
    return pairs_same, pairs_diff
    
def generate_pairs2(identities_imgs_name):

    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue
        
        pair_same = np.array(v)[np.random.choice(len(v), 80, replace=False)]
        
        for i in range(0, len(pair_same), 2):
            pairs_same.append(["{0}/{1}".format(k, pair_same[i]), "{0}/{1}".format(k, pair_same[i+1])])

        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
            
        other_ids = np.random.choice(id_list, 40, replace=True)
        
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
    with open(output, "w") as f:
        for i in range(len(identities_pairs)):
            f.write(identities_pairs[i][0] + " " + identities_pairs[i][1] + " 1")
            f.write('\n')
        
        for i in range(len(identities_pairs_not_same)):
            f.write(identities_pairs_not_same[i][0] + " " + identities_pairs_not_same[i][1] + " 0")
            f.write('\n')
            
            
def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", ethnicity="african"):
    df = pd.read_xml("VMER_dataset/finalTest.xml", parser="etree")
    if ethnicity == "african":
        ethnicity_ids = df[df["ethnicity"] == 1]["id"].values.tolist()
    elif ethnicity == "asian":
        ethnicity_ids = df[df["ethnicity"] == 2]["id"].values.tolist()
    elif ethnicity == "indian":
        ethnicity_ids = df[df["ethnicity"] == 4]["id"].values.tolist()
    elif ethnicity == "caucasian":
        ethnicity_ids = df[df["ethnicity"] == 3]["id"].values.tolist()
    
    all_identities_img_names = get_identities_imgs_name2(path=".", ethnicity_ids=None)
    identities_img_names = get_identities_imgs_name2(path=".", ethnicity_ids=ethnicity_ids)

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
            
