import os, sys
import numpy as np
import pandas as pd

def get_identities_imgs_name(path="LFR dataset", dir=None):
    
    identities = os.listdir(path)
    identities_imgs_name = {}
    for identity in identities:
        path_flr = os.path.join(path, identity)
        flr = os.listdir(path_flr)
        imgs_name = []
        for pose in flr:
            for img_name in os.listdir(os.path.join(path_flr, pose)):
                imgs_name.append(pose+"/"+img_name)
        identities_imgs_name[identity] = imgs_name
    if dir == None:
        return identities_imgs_name
    
    test_string = dir
    lfr_identities_imgs_name = {}
    for k, v in identities_imgs_name.items():
        lfr_imgs = []
        for img in v:
            if test_string in img:
                lfr_imgs.append(img)
        lfr_identities_imgs_name[k] = lfr_imgs
    
    return lfr_identities_imgs_name
    
 
def generate_pairs(identities_imgs_name):

    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue

        pair_same = np.array(v)[np.random.choice(len(v), 2, replace=False)]
        pairs_same.append(["{0}/{1}".format(k, pair_same[0]), "{0}/{1}".format(k, pair_same[1])])
        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
        other_id = np.random.choice(id_list, 1)[0]
        
        if len(identities_imgs_name[other_id]) == 0:
            continue
        first_elem = pair_same[0]
        second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
        pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)
    
    return pairs_same, pairs_diff

def generate_pairs_mixed(identities_imgs_name, dir1, dir2):

    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue
        lfr = {dir1: [], dir2: []}
        for img_name in v:

            if dir1 in img_name:
                lfr[dir1].append(img_name)
            elif dir2 in img_name:
                lfr[dir2].append(img_name)
        
        if len(lfr[dir1]) == 0 or len(lfr[dir2]) == 0:
            continue

        pair_same1 = np.array(lfr[dir1])[np.random.choice(len(lfr[dir1]), 1)]
        pair_same2 = np.array(lfr[dir2])[np.random.choice(len(lfr[dir2]), 1)]
        pairs_same.append(["{0}/{1}".format(k, pair_same1[0]), "{0}/{1}".format(k, pair_same2[0])])
        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
        other_id = np.random.choice(id_list, 1)[0]
        
        if len(identities_imgs_name[other_id]) == 0:
            continue

        lfr_other = {dir1: [], dir2: []}
        for img_name in identities_imgs_name[other_id]:

            if dir1 in img_name:
                lfr_other[dir1].append(img_name)
            elif dir2 in img_name:
                lfr_other[dir2].append(img_name)
        
        if len(lfr_other[dir2]) == 0: # or len(lfr_other[dir1]) == 0:
            continue

        first_elem = pair_same1[0]
        second_elem = np.array(lfr_other[dir2])[np.random.choice(len(lfr_other[dir2]), 1)[0]]
        pairs_diff.append(["{0}/{1}".format(k, first_elem), "{0}/{1}".format(other_id, second_elem)])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)
    
    return pairs_same, pairs_diff
    

def write_pairs(identities_pairs, identities_pairs_not_same, output="out.txt"):
    with open(output, "w") as f:
        for i in range(len(identities_pairs)):
            f.write(identities_pairs[i][0] + "," + identities_pairs[i][1] + ",1")
            f.write('\n')
        
        for i in range(len(identities_pairs_not_same)):
            f.write(identities_pairs_not_same[i][0] + "," + identities_pairs_not_same[i][1] + ",0")
            f.write('\n')
            

def generate_file_with_all_img_names_ids(path, output_file):
    map_id_imgs = get_identities_imgs_name(path)
    map_id_newid = dict(zip(map_id_imgs.keys(), range(len(map_id_imgs))))
    tups_img_id = [(id+"/"+img, map_id_newid[id]) for id, imgs in map_id_imgs.items() for img in imgs]

    with open(output_file, "w") as f:
        for i in range(len(tups_img_id)):
            f.write(tups_img_id[i][0] + "," + str(tups_img_id[i][1]))
            f.write('\n')

    print("{} generated".format(output_file))


def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", pose_gallery="front", pose_probe="left"):
    df_names_ids = pd.read_csv(path, sep=",", header=None, names=["img_name", "img_id"])
    map_id_names = df_names_ids.groupby("img_id").apply(lambda x: x["img_name"].values).to_dict()
    probe_set = []
    gallery_set = []
    ids_gallery = []
    ids_probe = []
    for id, names_list in map_id_names.items():
        if len(names_list) == 1:
            print(id)
            continue

        if pose_probe != pose_gallery:
            names_list_gallery = [name for name in names_list if pose_gallery in name]
            if len(names_list_gallery) < 1:
                print("gallery:" + str(id))
                continue
            
            img_name_gallery_choice = np.random.choice(names_list_gallery, 1)
            gallery_set.append(img_name_gallery_choice[0])
            ids_gallery.append(id)

            names_list_probe = [name for name in names_list if pose_probe in name]
            if len(names_list_probe) < 1:
                print(id)
                continue

            img_name_probe_choice = np.random.choice(names_list_probe, 1)
            probe_set.append(img_name_probe_choice[0])
            ids_probe.append(id)

        else:
            names_list_filter = [name for name in names_list if pose_gallery in name]
            if len(names_list_filter) < 1:
                print(id)
                continue
            
            if len(names_list_filter) > 1:
                img_names_choice = np.random.choice(names_list_filter, 2, replace=False)
                gallery_set.append(img_names_choice[0])
                probe_set.append(img_names_choice[1])
                ids_gallery.append(id)
                ids_probe.append(id)
            else:
                img_name_choice = np.random.choice(names_list_filter, 1)
                gallery_set.append(img_name_choice[0])
                ids_gallery.append(id)

    with open(output_gallery, "w") as f:
        for i in range(len(ids_gallery)):
            f.write("{0},{1}".format(gallery_set[i], ids_gallery[i]))
            f.write("\n")

    with open(output_probe, "w") as f:
        for i in range(len(ids_probe)):
            f.write("{0},{1}".format(probe_set[i], ids_probe[i]))
            f.write("\n")