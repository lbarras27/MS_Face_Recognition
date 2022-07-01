import os, sys
import numpy as np
import pandas as pd

def get_identities_imgs_name(path="rec2img_masked", mask=None):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values. mask filter the images to get only 
    images with the occlusion type specified.
    
        Parameters: 
            path (string): path to the folder that contains all the images
            dir (string): the occlusion type (original (without occ.), glasses, surgical_mask, all), if None, all images

        Returns: 
            masked_identities_imgs_name (dict): contains identities as keys and images of the identity as value
    '''
    
    identities = os.listdir(path)
    identities_imgs_name = {}
    for identity in identities:
        imgs_name = []
        for img_name in os.listdir(os.path.join(path, identity)):
            imgs_name.append(img_name)
        identities_imgs_name[identity] = imgs_name
    if mask == None:
        return identities_imgs_name
    
    test_string = "_" if mask == "all" else "_"+mask
    masked_identities_imgs_name = {}
    for k, v in identities_imgs_name.items():
        masked_imgs = []
        for img in v:
            if test_string in img:
                masked_imgs.append(img)
        masked_identities_imgs_name[k] = masked_imgs
    
    return masked_identities_imgs_name


def get_original_identities_imgs_name(path="rec2img_masked"):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values.
    
        Parameters: 
            path (string): path to the folder that contains all the images

        Returns: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value
    '''
    
    identities = os.listdir(path)
    identities_imgs_name = {}
    for identity in identities:
        imgs_name = []
        for img_name in os.listdir(os.path.join(path, identity)):
            if '_' not in img_name:
                imgs_name.append(img_name)
        identities_imgs_name[identity] = imgs_name
    
    return identities_imgs_name
    
 
def generate_pairs(identities_imgs_name):
    '''
    Generate randomly one positive pair and one negative pair for each identities.
    
        Parameters: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value

        Returns: 
            pairs_same (np.array of size n): all the positives pairs
            pairs_diff (np.array of size n): all the negatives pairs
    '''

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


def generate_file_with_all_identities_names(path, outpout_file):
    '''
    Generate a file that contains all the image names and an unique id for each identity.
    
        Parameters: 
            path (string): path to the folder that contains all the images
            output_file(string): name of the generated output file

    '''
    
    identities = os.listdir(path)
    
    all_img_names = []
    ids = []
    for i, id in enumerate(identities):
        imgs = os.listdir(os.path.join(path, id))
        img_names = [id+"/"+img for img in imgs]
        all_img_names += img_names
        ids += [i]*len(imgs)

    with open(outpout_file, "w") as f:
        for i in range(len(all_img_names)):
            f.write(all_img_names[i] + " " + str(ids[i]))
            f.write('\n')

    print("{} generated".format(outpout_file))


def get_only_identities_in_pair_list(path_to_pair_list):
    '''
    Get only the image names present in the pair list file and take care to not have duplicate.
    
        Parameters: 
            path_to_pair_list (string): path to the pair list file

        Returns: 
            unique_img_names (np.array of size n): array that contains all unique image names
            
    '''
    
    img_name_pair_label = pd.read_csv(path_to_pair_list, sep=" ", header=None, names=["img_name1", "img_name2", "label"])
    all_img_names = pd.concat([img_name_pair_label["img_name1"], img_name_pair_label["img_name2"]])
    unique_img_names = all_img_names.unique()
    return unique_img_names


def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", filter_gallery="original", filter_probe="glass"):
    '''
    Generate the probe set with images with occlusion type of filter_probe and the gallery set with images with occlusion type of filter_gallery 
    and write the probe and gallery set in two files respectively.
    
        Parameters: 
            path (string): path to the file that contains all the image names (file generated by the previous method: generate_file_with_all_identities_names())
            output_gallery (string): name of the output gallery set file
            output_probe (string): name of the output probe set file
            filter_gallery (string): occlusion_type in the gallery set (original (without occ.), glasses, surgical_mask, all)
            filter_probe (string): occlusion type in the probe set (original (without occ.), glasses, surgical_mask, all)

    '''
    
    df_names_ids = pd.read_csv(path, sep=" ", header=None, names=["img_name", "img_id"])
    map_id_names = df_names_ids.groupby("img_id").apply(lambda x: x["img_name"].values).to_dict()
    probe_set = []
    gallery_set = []
    ids_gallery = []
    ids_probe = []
    for id, names_list in map_id_names.items():
        if len(names_list) == 1:
            print(id)
            continue

        if filter_probe != filter_gallery:
            if filter_gallery == "all":
                names_list_gallery = [name for name in names_list if "_" in name.split("/")[1]]
            elif filter_gallery != "original":
                names_list_gallery = [name for name in names_list if filter_gallery in name.split("/")[1]]
            else:
                names_list_gallery = [name for name in names_list if "_" not in name.split("/")[1]]

            if len(names_list_gallery) < 1:
                print(id)
                continue
            
            img_name_gallery_choice = np.random.choice(names_list_gallery, 1)
            gallery_set.append(img_name_gallery_choice[0])
            ids_gallery.append(id)

            if filter_probe == "all":
                names_list_probe = [name for name in names_list if "_" in name.split("/")[1]]
            if filter_probe != "original":
                names_list_probe = [name for name in names_list if filter_probe in name.split("/")[1]]
            else:
                names_list_probe = [name for name in names_list if "_" not in name.split("/")[1]]

            if len(names_list_probe) < 1:
                print(id)
                continue

            img_name_probe_choice = np.random.choice(names_list_probe, 1)
            probe_set.append(img_name_probe_choice[0])
            ids_probe.append(id)
        else:
            if filter_gallery == "all":
                names_list_filter = [name for name in names_list if "_" in name.split("/")[1]]
            elif filter_gallery != "original":
                names_list_filter = [name for name in names_list if filter_gallery in name.split("/")[1]]
            else:
                names_list_filter = [name for name in names_list if "_" not in name.split("/")[1]]

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
                img_names_choice = np.random.choice(names_list_filter, 1)
                gallery_set.append(img_names_choice[0])
                ids_gallery.append(id)

    with open(output_gallery, "w") as f:
        for i in range(len(ids_gallery)):
            f.write("{0} {1}".format(gallery_set[i], ids_gallery[i]))
            f.write("\n")

    with open(output_probe, "w") as f:
        for i in range(len(ids_probe)):
            f.write("{0} {1}".format(probe_set[i], ids_probe[i]))
            f.write("\n")


# def generate_file_with_all_img_names_ids(path, output_file):
#     map_id_imgs = get_identities_imgs_name(path)
#     map_id_newid = dict(zip(map_id_imgs.keys(), range(len(map_id_imgs))))
#     tups_img_id = [(img, map_id_newid[id]) for id, imgs in map_id_imgs.items() for img in imgs]

#     with open(output_file, "w") as f:
#         for i in range(len(tups_img_id)):
#             f.write(tups_img_id[i][0] + " " + str(tups_img_id[i][1]))
#             f.write('\n')

#     print("{} generated".format(output_file))
            
