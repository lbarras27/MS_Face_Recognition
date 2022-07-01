import numpy as np
import pandas as pd
import os
import cv2

def get_identities_imgs_name(path="."):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values.
    
        Parameters: 
            path (string): path to the folder that contains all the images

        Returns: 
            identities_imgs_name (dict): contains identities as keys and images of the identity as value
    '''
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)

    # dic with identities as key and array of img names sorted by age as values
    identities_imgs_name = new_df.sort_values(["name", "age"]).groupby("name").apply(lambda x: x["img_name"].values.tolist()).to_dict()

    return identities_imgs_name


def get_identities_with_gap_pairs_imgs(path, gap=5):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values.
    
        Parameters: 
            path (string): path to the folder that contains all the images
            gap (int): age gap between the pairs

        Returns: 
            dic_final (dict): contains identities as keys and pair image names of the identity with gap n as value
    '''
    
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)
    new_df = new_df.sort_values(["name", "age"])

    dico = new_df.groupby("name").apply(lambda x: x[["age", "img_name"]].values.tolist()).to_dict()
    dic_final = {}
    for id in dico.keys():
        gap_n = []
        for age1 in dico[id]:
            for age2 in dico[id]:
                if (age1[0] - age2[0]) == gap:
                    gap_n.append([age1[1], age2[1]])
        dic_final[id] = gap_n

    return dic_final
    
# same that previous one but only have images that have landmarks points.
def get_identities_with_gap_pairs_imgs2(path, gap=5):
    '''
    Create a dictionnary that contains the identities as keys and the 
    images of these identities as values.
    
        Parameters: 
            path (string): path to the folder that contains all the images
            gap (int): age gap between the pairs

        Returns: 
            dic_final (dict): contains identities as keys and pair image names of the identity with gap n as value
    '''
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)
    new_df = new_df.sort_values(["name", "age"])
    
    with open(os.path.join(path, "metadata/landmarks.txt"), "r") as f:
        lines = f.readlines()
    img_names = []
    for line in lines:
        img_names.append(line.split()[0])

    dico = new_df.groupby("name").apply(lambda x: x[["age", "img_name"]].values.tolist()).to_dict()
    dic_final = {}
    for id in dico.keys():
        gap_5 = []
        for age1 in dico[id]:
            for age2 in dico[id]:
                if (age1[0] - age2[0]) == gap:
                    if age1[1] in img_names and age2[1] in img_names:
                        gap_5.append([age1[1], age2[1]])
        dic_final[id] = gap_5

    return dic_final

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


def generate_pairs_gap(path, output, gap=5):
    '''
    Generate randomly one positive and one negative pair with age gap n for each identities and save in 
    the output file.
    
        Parameters: 
            path (string): contains identities as keys and images of the identity as value
            output (string): name of the output file
            gap (int): the age gap between the pairs 
     
    '''
    
    identities_imgs_name = get_identities_with_gap_pairs_imgs2(path, gap)
    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0:
            continue

        pair_same = np.array(v)[np.random.choice(len(v), 1)[0]]
        pairs_same.append(pair_same)
        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
        other_id = np.random.choice(id_list, 1)[0]
        
        if len(identities_imgs_name[other_id]) == 0:
            continue
        first_elem = np.array(v)[np.random.choice(len(v), 1)[0]][0]
        second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]][0]
        pairs_diff.append([first_elem, second_elem])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)

    write_pairs(pairs_same, pairs_diff, output)


def generate_file_with_all_img_names_ids(path, output_file):
    '''
    Generate a file that contains all the image names and an unique id for each identity.
    
        Parameters: 
            path (string): path to the folder that contains all the images
            output_file(string): name of the generated output file

    '''
    identities_names = get_identities_imgs_name(path)
    with open(output_file, "w") as f:
        cnt = 0
        for name, img_names in identities_names.items():
            for img_name in img_names:
                f.write(img_name + " " + str(cnt))
                f.write('\n')
            cnt += 1

    print("{} generated".format(output_file))

def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", gap=5):
    '''
    Generate a probe set and a gallery set with an age gap of n between images of the same identity
    and write the probe and gallery set in two files respectively.
    
        Parameters: 
            path (string): path to the file that contains all the image names (file generated by the previous method)
            output_gallery (string): name of the output gallery set file
            output_probe (string): name of the output probe set file
            gap (int): the age gap in the probe set and the gallery set between the same identity

    '''
    identities_imgs_name = get_identities_with_gap_pairs_imgs2(path, gap)

    probe_set = []
    gallery_set = []
    
    for id, v in identities_imgs_name.items():
        if len(v) < 1:
            print(id)
            continue
        pair_same = np.array(v)[np.random.choice(len(v), 1)[0]]
        
        gallery_set.append(pair_same[0])
        probe_set.append(pair_same[1])
    
    ids = [i for i in range(len(gallery_set))]
        
    with open(output_gallery, "w") as f:
        for i in range(len(ids)):
            f.write("{0} {1}".format(gallery_set[i], ids[i]))
            f.write("\n")

    with open(output_probe, "w") as f:
        for i in range(len(ids)):
            f.write("{0} {1}".format(probe_set[i], ids[i]))
            f.write("\n")
            
def solve_caracter_encoding_img_names(path_img_names="imgs"):
    '''
    Solve caracter encoding for the image names in the dataset.
    
        Parameters: 
            path_img_names (string): path to the images

    '''
    
    for img_name in os.listdir(path_img_names):
        new_name = img_name
        if u"\u041a" in img_name:
            new_name = img_name.replace(u"\u041a", u"\u004b")
            os.rename(path_img_names + "/" +img_name, path_img_names + "/" + new_name)
        if u"\u0412" in img_name:
            new_name = img_name.replace(u"\u0412", u"\u0042")
            os.rename(path_img_names + "/" +img_name, path_img_names + "/" + new_name)
        if u"\u0430" in img_name:
            new_name = img_name.replace(u"\u0430", u"\u0061")
            os.rename(path_img_names + "/" +img_name, path_img_names + "/" + new_name)
        
        im = cv2.imread(path_img_names + "/" +new_name)
        
    for img_name in os.listdir(path_img_names):
        if " _" in img_name:
            new_name = img_name.replace(" _", "_")
            os.rename(path_img_names + "/" + img_name, path_img_names + "/" +new_name)