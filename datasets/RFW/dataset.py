import pandas as pd
import numpy as np
import os

def generate_probe_gallery_set(path, output_gallery="gallery_set.txt", output_probe="probe_set.txt", ethnicity="African"):
    '''
    Generate the probe set and the gallery set with images in the specified ethnicity
    and write the probe and gallery set in two files respectively.
    
        Parameters: 
            path (string): path to directory 'test'
            output_gallery (string): name of the output gallery set file
            output_probe (string): name of the output probe set file
            ethnicity (string): ethnicity for the probe and gallery set (African, Asian, Caucasian, Indian)

    '''
    path_landmark = path + "/txts/{0}/{1}_lmk.txt".format(ethnicity, ethnicity)
    img_names_id = []
    with open(path_landmark) as f:
        lines = f.readlines()

    for line in lines:
        name_id = line.split()[:2]
        img_names_id.append([name_id[0], int(name_id[1])])
    df = pd.DataFrame(img_names_id, columns=["img_name", "id"])

    map_id_names = df.groupby("id").apply(lambda x: x["img_name"].values.tolist()).to_dict()
    probe_set = []
    gallery_set = []
    ids = []
    for id, names_list in map_id_names.items():
        if len(names_list) == 1:
            print(id)
            continue
        img_names_choice = np.random.choice(names_list, 2, replace=False)
        gallery_set.append(img_names_choice[0])
        probe_set.append(img_names_choice[1])
        ids.append(id)
    
    with open(output_gallery, "w") as f:
        for i in range(len(ids)):
            f.write("{0} {1}".format(gallery_set[i], ids[i]))
            f.write("\n")

    with open(output_probe, "w") as f:
        for i in range(len(ids)):
            f.write("{0} {1}".format(probe_set[i], ids[i]))
            f.write("\n")