import numpy as np
import pandas as pd
import os

def get_identities_imgs_name(path="."):
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)

    # dic with identities as key and array of img names sorted by age as values
    identities_imgs_name = new_df.sort_values(["name", "age"]).groupby("name").apply(lambda x: x["img_name"].values.tolist()).to_dict()

    return identities_imgs_name


def get_identities_with_gap_pairs_imgs(path, gap=5):
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)
    new_df = new_df.sort_values(["name", "age"])

    dico = new_df.groupby("name").apply(lambda x: x[["age", "img_name"]].values.tolist()).to_dict()
    dic_final = {}
    for id in dico.keys():
        gap_5 = []
        for age1 in dico[id]:
            for age2 in dico[id]:
                if (age1[0] - age2[0]) == gap:
                    gap_5.append([age1[1], age2[1]])
        dic_final[id] = gap_5

    return dic_final
    
# same that previous one but only have images that have landmarks points.
def get_identities_with_gap_pairs_imgs2(path, gap=5):
    df = pd.DataFrame(os.listdir(os.path.join(path, "imgs")), columns=["img_name"])
    new_df = pd.DataFrame(df["img_name"].str.split("_").tolist(), columns=["id_img", "name", "age", "gender"])
    new_df["gender"] = new_df["gender"].str.slice(0, 1)
    new_df["img_name"] = df["img_name"]
    new_df["age"] = new_df["age"].astype(int)
    new_df = new_df.sort_values(["name", "age"])
    
    with open(os.path.join(path, "landmarks.txt"), "r") as f:
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
    with open(output, "w") as f:
        for i in range(len(identities_pairs)):
            f.write(identities_pairs[i][0] + " " + identities_pairs[i][1] + " 1")
            f.write('\n')
        
        for i in range(len(identities_pairs_not_same)):
            f.write(identities_pairs_not_same[i][0] + " " + identities_pairs_not_same[i][1] + " 0")
            f.write('\n')

def generate_pairs(path, output):
    identities_imgs_name = get_identities_imgs_name(path)
    pairs_same = []
    pairs_diff = []
    for k, v in identities_imgs_name.items():
        if len(v) == 0 or len(v) == 1:
            continue

        pair_same = [v[0], v[-1]]
        pairs_same.append(pair_same)
        
        id_list = list(identities_imgs_name.keys())
        id_list.remove(k)
        other_id = np.random.choice(id_list, 1)[0]
        
        if len(identities_imgs_name[other_id]) == 0:
            continue
        first_elem = np.array(v)[np.random.choice(len(v), 1)[0]]
        second_elem = np.array(identities_imgs_name[other_id])[np.random.choice(len(identities_imgs_name[other_id]), 1)[0]]
        pairs_diff.append([first_elem, second_elem])

    pairs_same = np.array(pairs_same)
    pairs_diff = np.array(pairs_diff)

    write_pairs(pairs_same, pairs_diff, output)

def generate_pairs_gap(path, output, gap=5):
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