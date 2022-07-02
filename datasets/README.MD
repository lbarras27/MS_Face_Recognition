## Structure of the dataset folders
Each dataset folder follow the same structure:
- `imgs`: folder that contains all the images of the dataset. When you downloaded the dataset, please rename the folder that contains all the images in `imgs` and replace the `imgs` folder present in the folder by this one.
- `metadata`: folder that contains all the metadata files of the dataset (pair_list_file, probe_set, gallery_set, landmarks, ...).
- `results`: folder that contains the results and scores for the dataset.
- `eval_{dataset}_util.py`: contain useful methods for the evaluation protocols but specific to the dataset not like `evaluation_util.py` that contains method useful for all the datasets.
- `dataset.py`: contains useful methods to generate the metadata files for the dataset as for example probe set, gallery set, pair list files.
- `metadata_generation.ipynb`: is a jupyter notebook that generate the metadata files with the help of the methods in `dataset.py`.
