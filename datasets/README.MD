## Structure of the dataset folders
Each dataset folder follow the same structure:
- `imgs`: folder that contains all the images of the dataset.
- `metadata`: folder that contains all the metadata files of the dataset (pair_list_file, probe_set, gallery_set, landmarks, ...).
- `eval_dataset_util.py`: contain useful methods for the evaluation protocols but specific to the dataset not like `evaluation_util.py` that contains method useful for all the datasets.
