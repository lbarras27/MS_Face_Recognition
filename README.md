# MS_Face_Recognition
The goal of this project is to measure the impact of influencing factors on the face recognition system Arcface. I will use 6 different datasets:
- Age variation: AgeDB, CALFW
- Pose variation: LFR
- Occlusion: Webface-OCC
- Ethnicity variation: VMER, RFW

I use a pretrain model trained on MS1M-V2 and use resnet-50 as feature extractor. The pretrained model can be found [here](https://drive.google.com/drive/folders/12xtGMMMbDf6VLX3unUjJNgjnoQF83tOe?usp=sharing).

## Project structure
- `model`: must contains the model parameters (ex: backbone.pth)
- `backbones`: contains the implementation of the features extractor resnet-50 (from Arcface repository)
- `datasets`: contains directory of each dataset. In each of these directories, we can find metadata files, script to generate these metadata files, results and dataset images.
- `examples`: contains a jupyter notebook that shows how to compute roc curves from the scores and the labels of the pairs list files and save the results roc curves data in a .npz file. It contains also a second jupyter notebook that shows how to do more beautiful roc curve plots.
- `evaluation_util.py`: contains all the usefull methods to do the evalutation protocols on all the datasets.
- `eval_{datasetname}_verification.py`: the script that allows to do the verification protocol on the {datasetname}.
- `eval_{datasetname}_identification.py`: the script that allows to do the identification protocol on the {datasetname}.

## Steps to do the evaluation protocols on the datasets
1. Download the dataset you want to use and put the parent folder that contains all the image in the corresponding dataset folder in `datasets` folder. For example if you download LFR dataset, rename the folder that contains all images in `imgs` and put inside `datasets/LFR dataset` folder. If the dataset contains some metata files, don't need to care about it, they are already in the `metadata` folder of the datasets.
2. Download the pretrained model [resnet-50 on MS1M-V2](https://drive.google.com/drive/folders/12xtGMMMbDf6VLX3unUjJNgjnoQF83tOe?usp=sharing) and put the model in `model` folder.
3. Install the conda envrionment with the help of `environment.yml` file. Use this command in anaconda to create the environment I used in this project: `conda env create --name envname --file=environment.yml`.
4. Now you can execute the corresponding script to the dataset you chose. If you want to do verification protocol, use `eval_{dataset_choose}_verification.py` and if you want to do identification protocol, use `eval_{dataset_choose}_identification.py`.  
The command for verification is: `python eval_{dataset_name}_verification.py` for example. In the next section I describe more in details and list the different argument that we can use for theses scripts.

## Execute scripts for verification and identification protocols
There is 5 argument use by all the script:  
- `--model-prefix`: the path to the pretrained model (backbone.pth). By default, the path is `./model/backbone.pth`.
- `--image-path`: the path to the directory that contains all the images. By default, the path is `./datasets/{dataset}/imgs`. For example in LFR dataset, this is the directory that contains all the folder identities.
- `--metadata-path`: the path to the metadata folder. By default, the path is `./datasets/{dataset}/metadata`.
- `--result-dir`: The path to the directory that will contains the results (ROC curves plot, ). By default, the path is `./datasets/{dataset}/results`.
- `--batch-size`: The batch size to use (allow to speed up the testing). By default, it is equal to 32.

After we can have one additional argument according to the dataset.
- AgeDB: we have the `--gap` argument that correspond to the age gap between the testing identities. By default is equal to `5` but you can choose between `5`, `10`, `20` and `30`.
- LFR: we have the `--pose` argument that correspond to the pose of the testing identities. Choice between: `left`, `front`, `right`.
- VMER, RFW: we have the `--ethnicity` argument that correspond to the ethnicity of the testing identities. Choice between: `african`, `caucasian`, `asian`, `indian` for VMER and for RFW choice between: `African`, `Caucasian`, `Asian`, `Indian` (note the upper case on the first letter).
- Webface-OCC: we have the `--occlusion` argument that correspond to the type of occlusion of the testing identities. Choice between: `original` (without occlusion), `glass`, `surgical`.

We have also specific argument for identification or verification.
- Verification: we have the `--pair-name` argument that correspond to the name of the pair list file. By default we use the name of the pair list file present in the `metadata` folder. 
- Identification: we have the `--gallery-set`and `probe-set` arguments that correspond to the names of the gallery set file and the probe set file respectively. By default we use the name of the gallery set file and probe set file present in the `metadata` folder.

If you followed the exactly same project structure, you just need to use the additional parameter of the dataset. For example if you want to do the identification protocol on LFR dataset and want to evaluate the left pose, you just need to execute this following command:  
`python eval_lfr_identification.py --pose left`. You may have a character encoding problem for AgeDB dataset, if this is the case, execute the `solve_caracter_encoding_img_names()` method that is in the file `dataset.py` in `AgeDB` folder (This file replace characters in image names).


