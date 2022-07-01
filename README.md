# MS_Face_Recognition
The goal of this project is to measure the impact of influencing factors on the face recognition system Arcface. I will use 6 different datasets:
- Age variation: AgeDB, CALFW
- Pose variation: LFR
- Occlusion: Webface-OCC
- Ethnicity variation: VMER, RFW

I use a pretrain model trained on MS1M-V2 and use resnet-50 as feature extractor. The pretrained model can be found here.

## Project structure
- `model`: must contains the model parameters (ex: backbone.pth)
- `backbones`: contains the implementation of the features extractor resnet-50 (from Arcface repository)
- `dataset`: contains directory of each dataset. In each of these directories, we can find metadata files, script to generate these metadata files, results and dataset images.
- `evaluation_util.py`: contains all the usefull methods to do the evalutation protocols on all the datasets.
- `eval_{datasetname}_verification.py`: the script that allows to do the verification protocol on the {datasetname}.
- `eval_{datasetname}_identification.py`: the script that allows to do the identification protocol on the {datasetname}.

## Steps to do the evaluation protocols on the datasets
1. Download the dataset you want to use and put the parent folder that contains all the image in the corresponding dataset folder in `datasets` folder. For example if you download LFR dataset, rename the folder that contains all images in `imgs` and put inside `datasets/LFR dataset` folder. If the dataset contains some metata files, don't need to care it, they are already in the `metadata` folder of the datasets.
2. Download the pretrained model resnet-50 on MS1MV2 and put the model in `model` folder.
3. Install the conda envrionment with the help of `environment.yml` file. Use this command in anaconda to create the environment I used in this project: `conda env create --name envname --file=environment.yml`.
4. Now you can execute the corresponding script to the dataset you chose. If you want to do verification protocol, use `eval_{dataset_choose}_verification.py` and if you want to do identification protocol, use `eval_{dataset_choose}_identification.py`.
The command for verification is: `python eval_{dataset_name}_verification.py`
There is 4 argument use by all the script: 
`--model-prefix`: the path to the pretrained model (backbone.pth)
