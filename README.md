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
