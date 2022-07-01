# MS_Face_Recognition
The goal of this project is to measure the impact of influencing factors on the face recognition system Arcface. I will use 6 different datasets:
- Age variation: AgeDB, CALFW
- Pose variation: LFR
- Occlusion: Webface-OCC
- Ethnicity variation: VMER, RFW

I use the a pretrain model trained on MS1M-V2 and use resnet-50 as feature extractor. The pretrained model can be found here.

## Project structure
- `model`: must contains the model parameters (ex: backbone.pth)
