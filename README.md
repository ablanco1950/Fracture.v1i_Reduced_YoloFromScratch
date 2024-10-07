# Fracture.v1i_Reduced_YoloFromScratch
From a selection of data from the Roboflow file https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1, which represents a reduced but homogeneous version of that file, a model is obtained using  an adaptation of an old version of the project https://www.kaggle.com/code/nyachhyonjinu/yolov3-test , that is the same as
 https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation, instead any yolo model 

The project: https://www.kaggle.com/code/nyachhyonjinu/yolov3-test produce   a yolov3 from scratch training with COCO dataset. Here is adapted so the training is  with the reduced roboflow dataset, getting a model with that custom dataset

By using only 147 images, training is allowed using a personal computer without GPU

===
Installation:

The packages used are the usual ones in a development environment with python. In case a module not found error appears, it will be enough to install it with a simple pip

To download the dataset you have to register as a roboflow user, but to simplify it, the compressed files for train, valid and test are attached: trainFractureOJumbo1.zip, validFractureOJumbo1.zip and testFractureOJumbo1.zip obtained by selecting the images that start with names 0 -_Jumbo-1 of the original file obtaining a reduced number of images that allow training with  a personal computer without GPU

Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

===
Training:

execute 

TRAINyolovFromScratch_kaggle.py 

A log with its execution in 50 epochs is attached: LOG_YolovFromScratch_50epoch.txt, and graph LossEpoch20240920.png showing the evolution of loss with epochs.

every 2 epoch a model is written to the project directory with the name Yolov3_epochNN.pth where NN is the epoch number. 

Due to their large size, over 240 MB, it has not been possible to upload some verification models to github

Models  can be evaluated by running the program:

TESTyolovFromScratch_kaggle.py modifying the name of the model that appears in instruction 666 according to the epoch to be considered.

The model that gives the best results may be  retained.

The test was done with the 9 images of the test file (directory testFractureOJumbo1) obtaining 7 hits out of 9. Better than the project https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10 (6 hits in 9 images) but worse than https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/model/1 (8 hits in 9 images).

Furthermore, many models are complementary, so the result can be improved with a combination of models.

When you run this test, the x-ray image appears with a green rectangle indicating the rectangle with which the image was labeled and in blue with the predicted rectangle.

#Comments and conclusions:

In the inference program, TESTyolovFromScratch_kaggle.py,  the non-max.suppression has been eliminated, instead the one with the highest probability has been chosen from all the boxes. In addition, only the first 3 anchors of scale 1 have been considered, since it has been proven that in practice, in this case, they do not provide improvements.

In any case, by activating instructions 715 to 931 (removing the “”” from both lines) it can be verified by performing a detection with all the anchors and the non-max-suppression,

A test with the first six images in test dataset, in red the predicted box, in green the labeled box


![Fig1](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_1.png)
![Fig2](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_2.png)
![Fig3](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_3.png)
![Fig4](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_4.png)
![Fig5](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_5.png)
![Fig6](https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch/blob/main/Figure_6.png)


===
References and citations:

https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation 

https://www.kaggle.com/code/nyachhyonjinu/yolov3-test

https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef

https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1

@misc{
                            fracture-ov5p1_dataset,
                            title = { fracture Dataset },
                            type = { Open Source Dataset },
                            author = { landy },
                            howpublished = { \url{ https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1 } },
                            url = { https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1 },
                            journal = { Roboflow Universe },
                            publisher = { Roboflow },
                            year = { 2024 },
                            month = { apr },
                            note = { visited on 2024-06-09 },
                            }


https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10
https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/model/1 
