# from  https://www.kaggle.com/code/nyachhyonjinu/yolov3-test
# and
# https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation
# Modified by Alfonso Blanco


# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

import tensorflow as tf

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

# YOLO v3 model architecture

"""
Architecture config:
- Tuple --> (filters, kernel_size, stride)
- List --> ['B', num_repeats] where 'B' is residual block
- 'S' --> scale prediction block. Also for computing yolo loss
- 'U' --> upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53

    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
    super(CNNBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # If batchnorm layer(bn_act) is true, then bias is False
    self.bn = nn.BatchNorm2d(out_channels)
    self.leaky = nn.LeakyReLU(0.1)
    self.use_bn_act = bn_act

  def forward(self, x):
    if self.use_bn_act:
      return self.leaky(self.bn(self.conv(x)))
    else:
      return self.conv(x)


class ResidualBlock(nn.Module):
  def __init__(self, channels, use_residual=True, num_repeats=1):
    super(ResidualBlock, self).__init__()
    self.layers = nn.ModuleList() # Like regular python list, but is container for pytorch nn modules

    for repeat in range(num_repeats):
      self.layers += [
          nn.Sequential(
            CNNBlock(channels, channels//2, kernel_size=1),
            CNNBlock(channels//2, channels, kernel_size=3, padding=1)
          )
      ]

    self.use_residual = use_residual
    self.num_repeats = num_repeats

  def forward(self, x):
    for layer in self.layers:
      if self.use_residual:
        x = x + layer(x)
      else:
        x = layer(x)

    return x

class ScalePrediction(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(ScalePrediction, self).__init__()
    self.pred = nn.Sequential(
        CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
        CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1), # (num_classes + 5) * 3 --> (20+5) for each anchor box which in total is 3
    )
    self.num_classes = num_classes

  def forward(self, x):
    return (
        self.pred(x)
        .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) # [batch_size, anchor_boxes, prediction(25), grid_h, grid_w]
        .permute(0, 1, 3, 4, 2) # [batch_size, anchor_boxes, grid_h, grid_w, prediction(25)]
      )

# UNA EXPLICACION CLARA DE YOLO EN LA PRIMERA IMAGEN DE
# ESTE ARTICULO
# https://medium.com/@chnwsw01/yolo-algorithm-c779b9b2018b

class YOLOv3(nn.Module):
  def __init__(self, in_channels=3, num_classes=20):
    super(YOLOv3, self).__init__()
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.layers = self._create_conv_layers()

  def forward(self, x):
    outputs = []
    route_connections = []

    for layer in self.layers:
      if isinstance(layer, ScalePrediction):
        outputs.append(layer(x))
        continue

      x = layer(x)

      if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
        route_connections.append(x)

      elif isinstance(layer, nn.Upsample):
        x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()

    return outputs


  def _create_conv_layers(self):
    layers = nn.ModuleList()
    in_channels = self.in_channels

    for module in config:
      if isinstance(module, tuple):
        out_channels, kernel_size, stride = module
        layers.append(CNNBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1 if kernel_size == 3 else 0
        ))
        in_channels = out_channels

      elif isinstance(module, list):
        num_repeats = module[1]
        layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

      elif isinstance(module, str):
        if module == "S":
          layers += [
              ResidualBlock(in_channels, use_residual=False, num_repeats=1),
              CNNBlock(in_channels, in_channels//2, kernel_size=1),
              ScalePrediction(in_channels//2, num_classes = self.num_classes)
          ]
          in_channels = in_channels // 2

        elif module == "U":
          layers.append(nn.Upsample(scale_factor=2))
          in_channels = in_channels * 3

    return layers

#num_classes = 20
num_classes = 1
class_list=["Fracture"]

IMAGE_SIZE = 640
model = YOLOv3(num_classes=num_classes)
#x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) # modified
x = torch.zeros((2, 3, IMAGE_SIZE, IMAGE_SIZE))
out = model(x)
assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)

print("Success!")

print(model(x)[0].shape)
print(model(x)[1].shape)
print(model(x)[2].shape)

#print(model)

# Count the total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 4
#BATCH_SIZE = 32
BATCH_SIZE = 3 
IMAGE_SIZE = 640
#NUM_CLASSES = 20
NUM_CLASSES = 1
#LEARNING_RATE = 1e-5
#LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-4
#NUM_EPOCHS = 80
NUM_EPOCHS = 20
CONF_THRESHOLD = 0.8
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

#IMG_DIR = "/kaggle/input/pascalvoc-yolo/images"
#LABEL_DIR = "/kaggle/input/pascalvoc-yolo/labels"

IMG_DIR = "trainFractureOJumbo1\\images"
LABEL_DIR = "trainFractureOJumbo1\\labels"
IMG_DIR_TEST = "validFractureOJumbo1\\images"
LABEL_DIR_TEST = "validFractureOJumbo1\\labels"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


PASCAL_CLASSES = [
    "Fracture"
]


# A CLEARER WAY WOULD BE IN THIS DIRECTION
# https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
# COMPARING THE W H OF THE LABEL WITH EACH ANCHOR BUT IT WOULD HAVE TO BE DONE BY COMPARING
# ONE BY ONE, HERE IT DOES IT IN A BLOCK AND IT IS MORE CONFUSING. Although using multiplication
# of matrices (tensors) it will be faster

# IOU width height
# Take in hxw of anchor boxe and bounding box to calc. IOU

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    """
   tensor([0.0656, 0.0562]) --> alto y ancho  del label del registro del dataset del train
   tensor([[0.2800, 0.2200],  --> anchors
        [0.3800, 0.4800],
        [0.9000, 0.7800],
        [0.0700, 0.1500],
        [0.1500, 0.1100],
        [0.1400, 0.2900],
        [0.0200, 0.0300],
        [0.0400, 0.0700],
        [0.0800, 0.0600]])
        tensor([0.0656, 0.0656, 0.0656, 0.0656, 0.0656, 0.0656, 0.0200, 0.0400, 0.0656]) --> min 0.0656 con la primera columna de anchors
        tensor([0.0562, 0.0562, 0.0562, 0.0562, 0.0562, 0.0562, 0.0300, 0.0562, 0.0562]) --> min 0.0562 con la 2ª columna de anchors
intersection _> 0.0656* 0.0562 = 0.0037
tensor([0.0037, 0.0037, 0.0037, 0.0037, 0.0037, 0.0037, 0.0006, 0.0022, 0.0037])
union  -->0.0656 * 0.0562 + 0.28* 0.22 - 0.0037 --> 0.0616
tensor([0.0616, 0.1824, 0.7020, 0.0105, 0.0165, 0.0406, 0.0037, 0.0042, 0.0048])
return intersection / union
0.0037/0.0616=0.06
tensor([0.0599, 0.0202, 0.0053, 0.3516, 0.2237, 0.0909, 0.1625, 0.5305, 0.7690])
fin iou anchors
    """
    #print(torch.min(boxes1[..., 0], boxes2[..., 0]))
    #print(torch.min(boxes1[..., 1], boxes2[..., 1]))
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    #print("intersection")
    #print(intersection)
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    #print("union")
    #print(union)
    #print("return")
    #print (intersection / union)
    return intersection / union

  # Intersection over union

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)



########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading images from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
     
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 for linea in f:
                      
                      indexFracture=int(linea[0])
                      Label=class_list[indexFracture]
                      xyxy=linea[2:]
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(xyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels



# Dataset

import numpy as np
import os
#import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# allows PIL to load images even if they are truncated or incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
  def __init__(self,  img_dir, label_dir, anchors,
               #image_size=640, S=[13,26,52], C=20, transform=None):
               
               #https://medium.com/@wilson.linzhe/digital-image-processing-in-c-chapter-9-thresholding-roberts-prewitt-sobel-and-edge-e7428405ede3
               
               image_size=640, S=[13,26,52], C=2, transform=None): # only 2 objects "Fracture" "no object"
    #self.annotations, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels=loadlabels(label_dir)
    ClassName, self.annotations, Tabxyxy, ContLabels, ContNoLabels=loadlabels(label_dir)
       
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.S = S

    # Suppose, anchors[0] = [a,b,c], anchors[1] = [d,e,f], anchors[2] = [g,h,i] : Each set of anchors for each scale
    # List addition gives shape 3x3
    # Anchors per scale suggests that there are three different aspect ratios for each anchor position.
    self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # For all 3 scales 3 anchors--> 9 anchors
    self.num_anchors = self.anchors.shape[0]
    self.num_anchors_per_scale = self.num_anchors // 3

    self.C = C # C=2 

    # If a cell has obj. then one anchor is responsible for outputting it,
    # one that's responsible is the one that has highest iou with ground truth box
    # but, there might be cases where there are several boxes in the same cell
    self.ignore_iou_thresh = 0.5

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
   
    NameImage=self.annotations[index]
    NameImage=NameImage[:len(NameImage)-4]
    ImageLabel=NameImage+".txt"
        
    
    label_path = os.path.join(self.label_dir, ImageLabel)

    
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.roll with shift 4
                                                                                               # on axis 1:
                                                                                               #[class, x, y, w, h] -->
                                                                                               #[x, y, w, h, class]

    Imagepath=NameImage+ ".jpg"
    img_path = os.path.join(self.img_dir, Imagepath)
    image = Image.open(img_path)

    #b = np.asarray(image)
    #print(b.shape)  
    
    if self.transform:
      image = self.transform(image)

    
    # self.S=[20,40,60] 
    
    #
    
    targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 because objectness score, bounding box coordinates (x, y, w, h), class label

    """
    targets[0] --> [3, 20, 20, 6]  
    targets[1] --> [3, 40, 40, 6]
    targets[2] --> [3, 80, 80, 6]
    """

    
    # labels from train dataset labeled
    # a image may have several labels
    for box in bboxes:   
      """For each box in bboxes,
      we want to assign which anchor should be responsible and
      which cell should be responsible for all the three different scales prediction"""
      #print("torch.tensor(box[2:4]")
      #print(torch.tensor(box[2:4]))
      #print("Fin torch.tensor(box[2:4]")
      #print(self.anchors)

      # It is about finding which of the anchors best adapts
      # to the w h of the label, which is achieved with iou_width_height
      iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) # IOU from height and width
      
      # returns the 9 indices of the 9 anchors ordered according to which one best fits the w h of the tag
      anchor_indices = iou_anchors.argsort(descending=True, dim=0) # Sorting sucht that the first is the best anchor

      box=box[:5]            
      #print(box)
      
      x, y, width, height, class_label = box
     

      for anchor_idx in anchor_indices:
        scale_idx = anchor_idx // self.num_anchors_per_scale # scale_idx is either 0,1,2: 0-->13x13, 1:-->26x26, 2:-->52x52
        
        anchor_on_scale = anchor_idx % self.num_anchors_per_scale # In each scale, choosing the anchor thats either 0,1,2

        S = self.S[scale_idx]
        i, j = int(S*y), int(S*x) # x=0.5, S=13 --> int(6.5) = 6 | i=y cell, j=x cell
        anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
        
        # Modified
        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
        x_cell, y_cell = S*x - j, S*y - i # 6.5 - 6 = 0.5 such that they are between [0,1]
        width_cell, height_cell = (
            width*S, # S=13, width=0.5, 6.5
            height*S
        )

        box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        
        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
        #print("targets[scale_idx][anchor_on_scale, i, j, 1:5]")
        #print(targets[scale_idx][anchor_on_scale, i, j, 1:5])
        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
        
        #break # GO OUT HERE, IT DOESN'T MAKE SENSE TO COMPARE WITH ALL THE INDEXES ONLY WITH THE FIRST ONE WHICH IS THE ONE THAT BEST FITS
    
    
    return image, tuple(targets)
    

# DataLoader

import torchvision.transforms as transforms
transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

def get_loaders():

    train_dataset = YOLODataset(        
        transform=transform,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    test_dataset = YOLODataset(        
        transform=transform,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=IMG_DIR_TEST,
        label_dir=LABEL_DIR_TEST,
        anchors=ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader
   

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

# Loss  

class YoloLoss(nn.Module):
  def __init__(self):
    super(YoloLoss, self).__init__()
    self.mse = nn.MSELoss() # For bounding box loss
    self.bce = nn.BCEWithLogitsLoss() # For multi-label prediction: Binary cross entropy
    self.entropy = nn.CrossEntropyLoss() # For classification
    self.sigmoid = nn.Sigmoid()

    # Constants for significance of obj, or no obj.
    self.lambda_class = 1
    self.lambda_noobj = 10
    self.lambda_obj = 1
    self.lambda_box = 10

  def forward(self, predictions, target, anchors):
    obj = target[..., 0] == 1
    noobj = target[..., 0] == 0

    # No object Loss
    ################
    no_object_loss = self.bce(
        (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
    )

    # Object Loss
    #############
    anchors = anchors.reshape(1,3,1,1,2) # Anchors initial shape 3x2 --> 3 anchor boxes each of certain hxw (2)

    # box_preds = [..., sigmoid(x), sigmoid(y), [p_w * exp(t_w)], [p_h * exp(t_h)], ...]
    box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)

    # iou between predicted box and target box
    ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()

    object_loss = self.bce(
        (predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]) # target * iou because only intersected part object loss calc
    )

    # Box Coordinate Loss
    #####################
    predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between [0,1]
    target[..., 3:5] = torch.log(
        (1e-6 + target[..., 3:5] / anchors)
    ) # Exponential of hxw (taking log because opp. of exp)

    box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

    # Class Loss
    ############
    class_loss = self.entropy(
        (predictions[..., 5:][obj]), (target[..., 5][obj].long())
    )

    return(
        self.lambda_box * box_loss
        + self.lambda_obj * object_loss
        + self.lambda_noobj * no_object_loss
        + self.lambda_class * class_loss
    )


# TRAIN

# Instantiate the model
model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)

# Compile the model
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)
loss_fn = YoloLoss()

# Scaler
scaler = torch.cuda.amp.GradScaler()
import cv2

print("Start loader")
train_loader, test_loader = get_loaders()
print("End loader")
#print(test_loader)
#https://discuss.pytorch.org/t/how-to-find-shape-and-columns-for-dataloader/34901/2

# Anchors
scaled_anchors = (
    torch.tensor(ANCHORS) * torch.tensor([13,26,52]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    #torch.tensor(ANCHORS) * torch.tensor([20,40,80]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)# MUY MAL
).to(DEVICE)

# Save test loader to a file
torch.save(test_loader, '/kaggle/working/PRUEBAtest_loader.pth')

import torch.optim as optim

from tqdm import tqdm
import time

history_loss = [] # To plot the epoch vs. loss

for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
  model.train()

  losses = []

  start_time = time.time() # Start time of the epoch

  for batch_idx, (x,y) in enumerate(train_loader):
    x = x.to(DEVICE)
    y0, y1, y2 = (y[0].to(DEVICE),
                  y[1].to(DEVICE),
                  y[2].to(DEVICE))

    # context manager is used in PyTorch to automatically handle mixed-precision computations on CUDA-enabled GPUs
    with torch.cuda.amp.autocast():
      out = model(x)
     
      loss = (
          loss_fn(out[0], y0, scaled_anchors[0])
          + loss_fn(out[1], y1, scaled_anchors[1])
          + loss_fn(out[2], y2, scaled_anchors[2])
      )

    losses.append(loss.item())
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

  end_time = time.time()  # End time of the epoch
  epoch_duration = end_time - start_time  # Duration of the epoch
    
  history_loss.append(sum(losses)/len(losses))

  #if (epoch+1) % 10 == 0:
  if (epoch+1) % 2 == 0:  
    # Print the epoch duration
    tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    # Print the loss and accuracy for training and validation data
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Loss: {sum(losses)/len(losses):.4f}")

    # save the model after every 10 epoch
    # RESPUESTA COPILOT Reducing the size of a .pth
    torch.save(model.state_dict(), f'YoloFromScratch_epoch{epoch+1}.pth', _use_new_zipfile_serialization=False)

import matplotlib.pyplot as plt
epochs = range(1, len(history_loss)+1)

# Plot losses
plt.plot(epochs, history_loss)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training Loss")
plt.show()

# inference in a separate program

