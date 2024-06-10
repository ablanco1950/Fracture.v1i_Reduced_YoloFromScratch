# https://www.kaggle.com/code/nyachhyonjinu/yolov3-test

# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

import torch

from collections import Counter
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

num_classes = 1

IMAGE_SIZE = 640

import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
IMAGE_SIZE = 640

NUM_CLASSES = 1

CONF_THRESHOLD = 0.8
CONF_THRESHOLD = 0.532 # quede
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

IMG_DIR = "testFractureOJumbo1\\images"
LABEL_DIR = "testFractureOJumbo1\\labels"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

PASCAL_CLASSES = [
    "Fracture"
]

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
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
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

# Non-max Supression

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
     Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

# Dataset

import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# allows PIL to load images even if they are truncated or incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
  def __init__(self, img_dir, label_dir, anchors,
               image_size=640, S=[13,26,52], C=20, transform=None):
               #image_size=640, S=[13,26,52], C=2, transform=None):
   
    dirnameCV="testFractureOJumbo1\\images"
   
    images, self.annotations =loadimages(dirnameCV)
        
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.S = S

    # Suppose, anchors[0] = [a,b,c], anchors[1] = [d,e,f], anchors[2] = [g,h,i] : Each set of anchors for each scale
    # List addition gives shape 3x3
    # Anchors per scale suggests that there are three different aspect ratios for each anchor position.
    self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # For all 3 scales
    self.num_anchors = self.anchors.shape[0]
    self.num_anchors_per_scale = self.num_anchors // 3

    self.C = C

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

        
    #print(label_path)

    
    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # np.roll with shift 4 on axis 1: [class, x, y, w, h] --> [x, y, w, h, class]

    Imagepath=NameImage+ ".jpg"
    img_path = os.path.join(self.img_dir, Imagepath)
    image = Image.open(img_path)

    
    if self.transform:
      image = self.transform(image)

   

    targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 because objectness score, bounding box coordinates (x, y, w, h), class label

    for box in bboxes:
      """For each box in bboxes,
      we want to assign which anchor should be responsible and
      which cell should be responsible for all the three different scales prediction"""
      iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors) # IOU from height and width
      anchor_indices = iou_anchors.argsort(descending=True, dim=0) # Sorting sucht that the first is the best anchor

      box=box[:5]            
      #print(box)
      
      x, y, width, height, class_label = box
      has_anchor = [False, False, False] # Make sure there is an anchor for each of three scales for each bounding box

      for anchor_idx in anchor_indices:
        scale_idx = anchor_idx // self.num_anchors_per_scale # scale_idx is either 0,1,2: 0-->13x13, 1:-->26x26, 2:-->52x52
        anchor_on_scale = anchor_idx % self.num_anchors_per_scale # In each scale, choosing the anchor thats either 0,1,2

        S = self.S[scale_idx]
        i, j = int(S*y), int(S*x) # x=0.5, S=13 --> int(6.5) = 6 | i=y cell, j=x cell
        anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

        if not anchor_taken and not has_anchor[scale_idx]:
          targets[scale_idx][anchor_on_scale, i, j, 0] = 1
          x_cell, y_cell = S*x - j, S*y - i # 6.5 - 6 = 0.5 such that they are between [0,1]
          width_cell, height_cell = (
              width*S, # S=13, width=0.5, 6.5
              height*S
          )

          box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

          targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
          targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
          has_anchor[scale_idx] = True

        # Even if the same grid shares another anchor having iou>ignore_iou_thresh then,
        elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
          targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction

    return image, tuple(targets)

# DataLoader

import torchvision.transforms as transforms
transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

def get_loaders():

   
    test_dataset = YOLODataset(       
        transform=transform,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    return  test_loader

# Mean Average Precision

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=4
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
  
def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    
    x, labels = next(iter(loader))
    x = x.float().to(DEVICE)
    
    print("LABELS ITER " + str(len(labels)))
    all_pred_boxes = []
    all_true_boxes = []
    
    #for batch_idx, (x, labels) in enumerate(loader):
    for batch_idx in range(len(x)):  
        #print("PASAAAAAAAAAAAAAAAAAAAAAAA")
        x = x.float().to(device)

        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2] # grid cell size for each predictions
            anchor = torch.tensor([*anchors[i]]).to(device) * S # anchor for each grid, prediction type
            boxes_scale_i = cells_to_bboxes( # get bboxes for each image in the batch
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i): # for each image, append the bbox to corr. bboxes[idx]
                bboxes[idx] += box
                

        # we just want one bbox for each label, not one for each scale
        #print("labels[2]")
        #print(labels[2])
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )
       
        #added
        assert type(bboxes) == list

        #bboxes = [box for box in bboxes if box[1] > threshold]
        bboxes = [box for box in bboxes]
        print("BBOSES DE CER")
        print(bboxes[batch_idx][0])
        bboxes[batch_idx] = sorted(bboxes[batch_idx], key=lambda x: x[1], reverse=True)
        print("BBOSES DE CER clasificadas")
        print(bboxes[batch_idx][0])
        
        #print(len(bboxes[batch_idx][0]))
        #print("BBOXES")
        #print(bboxes[0][0])
        all_pred_boxes.append(bboxes[batch_idx][0])
        all_true_boxes.append(true_bboxes[batch_idx][0])
        #print("LLEGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    model.train()
    return all_pred_boxes, all_true_boxes

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


# Plot

def plot_image(image, boxes, boxesTrue, imageCV):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    Cont=0
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf=box[1]
        conf=str(conf)
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)] + " conf: " + str(conf[:3]),
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
      
        

        
        Cont+=1
        #if Cont > 1: break # only the most predicted box
        break
      # rect with true fracture
    upper_left_x_true = boxesTrue[2] - boxesTrue[4] / 2
    upper_left_y_true = boxesTrue[3] - boxesTrue[5] / 2
    rect1 = patches.Rectangle(
            (upper_left_x_true * width, upper_left_y_true * height),
            boxesTrue[4] * width,
            boxesTrue[5] * height,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
    # Add the patch to the Axes
    ax.add_patch(rect1)  
    plt.show()
   
dirnameCV="testFractureOJumbo1\\images"
import re    
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
                 #if Cont > 1:break
     print("Readed " + str(len(images)))
     #cv2.imshow('True', images[0])
     #cv2.waitKey(0)
   
     return images, TabFileName

dirnameLabels="testFractureOJumbo1\\labels"
def loadlabels (dirname ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
     Conta=0
     print("Reading labels from ",lblpath)
     
     
     
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                 Conta=Conta+1
                 # case test
                 
                 filepath = os.path.join(root, filename)
                 #License=filename[:len(filename)-4]
                 #if Detect_Spanish_LicensePlate(License)== -1: continue
               
                 f=open(filepath,"r")

                 ContaLin=0
                 for linea in f:
                     lineadelTrain=[]
                     lineadelTrain1 =linea.split(" ")
                     lineadelTrain.append(0.0)
                     lineadelTrain.append(1.0)
                     lineadelTrain.append(float(lineadelTrain1[1]))
                     lineadelTrain.append(float(lineadelTrain1[2]))
                     lineadelTrain.append(float(lineadelTrain1[3]))
                     lineadelTrain.append(float(lineadelTrain1[4]))
                     labels.append(lineadelTrain)
                     
                 f.close() 
                 #if ContaLin==0:
                 #    print("Rare labels without tag 0 on " + filename )
                   
                 
 
     return labels

imagesCV, TabFileName=loadimages(dirnameCV)

labelsTrue=loadlabels(dirnameLabels)

test_loader = get_loaders()


# Inference


# Load the model
model = YOLOv3(num_classes=NUM_CLASSES)

#model_path = "/kaggle/working/Yolov3_epoch40.pth"
model_path = "/kaggle/working/Yolov3_epoch30.pth"
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.to(DEVICE)

# COPIAND DEL ORIGINAL
model.eval()
x, y = next(iter(test_loader))
x = x.float().to(DEVICE)

with torch.no_grad():
    out = model(x)
    bboxes = [[] for _ in range(x.shape[0])]
    batch_size, A, S, _, _ = out[0].shape
    anchor = torch.tensor([*ANCHORS[0]]).to(DEVICE) * S
    boxes_scale_i = cells_to_bboxes(
        out[0], anchor, S=S, is_preds=True
    )
    for idx, (box) in enumerate(boxes_scale_i):
        bboxes[idx] += box

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            #bboxes[i], iou_threshold=0.5, threshold=0.6, box_format="midpoint",
            bboxes[i], iou_threshold=0.45, threshold=0.45, box_format="midpoint",
        )
        boxesTrue=labelsTrue[i]
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, boxesTrue, imagesCV[i])
