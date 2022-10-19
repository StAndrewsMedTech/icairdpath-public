from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from repath.utils.paths import project_root
import repath.data.datasets.endometrial_algo2 as endometrial
from repath.utils.seeds import set_seed
from repath.postprocess.results import SlidesIndexResults

experiment_name = "endo_set2"
experiment_root = project_root() / "experiments" / experiment_name

global_seed = 123

set_seed(global_seed)

results_dir_name = "results"
heatmap_dir_name = "heatmaps"

trainresultsin = experiment_root / "patch_results" / "train" 
validresultsin = experiment_root / "patch_results" / "valid"  

train_results = SlidesIndexResults.load(endometrial.algo2(), trainresultsin, "results", "heatmaps")
valid_results = SlidesIndexResults.load(endometrial.algo2(), validresultsin, "results", "heatmaps")

train_images = []
trainlabels = []
for result in train_results:
    csvpath = result.slide_path.with_suffix('.csv')
    malig_hm_path = trainresultsin / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
    malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
    other_hm_path = trainresultsin / 'heatmaps' / (Path(csvpath).stem + '_other_benign.png')
    other_hm = np.asarray(Image.open(other_hm_path)) / 255
    train_image = np.dstack((malig_hm, other_hm))
    train_images.append(train_image)
    trainlabels.append(result.label)

valid_images = []
validlabels = []
for result in valid_results:
    csvpath = result.slide_path.with_suffix('.csv')
    malig_hm_path = validresultsin / 'heatmaps' / (Path(csvpath).stem + '_malignant.png')
    malig_hm = np.asarray(Image.open(malig_hm_path)) / 255
    other_hm_path = validresultsin / 'heatmaps' / (Path(csvpath).stem + '_other_benign.png')
    other_hm = np.asarray(Image.open(other_hm_path)) / 255
    valid_image = np.dstack((malig_hm, other_hm))
    valid_images.append(valid_image)
    validlabels.append(result.label)

trainlabels = np.array(trainlabels)
trainlabels = np.where(trainlabels=='insufficient',0,np.where(trainlabels=='other_benign', 1, 2))
validlabels = np.array(validlabels)
validlabels = np.where(validlabels=='insufficient',0,np.where(validlabels=='other_benign', 1, 2))

newtrainimages = []
for img in train_images:
    imgrow, imgcol, imgch = img.shape
    imgasp = imgcol / imgrow
    if imgasp < 2:
        newimgcol = imgrow * 2
        addcols = newimgcol - imgcol
        imgcolleft = np.random.randint(0, addcols + 1)
        imgcolright = addcols - imgcolleft
        newimg = np.hstack((np.zeros((imgrow, imgcolleft, imgch)), img, np.zeros((imgrow, imgcolright, imgch))))
    if imgasp > 2:
        newimgrow = int(imgcol / 2)
        addrows = newimgrow - imgrow
        imgrowlower = np.random.randint(0, addrows + 1)
        imgrowupper = addrows - imgrowlower
        newimg = np.vstack((np.zeros((imgrowupper, imgcol, imgch)), img, np.zeros((imgrowlower, imgcol, imgch))))  
    newimg = Image.fromarray(np.array(newimg*255, dtype=np.uint8))
    newimg = newimg.resize((384, 192))
    #newtrainimages.append(np.asarray(newimg))
    newtrainimages.append(newimg)

newvalidimages = []
for img in valid_images:
    imgrow, imgcol, imgch = img.shape
    imgasp = imgcol / imgrow
    if imgasp < 2:
        newimgcol = imgrow * 2
        addcols = newimgcol - imgcol
        imgcolleft = np.random.randint(0, addcols + 1)
        imgcolright = addcols - imgcolleft
        newimg = np.hstack((np.zeros((imgrow, imgcolleft, imgch)), img, np.zeros((imgrow, imgcolright, imgch))))
    if imgasp > 2:
        newimgrow = int(imgcol / 2)
        addrows = newimgrow - imgrow
        imgrowlower = np.random.randint(0, addrows + 1)
        imgrowupper = addrows - imgrowlower
        newimg = np.vstack((np.zeros((imgrowupper, imgcol, imgch)), img, np.zeros((imgrowlower, imgcol, imgch))))  
    newimg = Image.fromarray(np.array(newimg*255, dtype=np.uint8))
    newimg = newimg.resize((384, 192))
    #newvalidimages.append(np.asarray(newimg))
    newvalidimages.append(newimg)



class SlideClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(120 * 6 * 12, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        output = self(x)
        pred = torch.log_softmax(output, dim=1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        output = self(x)
        pred = torch.log_softmax(output, dim=1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("valid_accuracy", accu, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class HeatmapDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = Compose([ToTensor()])

batch_size = 32
train_set = HeatmapDataset(newtrainimages, trainlabels, transform=transform)
valid_set = HeatmapDataset(newvalidimages, validlabels, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath= experiment_root / 'slide_model_cnn',
    filename=f"checkpoint",
    save_top_k=1,
    mode="max",
)
csv_logger = pl_loggers.CSVLogger(experiment_root / 'slide_model_logs', name='slide_classifier', version=0)

# train our model
classifier = SlideClassifier()
trainer = Trainer(callbacks=[checkpoint_callback], gpus=1, accelerator="ddp", max_epochs=50, logger=csv_logger)
trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)