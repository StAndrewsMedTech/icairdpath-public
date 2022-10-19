import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import GoogLeNet

from repath.preprocess.tissue_detection import TissueDetectorGreyScale, SizedClosingTransform, FillHolesTransform
from repath.utils.paths import project_root


# Endometrial patching parameters
feature_level = 5
patch_level = 0
patch_size = 1024
stride = 1024
pool_mode = 'mode'

# Endometrial tissue detection parameters
morphology_transform1 = SizedClosingTransform(level_in=feature_level)
morphology_transform2 = FillHolesTransform(level_in=feature_level)
morphology_transforms = [morphology_transform1, morphology_transform2]
tissue_detector = TissueDetectorGreyScale(grey_level=0.85, morph_transform = morphology_transforms)
bm_clf_path = project_root() / 'results' / 'blood_mucus_classifier' / 'bloodm_clf.joblib'

# Endometrial patch model parameters
batch_size = 32
num_classes = 2    
class_names = ['malignant', 'other_benign']

# Endometrial model parameters
class PatchClassifier(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=2)
        self.model.dropout = nn.Dropout(0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux2, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        loss3 = criterion(aux2, y)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss, on_epoch=True)

        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("val_accuracy", accu, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.01, 
                                    momentum=0.9, 
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
            'interval': 'epoch' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

cp_path = project_root() / 'results' / "endo_1024_bmmode" / "patch_model" / "checkpoint.ckpt"


# Endometrial slide level parameters
trained_model = project_root() / 'results' / "endo_1024_bmmode" / 'slide_results' / 'train_update' / 'slide_model.joblib'
slide_class_names = ['insufficient', 'malignant', 'other_benign']
