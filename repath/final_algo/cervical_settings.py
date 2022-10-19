import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import GoogLeNet

from repath.preprocess.tissue_detection import TissueDetectorGreyScale
from repath.utils.paths import project_root


# Cervical patching parameters
feature_level=5
patch_level=0
patch_size=256
stride=256
pool_mode = 'max'

# Cervical tissue detection parameters
tissue_detector = TissueDetectorGreyScale()

# Cervical patch model parameters
batch_size = 128
num_classes = 4
class_names = ['high_grade', 'low_grade', 'malignant', 'normal']

# Cervical model parameters
cp_path = project_root() / 'results' / "cervical_final_May2022" / "patch_model" / "checkpoint.ckpt"


# Cervical slide level parameters
trained_model = project_root() / 'results' / "cervical_final_May2022" / 'slide_results' / 'train_update' / 'slide_model.joblib'
slide_class_names = class_names


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class PatchClassifier(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=4)
    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux2, aux1 = self.model(x)

        pred = torch.log_softmax(output, dim=1)

        #criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        loss3 = criterion(aux2, y)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
        self.log("train_loss", loss, on_step= False, on_epoch= True, prog_bar = True, sync_dist=True)

        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)
        accu = correct / total
        self.log("train_accuracy", accu, on_step= False, on_epoch= True, prog_bar = True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        #criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss, on_step= False, on_epoch= True, prog_bar = True, sync_dist=True)

        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)
        accu = correct / total
        self.log("val_accuracy", accu, on_step= False, on_epoch= True, prog_bar = True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=0.001,
                                    momentum=0.9,
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)