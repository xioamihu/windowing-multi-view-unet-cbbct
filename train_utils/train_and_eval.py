import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import torch.nn.functional as F
import numpy as np


class BinaryFocalLoss(nn.Module):
    """
    https://github.com/shuxinyin/NLP-Loss-Pytorch/tree/master
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss
  

def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        lossfocal = BinaryFocalLoss(alpha=1, gamma=2)
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight) #+ lossfocal(x, target)
        # print(lossfocal(x, target))

        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            # lossFocal = FocalLoss(alpha=0.25, ignore_index=ignore_index, weight=loss_weight, gamma=2)
            # loss = loss + lossfocal(x, target)  # target.long()
            
        losses[name] = loss

        # dice_target = build_target(target, num_classes, ignore_index)
        # losses = focal_loss(x, dice_target, ignore_index=ignore_index)
    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, model1, model2, model3, data_loader, device, num_classes):
    model.eval()

    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'val:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def train_one_epoch(model, model1, model2, model3, optimizer, data_loader, device, epoch, num_classes, lr_scheduler, print_freq=10,
                    scaler=None, teacher_model=None, data_loader1=None, dice=True):
    model.train()
                        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for index, [image, target] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = image.to(device), target.to(device)

        soft_loss = nn.KLDivLoss(reduction="mean")
        temp = 1
        alpha = 0.3

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255, dice=dice)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_preds = teacher_model(image)
                ditillation_loss = soft_loss(
                    F.log_softmax(output['out'] / temp, dim=1),
                    F.softmax(teacher_preds['out'] / temp, dim=1)
                )
                loss = alpha * loss + (1 - alpha) * ditillation_loss

        optimizer.zero_grad()

        if teacher_model is None:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        else:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            for index1, [image1, _] in enumerate(data_loader1):
                if index1 == index:
                    image1 = image1.to(device)
                    break

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output1 = model(image1)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_preds1 = teacher_model(image1)
                    ditillation_loss1 = soft_loss(
                        F.log_softmax(output1['out'] / temp, dim=1),
                        F.softmax(teacher_preds1['out'] / temp, dim=1)
                    )

            loss += ditillation_loss1

            if scaler is not None:
                scaler.scale(ditillation_loss1).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ditillation_loss1.backward()
                optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=100,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        # return (1.0 - x / (epochs * num_step)) ** 0.9
        
        if x > warmup_epochs * num_step:
            return 0.1
        # elif x > 200 * num_step:
        #     return 0.01
        # elif x > 300 * num_step:
        #     return 0.001
        else:
            return 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

