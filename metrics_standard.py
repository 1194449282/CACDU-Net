import numpy
import numpy as np
import torch
from medpy import metric
from sklearn.metrics import confusion_matrix

# TP 病变区域  两个交集
# TN 背景区域  除了A 交 除了B
# FP 病变误判区域  A 交 除了B
# FN 背景误判区域  除了A 交 B



# sensitivity =recall

def sensitivity_socre(output, target):
    return metric.binary.sensitivity(torch.sigmoid(output).data.cpu().numpy() > 0.5, target.data.cpu().numpy() > 0.5)
def dice_score(output, target):
    return metric.binary.dc(torch.sigmoid(output).data.cpu().numpy()> 0.5, target.data.cpu().numpy()> 0.5)
def specificity_socre(output, target):
    return metric.binary.specificity(torch.sigmoid(output).data.cpu().numpy()> 0.5, target.data.cpu().numpy()> 0.5)
# precision =ppv
def precision_socre(output, target):
    return metric.binary.precision(torch.sigmoid(output).data.cpu().numpy()> 0.5, target.data.cpu().numpy()> 0.5)
# Jaccard Coefficient
def jaccard_socre(output, target):
    return metric.binary.jc(torch.sigmoid(output).data.cpu().numpy()> 0.5, target.data.cpu().numpy()> 0.5)
def iou_score(result, reference):
    result =torch.sigmoid(result).data.cpu().numpy() > 0.5
    reference= reference.data.cpu().numpy()> 0.5
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    # tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
        return tp / float(tp + fn + fp)
    except ZeroDivisionError:
        return 0.0
def acc_socre(result, reference):
    result = torch.sigmoid(result).data.cpu().numpy()> 0.5
    reference = reference.data.cpu().numpy()> 0.5
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)

    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)

    try:
       return float(tp + tn) / float(tp + fn + fp + tn)
    except ZeroDivisionError:
       return 0.0




#  output = tp+fp
#  target = tp+fn
#  tp = output * target
#   output + target =2 *tp +FN +FP


def iou_score1(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # intersection = (output * target).sum()
    intersection = (output * target).sum()
    total = (output + target).sum()
    union = total - intersection

    return(intersection + smooth) / (union + smooth)
    # return (intersection + smooth) / (output.sum() + target.sum() -intersection + smooth)
def specificity_score1(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    tp = (output * target).sum()
    fp = output.sum() - tp
    fn = target.sum() - tp
    tn = np.size(target)-output.sum()-target.sum()+tp

    return tn /(np.size(target)-target.sum())


def acc_score1(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    tp = (output * target).sum()
    fp = output.sum()-tp
    fn = target.sum()-tp
    return (np.size(target)-fp-fn)/np.size(target)



def dice_score1(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def sensitivity_score1(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    tp = (output * target).sum()

    return (tp + smooth) / (target.sum() + smooth)
def precision_score1(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (output.sum() + smooth)



