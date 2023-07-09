import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
import acfnet
import albumentations as albu
from dataset import Dataset
from metrics_standard import iou_score, jaccard_socre, acc_socre, sensitivity_socre, specificity_socre, precision_socre, dice_score,iou_score1,  sensitivity_score1,  precision_score1, dice_score1,specificity_score1,acc_score1

from utils import AverageMeter

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
用于测试ph2数据集
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="final_skin2018_UNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = acfnet.Seg_Model(config['num_classes'])
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        #                                    config['deep_supervision']
    )

    model = model.cuda()

    # Data loading code
    test_img_ids = glob(os.path.join('inputs', 'ph2', 'images', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    test_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', 'ph2', 'images'),
        mask_dir=os.path.join('inputs', 'ph2', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # avg_meter = AverageMeter()
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'specificity': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'jaccard': AverageMeter(),
                  'precision': AverageMeter(),
                  'iou1': AverageMeter(),
                  'dice1': AverageMeter(),
                  'sensitivity1': AverageMeter(),
                  'precision1': AverageMeter(),
                  'specificity1': AverageMeter(),
                  'accuracy1': AverageMeter()
                  }
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                if (config['arch'] == 'U2NET'):
                    output, d1, d2, d3, d4, d5, d6 = model(input)
                    # loss2, loss = muti_bce_loss_fusion(output, d1, d2, d3, d4, d5, d6, target)
                    iou = iou_score(output, target)
                    dice = dice_score(output, target)
                    accuracy = acc_socre(output, target)
                    sensitivity = sensitivity_socre(output, target)
                    specificity = specificity_socre(output, target)
                    jaccard = jaccard_socre(output, target)
                    precision = precision_socre(output, target)
                else:
                    output = model(input)
                    iou = iou_score(output, target)
                    dice = dice_score(output, target)
                    accuracy = acc_socre(output, target)
                    sensitivity = sensitivity_socre(output, target)
                    specificity = specificity_socre(output, target)
                    jaccard = jaccard_socre(output, target)
                    precision = precision_socre(output, target)
                    iou1 = iou_score1(output, target)
                    dice1 = dice_score1(output, target)
                    sensitivity1= sensitivity_score1(output, target)
                    precision1 = precision_score1(output, target)
                    specificity1 = specificity_score1(output, target)
                    accuracy1 = acc_score1(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['sensitivity'].update(sensitivity, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['jaccard'].update(jaccard, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['iou1'].update(iou1, input.size(0))
            avg_meters['dice1'].update(dice1, input.size(0))
            avg_meters['sensitivity1'].update(sensitivity1, input.size(0))
            avg_meters['precision1'].update(precision1, input.size(0))
            avg_meters['specificity1'].update(specificity1, input.size(0))
            avg_meters['accuracy1'].update(accuracy1, input.size(0))
            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meters['iou'].avg)
    print('IoU1: %.4f' % avg_meters['iou1'].avg)
    print('dice: %.4f' % avg_meters['dice'].avg)
    print('dice1: %.4f' % avg_meters['dice1'].avg)
    print('accuracy: %.4f' % avg_meters['accuracy'].avg)
    print('accuracy1: %.4f' % avg_meters['accuracy1'].avg)
    print('sensitivity: %.4f' % avg_meters['sensitivity'].avg)
    print('sensitivity1: %.4f' % avg_meters['sensitivity1'].avg)
    print('specificity: %.4f' % avg_meters['specificity'].avg)
    print('specificity1: %.4f' % avg_meters['specificity1'].avg)
    # print('jaccard: %.4f' % avg_meters['jaccard'].avg)
    print('precision: %.4f' % avg_meters['precision'].avg)
    print('precision1: %.4f' % avg_meters['precision1'].avg)
    # plot_examples(input, target, model, num_examples=3)


    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
