import argparse
import os
from collections import OrderedDict
from glob import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
# pip install PyYaml
import yaml
# https://github.com/albumentations-team/albumentations
# pip install -U albumentations
# python3.6+
from albumentations.augmentations import transforms

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import albumentations as albu

import MISS
import archs
import archs_swin_unet
import archs_u2uet
import archs_vit
import losses
from dataset import Dataset
from metrics_standard import iou_score,  acc_socre, sensitivity_socre, specificity_socre, precision_socre, dice_score,iou_score1,  sensitivity_score1,  precision_score1, dice_score1,specificity_score1,acc_score1


from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset dsb2018_96 
--arch NestedUNet

"""

# 参数
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=6, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
    # parser.add_argument('--arch', default='AttentionUnet',
    # parser.add_argument('--arch', default='UNet',
    # parser.add_argument('--arch', default='MISS',
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='U2NET',
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='SegFormer',
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='swin',
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='Doubleunet',
    parser.add_argument('--arch', '-a', metavar='ARCH', default='CACDUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
    # parser.add_argument('--input_w', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
    # parser.add_argument('--input_h', default=224, type=int,
                        help='image height')
    
    # loss
    # parser.add_argument('--loss', default='BCEDiceLoss',
    # parser.add_argument('--loss', default='BCEWithLogitsLoss',
    parser.add_argument('--loss', default='BCEDiceLoss',
    # parser.add_argument('--loss', default='IoULoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='final_skin2018',
                        help='dataset name')
    # parser.add_argument('--img_ext', default='.png',
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    # parser.add_argument('--optimizer', default='SGD',
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'AdamW']) +
                        ' (default: Adam)')
    # 当使用Adam优化器时建议设置  Init_lr=1e-4
    # 当使用SGD优化器时建议设置   Init_lr=1e-2
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    #weight_decay lr的0.01倍
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    config = parser.parse_args()

    return config





def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  # 'sensitivity': AverageMeter(),
                  # 'accuracy': AverageMeter()
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice1 = dice_score(output, target)
        # iou = iou_score1(output, target)
        # dice1 = dice_score1(output, target)


        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice1, input.size(0))
        # avg_meters['sensitivity'].update(sensitivity1, input.size(0))
        # avg_meters['accuracy'].update(accuracy1, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            # ('accuracy', avg_meters['accuracy'].avg),
            # ('sensitivity', avg_meters['sensitivity'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        # del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        # ('accuracy', avg_meters['accuracy'].avg),
                        # ('sensitivity', avg_meters['sensitivity'].avg),
                        ])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter(),
                  # 'compute_acc': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice1 = dice_score(output[-1], target)
                accuracy1 = acc_socre(output[-1], target)
                # compute_acc = compute_acc(output[-1], target)
                sensitivity1 = sensitivity_socre(output[-1], target)

            else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou = iou_score(output, target)
                    dice1 = dice_score(output, target)
                    accuracy1 = acc_socre(output, target)
                    sensitivity1 = sensitivity_socre(output, target)
                    # iou = iou_score1(output, target)
                    # dice1 = dice_score1(output, target)
                    # accuracy1 = acc_score1(output, target)
                    # sensitivity1 = sensitivity_score1(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice1, input.size(0))
            avg_meters['accuracy'].update(accuracy1, input.size(0))
            # avg_meters['compute_acc'].update(compute_acc, input.size(0))
            avg_meters['sensitivity'].update(sensitivity1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('accuracy', avg_meters['accuracy'].avg),
                # ('compute_acc', avg_meters['compute_acc'].avg),
                ('sensitivity', avg_meters['sensitivity'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
            # del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        # ('compute_acc', avg_meters['compute_acc'].avg),
                        ('sensitivity', avg_meters['sensitivity'].avg)
                        ])


def main():
    #  第一步 参数获取
    config = vars(parse_args())
    # random.seed(config['seed'])
    # np.random.seed(config['seed'])
    # torch.manual_seed(config['seed'])
    # torch.cuda.manual_seed((config['seed']))
    #  输出地址
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    #  打印参数
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    # 输出当前模型的参数到yml文件
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()#WithLogits 就是先将输出结果经过sigmoid再交叉熵
    elif config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()


    # 激活函数定义

    cudnn.benchmark = True

    # create model          =========================================   模型核心++++++++++
    print("=> creating model %s" % config['arch'])
    if (config['arch'] == 'U2NET'):
        model = archs_u2uet.U2NET()
    elif (config['arch'] == 'swin'):
        model = archs_swin_unet.SwinUnet()
    elif (config['arch'] == 'MISS'):
        model = MISS.MISSFormer()
    else:
        model = archs.__dict__[config['arch']](
            config['num_classes'],
                                               # config['input_channels']
                                               # ,
                                               # config['deep_supervision']
                                               )
        # model = archs.__dict__[config['arch']](config['num_classes'])
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # 原始图像和标注后的
    train_img_ids = glob(os.path.join('inputs', config['dataset'], 'train/images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', config['dataset'], 'val/images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    test_img_ids = glob(os.path.join('inputs', config['dataset'], 'test/images', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    #                     RandomRotation(degrees=10),#随机旋转0到10度
    #                     RandomHorizontalFlip(),#随机翻转
    #                     ContrastTransform(0.1),#随机调整图片的对比度
    #                     BrightnessTransform(0.1),#随机调整图片的亮度


    #     albu.HorizontalFlip(),
    #     albu.OneOf([
    #         albu.RandomContrast(),
    #         albu.RandomGamma(),
    #         albu.RandomBrightness(),
    #         ], p=0.3),
    #     albu.OneOf([
    #         albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #         albu.GridDistortion(),
    #         albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    #         ], p=0.3),
    #     albu.ShiftScaleRotate(),
    #     albu.Resize(img_size,img_size,always_apply=True),
    #数据增强：
    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        OneOf([
            albu.HueSaturationValue(),
            albu.RandomBrightness(),
            albu.RandomContrast(),
        ], p=1),#按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'train/images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'train/masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'val/images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'val/masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)#不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs',  config['dataset'], 'test/images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test/masks'),
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
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),

        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'],   val_log['loss'], val_log['iou'], val_log['dice'] ))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
            #     直接验证test数据

            # avg_meters = {'loss': AverageMeter(),
            #               'iou': AverageMeter(),
            #               'dice': AverageMeter(),
            #               'accuracy': AverageMeter(),
            #               'specificity': AverageMeter(),
            #               'sensitivity': AverageMeter(),
            #               'jaccard': AverageMeter(),
            #               'precision': AverageMeter(),
            #               'iou1': AverageMeter(),
            #               'dice1': AverageMeter(),
            #               'sensitivity1': AverageMeter(),
            #               'precision1': AverageMeter(),
            #               'specificity1': AverageMeter(),
            #               'accuracy1': AverageMeter()
            #               }
            # for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            #     input = input.cuda()
            #     target = target.cuda()
            #     # compute output
            #     if config['deep_supervision']:
            #         output = model(input)[-1]
            #     else:
            #             output = model(input)
            #             iou = iou_score(output, target)
            #             dice = dice_score(output, target)
            #             accuracy = acc_socre(output, target)
            #             sensitivity = sensitivity_socre(output, target)
            #             specificity = specificity_socre(output, target)
            #             precision = precision_socre(output, target)
            #             iou1 = iou_score1(output, target)
            #             dice1 = dice_score1(output, target)
            #             sensitivity1 = sensitivity_score1(output, target)
            #             precision1 = precision_score1(output, target)
            #             specificity1 = specificity_score1(output, target)
            #             accuracy1 = acc_score1(output, target)
            #     avg_meters['iou'].update(iou, input.size(0))
            #     avg_meters['dice'].update(dice, input.size(0))
            #     avg_meters['accuracy'].update(accuracy, input.size(0))
            #     avg_meters['sensitivity'].update(sensitivity, input.size(0))
            #     avg_meters['specificity'].update(specificity, input.size(0))
            #     avg_meters['precision'].update(precision, input.size(0))
            #     avg_meters['iou1'].update(iou1, input.size(0))
            #     avg_meters['dice1'].update(dice1, input.size(0))
            #     avg_meters['sensitivity1'].update(sensitivity1, input.size(0))
            #     avg_meters['precision1'].update(precision1, input.size(0))
            #     avg_meters['specificity1'].update(specificity1, input.size(0))
            #     avg_meters['accuracy1'].update(accuracy1, input.size(0))
            #
            #
            # print('IoU: %.4f' % avg_meters['iou'].avg)
            # print('IoU1: %.4f' % avg_meters['iou1'].avg)
            # print('dice: %.4f' % avg_meters['dice'].avg)
            # print('dice1: %.4f' % avg_meters['dice1'].avg)
            # print('accuracy: %.4f' % avg_meters['accuracy'].avg)
            # print('accuracy1: %.4f' % avg_meters['accuracy1'].avg)
            # print('sensitivity: %.4f' % avg_meters['sensitivity'].avg)
            # print('sensitivity1: %.4f' % avg_meters['sensitivity1'].avg)
            # print('specificity: %.4f' % avg_meters['specificity'].avg)
            # print('specificity1: %.4f' % avg_meters['specificity1'].avg)
            # print('precision: %.4f' % avg_meters['precision'].avg)
            # print('precision1: %.4f' % avg_meters['precision1'].avg)

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
