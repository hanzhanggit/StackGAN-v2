from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


from miscc.config import cfg, cfg_from_file
from miscc.classes import CLASS_DIC


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/fashion_3stages.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    output_dir_root = cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else '../output'
    output_dir = '%s/%s_%s_%s' % \
            (output_dir_root, cfg.EXPERIMENT_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    image_transform = None

    if cfg.GAN.B_CONDITION:  # text to image task
        from datasets import SsenseDataset, SplitType
        dataset = SsenseDataset(
            cfg.DATA_DIR,
            cfg.TEXT.VOCAB_PATH,
            cfg.TEXT.MAX_LEN,
            cfg.DATASET_NAME,
            split_name=SplitType.baby,
            base_size=cfg.TREE.BASE_SIZE,
            transform=image_transform)
    else:
        from datasets import ImageFolder
        dataset = ImageFolder(
            cfg.DATA_DIR, split_dir='train',
            custom_classes=CLASS_DIC[cfg.DATASET_NAME],
            base_size=cfg.TREE.BASE_SIZE,
            transform=image_transform)

    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    if cfg.GAN.B_CONDITION:
        from trainer import condGANTrainer as trainer
    else:
        from trainer import GANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    ''' Running time comparison for 10epoch with batch_size 24 on birds dataset
        T(1gpu) = 1.383 T(2gpus)
            - gpu 2: 2426.228544 -> 4min/epoch
            - gpu 2 & 3: 1754.12295008 -> 2.9min/epoch
            - gpu 3: 2514.02744293
    '''
