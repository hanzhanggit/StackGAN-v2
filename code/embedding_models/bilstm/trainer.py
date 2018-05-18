import argparse
import os
import random
import yaml

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.autograd import Variable
from pprint import pprint

from embedding_models.bilstm.model import BiLSTMClassifier
from miscc.datasets import TextClassesDataset, SplitType
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p, get_output_dir


from tensorboardX import SummaryWriter as FileWriter


class BiLSTMTrainer(object):
    def __init__(self, output_dir, experiment_cfg, model_cfg):
        if cfg.TRAIN.FLAG:
            print('Creating the directories...')
            self.outputdir = output_dir
            self.model_dir = os.path.join(output_dir, 'Model')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            with open(os.path.join(output_dir, 'config.yaml'), 'w') as currect_cfg_file:
                yaml.dump(cfg, currect_cfg_file, default_flow_style=False)

            self.summary_writer = FileWriter(self.log_dir)
            self.model = BiLSTMClassifier.from_configs(experiment_cfg, model_cfg)
            if cfg.CUDA:
                self.model.cuda()

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.snapshot_interval_validation = cfg.TRAIN.SNAPSHOT_INTERVAL_VALIDATION
        self.checkpoint_step = None

        self.__init_gpus(experiment_cfg)

    def __init_gpus(self, cfg):
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])

    def train(self, train_dataloader, valid_dataloader):
        pprint(self.model)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(
            self.model.parameters(), betas=(0.9, 0.98), eps=1e-09,
            weight_decay=0.001
        )

        train_losses = []
        eval_losses = []
        for epoch in range(1, self.max_epoch + 1):
            train_loss, train_acc = self.step(
                epoch, loss_function, optimizer, train_dataloader, 'train', train_losses)
            print('[%d] Train: loss: %.3f acc: %.3f' % (epoch, train_loss, train_acc))

            if epoch % self.snapshot_interval == 0 or epoch == 1:
                val_loss, val_acc = self.step(
                    epoch, loss_function, optimizer, valid_dataloader, 'valid', eval_losses, update=False)
                print('[%d] Validation: loss: %.3f acc: %.3f' % (epoch, val_loss, val_acc))
                print('[%d] Saving the model' % epoch)
                self.save_model(epoch)

        self.summary_writer.close()

    def step(self, epoch, loss_function, optimizer, data_loader, desc, losses=None, update=True):
        num_loss = 0
        n_total_examples = 0.
        n_total_correct = 0.

        for i, (txt, cls) in enumerate(tqdm(data_loader, desc, leave=False), 0):
            ######################################################
            # (1) Prepare training data
            ######################################################
            txt = Variable(txt)
            cls = Variable(cls)
            if cfg.CUDA:
                txt = txt.cuda()
                cls = cls.cuda()

            #######################################################
            # (2) Make predictions
            ######################################################
            log_probs = self.model(txt)
            pred = log_probs.max(1)[1]

            ############################
            # (3) Update the  network
            ############################
            if update:
                self.model.zero_grad()

            loss = loss_function(log_probs, cls)
            losses.append(loss.data[0])

            if update:
                loss.backward()
                optimizer.step()

            # calculate running loss and accuracy
            num_loss += loss.data[0]
            n_total_correct += pred.data.eq(cls.data).sum()
            n_total_examples += pred.size(0)
            num_loss / float(n_total_examples)
            num_acc = float(n_total_correct) / float(n_total_examples)

            if i % self.snapshot_interval == 0 and losses:
                self.summary_writer.add_scalar("%s_loss" % desc, losses[-1], len(losses))
                self.summary_writer.add_scalar("%s_acc" % desc, num_acc, len(losses))

        return num_loss, num_acc

    def save_model(self, epoch):
        print('Saving encoder and classifier')
        torch.save(
            self.model.encoder.state_dict(),
            '%s/encoder_%d.pth' % (self.model_dir, epoch))
        torch.save(
            self.model.state_dict(),
            '%s/classifier_%d.pth' % (self.model_dir, epoch))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/fashion_bilstm_cls.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        print("Loading configs from file")
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    print('Using config:')
    pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        train_dataset = TextClassesDataset(
            cfg.DATA_DIR,
            cfg.DATASET_NAME,
            cfg.TEXT.VOCAB_PATH,
            cfg.TEXT.MAX_LEN,
            split_name=SplitType.train)

        valid_dataset = TextClassesDataset(
            cfg.DATA_DIR,
            cfg.DATASET_NAME,
            cfg.TEXT.VOCAB_PATH,
            cfg.TEXT.MAX_LEN,
            split_name=SplitType.valid)

        assert train_dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        output_dir = get_output_dir(cfg)
        with open(cfg.TEXT_EMBEDDING_MODEL_CFG, 'rt') as file_:
            model_config = yaml.load(file_)
            print('Using following config for embedding:')
            pprint(model_config)
            model_config['n_src_vocab'] = train_dataset.vocab_size
            model_config['n_classes'] = train_dataset.n_classes

        algo = BiLSTMTrainer(output_dir, cfg, model_config)
        algo.train(train_dataloader, valid_dataloader)
    else:
        print("OR ELSE!!")
