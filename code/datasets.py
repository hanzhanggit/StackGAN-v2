from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import os.path
import random
import six
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from miscc.config import cfg


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def transform_img(img, imsize, transform=None, normalize=None):
    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


class ImageFolder(data.Dataset):
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)
        imgs = self.make_dataset(classes, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print('num_classes', self.num_classes)

    def find_classes(self, dir, custom_classes):
        classes = []

        for d in os.listdir(dir):
            if os.path.isdir:
                if custom_classes is None or d in custom_classes:
                    classes.append(os.path.join(dir, d))
        print('Valid classes: ', len(classes), classes)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, classes, class_to_idx):
        images = []
        for d in classes:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[d])
                        images.append(item)
        print('The number of images: ', len(images))
        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs_list = get_imgs(path, self.imsize,
                             transform=self.transform,
                             normalize=self.norm)

        return imgs_list

    def __len__(self):
        return len(self.imgs)


class LSUNClass(data.Dataset):
    def __init__(self, db_path, base_size=64,
                 transform=None, target_transform=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            print('length: ', self.length)
        cache_file = db_path + '/cache'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
            print('Load:', cache_file, 'keys: ', len(self.keys))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        imgs = get_imgs(buf, self.imsize,
                        transform=self.transform,
                        normalize=self.norm)
        return imgs

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = []

        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % \
            (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs, embedding, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


class SplitType:
    train = 'train'
    valid = 'validation'
    baby = 'baby'


class Dictionary(object):
    ''' Holds info about vocabulary '''
    def __init__(self, path):
        print(path)
        self.word2idx = {'UNK': 0, '<eos>': 1, '<pad>': 2}
        self.idx2word = ['UNK', '<eos>', '<pad>']
        self.counter = Counter()
        self.total = len(self.idx2word)
        with open(path, 'r') as vocab:
            for i, line in enumerate(vocab.readlines()):
                word = line.decode('latin1').strip().split('\t')[0]
                self.add_word(word)
        print("Loaded dictionary with %d words" % len(self.idx2word))

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def _word2id(self, word):
        if word not in self.word2idx:
            # print "It's unkown"
            return self.word2idx['UNK']
        return self.word2idx[word]

    def words2ids(self, words):
        ids = np.asarray(map(self._word2id, words))
        return ids

    def __len__(self):
        return len(self.idx2word)


class SsenseDataset(data.Dataset):
    def __init__(self, data_dir, vocab_path, max_len, dataset_name,
                 base_size=64, split_name='train', transform=None):
        self.category2idx = {
            'POCKET SQUARES & TIE BARS': 38, 'WALLETS & CARD HOLDERS': 48, 'FINE JEWELRY': 19, 'JACKETS & COATS': 5,
            'HATS': 10, 'TOPS': 0, 'SOCKS': 39, 'SHOULDER BAGS': 21, 'LOAFERS': 37, 'SHIRTS': 1, 'TIES': 8,
            'BRIEFCASES': 40, 'BELTS & SUSPENDERS': 14, 'TOTE BAGS': 27, 'TRAVEL BAGS': 47,
            'DUFFLE & TOP HANDLE BAGS': 32, 'BAG ACCESSORIES': 46, 'KEYCHAINS': 26,
            'DUFFLE BAGS': 45, 'SNEAKERS': 17, 'PANTS': 3, 'SWEATERS': 4,
            'JEWELRY': 23, 'SHORTS': 2, 'ESPADRILLES': 43, 'MESSENGER BAGS': 44,
            'EYEWEAR': 31, 'HEELS': 41, 'MONKSTRAPS': 36, 'MESSENGER BAGS & SATCHELS': 42,
            'FLATS': 33, 'BLANKETS': 22, 'POUCHES & DOCUMENT HOLDERS': 29,
            'DRESSES': 11, 'JUMPSUITS': 13, 'UNDERWEAR & LOUNGEWEAR': 25,
            'BOAT SHOES & MOCCASINS': 28, 'CLUTCHES & POUCHES': 20, 'JEANS': 6,
            'SWIMWEAR': 12, 'SUITS & BLAZERS': 7, 'LINGERIE': 16, 'GLOVES': 18, 'BOOTS': 34,
            'LACE UPS': 35, 'SCARVES': 15, 'SANDALS': 30, 'BACKPACKS': 24, 'SKIRTS': 9
        }

        self.max_desc_length = max_len
        self.dictionary = Dictionary(vocab_path)
        self.split_name = split_name

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = []

        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data_size = 0
        self.dataset_name = dataset_name

        split_dir = os.path.join(data_dir, self.split_name)
        print("Split Dir: %s" % split_dir)

        self.images = self.load_h5_images(split_dir)
        self.categories = self.load_categories(split_dir)
        self.descriptions = self.load_descriptions(split_dir)

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def pad_sequence(self, seq):
        eos_id = self.dictionary.word2idx['<eos>']
        pad_id = self.dictionary.word2idx['<pad>']
        if len(seq) < self.max_desc_length:
            seq = np.concatenate([seq, [eos_id], [pad_id] * (self.max_desc_length - len(seq) - 1)])
            # seq = np.concatenate([seq, [eos_id] * (self.max_desc_length - len(seq))])
            return seq
        elif len(seq) >= self.max_desc_length:
            seq = np.concatenate([seq[:self.max_desc_length - 1], [eos_id]])
        return seq

    def load_descriptions(self, data_dir):
        filename = '%s_%s.h5' % (self.dataset_name, self.split_name)
        print("Loading descriptions file from %s" % filename)

        with h5py.File(os.path.join(data_dir, filename)) as data_file:
            descriptions = np.asarray(data_file['input_description'].value)

        print('Loaded descriptions, shape: ', descriptions.shape)

        return descriptions

    def load_categories(self, data_dir):
        filename = '%s_%s.h5' % (self.dataset_name, self.split_name)
        print("Loading Categories file from %s" % filename)
        with h5py.File(os.path.join(data_dir, filename)) as data_file:
            categories = np.asarray(data_file['input_category'].value)
            print('loaded Categories, shape: ', categories.shape)

        return categories

    def load_h5_images(self, data_dir):
        filename = '%s_%s.h5' % (self.dataset_name, self.split_name)
        print("Loading image file from %s" % filename)
        with h5py.File(os.path.join(data_dir, filename)) as data_file:
            images = np.asarray(data_file['input_image'].value)
            print('loaded images, shape: ', images.shape)
            self.data_size = images.shape[0]
        return images

    def old__getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGB')
        img = self.get_img(img)

        desc = self.descriptions[index][0].decode('latin1')
        desc_ids = self.dictionary.words2ids(desc.split())
        desc_ids = self.pad_sequence(desc_ids)
        desc_tensor = torch.from_numpy(desc_ids).type(torch.LongTensor)

        img_tensor = img.type(torch.FloatTensor)

        return img_tensor, desc_tensor, desc

    def prepair_training_pairs(self, index):
        img = self.images[index]
        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGB')
        imgs = transform_img(img, self.imsize, self.transform, normalize=self.norm)

        desc = self.descriptions[index][0].decode('latin1')
        desc_ids = self.dictionary.words2ids(desc.split())
        desc_ids = self.pad_sequence(desc_ids)
        desc_tensor = torch.from_numpy(desc_ids).type(torch.LongTensor)

        wrong_ix = random.randint(0, self.data_size - 1)
        if(self.categories[index] == self.categories[wrong_ix]):
            wrong_ix = random.randint(0, self.data_size - 1)
        wrong_img = self.images[index]
        wrong_img = Image.fromarray(wrong_img.astype('uint8'), 'RGB').convert('RGB')
        wrong_imgs = transform_img(wrong_img, self.imsize, self.transform, normalize=self.norm)

        return imgs, wrong_imgs, desc_tensor, desc  # captions

    def prepair_test_pairs(self, index):
        img = self.images[index]
        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGB')
        imgs = transform_img(img, self.imsize, self.transform, normalize=self.norm)

        desc = self.descriptions[index][0].decode('latin1')
        desc_ids = self.dictionary.words2ids(desc.split())
        desc_ids = self.pad_sequence(desc_ids)
        desc_tensor = torch.from_numpy(desc_ids).type(torch.LongTensor)

        return imgs, desc_tensor, desc  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return self.data_size
