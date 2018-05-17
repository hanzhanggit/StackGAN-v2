import re
import numpy as np
import scipy.misc
import os

from PIL import Image, ImageDraw, ImageFont


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass


def drawCaption(img, caption, max_len):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('/eai/project/.fonts/FreeMono.ttf', 30)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    d.text((10, 256), 'Stage-I', font=fnt, fill=(0, 0, 0, 255))
    d.text((10, 512), 'Stage-II', font=fnt, fill=(0, 0, 0, 255))
    d.text((10, 768), 'Stage-III', font=fnt, fill=(0, 0, 0, 255))

    caption = caption.split(' ')

    cap1 = ' '.join(caption[:max_len])
    cap2 = ' '.join(caption[max_len + 1:])
    d.text((256, 10), cap1, font=fnt, fill=(0, 0, 0, 255))
    d.text((256, 60), cap2, font=fnt, fill=(0, 0, 0, 255))

    return img_txt


def save_images_with_captions(
        lr_sample_batchs, hr_sample_batchs, sr_sample_batchs,
        texts_batch, batch_size, max_len,
        startID, save_dir=None):

    if save_dir and not os.path.isdir(save_dir):
        print('Make a new folder: ', save_dir)
        mkdir_p(save_dir)

    # Save up to 16 samples for each text embedding/sentence
    img_shape = sr_sample_batchs[0][0].shape
    super_images = []
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', texts_batch[j]):
            continue

        padding = 255 + np.zeros(img_shape)
        row1 = [padding]
        row2 = [padding]
        row3 = [padding]
        # First row with up to 8 samples
        for i in range(np.minimum(8, len(lr_sample_batchs))):
            lr_img = lr_sample_batchs[i][j]
            hr_img = hr_sample_batchs[i][j]
            sr_img = sr_sample_batchs[i][j]

            sr_img = (sr_img + 1.0) * 127.5
            lr_re_sample = scipy.misc.imresize(lr_img, sr_img.shape[:2])
            hr_re_sample = scipy.misc.imresize(hr_img, sr_img.shape[:2])
            row1.append(lr_re_sample)
            row2.append(hr_re_sample)
            row3.append(sr_img)

        row1 = np.concatenate(row1, axis=1)
        row2 = np.concatenate(row2, axis=1)
        row3 = np.concatenate(row3, axis=1)
        superimage = np.concatenate([row1, row2, row3], axis=0)

        # Second 8 samples with up to 8 samples
        if len(lr_sample_batchs) > 8:
            row1 = [padding]
            row2 = [padding]
            for i in range(8, len(lr_sample_batchs)):
                lr_img = lr_sample_batchs[i][j]
                hr_img = hr_sample_batchs[i][j]
                hr_img = (hr_img + 1.0) * 127.5
                re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                row1.append(re_sample)
                row2.append(hr_img)
            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            super_row = np.concatenate([row1, row2], axis=0)

            superimage2 = np.zeros_like(superimage)

            superimage2[:super_row.shape[0],
                        :super_row.shape[1],
                        :super_row.shape[2]] = super_row
            mid_padding = np.zeros((64, superimage.shape[1], 3))
            superimage =\
                np.concatenate([superimage, mid_padding, superimage2], axis=0)

        top_padding = 255 + np.zeros((128, superimage.shape[1], 3))
        superimage = np.concatenate([top_padding, superimage], axis=0)

        fullpath = '%s/sentence_%04d.jpg' % (save_dir, startID + j)
        superimage = drawCaption(np.uint8(superimage), texts_batch[j], max_len)

        if save_dir:
            scipy.misc.imsave(fullpath, superimage)
        super_images.append(superimage)

    return super_images
