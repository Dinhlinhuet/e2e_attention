from __future__ import absolute_import

import logging
import re

import tensorflow as tf
import cv2
import sys
# sys.path.append('/home/ubuntu/da_lih/gen_hw/')
sys.path.append('/media/warrior/Ubuntu/corp_workspace/attention-ocr/')

from six import b
# from dataloader.generate.image import HandwrittenLineGenerator
# import dataloader.utils.constants as constants

img_dir = '%s/datasets/test_on_real/'%(sys.path[-1])
ALLOWED_CHARS='%s/dataloader/dev/charset_codes.txt'%(sys.path[-1])
TEST_PKL_FILE='%s/dataloader/dev/datefield.pkl'%(sys.path[-1])
PRINTED_DIR = '%s/dataloader/dev/PRINTED/'%(sys.path[-1])
print('TEST_PKL_FILE',TEST_PKL_FILE)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_printed_data():
    upper_chars = {char:char for char in constants.UPPER_CASES_CHARS}
    chars = {'.':'dot','-':'hyphen','/':'slash','~':'tilde'}
    chars.update(upper_chars)
    print('chars', chars)
    dict_printed = {char:cv2.imread('%s%s.png'%(PRINTED_DIR, char_name),0) for (char, char_name) in chars.items()}
    return dict_printed	

def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=True, save_filename=False):

    logging.info('Building a datasets from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0
    # lineocr = HandwrittenLineGenerator(allowed_chars=ALLOWED_CHARS)
    # lineocr.load_character_database(TEST_PKL_FILE)
    # lineocr.initialize()
    # print('lineocr', lineocr.char_2_imgs.keys())
    # printed_data = read_printed_data()
    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'(\S+)\s(.*)', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()
            img_path = '%s%s'%(img_dir, img_path)
            img_tmp = cv2.imread(img_path)
            hei, wid,_ = img_tmp.shape
            resize = cv2.resize(img_tmp,(160, int(hei * 160/wid)))
            cv2.imwrite(img_path, resize)
            print('imgpath',img_path)
            # cv2.imshow(img_path,resize)
            # cv2.waitKey(0)
            with open(img_path, 'rb') as img_file:
               img = img_file.read()
            # img = lineocr._generate_sequence_image(label,printed_data)[0]
            # tmp_path = '/home/ubuntu/da_lih/gen_hw/dataloader/dev/tmp.png'
            # cv2.imwrite(tmp_path,img)
            # with open(tmp_path, 'rb') as img_file:
            #     img = img_file.read()
            if force_uppercase:
                label = label.upper()

            if len(label) > len(longest_label):
                longest_label = label

            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(label.encode())
            if save_filename:
                feature['comment'] = _bytes_feature(img_path.encode())

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()
