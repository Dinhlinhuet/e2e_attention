from __future__ import absolute_import

import logging
import re

import tensorflow as tf
import cv2
import sys

from six import b

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
    
    prefix_img ='/data/fixed_form_hw_data/data/'
    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0
    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image path
            #print(line)
            line_match = re.match(r'(\S+)\s(.*)', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()
#            (img_path, label) = line.split()
            img_path = prefix_img+img_path
#            with open(img_path, 'rb') as img_file:
#                img = img_file.read()
            with open(img_path, 'rb') as img_file:
                img = img_file.read()
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
