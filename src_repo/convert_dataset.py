from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import shutil
import pathlib
import random

import PIL
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
import io
from collections import namedtuple, OrderedDict

from global_config import *
from tf_models.research.object_detection.utils import dataset_util

data_dir = '/home/data/12'
train_data_dir = os.path.join(project_root, 'dataset/images/train/')
valid_data_dir = os.path.join(project_root, 'dataset/images/valid')
annotations_dir = os.path.join(project_root, 'dataset/annotations')
supported_fmt = ['.jpg']


def class_text_to_int(row_label):
    if row_label == 'mask':  # 标注文件里面的标签名称
        return 1
    if row_label == 'head':
        return 2
    if row_label == 'back':
        return 3
    if row_label == 'mid_mask':
        return 4
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def change_pic_jpg():
    th_list = os.listdir(data_dir)
    for pic in th_list:
        # 分离xml文件
        if os.path.splitext(pic)[1] == '.xml':
            pass
        else:
            # 转换并分离jpg文件
            img = Image.open(os.path.join(data_dir, pic))
            img = img.convert('RGB')
            if os.path.splitext(pic)[1] == '.PNG':
                img.save(os.path.join(data_dir, pic.replace('.PNG', '.jpg')))
            elif os.path.splitext(pic)[1] == '.JPG':
                img.save(os.path.join(data_dir, pic.replace('.JPG', '.jpg')))
            elif os.path.splitext(pic)[1] == '.png':
                img.save(os.path.join(data_dir, pic.replace('.png', '.jpg')))
            elif os.path.splitext(pic)[1] == '.jpg':
                img.save(os.path.join(data_dir, pic))
            else:
                print('wrong!', os.path.splitext(pic)[1], 'is exist!')
                break


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def csv_to_record(output_path, img_path, csv_input):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), img_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def xml_to_csv(data_list):
    """将data_list表示的(图片, 标签)对转换成pandas.Dataframe记录
    """
    xml_list = []
    for data in data_list:
        tree = ET.parse(data['label'])
        root = tree.getroot()
        try:
            img = Image.open(data['image'])
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            print(f'打开{data["image"]}出错!')
            continue
        width, height = img.size
        img.close()
        for member in root.findall('object'):
            value = (data['image'],
                     width,
                     height,
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    os.makedirs(project_root, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(valid_data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    if not os.path.exists(data_dir):
        print(f'{data_dir} 不存在!')
        exit(-1)

    change_pic_jpg()
    print('create all.jpg!!')

    # 遍历数据集目录下所有xml文件及其对应的图片
    dataset_path = pathlib.Path(data_dir)
    found_data_list = []
    for xml_file in dataset_path.glob('**/*.xml'):
        possible_images = [xml_file.with_suffix(suffix) for suffix in supported_fmt]
        supported_images = list(filter(lambda p: p.is_file(), possible_images))
        if len(supported_images) == 0:
            print(f'找不到对应的图片文件：`{xml_file.as_posix()}`')
            continue
        found_data_list.append({'image': supported_images[0], 'label': xml_file})

    # 随机化数据集，将数据集拆分成训练集和验证集，并将其拷贝到/project/train/src_repo/dataset下
    random.shuffle(found_data_list)
    train_data_count = len(found_data_list) * 4 / 5
    train_data_list = []
    valid_data_list = []
    for i, data in enumerate(found_data_list):
        if i < train_data_count:  # 训练集
            dst = train_data_dir
            data_list = train_data_list
        else:  # 验证集
            dst = valid_data_dir
            data_list = valid_data_list
        image_dst = (pathlib.Path(dst) / data['image'].name).as_posix()
        label_dst = (pathlib.Path(dst) / data['label'].name).as_posix()
        shutil.copy(data['image'].as_posix(), image_dst)
        shutil.copy(data['label'].as_posix(), label_dst)
        data_list.append({'image': image_dst, 'label': label_dst})

    # 将XML转换成CSV格式
    train_xml_df = xml_to_csv(train_data_list)
    train_xml_df.to_csv(os.path.join(annotations_dir, 'train_labels.csv'), index=False)
    valid_xml_df = xml_to_csv(valid_data_list)
    valid_xml_df.to_csv(os.path.join(annotations_dir, 'valid_labels.csv'), index=False)
    print('Successfully converted xml to csv.')

    # 将数据集转换成tf record格式
    csv_to_record(os.path.join(annotations_dir, 'train.record'), train_data_dir,
                  os.path.join(annotations_dir, 'train_labels.csv'))
    csv_to_record(os.path.join(annotations_dir, 'valid.record'), valid_data_dir,
                  os.path.join(annotations_dir, 'valid_labels.csv'))

    #
    with open(os.path.join(annotations_dir, 'label_map.pbtxt'), 'w') as f:
        label_map = """
item {
    id: 1
    name: "mask"
}
item {
    id: 2
    name: "head"
}
item {
    id: 3
    name: "back"
}
item {
    id: 4
    name: "mid_mask"
}
        """
        f.write(label_map)
