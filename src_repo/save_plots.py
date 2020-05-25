import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn
import matplotlib.pyplot as plt
import sys
import glob
from global_config import *


def save_plot(event_file, tags, output_dir):
    """从event_file读取Tensorboard数据，并将其保存成图标
    """
    if not os.path.isfile(event_file):
        print(f'{event_file} 不存在')
        return
    os.makedirs(output_dir, exist_ok=True)
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 1,
        'scalars': 100,
        'histograms': 1
    }
    event_acc = EventAccumulator(event_file, tf_size_guidance)
    event_acc.Reload()
    for tag in tags:
        try:
            event = event_acc.Scalars(tag)
        except KeyError:
            print(f'Tag {tag} does not exist!')
            continue
        steps = []
        data = []
        for event in event:
            steps.append(event[1])
            data.append(event[2])
        seaborn.set()
        plt.clf()
        plt.rcParams['figure.figsize'] = (16, 8)
        plt.plot(steps, data)
        plt.xlabel('steps')
        plt.ylabel(f'{tag}')
        plt.savefig(os.path.join(output_dir, f"{tag.replace('/', '_')}.png"))


if __name__ == '__main__':
    tag_list = ['LearningRate/LearningRate/learning_rate', 'Losses/TotalLoss', 'Losses/Loss/localization_loss',
                'Losses/Loss/classification_loss']
    event_file_list = glob.glob(f'{project_root}/training/events.out.tfevents.*')
    if len(event_file_list) == 0:
        print('Tensorboard event file not found!')
        exit(-1)
    save_plot(event_file_list[0], tag_list, '/project/train/result-graphs')
