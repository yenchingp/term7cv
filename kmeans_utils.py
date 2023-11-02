import os
import random


input_path = "/home/zach/PycharmProjects/term7cv/dataset/objects"
subset_ratio = 0.01
seed = 12345
data_set = 'train'


def create_subset(input_path: str, data_set: str, subset_ratio: float, seed: int) -> list:
    set_list = [x for x in os.listdir(input_path) if data_set in x]
    random.Random(seed).shuffle(set_list)

    subset_size = int(subset_ratio * len(set_list))

    subset_list = set_list[:subset_size]

    return subset_list


subset_list = create_subset(input_path, data_set, subset_ratio, seed)

print(subset_list)
