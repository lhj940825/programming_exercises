import random

from main import load_dataset, build_decision_tree
import os
import json
from tree_graph import generate_graph, dot
import numpy as np


def process_data_with_random(data):
    data = np.copy(data)
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            if np.isnan(data[i][j]):
                data[i][j] = random.random()
    return data


def process_data_with_average(data):
    data = np.copy(data)
    num_row = len(data)
    num_col = len(data[0] - 1)

    for j in range(num_col):
        has_nan = False
        total = 0
        count = 0
        for i in range(num_row):
            d = data[i][j]
            if np.isnan(d):
                has_nan = True
            else:
                total += d
                count += 1

        average = total / count
        if has_nan:
            for i in range(num_row):
                if np.isnan(data[i][j]):
                    data[i][j] = average
    return data


def process_data_with_median(data):
    data = np.copy(data)
    num_row = len(data)
    num_col = len(data[0] - 1)

    for j in range(num_col):
        has_nan = False
        arr = []
        for i in range(num_row):
            d = data[i][j]
            if np.isnan(d):
                has_nan = True
            else:
                arr.append(d)

        arr.sort()
        median = arr[int(len(arr) / 2)]
        if has_nan:
            for i in range(num_row):
                if np.isnan(data[i][j]):
                    data[i][j] = median
    return data


def work(data, out_json_filename, out_dot_filename):
    indexes = [i for i in range(len(data[0]) - 1)]
    tree = build_decision_tree(headers, indexes, data)

    # save tree to file
    jsonstr = json.dumps(tree)
    with open(out_json_filename, 'w') as f:
        f.write(jsonstr)

    dot.clear()
    generate_graph(-1, '<', tree, 0)
    dot.format = 'png'
    dot.view(out_dot_filename, 'missing_value_output')


if __name__ == '__main__':
    folder = 'data/'
    output_folder = "missing_value_output"
    train_data_filename = os.path.join(folder, 'gene_expression_with_missing_values.csv')

    headers, origin_data = load_dataset(train_data_filename)

    train_data_random = process_data_with_random(origin_data)
    work(train_data_random, os.path.join(output_folder, 'random.json'), "random.dot")

    train_data_average = process_data_with_average(origin_data)
    work(train_data_average, os.path.join(output_folder, 'average.json'), "average.dot")

    train_data_median = process_data_with_median(origin_data)
    work(train_data_median, os.path.join(output_folder, 'median.json'), "median.dot")
