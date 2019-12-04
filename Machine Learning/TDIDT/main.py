#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from tree_graph import generate_graph, dot
import json


def load_dataset(filename):
    """
    load dataset from csv file
    :param filename:
    :return: headers of name
    """
    dataset = pd.read_csv(filename, delimiter=',')
    headers = list(dataset.columns.values)
    data = dataset.values
    return headers, data


def cal_entropy(data):
    """
    calculate the entropy for the given data
    :param data:
    :return:
    """
    if len(data) <= 0:
        return 0
    data = np.array(data)
    labels = data[:, len(data[0]) - 1].astype("uint8")
    label_count = {}
    for label in labels:
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1
    # number of instances of dataset
    num_data = len(labels)
    entropy = 0
    for key, value in label_count.items():
        prob = value / num_data
        entropy += -prob * np.log2(prob)
    return entropy


def cal_condition_entropy(index, feature_value, data):
    num_data = len(data)
    subdata1, subdata2 = [], []
    for i in range(len(data)):
        d = data[i]
        if d[index] < feature_value:
            subdata1.append(d)
        else:
            subdata2.append(d)

    prob1 = len(subdata1) / num_data
    prob2 = len(subdata2) / num_data
    condition_entropy = prob1 * cal_entropy(subdata1) + prob2 * cal_entropy(subdata2)
    return condition_entropy, subdata1, subdata2


def get_feature_with_min_condition_entropy(index, data):
    """
    calculate and choose the minimum conditional entropy for a specific feature
    :param index: data column index
    :param data:
    :return: result_subdata1, result_subdata2, feature_value
    """
    min_condition_entropy = 100
    result_subdata1, result_subdata2 = [], []
    feature_value = 0

    unique_values = set()
    for value in data[:, index]:
        unique_values.add(value)

    for uni_value in unique_values:
        # split the data into two(bigger or smaller)
        condition_entropy, subdata1, subdata2 = cal_condition_entropy(index, uni_value, data)

        if condition_entropy < min_condition_entropy:
            min_condition_entropy = condition_entropy
            result_subdata1 = subdata1
            result_subdata2 = subdata2
            feature_value = uni_value
    return min_condition_entropy, result_subdata1, result_subdata2, feature_value


def is_perfectly_classified(data):
    """
    check if all the data belong to the same class
    :param data:
    :return:
    """
    temp_label = data[0][-1]
    for d in data:
        if d[-1] != temp_label:
            return False, None
    return True, temp_label


def majority_label(labels):
    """
    choose the class label which has a major count
    :param labels:
    :return:
    """
    dict = {}
    for d in labels:
        if dict.__contains__(d):
            dict[d] += 1
        else:
            dict[d] = 1
    max_key, max_value = 0, 0
    for key, value in dict.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key


def choose_best_feature(headers, indexes, data):
    data = np.array(data)
    entropy = cal_entropy(data)
    information_gain = 0
    result_subdata1, result_subdata2, feature_value = [], [], 0
    feature_name, feature_index = '', 0
    for i in indexes:
        min_condition_entropy, subdata1, subdata2, value = get_feature_with_min_condition_entropy(i, data)
        temp = entropy - min_condition_entropy
        if temp > information_gain:
            information_gain = temp
            result_subdata1 = subdata1
            result_subdata2 = subdata2
            feature_value = value
            feature_name = headers[i]
            feature_index = i
    return feature_name, feature_index, feature_value, result_subdata1, result_subdata2


def build_decision_tree(headers, indexes, data):
    # check if perfectly classified, if do, return the corresponding label
    perfectly_classified, label = is_perfectly_classified(data)
    if perfectly_classified:
        return label

    # check if there is no more split features
    if len(data) == 1:
        return majority_label(data[:, -1])
    if len(data) < 1:
        print("oh my god")
    best_feature_name, best_feature_index, best_feature_value, subdata1, subdata2 = choose_best_feature(headers,
                                                                                                        indexes, data)
    tree = {"index": best_feature_index, "value": best_feature_value, "name": best_feature_name}
    indexes.remove(best_feature_index)
    if len(subdata1) > 0:
        tree["left"] = build_decision_tree(headers, indexes, subdata1)
    if len(subdata2) > 0:
        tree["right"] = build_decision_tree(headers, indexes, subdata2)

    return tree


def predict(tree, d):
    if tree == 0 or tree == 1:
        return tree
    index = tree['index']
    threshold = tree['value']
    left = tree['left']
    right = tree['right']
    if d[index] < threshold:
        return predict(left, d)
    else:
        return predict(right, d)


def do_test(tree, data):
    correct_num = 0
    for d in data:
        predicted_label = predict(tree, d)
        # print('正确：', d[-1], ',预测：', predicted_label)
        if d[-1] == predicted_label:
            correct_num += 1

    accuracy = correct_num / len(data)
    print('correct number：', correct_num, " ,total number：", len(data), " ,accuracy：", accuracy)


def test_use_tree(json_filename, train_data_filename, test_data_filename):
    """
    test the accuracy on training data and test data using the decision tree
    :type json_filename: str
    :type train_data_filename: str
    :type test_data_filename: str
    :return:
    """
    with open(json_filename) as f:
        # load tree from json file
        lines = f.readlines()
        text = ''
        for l in lines:
            text += l.replace("\n", "")
        tree = json.loads(text)

        if train_data_filename is not None:
            _, train_data = load_dataset(train_data_filename)
            print('Test the accuracy on training data')
            do_test(tree, train_data)

        if test_data_filename is not None:
            _, test_data = load_dataset(test_data_filename)
            print('Test the accuracy on test data')
            do_test(tree, test_data)


if __name__ == '__main__':
    folder = 'data/'
    output_folder = 'main_output'
    json_filename = os.path.join(output_folder, "tree.json")
    train_data_filename = os.path.join(folder, 'gene_expression_training.csv')
    test_data_filename = os.path.join(folder, 'gene_expression_test.csv')

    headers, train_data = load_dataset(train_data_filename)
    indexes = [i for i in range(len(train_data[0]) - 1)]
    tree = build_decision_tree(headers, indexes, train_data)

    # save tree to file
    jsonstr = json.dumps(tree)
    with open(json_filename, 'w') as f:
        f.write(jsonstr)

    generate_graph(-1, '<', tree, 0)
    dot.format = 'png'
    dot.view('output.dot', 'main_output')

    test_use_tree(json_filename, train_data_filename, test_data_filename)
