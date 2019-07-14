import numpy as np
import csv
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_label():
    path = '/home/luchixiang/Desktop/laji.csv'
    reader = csv.reader(open(path, 'r'))
    data = list(reader)
    x_text = []
    harm_label = []
    dry_label = []
    recycleable_label = []
    wet_label = []
    for i in range(1, len(data)):
        if '\u4e00' <= data[i][0] <= '\u9fa5':
            continue
        temp = data[i][0].lower().split(" ")
        x_text.append(temp)
        if data[i][1] == '1':
            wet_label.append([0, 0, 0, 1])
        elif data[i][1] == '2':
            dry_label.append([0, 0, 1, 0])
        elif data[i][1] == '3':
            recycleable_label.append([0, 1, 0, 0])
        else:
            harm_label.append([1, 0, 0, 0])
    # x_text = [clean_str(sent) for sent in x_text]
    # harm_label = [[1, 0, 0, 0] for i in range(1, len(data)) if data[i][1] == '4']
    # recycleable_label = [[0, 1, 0, 0] for i in range(1, len(data)) if data[i][1] == '3']
    # dry_label = [[0, 0, 1, 0] for i in range(1, len(data)) if data[i][1] == '2']
    # wet_label = [[0, 0, 0, 1] for i in range(1, len(data)) if data[i][1] == '1']
    y_label = np.concatenate([wet_label, dry_label, recycleable_label, harm_label], 0)
    return [x_text, y_label]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
