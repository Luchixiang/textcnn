import gensim
import csv

class Word2vecHelper(object):
    def __init__(self):
        path = "/home/luchixiang/Desktop/python/machine-learning/all_words.txt"
        csv_oath = '/home/luchixiang/Desktop/laji.csv'
        reader = csv.reader(open(csv_oath, 'r'))
        data = list(reader)
        self.all_words = list(open(path, "r", encoding='utf-8').readlines())
        self.all_words = [s.lower().split(" ") for s in self.all_words]
        for i in range(1, len(data)):
            if '\u4e00' <= data[i][0] <= '\u9fa5':
                continue
            temp = data[i][0].lower().split(" ")
            self.all_words.append(temp)
        # self.all_words = [data_loader.clean_str(s) for s in self.all_words]
        self.model = gensim.models.Word2Vec(self.all_words, min_count=1)

    def get_vector(self, word):
        word = [s.lower() for s in word]
        return self.model[word]
