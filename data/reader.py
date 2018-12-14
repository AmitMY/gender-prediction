from functools import lru_cache
import random
from utils.file_system import listdir
from re import findall
import json

import os
import spacy

script_dir = os.path.dirname(__file__)

nlp = spacy.load("nl_core_news_sm")

from spacy.lang.nl import Dutch

tokenizer = Dutch().Defaults.create_tokenizer(nlp)


@lru_cache(maxsize=None)
def cached_tokenizer(text):
    return " ".join(map(str, tokenizer(text)))


class Data:
    def __init__(self, name, source, specific=None, tokenize=True):
        # Now loading cluster data
        self.clusterdata = self.loadClusters()

        if isinstance(source, str):
            files = listdir(os.path.join(script_dir, source))
            self.categories = {f.split("_")[1].split(".")[0].lower(): self.parse_file(f, tokenize) for f in files}
        else:
            self.categories = source

        self.categories = {w: c for w, c in self.categories.items() if not specific or w in specific}

        print(name, "found", {w: len(c) for w, c in self.categories.items()})

    def parse_file(self, f, tokenize=False):
        raw = open(f, "r", encoding="utf-8").read()

        matches = findall('<doc id="(\d*?)" genre="(.*?)" gender="(M|F|\?)">\n([\s\S]*?)\n<\/doc>', raw)

        data = list(map(lambda m: {"id": m[0], "genre": m[1], "gender": m[2],
                                   "text": m[3] if not tokenize else cached_tokenizer(m[3])}, matches))
        random.Random(1234).shuffle(data)  # Same shuffle seed
        return data

    def loadClusters(self, clusterF='../models/sklearn_based/glove_clusters_0.011442950_words.json'):
        clusterFileName = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', clusterF))
        with open(clusterFileName) as json_file:
            data = json.load(json_file)
        data_inv = {}
        for k, v in data.items():
            for i in v:
                data_inv[i] = k
        return data_inv

    def split(self, percent=0.9):
        data1 = Data("train", {w: c[:int(len(c) * percent)] for w, c in self.categories.items()})
        data2 = Data("dev", {w: c[int(len(c) * percent):] for w, c in self.categories.items()})

        return data1, data2

    def getClusters(self, text):
        ''' For every word it gives back the cluster
        
            :param text: the document with sentences to get words from and then the clusters
            :returns: string of clusters (same as the input)
        '''
        clusterS = [self.clusterdata[sent] if sent in self.clusterdata else "0" for sent in text.split()]
        return " ".join(clusterS)

    def export(self, prefix=False, lowercase=False, clusters=False):

        def preprocess(t, w):
            if prefix:
                t = w + ": " + t

            if lowercase:
                t = t.lower()

            if clusters:
                t = self.getClusters(t)

            return t

        pairs = [(preprocess(c["text"], w), None if c["gender"] == "?" else (0 if c["gender"] == "M" else 1), int(c["id"]))
                 for w, co in self.categories.items() for c in co]

        return list(zip(*pairs))  # List of texts, list of tags


if __name__ == "__main__":
    data = Data("All Train", "train", tokenize=False)
    train, dev = data.split()
    test = Data("All Test", "test", tokenize=False)

    texts, categories, ids = train.export()
    print("\n", texts[-1], "|", categories[-1], "|", ids[-1])

    texts, categories, ids = test.export()
    print("\n", texts[-1], "|", categories[-1], "|", ids[-1])
