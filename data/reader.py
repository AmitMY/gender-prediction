from functools import lru_cache
import random
from utils.file_system import listdir
from re import findall


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
    def __init__(self, name, source, specific=None, tokenize=True, clusterF='/home/evanmas2/NMTdata/data/ClinSharedTask/cluster-semantic-vectors/glove_clusters_0.011442950_words.json'):
        
        self.clusterF=clusterF
        self.clusterdata=self.getClusters()
        
        if isinstance(source, str):
            files = listdir(os.path.join(script_dir, source))
            self.categories = {f.split("_")[1].split(".")[0].lower(): self.parse_file(f, tokenize) for f in files}
        else:
            self.categories = source

        self.categories = {w: c for w, c in self.categories.items() if not specific or w in specific}

        print(name, "found", {w: len(c) for w, c in self.categories.items()})

    def parse_file(self, f, tokenize=False):
        raw = open(f, "r", encoding="utf-8").read()

        matches = findall('<doc id="(\d*?)" genre="(.*?)" gender="(M|F)">\n([\s\S]*?)\n<\/doc>', raw)

        data = list(map(lambda m: {"id": m[0], "genre": m[1], "gender": m[2],
                                   "text": m[3] if not tokenize else cached_tokenizer(m[3])}, matches))
        random.Random(1234).shuffle(data)  # Same shuffle seed
        return data

    def getClusters(self):
        with open(self.clusterF) as json_file:
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

    def getclusters(self,t,y=None):
        clusterS = [self.clusterdata[s] if s in self.clusterdata else "0" for s in t.split()]
        return " ".join(clusterS)

    def export(self, prefix=False, lowercase=False, clusters=True):
        def preprocess(t, w):
            if prefix:
                t = w + ": " + t

            if lowercase:
                t = t.lower()
            
            if clusters:
                t = getclusters(t) 
                print(t)
                exit()
            return t

        pairs = [(preprocess(c["text"], w), 0 if c["gender"] == "M" else 1)
                 for w, co in self.categories.items() for c in co]

        return list(zip(*pairs))  # List of texts, list of tags


if __name__ == "__main__":
    data = Data("All", "train", tokenize=True)
    train, dev = data.split()
    texts, categories = train.export()
    print("\n", texts[-1], "|", categories[-1])
