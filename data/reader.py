from itertools import chain
from random import shuffle

from utils.file_system import listdir
from regex import findall

import os

script_dir = os.path.dirname(__file__)


class Data:
    def __init__(self, name, source, specific=None):
        if isinstance(source, str):
            files = listdir(os.path.join(script_dir, source))
            self.categories = {f.split("_")[1].split(".")[0].lower(): self.parse_file(f) for f in files}
        else:
            self.categories = source

        self.categories = {w: c for w, c in self.categories.items() if not specific or w in specific}

        print(name, "found", {w: len(c) for w, c in self.categories.items()})

    def parse_file(self, f):
        raw = open(f, "r", encoding="utf-8").read()

        matches = findall('<doc id="(\d*?)" genre="(.*?)" gender="(M|F)">\n([\s\S]*?)\n<\/doc>', raw)

        data = list(map(lambda m: {"id": m[0], "genre": m[1], "gender": m[2], "text": m[3]}, matches))
        shuffle(data)
        return data

    def split(self, percent=0.9):
        data1 = Data("train", {w: c[:int(len(c) * percent)] for w, c in self.categories.items()})
        data2 = Data("dev", {w: c[int(len(c) * percent):] for w, c in self.categories.items()})

        return data1, data2

    def export(self):
        pairs = [[w + ": " + c["text"], 0 if c["gender"] == "M" else 1]
                 for w, co in self.categories.items() for c in co]

        return list(zip(*pairs))  # List of texts, list of tags


if __name__ == "__main__":
    data = Data("All", "train")
    train, dev = data.split()

    texts, categories = train.export()
    print("\n", texts[-1], "|", categories[-1])
