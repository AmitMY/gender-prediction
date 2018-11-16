from utils.file_system import listdir
from regex import findall

import os

script_dir = os.path.dirname(__file__)


class Data:
    def __init__(self, dir_name):
        files = listdir(os.path.join(script_dir, dir_name))
        self.categories = {f.split("_")[1].split(".")[0].lower(): self.parse_file(f) for f in files}

        print(dir_name, "found", {w: len(c) for w, c in self.categories.items()})

    def parse_file(self, f):
        raw = open(f, "r", encoding="utf-8").read()

        matches = findall('<doc id="(\d*?)" genre="(.*?)" gender="(M|F)">\n([\s\S]*?)\n<\/doc>', raw)

        return list(map(lambda m: {"id": m[0], "genre": m[1], "gender": m[2], "text": m[3]}, matches))


if __name__ == "__main__":
    data = Data("train")
