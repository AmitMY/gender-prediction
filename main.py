from json import load, dump
from random import shuffle

from models.spacy.main import ModelRunner as spacy_runner
from models.pytorch.main import ModelRunner as pytorch_runner
from models.lm_based.main import ModelRunner as kenlm_runner
from models.sklearn_based.main import ModelRunner as sklearn_runner
from data.reader import Data

from utils.file_system import makedir, rmfile

import hashlib

data = {
    # "original": Data("All", "train", ["twitter", "youtube", "news"]),

    "twitter": Data("Twitter", "train", ["twitter"]),
    "youtube": Data("Youtube", "train", ["youtube"]),
    "news": Data("News", "train", ["news"]),

    # "twitter+twisty": Data("Twitter, Twisty", "train", ["twitter", "twisty1"]),

    "twitter+news": Data("Twitter, News", "train", ["twitter", "news"]),
    "twitter+youtube": Data("Twitter, Youtube", "train", ["twitter", "youtube"]),
    "news+youtube": Data("News, Youtube", "train", ["news", "youtube"]),

    # "twitter+news+external": Data("Twitter, News, External", "train", ["twitter", "news", "csi", "twisty1"]),
    # "twitter+youtube+external": Data("Twitter, Youtube, External", "train", ["twitter", "youtube", "csi", "twisty1"]),
    # "news+youtube+csi": Data("News, Youtube, CSI", "train", ["news", "youtube", "csi"]),
}

scenarios = {
    # # "Original 90%|Original 10%": data["original"].split(),
    # In domain
    "Twitter 90%|Twitter 10%": data["twitter"].split(),
    "YouTube 90%|YouTube 10%": data["youtube"].split(),
    "News 90%|News 10%": data["news"].split(),
    #
    # # In domain + external data
    # "Twitter 90%, Twisty|Twitter 10%": data["twitter+twisty"].split(),
    #
    # Out of domain
    "Twitter, News|YouTube": (data["twitter+news"], data["youtube"]),
    "Twitter, YouTube|News": (data["twitter+youtube"], data["news"]),
    "YouTube, News|Twitter": (data["news+youtube"], data["twitter"]),
    #
    # # Out of domain + external data
    # "Twitter, News, External|YouTube": (data["twitter+news+external"], data["youtube"]),
    # "Twitter, YouTube, External|News": (data["twitter+youtube+external"], data["news"]),
    # "YouTube, News, CSI|Twitter": (data["news+youtube+csi"], data["twitter"]),
}

models = {
#    "Spacy": (spacy_runner, "nl_core_news_sm", {"clusters": False}),
#    "Spacy-c": (spacy_runner, "nl_core_news_sm", {"clusters": True}),
}

# for ngram in range(3, 7):
#     models["KENLM." + str(ngram)] = (kenlm_runner, "KENLM", {"ngram": ngram})

for t in ['svm', 'log']:#, 'rf', 'nb', 'knn']:
    models["SKLearn-" + t] = (sklearn_runner, t, {"clusters": False})
    models["SKLearn-" + t + "-c"] = (sklearn_runner, t, {"clusters": True})

# # Add all of the pytorch models
# for m in ["RNN", "CNN", "RCNN", "LSTM", "LSTMAttention", "SelfAttention"]:
#     models[m + ""] = (pytorch_runner, m, {})
#     models[m + "-c"] = (pytorch_runner, m, {"clusters": True})
#     models[m + "+"] = (pytorch_runner, m, {"pretrained": "fasttext"})

NUM_RUNS = 2
models = {w + "." + str(i): c for w, c in models.items() for i in range(NUM_RUNS)}

checkpoints_dir = "models/checkpoints/"

if __name__ == "__main__":
    res_file_name = "results.json"

    results = {}
    try:
        results = load(open(res_file_name, "r"))
    except:
        pass

    print("Current Results")
    print(results)

    makedir(checkpoints_dir)

    scenarios_shuffled = list(scenarios.items())
    shuffle(scenarios_shuffled)
    print(scenarios_shuffled[0])

    for name, (train, dev) in scenarios_shuffled:

        hashed = hashlib.md5(name.encode('utf-8')).hexdigest()
        checkpoints_dir_scenario = checkpoints_dir + hashed + "/"
        makedir(checkpoints_dir_scenario)

        if name not in results:
            results[name] = {}

        models_shuffled = list(models.items())
        shuffle(models_shuffled)
        for model_name, (runner, model, options) in models_shuffled:
            print(name, model_name, model, options, hashed)
            if model_name in results[name] and results[name][model_name]["best"] > 0:
                print("Skipping", model_name, name, options)
            else:
                all_scores = []
                best_score = 0
                early_stop = 0

                inst = runner(model=model, train=train, dev=dev, opt=options)

                for score in inst.train():
                    all_scores.append(score)
                    if score > best_score:
                        best_score = score
                        early_stop = 0

                        # Save
                        print("Saving...")
                        f_name = checkpoints_dir_scenario + model_name
                        rmfile(f_name)
                        inst.save(f_name)
                    else:
                        early_stop += 1

                    if early_stop >= 5:
                        break

                try:
                    results = load(open(res_file_name, "r"))
                except:
                    pass

                if name not in results:
                    results[name] = {}

                results[name][model_name] = {
                    "scores": all_scores,
                    "best": best_score
                }
                dump(results, open(res_file_name, "w"), indent=2)
                print(model_name, name, options, best_score)
