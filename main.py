from json import load, dump
from random import shuffle

from data.reader import Data
from models.spacy.main import main as spacy_main
from models.pytorch.main import main as pytorch_main

data = {
    "original": Data("All", "train", ["twitter", "youtube", "news"]),

    "twitter": Data("Twitter", "train", ["twitter"]),
    "youtube": Data("Youtube", "train", ["youtube"]),
    "news": Data("News", "train", ["news"]),

    "twitter+news": Data("Twitter, News", "train", ["twitter", "news"]),
    "twitter+youtube": Data("Twitter, Youtube", "train", ["twitter", "youtube"]),
    "news+youtube": Data("News, Youtube", "train", ["news", "youtube"]),

}

scenarios = {
    "Original 90%|Original 10%": data["original"].split(),
    "Twitter 90%|Twitter 10%": data["twitter"].split(),
    "YouTube 90%|YouTube 10%": data["youtube"].split(),
    "News 90%|News 10%": data["news"].split(),
    "Twitter, News|YouTube": (data["twitter+news"], data["youtube"]),
    "Twitter, YouTube|News": (data["twitter+youtube"], data["news"]),
    "YouTube, News|Twitter": (data["news+youtube"], data["twitter"]),
}

models = {
    "Spacy": (spacy_main, "nl_core_news_sm", {}),
}

# Add all of the pytorch models
for m in ["RNN", "CNN", "RCNN", "LSTM", "LSTMAttention", "SelfAttention"]:
    models[m] = (pytorch_main, m, {})
    models[m + "+"] = (pytorch_main, m, {"pretrained": "fasttext"})

NUM_RUNS = 3
models = {w + "." + str(i): c for w, c in models.items() for i in range(NUM_RUNS)}

res_file_name = "results.json"

results = {}
try:
    results = load(open(res_file_name, "r"))
except:
    pass

print("Current Results")
print(results)

scenarios_shuffled = list(scenarios.items())
shuffle(scenarios_shuffled)
for name, (train, dev) in scenarios_shuffled:
    if name not in results:
        results[name] = {}
    for model_name, (runner, model, options) in models.items():
        print(name, model_name, model, options)
        if model_name in results[name]:
            print("Skipping", model_name, name, options)
        else:
            all_scores = []
            best_score = 0
            early_stop = 0
            for score in runner(model=model, train=train, dev=dev, opt=options):
                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    early_stop = 0
                else:
                    early_stop += 1

                if early_stop > 5:
                    break

            try:
                results = load(open(res_file_name, "r"))
            except:
                pass
            results[name][model_name] = {
                "scores": all_scores,
                "best": best_score
            }
            dump(results, open(res_file_name, "w"), indent=2)
            print(model_name, name, options, best_score)
