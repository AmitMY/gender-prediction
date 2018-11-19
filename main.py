from json import load, dump

from data.reader import Data
from models.spacy.main import main as spacy_main

scenarios = {
    "All 90%|All 10%": Data("All", "train").split(),
    "Twitter 90%|Twitter 10%": Data("Train", "train", ["twitter"]).split(),
    "YouTube 90%|YouTube 10%": Data("Train", "train", ["youtube"]).split(),
    "News 90%|News 10%": Data("Train", "train", ["news"]).split(),
    "Twitter, News|YouTube": (Data("Train", "train", ["twitter", "news"]), Data("Train", "train", ["youtube"])),
    "Twitter, YouTube|News": (Data("Train", "train", ["twitter", "youtube"]), Data("Train", "train", ["news"])),
    "YouTube, News|Twitter": (Data("Train", "train", ["youtube", "news"]), Data("Train", "train", ["twitter"])),
}

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

models = {
    "Spacy": (spacy_main, "nl_core_news_sm")
}

res_file_name = "results.json"

results = {}
try:
    results = load(open(res_file_name, "r"))
except:
    pass

print("Current Results")
print(results)

for name, (train, dev) in scenarios.items():
    if name not in results:
        results[name] = {}
    for model_name, (runner, model) in models.items():
        if model_name in results[name]:
            print("Skipping", model_name, name)
        else:
            all_scores = []
            best_score = 0
            early_stop = 0
            for score in runner(model=model, train=train, dev=dev):
                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    early_stop = 0
                else:
                    early_stop += 1

                if early_stop > 3:
                    break

            results[name][model_name] = {
                "scores": all_scores,
                "best": best_score
            }
            dump(results, open(res_file_name, "w"), indent=2)
            print(model_name, name, best_score)
