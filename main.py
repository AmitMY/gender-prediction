from json import load, dump
from random import shuffle

from models.spacy.main import ModelRunner as spacy_runner
from models.pytorch.main import ModelRunner as pytorch_runner
from models.lm_based.main import ModelRunner as kenlm_runner
from models.sklearn_based.main import ModelRunner as sklearn_runner
from data.reader import Data

from utils.file_system import makedir, rmfile

import hashlib

# train, dev = Data("All", "train", ["news"], tokenize=True).split()
#
# runner = pytorch_runner("SelfAttention", train, dev, {"lowercase": True, "prefix": False})
# runner.load("models/checkpoints/bcfebe661493938d7e398691c9943ea8/SelfAttention.0")
# print(round(runner.eval_one(
#     "Zelzate - Na de Jupiler Pro League en de Proximus League start dit weekend ook de voetbalcompetitie in de amateurreeksen. Voor het eerste in zijn tienjarige geschiedenis treedt KVV Zelzate aan in de nationale reeksen. Coach Bjorn De Neve hoopt op een goede seizoensstart, “maar dat doet wel elke coach”, blijft de woensdag veertig jaar geworden coach nuchter. De voorbereiding van KVV Zelzate op dit maidenseizoen in het nationaal voetbal kende hoogtes en laagtes, ook al gingen de resultaten crescendo. De nederlaag tegen Ertvelde buiten beschouwing gelaten kan er gesteld worden dat het zeer zwak begon met de nederlaag voor de beker van België in het verre Saint-Léger en eindigde met mooie overwinningen tegen Knokke en Elene/Grotenberge. “Ik ben het gedeeltelijk eens met die analyse”, zegt de coach, ‘het klopt dat er mindere resultaten bij waren maar dit is eigen aan een voorbereiding omdat ik zoveel mogelijk spelers zoveel mogelijk speelminuten wil gunnen. Tegen Ertvelde startten we met enkele beloften en probeerden we, net als tegen Wachtebeke, een systeem met drie centrale verdedigers uit. Dat resultaat is voor mij niet het belangrijkste. Ik vind het veel belangrijker dat we top waren tegen ploegen zoals Cappellen (tweede amateur), Knokke (eerste amateur) en Elene/Grotenberge (eerste provinciale)”. Over het komende seizoen blijft De Neve op de vlakte. “Iedere coach, iedere clubleiding en elke supporter hoopt op een goede start maar voor ons valt die toch wel zwaar uit. Ik heb er alle vertrouwen in dat we elke wedstrijd competitief kunnen zijn en kans maken op puntengewin als we de stijgende lijn van het einde van de voorbereiding kunnen doortrekken” Groen/wit heeft in zijn eerste vijf wedstrijden drie uitmatchen naar Wingene, Wolvertem/Merchtem en Wetteren; tussendoor ontvangt het thuis Oostkamp en Melsele. “Het is nu onze uitdaging om de tegenstanders zo vlug mogelijk in kaart te brengen. Wingene is door mijn assistent Jens Pauwels als door mij gescout en via verschillende kanalen kregen we extra informatie.” Tegenstander Wingene eindigde vorig seizoen op de derde plaats op vijf punten van kampioen Menen. Thuis haalde het 33 op 45, uit was dit 24 op 45. In de voorbereiding haalde het de derde ronde van de beker van België; in de eerste ronde werd Wielsbeke met 1-0 verslagen, in de tweede ronde was het met 2-0 te sterk voor Taminoise. In de derde ronde gingen de West-Vlamingen met 3-0 onderuit bij Givry. De wedstrijd wordt zondag om 15 uur afgetrapt aan de Rozendalestraat in Wingene. Voor de eerste competitiewedstrijd selecteerde de trainersstaff donderdag veertien spelers. Daar kwamen na de beloftenwedstrijd van vrijdagavond, thuis tegen Melsele, nog twee spelers bij. De geselecteerden zijn: Thomas Ampe, Emmanuel Annor (?), Richard Antwi-Manu, Nico Blondeel, Roy Broeckaert, Stijn Cools-Ceuppens, Jelte De Coninck, Fréderique De Vleesschauwer, Arno Decraecker, Pieter Grootaert, Ennio Martens, Doran Mertens, Tars Notteman, Tjorven Quintelier, Nicolas Van Buyten en Jens Vandenhende (?) Zondag voor de wedstrijd valt één speler af.")))


data = {
    "all": Data("All", "train"),

    "twitter": Data("Twitter", "train", ["twitter"]),
    "youtube": Data("Youtube", "train", ["youtube"]),
    "news": Data("News", "train", ["news"]),

    "twitter+twisty": Data("Twitter, Twisty", "train", ["twitter", "twisty1"]),

    "twitter+news": Data("Twitter, News", "train", ["twitter", "news"]),
    "twitter+youtube": Data("Twitter, Youtube", "train", ["twitter", "youtube"]),
    "news+youtube": Data("News, Youtube", "train", ["news", "youtube"]),

    "twitter+news+external": Data("Twitter, News, External", "train", ["twitter", "news", "csi", "twisty1"]),
    "twitter+youtube+external": Data("Twitter, Youtube, External", "train", ["twitter", "youtube", "csi", "twisty1"]),
    "news+youtube+csi": Data("News, Youtube, CSI", "train", ["news", "youtube", "csi"]),
}

scenarios = {
    "All 90%|All 10%": data["all"].split(),
    # In domain
    "Twitter 90%|Twitter 10%": data["twitter"].split(),
    "YouTube 90%|YouTube 10%": data["youtube"].split(),
    "News 90%|News 10%": data["news"].split(),

    # In domain + external data
    "Twitter 90%, Twisty|Twitter 10%": data["twitter+twisty"].split(),

    # Out of domain
    "Twitter, News|YouTube": (data["twitter+news"], data["youtube"]),
    "Twitter, YouTube|News": (data["twitter+youtube"], data["news"]),
    "YouTube, News|Twitter": (data["news+youtube"], data["twitter"]),

    # Out of domain + external data
    "Twitter, News, External|YouTube": (data["twitter+news+external"], data["youtube"]),
    "Twitter, YouTube, External|News": (data["twitter+youtube+external"], data["news"]),
    "YouTube, News, CSI|Twitter": (data["news+youtube+csi"], data["twitter"]),
}

models = {
    "Spacy": (spacy_runner, "nl_core_news_sm", {"clusters": False}),
    # "Spacy-c": (spacy_runner, "nl_core_news_sm", {"clusters": True}),
}

# for ngram in range(3, 7):
#     models["KENLM." + str(ngram)] = (kenlm_runner, "KENLM", {"ngram": ngram})

# for t in ['svm', 'log', 'nb', 'knn']:  # 'rf',
#     models["SKLearn-" + t] = (sklearn_runner, t, {"clusters": False})
#     models["SKLearn-" + t + "-c"] = (sklearn_runner, t, {"clusters": True})

# Add all of the pytorch models
for m in ["SelfAttention", "LSTMAttention"]:  # "CNN", "LSTMAttention", "RNN", "RCNN", "LSTM",
    models[m] = (pytorch_runner, m, {})
    models[m + "+"] = (pytorch_runner, m, {"pretrained": "fasttext"})
    # models[m + "-c"] = (pytorch_runner, m, {"clusters": True})

NUM_RUNS = 5
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
