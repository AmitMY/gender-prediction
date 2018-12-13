import hashlib
import os

from data.reader import Data
from main import scenarios, checkpoints_dir, models
from utils.file_system import makedir

from models.ensemble.main_naive import ModelRunner as ensemble

test_data = {
    "twitter": Data("Twitter", "test", ["twitter"]),
#    "youtube": Data("Youtube", "test", ["youtube"]),
#    "news": Data("News", "test", ["news"]),
}

TEAM = "ABI"

test_runs = {
    TEAM + "_IN_twitter_1": ("Twitter 90%|Twitter 10%", test_data["twitter"]),
#    TEAM + "_IN_youtube_1": ("YouTube 90%|YouTube 10%", test_data["youtube"]),
#    TEAM + "_IN_news_1": ("News 90%|News 10%", test_data["news"]),
#    TEAM + "_CROSS_twitter_1": ("YouTube, News|Twitter", test_data["twitter"]),
#    TEAM + "_CROSS_youtube_1": ("Twitter, News|YouTube", test_data["youtube"]),
#    TEAM + "_CROSS_news_1": ("Twitter, YouTube|News", test_data["news"]),
}

results_dir = "models/results/"
makedir(results_dir)


def male_female(score):
    ''' Return the label based on the score
        0 = M
        1 = F

        :param score: the score -a number int or float
        :returns: label
    '''
    return 'M' if score < 0.5 else 'F'


def parse_results_file(results_f):
    results = open(results_f).read().splitlines()
    return {int(id): g for (id, g) in [r.split() for r in results]}


def gender_eval(results, data):
    total_male = total_female = 0
    correct_male = correct_female = 0
    for text, gender, id in zip(*data.export()):
        if results[id] == "M":
            total_male += 1
            if gender == 0:
                correct_male += 1
        elif results[id] == "F":
            total_female += 1
            if gender == 1:
                correct_female += 1

    return {
        "all": (correct_male + correct_female) / (total_male + total_female),
        "male": correct_male / (total_male + 0.0000001),
        "female": correct_female / (total_female + 0.0000001),
    }


for test_run, (scenario_name, test_data) in test_runs.items():
    hashed = hashlib.md5(scenario_name.encode('utf-8')).hexdigest()
    checkpoints_dir_scenario = os.path.join(checkpoints_dir, hashed)
    results_dir_scenario = os.path.join(results_dir, hashed)
    makedir(results_dir_scenario)

    model_list = []  # A list to contain all trained models (or loaded models)

    train_data, dev_data = scenarios[scenario_name]
    for model_name, (runner, model, options) in models.items():
        t_data = {"train": train_data, "dev": dev_data, "test": test_data}
        # First check if everything is evaluated
        if all([os.path.isfile(os.path.join(results_dir_scenario, t, model_name)) for t in t_data.keys()]):
            print(test_run, "Skipping", model_name)
            print("\n")
            continue

        print(test_run, "Loading", model_name)

        inst = runner(model=model, train=train_data, dev=dev_data, opt=options)
        invert_op = getattr(inst, "eval_one", None)
        eval_all = callable(getattr(inst, "eval_all", None))
        if not callable(getattr(inst, "eval_one", None)) and not eval_all:
            print("No eval_one/eval_all method!")
        else:
            inst.load(os.path.join(checkpoints_dir_scenario, model_name))
            model_list.append(inst)

            for t, data in {"train": train_data, "dev": dev_data, "test": test_data}.items():
                results_dir_scenario_corpus = os.path.join(
                    results_dir_scenario, t)
                makedir(results_dir_scenario_corpus)
                model_res = os.path.join(
                    results_dir_scenario_corpus, model_name)
                if not os.path.isfile(model_res):
                    out = []
                    out_prob = []

                    export = list(zip(*data.export(options)))
                    correct = 0
                    if eval_all:
                        texts, labels, ids = list(zip(*export))
                        all_scores = inst.eval_all(texts)
                        for id, label, score in zip(ids, labels, all_scores):
                            out.append(id + " " + male_female(score))
                            out_prob.append(id + " " + str(score))
                            if label is not None and round(score) == label:
                                correct += 1
                    else:
                        for text, label, id in export:
                            score = inst.eval_one(text)
                            out.append(id + " " + male_female(score))
                            out_prob.append(id + " " + str(score))
                            if label is not None and round(score) == label:
                                correct += 1

                    f = open(model_res, "w")
                    f.write("\n".join(out))
                    f.close()

                    f = open(model_res + ".prob", "w")
                    f.write("\n".join(out_prob))
                    f.close()

                    print("Evaluated", t, correct / len(export))

        print("\n")

    print("Evaluating...")
    gender_based_ensemble = []
    for model_name, _ in models.items():
        f_name = os.path.join(results_dir_scenario, "dev", model_name)
        res = gender_eval(parse_results_file(f_name), dev_data)
        gender_based_ensemble.append(("M", f_name, res["male"]))
        gender_based_ensemble.append(("F", f_name, res["female"]))
        print(model_name, res)

    new_res = {}
    for g, f, _ in sorted(gender_based_ensemble, key=lambda k: k[2]):
        results = parse_results_file(f)
        for res_id, res_g in results.items():
            if res_g == g:
                new_res[res_id] = res_g

    print("Gender Ensemble", test_run, gender_eval(new_res, dev_data))
    
    print("Ensembling...")

    ens = ensemble('Ensemble_Naive', model_list=model_list, test_data=test_data)
    _, results = ens.evaluate()

    # Now let's also compute the dev accuracy of the ensembel
    dev_accuracy, _ = ens.evaluate(dev_data)
    print(" ".join(['Ensembele Naive', scenario_name, 'dev', str(dev_accuracy[0])]))

    model_res_dir = os.path.join(results_dir_scenario, 'ensemble')
    makedir(model_res_dir)

    model_res_fname = os.path.join(model_res_dir, test_run)
    with open(model_res_fname, "w") as f:
        f.write(
            "\n".join([str(id) + " " + male_female(results[id]) for id in results]))

    with open(model_res_fname + ".prob", "w") as f:
        f.write("\n".join([str(id) + " " + str(results[id])
                           for id in results]))
