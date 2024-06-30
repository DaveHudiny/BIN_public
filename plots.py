# Author = David Hudak
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = plots.py file of project to subject Biology Inspired Computers. There I plot figures.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


def load_history(path: str):
    """Reads experiments folder history file.

    Args:
        path (str): Path where to history file (eg. ../experimenty/two_outputs/histories.txt)

    Returns:
        list, list, list: Lists with complete histories, seeds (useless) and final results.
    """
    histories = []
    results = []
    with open(path, "r") as file:
        for line in file.readlines():
            splitor = line.split(" ")
            results = list(map(float, splitor))
    return histories, seeds, results


def plt_boxplots(results):
    """Plots boxplots of result fitnesses.

    Args:
        results (dict): Dictionary of keys by strategy with lists with resulting fitnesses of runs.
    """
    MAX_FIT = 1
    dfr = pd.DataFrame.from_dict(results)
    dfr = dfr.rename(columns={"lenet_results_backmem=False_basic=False_layers=both_mem_divisor=5": "Paměť 20 %",
                              "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=10": "Pokročilé fce",
                              "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=20": "Paměť 5 %",
                              "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=50": "Paměť 2 %",
                              "lenet_results_backmem=False_basic=True_layers=both_mem_divisor=10": "Základní fce",
                              "lenet_results_backmem=True_basic=False_layers=both_mem_divisor=10": "Prop pokročilý",
                              "lenet_results_backmem=True_basic=True_layers=both_mem_divisor=10": "Prop základní"})
    ax = sns.boxplot(data=dfr)

    plt.xlabel("Zvolená strategie")
    plt.ylabel("Úspěšnost na MNISTu [%]")
    plt.savefig("./boxplots.pdf")
    plt.show()


def statistic_compare_results(results1, results2):
    """Pair t-test for two result array.

    Args:
        results1 (np.ndarray): First results.
        results2 (np.ndarray): Second results.
    """
    t, p = st.ttest_ind(results1, results2)
    print(f"P-hodnota pro dva zvolené výběry = {p}")


if __name__ == "__main__":
    seeds = {}
    histories = {}
    results = {}
    for name in [  # "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=5",
        "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=10",
        # "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=20",
        # "lenet_results_backmem=False_basic=False_layers=both_mem_divisor=50",
        "lenet_results_backmem=False_basic=True_layers=both_mem_divisor=10",
        "lenet_results_backmem=True_basic=False_layers=both_mem_divisor=10",
            "lenet_results_backmem=True_basic=True_layers=both_mem_divisor=10"]:
        histories[name], seeds[name], results[name] = load_history(
            f"./results/{name}.txt")

    plt_boxplots(results)
