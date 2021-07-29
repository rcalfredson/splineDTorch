import argparse
import itertools
import json
from lib.error_manager import ErrorManager
from lib.statistics import ttest_ind
import numpy as np
import os
from pathlib import Path
from util import p2stars

parser = argparse.ArgumentParser(
    description="Generate rankings of best nets"
    " within a given experiment and"
    "/or best overall experiments."
)
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "--dirs",
    nargs="*",
    help="One or more paths to directories containing error results, with one "
    "experiment per directory. If only one path is entered, the nets within it will"
    " be ranked, but if multiple directories are passed in, the directories will be"
    " ranked instead.",
)
group.add_argument(
    "--dirlist", help="Newline-separated list of experiment folders to check"
)

parser.add_argument(
    "--subdir",
    help="Sub-folder in which to look for error results"
    " files (i.e., to append to the folders specified by --dirs or --dirlist)",
)

parser.add_argument(
    "--model_list",
    nargs="?",
    default=False,
    const=True,
    help="Path to newline-separated list of models for which to measure"
    " error within the given directories (all other models will be ignored)."
    " Note: the '.pth' extension should be omitted. --model_list is given with"
    " no path, the script looks for file nets_to_check.txt inside the first"
    " experiment directory.",
)

parser.add_argument(
    "--t_test",
    action="store_true",
    help="Run t-tests to compare every pairwise combination of experiments",
)

parser.add_argument(
    "--best_by_min_cat",
    action="store_true",
    help="Use alternate criterion to select best nets within an experiment: having a"
    " minimum value for one or more categories of error, as opposed to ranking nets"
    " in ascending order by their weighted average error.",
)

opts = parser.parse_args()
if opts.dirlist:
    with open(opts.dirlist, "r") as f:
        opts.dirs = f.read().splitlines()
if opts.subdir:
    opts.dirs = [os.path.join(d, opts.subdir) for d in opts.dirs]
if len(opts.dirs) == 0:
    exit("Exiting early because no experiments were specified.")
if opts.model_list:
    if type(opts.model_list) == bool:
        opts.model_list = os.path.join(opts.dirs[0], "nets_to_check.txt")
    with open(opts.model_list) as f:
        opts.model_list = f.read().splitlines()
err_manager = ErrorManager(
    opts.dirs,
    analyze_by_wt_avg=True,
    best_by_min_cat=opts.best_by_min_cat,
    model_list=opts.model_list,
)
CATEGORY_ABBREVS = dict(
    zip(
        err_manager.error_categories,
        ("MA", "MR", "MR0", "MR1", "MR2", "MX"),
    )
)
# how to select
for i, dir_name in enumerate(opts.dirs):
    err_manager.parse_error_for_dir(dir_name, i)
per_experiment_averages = {
    opts.dirs[i]: np.mean(list(avg_err_list.values()))
    for i, avg_err_list in enumerate(err_manager.avgErrsByNet)
}
if len(opts.dirs) == 1:
    ranked_results = sorted(
        err_manager.avgErrsByNet[0].items(), key=lambda kv: kv[1], reverse=False
    )
    with open("ranked_results_single_exp.json", "w") as f:
        json.dump(ranked_results, f, ensure_ascii=False, indent=4)
    print("Names of the top three nets:")
    [print(rr[0]) for rr in ranked_results[:3]]
    print("Top three individual weighted error averages:")
    [print(rr[1]) for rr in ranked_results[:3]]
    print(
        "Mean of the top three weighted error averages:",
        np.mean([rr[1] for rr in ranked_results[:3]]),
    )
    print()
    if opts.best_by_min_cat:
        print("Best nets as decided by minimum values in one or more error categories:")
        sorted_keys = sorted(
            err_manager.best_nets[0],
            key=lambda kv: err_manager.best_nets[0][kv]["wt_avg"],
        )
        for k in sorted_keys:
            print(k)
        print()
        print("categories:")
        for k in sorted_keys:
            [
                print(
                    ",".join(
                        [
                            CATEGORY_ABBREVS[cat]
                            for cat in err_manager.best_nets[0][k]["cats"]
                        ]
                    )
                )
            ]
        print()
        print("weighted averages:")
        for k in sorted_keys:
            print(err_manager.best_nets[0][k]["wt_avg"])
        print("\n")
        print(
            "mean of weighted averages:",
            np.mean([err_manager.best_nets[0][k]["wt_avg"] for k in sorted_keys]),
        )
    with open("errors_by_net.csv", "w") as f:
        f.write("net name,experiment name,weighted error\n")
        for i, exp in enumerate(err_manager.avgErrsByNet):
            for net_key in exp:
                print("key of a net:", net_key)
                print("average weighted error:", exp[net_key])
                f.write(f"{net_key},{opts.dirs[i]},{exp[net_key]}\n")
else:
    if opts.t_test:
        # create all combinations
        t_test_results = {}
        experiment_pairs = list(
            itertools.combinations(range(err_manager.n_experiments), 2)
        )
        for pair in experiment_pairs:
            dir1, dir2 = [os.path.basename(err_manager.error_dirs[i]) for i in pair]
            print("comparison of these data points:")
            print(np.array(list(err_manager.avgErrsByNet[pair[0]].values())))
            print("to these:")
            print(np.array(list(err_manager.avgErrsByNet[pair[1]].values())))
            results_key = f"{dir1}_vs_{dir2}"
            t_test_results[results_key] = list(
                ttest_ind(
                    np.array(list(err_manager.avgErrsByNet[pair[0]].values())),
                    np.array(list(err_manager.avgErrsByNet[pair[1]].values())),
                )
            )
            t_test_results[results_key].append(p2stars(t_test_results[results_key][1]))
        with open("ttest_results.json", "w") as f:
            json.dump(t_test_results, f, ensure_ascii=False, indent=4)
    print("average max abs. error by experiment:")
    for i in range(err_manager.n_experiments):
        # print(err_manager.error_dirs[i], ":")
        print(np.mean(err_manager.errors[i]["max_abs_error"]))
    ranked_results = sorted(
        per_experiment_averages.items(), key=lambda kv: kv[1], reverse=False
    )
    with open("ranked_results_multi_exp.json", "w") as f:
        json.dump(ranked_results, f, ensure_ascii=False, indent=4)
    with open("errors_by_net.csv", "w") as f:
        f.write("net name,experiment name,weighted error\n")
        for i, exp in enumerate(err_manager.avgErrsByNet):
            for net_key in exp:
                print("key of a net:", net_key)
                print("average weighted error:", exp[net_key])
                f.write(f"{net_key},{opts.dirs[i]},{exp[net_key]}\n")
