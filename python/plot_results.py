"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../../LICENSE)*** <--

Script to plot results of numerical experiments.
"""


# ==== IMPORTS ============================================================== #

import math
import os
import sys
sys.path.append("python/")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==== CONSTANTS ============================================================ #

PATH_2_DATA = "../data/results_dataframe.csv"
PATH_2_FIGURES = "../tex/figures/results/"


# ==== MAIN ================================================================= #

if __name__ == "__main__":
    #   1. load data into DataFrame
    data_frame = pd.read_csv(PATH_2_DATA)
    feature_list = list(data_frame)

    print("feature_list:", feature_list)
    print()

    #   2. get unique lists
    benchmark_list = list(data_frame[feature_list[0]].unique())
    sampling_list = list(data_frame[feature_list[1]].unique())
    dimensionality_list = list(data_frame[feature_list[2]].unique())

    print("benchmark_list:", benchmark_list)
    print("sampling_list:", sampling_list)
    print("dimensionality_list:", dimensionality_list)
    print()

    #   3. plot results
    #benchmark = benchmark_list[0]
    #sampling = sampling_list[0]
    #dimensionality = dimensionality_list[0]

    for benchmark in benchmark_list:
        for sampling in sampling_list:
            for dimensionality in dimensionality_list:

                save_str = "_".join(benchmark.split(" "))
                save_str += "/results_sampling_"
                save_str += "_".join(sampling.split(" "))
                save_str += "_dimensionality_"
                save_str += str(dimensionality)
                save_str += ".png"

                boolean_mask = np.logical_and(
                    data_frame[feature_list[0]].values == benchmark,
                    data_frame[feature_list[1]].values == sampling,
                    data_frame[feature_list[2]].values == dimensionality
                )

                sub_frame = data_frame[boolean_mask]
                n_rows = sub_frame.shape[0]

                num_samples_array = sub_frame[feature_list[3]].values
                surrogate_efficiency_array = sub_frame[feature_list[-1]].values

                plotting_dict = {}

                for i in range(0, n_rows):
                    num_samples = num_samples_array[i]
                    surrogate_efficiency = surrogate_efficiency_array[i]

                    if num_samples not in plotting_dict.keys():
                        plotting_dict[num_samples] = [surrogate_efficiency]

                    else:
                        plotting_dict[num_samples].append(surrogate_efficiency)

                plot_x = np.array([
                    math.pow(num_samples, 1 / dimensionality)
                    for num_samples in plotting_dict.keys()
                ])

                idx_sort = np.argsort(plot_x)
                plot_x = plot_x[idx_sort]

                plot_y = np.array([
                    np.mean(plotting_dict[num_samples])
                    for num_samples in plotting_dict.keys()
                ])

                plot_y = plot_y[idx_sort]

                plt.figure(figsize=(8, 6))
                plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

                for num_samples in plotting_dict.keys():
                    points_y = plotting_dict[num_samples]

                    points_x = math.pow(num_samples, 1 / dimensionality) * np.ones(
                        len(points_y)
                    )

                    plt.scatter(
                        points_x,
                        points_y,
                        marker=".",
                        color="C0",
                        alpha=0.333,
                        zorder=2
                    )

                plt.scatter(
                    [-1],
                    [-1],
                    marker=".",
                    color="C0",
                    alpha=0.333,
                    zorder=2,
                    label="Monte Carlo data"
                )

                plt.plot(
                    plot_x,
                    plot_y,
                    linestyle="--",
                    color="black",
                    alpha=0.5,
                    zorder=3
                )

                plt.scatter(
                    plot_x,
                    plot_y,
                    marker="*",
                    color="black",
                    zorder=4,
                    label="point stack means"
                )

                plt.xlim(0, 1.05 * np.max(plot_x))
                plt.xlabel(r"$\sqrt[D]{N}$ [  ]")
                plt.ylim(0, 1)
                plt.ylabel(r"$\eta_{SM}$ [  ]")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    PATH_2_FIGURES + save_str,
                    format="png",
                    dpi=128
                )
                print("plot saved:", save_str)
                plt.close()

                #plt.show()
