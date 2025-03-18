"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../../LICENSE)*** <--

Script of numerical experiments.
"""


# ==== IMPORTS ============================================================== #

import os
import sys
sys.path.append("python/")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.model_selection as sklms
import sklearn.preprocessing as sklpp
import torch

import funclib as flib


# ==== CONSTANTS ============================================================ #

MAX_SAMPLES = 10**5

PATH_2_CHECKPOINT = "../data/checkpoint.torch"
PATH_2_DATAFRAME = "../data/results_dataframe.csv"


# ==== FUNCTIONS ============================================================ #

def sampleObjective(
    number_of_samples: int,
    dimensionality: int,
    objective_func: str,
    sampling_scheme: str,
    space_bounds: tuple = (-5, 5),
    print_flag: bool = False
) -> np.array:
    """
    Helper function to handle the sampling of the objective function.

    Ref:  <https://docs.scipy.org/doc/scipy/reference/stats.qmc.html>

    Parameters
    ----------
    number_of_samples: int
        The number of samples taken in constructing the surrogate.

    dimensionality: int
        The dimensionality (input space) of the surrogate.

    objective_func: str
        A string naming the objective function. Should be one of `rastrigin`,
        `rosenbrock`, `griewank`, or `styblinskitang`.

    sampling_scheme: str
        A string naming the sampling scheme. Should be either `simple random`
        or `latin hypercube`.

    space_bounds: tuple, optional, default (-5, 5)
        A tuple that defines the bounds on each dimension of the sample space.

    print_flag: bool, optional, default False
        A boolean which indicates whether or not to do running prints.

    Returns
    -------
    np.array
        An np.array of the sample data, with each row
        being [x_1, x_2, ..., x_D, y].
    """

    #   1. generate sample (from input space)
    if sampling_scheme.lower() == "simple random":
        sample_array = np.random.rand(number_of_samples, dimensionality)

    elif sampling_scheme.lower() == "latin hypercube":
        lhs_sampler = sp.stats.qmc.LatinHypercube(d=dimensionality)
        sample_array = lhs_sampler.random(n=number_of_samples)

    else:
        error_string = "ERROR:  sampleObjective()\t"
        error_string += "expected either 'simple random' or 'latin hypercube' "
        error_string += "for sampling scheme, but received '"
        error_string += sampling_scheme
        error_string += "' instead"

        raise RuntimeError(error_string)

    lower_bounds = [space_bounds[0] for i in range(0, dimensionality)]
    upper_bounds = [space_bounds[1] for i in range(0, dimensionality)]

    sample_array = sp.stats.qmc.scale(
        sample_array, lower_bounds, upper_bounds
    )

    #   2. map sample through objective function (to output space)
    if objective_func.lower() == "rastrigin":
        func = flib.rastrigin

    elif objective_func.lower() == "rosenbrock":
        func = flib.rosenbrock

    elif objective_func.lower() == "griewank":
        func = flib.griewank

    elif objective_func.lower() == "styblinskitang":
        func = flib.styblinskitang

    else:
        error_string = "ERROR:  sampleObjective()\t"
        error_string += "expected one of 'rastrigin', 'rosenbrock', "
        error_string += "'griewank', or 'styblinskitang' for objective "
        error_string += "function, but received '"
        error_string += objective_func
        error_string += "' instead"

        raise RuntimeError(error_string)

    output_array = np.zeros(number_of_samples)

    for i in range(0, number_of_samples):
        output_array[i] = func(sample_array[i])

    #   3. assemble data array
    data_array = np.hstack((sample_array, output_array.reshape(-1, 1)))

    #   4. running print
    if print_flag:
        print("sampleObjective():")
        print("\tdata_array.shape:", data_array.shape)
        print()

        for col in range(0, dimensionality + 1):
            if col < dimensionality:
                print("\tmin(x_{}):".format(col), np.min(data_array[:, col]))
                print("\tmax(x_{}):".format(col), np.max(data_array[:, col]))
                print()

            else:
                print("\tmin(y):".format(col), np.min(data_array[:, col]))
                print("\tmax(y):", np.max(data_array[:, col]))
                print()

        print("# ================ #")
        print()

    return data_array


def trainSurrogate(
    input_array_train_norm: np.array,
    target_array_train: np.array,
    validation_size: float = 0.15,
    hidden_layer_neurons: int = 100,
    num_epochs: int = 10000,
    patience_epochs: int = 1000,
    batch_size: int = int(1e120),
    print_flag: bool = False
) -> torch.nn.Sequential:
    """
    Helper function to set up and train a surrogate model.

    Ref:  <https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/>
    Ref:  <https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/>

    Parameters
    ----------
    input_array_train_norm: np.array
        An array of normalized training inputs.

    target_array_train: np.array
        An array of training targets.

    validation_size: float, optional, default 0.15
        An ratio of the training data to hold back for validation (early
        stopping).

    hidden_layer_neurons: int, optional, default 100
        The number of neurons in each hidden layer of the surrogate.

    num_epochs: int, optional, default 10000
        The maximum number of training epochs.

    patience_epochs: int, optional, default 1000
        Tolerance on number epochs for which there is no reduction in the
        validaction loss metric. Used to trigger early stopping.

    batch_size: int, optional, default "infinity"
        The batch size to use in training. Defaults to using the entire data
        set in one shot (i.e., no batching).

    print_flag: bool, optional, default False
        A boolean which indicates whether or not to do running prints.

    Returns
    -------
    torch.nn.Sequential
        A trained surrogate (PyTorch Sequential, a.k.a multilayer perceptron).
    """

    #   1. infer input and target dimensionalities
    if len(input_array_train_norm.shape) == 1:
        input_array_train_norm = input_array_train_norm.reshape(-1, 1)

    if len(target_array_train.shape) == 1:
        target_array_train = target_array_train.reshape(-1, 1)

    input_dimen = input_array_train_norm.shape[1]
    target_dimen = target_array_train.shape[1]

    if print_flag:
        print("trainSurrogate():")
        print("\tinput_dimen:", input_dimen)
        print("\ttarget_dimen:", target_dimen)
        print()

    #   2. init surrogate
    surrogate = torch.nn.Sequential()

    #   3. add input layer
    surrogate.add_module(
        "input layer",
        torch.nn.Linear(input_dimen, hidden_layer_neurons)
    )
    surrogate.add_module("input activation", torch.nn.ReLU())

    #   4. add hidden layers
    for layer in range(0, input_dimen):
        surrogate.add_module(
            "hidden layer {}".format(layer),
            torch.nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        )
        surrogate.add_module(
            "hidden activation {}".format(layer),
            torch.nn.ReLU()
        )

    #   5. add output layer
    surrogate.add_module(
        "output layer",
        torch.nn.Linear(hidden_layer_neurons, target_dimen)
    )

    if print_flag:
        print("\tsurrogate:", surrogate)
        print()

    #   6. init optimizer
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.001)

    #   7. hold back validation data
    (
        input_array_train_norm,
        input_array_valid_norm,
        target_array_train,
        target_array_valid
    ) = sklms.train_test_split(
        input_array_train_norm,
        target_array_train,
        test_size=validation_size
    )

    if print_flag:
        print("\ttrain_test_split():")
        print("\t\tinput_array_train_norm.shape:", input_array_train_norm.shape)
        print("\t\tinput_array_valid_norm.shape:", input_array_valid_norm.shape)
        print("\t\ttarget_array_train.shape:", target_array_train.shape)
        print("\t\ttarget_array_valid.shape:", target_array_valid.shape)
        print()

    #   8. cast to PyTorch tensors (float32)
    input_tensor_train_norm = torch.from_numpy(
        input_array_train_norm.astype(np.float32, casting="same_kind")
    )

    input_tensor_valid_norm = torch.from_numpy(
        input_array_valid_norm.astype(np.float32, casting="same_kind")
    )

    target_tensor_train = torch.from_numpy(
        target_array_train.astype(np.float32, casting="same_kind")
    )

    target_tensor_valid = torch.from_numpy(
        target_array_valid.astype(np.float32, casting="same_kind")
    )

    #   9. train model (using validation and early stopping)
    if print_flag:
        print("\tTraining surrogate ...")

    loss_func = surrogateLoss()

    no_improvement_epochs = 0
    best_valid_loss = np.inf

    for epoch in range(num_epochs):
        batch_start_idx = 0
        batch_end_idx = batch_size
        batch_max_idx = input_tensor_train_norm.shape[0]

        if batch_end_idx > batch_max_idx:
            batch_end_idx = batch_max_idx

        #   9.1. back propogation using batch training loss
        while batch_end_idx <= batch_max_idx:
            input_tensor_train_norm_batch = (
                input_tensor_train_norm[batch_start_idx : batch_end_idx, :]
            )

            target_tensor_train_batch = (
                target_tensor_train[batch_start_idx : batch_end_idx, :]
            )

            if (
                input_tensor_train_norm_batch.shape[0] == 0
                or target_tensor_train_batch.shape[0] == 0
            ):
                break

            prediction_tensor_train_batch = surrogate(input_tensor_train_norm_batch)

            train_loss = loss_func(
                prediction_tensor_train_batch,
                target_tensor_train_batch
            )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_start_idx = batch_end_idx
            batch_end_idx += batch_size

            if batch_end_idx > batch_max_idx:
                batch_end_idx = batch_max_idx

            if print_flag:
                print("\t\t(trial {})".format(trial), end="")
                print("  epoch:", epoch + 1, end="")

                print(
                    "  training loss:",
                    round(train_loss.item(), 5),
                    end=""
                )

                print(
                    "  ({} % of batches)".format(
                        round(100 * (batch_end_idx / batch_max_idx))
                    ),
                    end="\r"
                )

        #   9.2. compute validation loss
        prediction_tensor_valid = surrogate(input_tensor_valid_norm)

        valid_loss = loss_func(
            prediction_tensor_valid,
            target_tensor_valid
        )

        if print_flag:
            print("\t\t(trial {})".format(trial), end="")
            print("  epoch:", epoch + 1, end="")
            print("  training loss:", round(train_loss.item(), 5), end="")
            print("  validation loss:", round(valid_loss.item(), 5), end="")

        #   9.3. handle checkpoints
        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            no_improvement_epochs = 0

            torch.save(surrogate.state_dict(), PATH_2_CHECKPOINT)

        else:
            no_improvement_epochs += 1

        if print_flag:
            print()

        #   9.4. handle early stopping
        if no_improvement_epochs > patience_epochs:
            break

    surrogate.load_state_dict(torch.load(PATH_2_CHECKPOINT))

    if print_flag:
        print()
        print()
        print("# ================ #")
        print()

    return surrogate


def testSurrogate(
    surrogate: torch.nn.Sequential,
    input_array_test_norm: np.array,
    print_flag: bool = False
) -> np.array:
    """
    Helper function to test a surrogate model.

    Parameters
    ----------
    surrogate: torch.nn.Sequential
        A trained surrogate (PyTorch Sequential, a.k.a multilayer perceptron).

    input_array_test_norm: np.array
        An array of normalized test inputs.

    print_flag: bool, optional, default False
        A boolean which indicates whether or not to do running prints.

    Returns
    -------
    np.array
        An array of surrogate predictions.
    """

    #   1. cast to PyTorch tensor (float32)
    input_tensor_test_norm = torch.from_numpy(
        input_array_test_norm.astype(np.float32, casting="same_kind")
    )

    #   2. get prediction tensor
    prediction_tensor_test = surrogate(input_tensor_test_norm)

    #   3. cast prediction tensor to np.array
    prediction_array_test = prediction_tensor_test.detach().numpy()

    if print_flag:
        print("testSurrogate():")
        print()
        print("# ================ #")
        print()

    return prediction_array_test


def logResults(
    target_array_test: np.array,
    prediction_array_test: np.array,
    trial: int,
    number_of_samples: int,
    dimensionality: int,
    objective_func: str,
    sampling_scheme: str,
    logging_dict: dict,
    force_write: bool = False,
    print_flag: bool = False
) -> dict:
    """
    Helper function to log results.

    Parameters
    ----------
    target_array_test: np.array
        An array of test targets.

    prediction_array_test: np.array
        An array of test predictions.

    trial: int
        A counter for the number of trials completed.

    number_of_samples: int
        The number of samples taken in constructing the surrogate.

    dimensionality: int
        The dimensionality (input space) of the surrogate.

    objective_func: str
        A string naming the objective function. Should be one of `rastrigin`,
        `rosenbrock`, `griewank`, or `styblinskitang`.

    sampling_scheme: str
        A string naming the sampling scheme. Should be either `simple random`
        or `latin hypercube`.

    logging_dict: dict
        A dictionary for logging trial results.

    force_write: bool, optional, default False
        A boolean which indicates whether or not to force a disk write.

    print_flag: bool, optional, default False
        A boolean which indicates whether or not to do running prints.

    Returns
    -------
    dict
        A dictionary for logging trial results.
    """

    #   1. init logging dict if empty
    if len(logging_dict) == 0:
        logging_dict = {
            "Benchmark Problem": [],
            "Sampling Scheme": [],
            "Dimensionality [  ]": [],
            "Number of Samples [  ]": [],
            "d-APE (0-perc) [  ]": [],
            "d-APE (5-perc) [  ]": [],
            "d-APE (10-perc) [  ]": [],
            "d-APE (20-perc) [  ]": [],
            "d-APE (25-perc) [  ]": [],
            "d-APE (40-perc) [  ]": [],
            "d-APE (50-perc) [  ]": [],
            "d-APE (60-perc) [  ]": [],
            "d-APE (75-perc) [  ]": [],
            "d-APE (80-perc) [  ]": [],
            "d-APE (90-perc) [  ]": [],
            "d-APE (95-perc) [  ]": [],
            "d-APE (100-perc) [  ]": [],
            "d-APE (mean) [  ]": [],
            "Surrogate Efficiency [  ]": []
        }

    #   2. compute d-APE percentiles
    dAPE_percentiles_dict = flib.getDampedAPEPercentiles(
        prediction_array_test,
        target_array_test,
        percentiles_list=[0, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 100]
    )

    #   3. compute surrogate efficiency
    mean_dAPE = dAPE_percentiles_dict["mean"]
    IQR_dAPE = dAPE_percentiles_dict[75] - dAPE_percentiles_dict[25]

    surrogate_efficiency = flib.getSurrogateEfficiency(
        mean_dAPE,
        IQR_dAPE,
        number_of_samples,
        dimensionality
    )

    #   4. write to logging dict
    logging_dict["Benchmark Problem"].append(objective_func)
    logging_dict["Sampling Scheme"].append(sampling_scheme)
    logging_dict["Dimensionality [  ]"].append(dimensionality)
    logging_dict["Number of Samples [  ]"].append(number_of_samples)

    for key in dAPE_percentiles_dict.keys():
        if key == "mean":
            continue

        logging_dict["d-APE ({}-perc) [  ]".format(key)].append(
            dAPE_percentiles_dict[key]
        )

    logging_dict["d-APE (mean) [  ]"].append(dAPE_percentiles_dict["mean"])
    logging_dict["Surrogate Efficiency [  ]"].append(surrogate_efficiency)

    #   5. write to disk
    if trial % 10 == 0:
        logging_dataframe = pd.DataFrame(logging_dict)
        logging_dataframe.to_csv(PATH_2_DATAFRAME, index=False)

    if print_flag:
        print("logResults():")
        print(logging_dict)
        print()
        print("# ================ #")
        print()

    return logging_dict


def runTrial(
    trial: int,
    number_of_samples: int,
    dimensionality: int,
    objective_func: str,
    sampling_scheme: str,
    logging_dict: dict,
    space_bounds: tuple = (-5, 5),
    test_size: float = 0.15,
    validation_size: float = 0.15,
    hidden_layer_neurons: int = 100,
    num_epochs: int = 10000,
    patience_epochs: int = 1000,
    print_flag: bool = False
) -> dict:
    """
    Function to run a single Monte Carlo trial.

    Ref:  <https://scikit-learn.org/stable/index.html>

    Parameters
    ----------
    trial: int
        A counter for the number of trials completed.

    number_of_samples: int
        The number of samples taken in constructing the surrogate.

    dimensionality: int
        The dimensionality (input space) of the surrogate.

    objective_func: str
        A string naming the objective function. Should be one of `rastrigin`,
        `rosenbrock`, `griewank`, or `styblinskitang`.

    sampling_scheme: str
        A string naming the sampling scheme. Should be either `simple random`
        or `latin hypercube`.

    logging_dict: dict
        A dictionary for logging trial results.

    space_bounds: tuple, optional, default (-5, 5)
        A tuple that defines the bounds on each dimension of the sample space.

    test_size: float, optional, default 0.15
        A ratio of the data to hold back for testing.

    validation_size: float, optional, default 0.15
        An ratio of the training data to hold back for validation (early
        stopping).

    hidden_layer_neurons: int, optional, default 100
        The number of neurons in each hidden layer of the surrogate.

    num_epochs: int, optional, default 10000
        The maximum number of training epochs.

    patience_epochs: int, optional, default 1000
        Tolerance on number epochs for which there is no reduction in the
        validaction loss metric. Used to trigger early stopping.

    print_flag: bool, optional, default False
        A boolean which indicates whether or not to do running prints.

    Returns
    -------
    dict
        A dictionary for logging trial results.
    """

    if print_flag:
        print("runTrial():")
        print()
        print("# ================ #")
        print()

    #   1. sample objective, get data
    data_array = sampleObjective(
        number_of_samples,
        dimensionality,
        objective_func,
        sampling_scheme,
        space_bounds=space_bounds,
        print_flag=print_flag
    )

    #   2. get training/test split
    input_array = data_array[:, 0:dimensionality]
    target_array = data_array[:, -1]

    (
        input_array_train,
        input_array_test,
        target_array_train,
        target_array_test
    ) = sklms.train_test_split(
        input_array, target_array, test_size=test_size
    )

    if print_flag:
        print("train_test_split():")
        print("\tinput_array_train.shape:", input_array_train.shape)
        print("\tinput_array_test.shape:", input_array_test.shape)
        print("\ttarget_array_train.shape:", target_array_train.shape)
        print("\ttarget_array_test.shape:", target_array_test.shape)
        print()
        print("# ================ #")
        print()

    #   3. normalize inputs (training and test)
    scaler = sklpp.StandardScaler()
    scaler.fit(input_array_train)

    input_array_train_norm = scaler.transform(input_array_train)
    input_array_test_norm = scaler.transform(input_array_test)

    if print_flag:
        print("StandardScaler():")
        print("\tscaler.mean_:", scaler.mean_)
        print("\tsqrt(scaler.var_):", np.sqrt(scaler.var_))
        print("\tinput_array_train_norm.shape:", input_array_train_norm.shape)
        print("\tinput_array_test_norm.shape:", input_array_test_norm.shape)
        print()
        print("# ================ #")
        print()

    #   4. train surrogate
    surrogate = trainSurrogate(
        input_array_train_norm,
        target_array_train,
        validation_size=validation_size,
        hidden_layer_neurons=hidden_layer_neurons,
        num_epochs=num_epochs,
        patience_epochs=patience_epochs,
        print_flag=print_flag
    )

    #   5. test surrogate
    prediction_array_test = testSurrogate(
        surrogate,
        input_array_test_norm,
        print_flag=print_flag
    )

    prediction_array_test = prediction_array_test.flatten()

    #   6. log results
    logging_dict = logResults(
        target_array_test,
        prediction_array_test,
        trial,
        number_of_samples,
        dimensionality,
        objective_func,
        sampling_scheme,
        logging_dict,
        print_flag=print_flag
    )

    return logging_dict


# ==== CLASSES ============================================================== #

class surrogateLoss(torch.nn.Module):
    """
    Custom surrogate loss class.

    Ref:  <https://machinelearningmastery.com/creating-custom-layers-loss-functions-pytorch/>
    """

    def __init__(self):
        super(surrogateLoss, self).__init__()

    def forward(self, prediction_tensor, target_tensor):
        #   1. construct damped absolute percentage error (d-APE) tensor
        unit_tensor = torch.tensor(
            np.ones(prediction_tensor.shape, dtype=np.float32)
        )

        denominator_tensor = torch.maximum(
            torch.abs(target_tensor),
            unit_tensor
        )

        dAPE_tensor = torch.abs(
            torch.divide(
                torch.subtract(prediction_tensor, target_tensor),
                denominator_tensor
            )
        )

        #   2. construct loss_tensor (mean d-APE + IQR d-APE)
        mean_dAPE_tensor = torch.mean(dAPE_tensor)

        dAPE_quantile_tensor = torch.quantile(
            dAPE_tensor, torch.tensor([0.25, 0.75])
        )

        IQR_dAPE_tensor = dAPE_quantile_tensor[1] - dAPE_quantile_tensor[0]

        loss_tensor = mean_dAPE_tensor + IQR_dAPE_tensor

        return loss_tensor


# ==== MAIN ================================================================= #

if __name__ == "__main__":
    benchmark_problem_list = [
        "rastrigin",
        "rosenbrock",
        "griewank",
        "styblinskitang"
    ]

    sampling_scheme_list = ["simple random", "latin hypercube"]

    dimensionality_list = [2, 3, 4, 5, 6]

    if os.path.isfile(PATH_2_DATAFRAME):
        logging_dataframe = pd.read_csv(PATH_2_DATAFRAME)
        logging_dict = logging_dataframe.to_dict(orient="list")
        skip_trials = len(logging_dict["Benchmark Problem"])

    else:
        logging_dict = {}
        skip_trials = 0

    trial = 1

    for benchmark_problem in benchmark_problem_list:
        for sampling_scheme in sampling_scheme_list:
            for D in dimensionality_list:
                samples_list_1 = [i * D for i in range(3, 11)]
                samples_list_2 = [i ** D for i in range(3, 11)]
                samples_list = samples_list_1 + samples_list_2

                if samples_list[-1] > MAX_SAMPLES:
                    samples_list[-1] = MAX_SAMPLES

                samples_list = [
                    samples for samples in samples_list
                    if samples <= MAX_SAMPLES
                ]

                for N in samples_list:
                    for monte_carlo_iteration in range(0, 50):
                        if trial > skip_trials:
                            """
                            runTrial(
                                trial: int,
                                number_of_samples: int,
                                dimensionality: int,
                                objective_func: str,
                                sampling_scheme: str,
                                logging_dict: dict,
                                space_bounds: tuple = (-5, 5),
                                test_size: float = 0.15,
                                validation_size: float = 0.15,
                                hidden_layer_neurons: int = 100,
                                num_epochs: int = 10000,
                                patience_epochs: int = 1000,
                                print_flag: bool = False
                            ) -> dict:
                            """

                            logging_dict = runTrial(
                                trial,
                                N,
                                D,
                                benchmark_problem,
                                sampling_scheme,
                                logging_dict,
                                print_flag=False
                            )

                        trial += 1

                        print("trial:", trial, end="\r", flush=True)
    print()
