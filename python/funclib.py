"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../../LICENSE)*** <--

Library of optimization test functions.
"""


# ==== IMPORTS ============================================================== #

import math

import numpy as np


# ==== CONSTANTS ============================================================ #

FLOAT_TOLERANCE = 1e-6


# ==== FUNCTIONS ============================================================ #

def rastrigin(
    input_array: np.array,
    A: float = 10
) -> float:
    """
    The Rastrigin function (R^D -> R).

    Ref: <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
    Ref: <https://en.wikipedia.org/wiki/Rastrigin_function>

    Parameters
    ----------
    input_array: np.array
        The vector input to the function. Input dimensionality is inferred from
        the array length.

    A: float, optional, default 10
        A parameter of the Rastrigin function.

    Returns
    -------
    float
        The scalar output of the function.
    """

    #   1. infer dimensionality
    D = len(input_array)

    #   2. check dimensionality
    if D <= 0:
        error_string = "ERROR:  rastrigin()\t"
        error_string += "received an empty input array"

        raise RuntimeError(error_string)

    #   3. compute output
    input_array = np.array(input_array)

    constant = A * D
    array_sum_1 = np.sum(np.power(input_array, 2))
    array_sum_2 = A * np.sum(np.cos(2 * math.pi * input_array))

    return constant + array_sum_1 - array_sum_2


def rosenbrock(
    input_array: np.array,
    A: float = 100
) -> float:
    """
    The Rosenbrock function (R^D -> R).

    Ref: <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
    Ref: <https://en.wikipedia.org/wiki/Rosenbrock_function>

    Parameters
    ----------
    input_array: np.array
        The vector input to the function. Input dimensionality is inferred from
        the array length.

    A: float, optional, default 100
        A parameter of the Rosenbrock function.

    Returns
    -------
    float
        The scalar output of the function.
    """

    #   1. infer dimensionality
    D = len(input_array)

    #   2. check dimensionality
    if D <= 0:
        error_string = "ERROR:  rosenbrock()\t"
        error_string += "received an empty input array"

        raise RuntimeError(error_string)

    #   3. compute output
    input_array = np.array(input_array)

    left_array = input_array[:-1]
    right_array = input_array[1:]

    array_sum_1 = A * np.sum(np.power(right_array, 2))

    array_sum_2 = 2 * A * np.sum(
        np.multiply(right_array, np.power(left_array, 2))
    )

    array_sum_3 = A * np.sum(np.power(left_array, 4))
    array_sum_4 = np.sum(np.power(-1 * left_array + 1, 2))

    return array_sum_1 - array_sum_2 + array_sum_3 + array_sum_4


def griewank(
    input_array: np.array,
    A: float = 4000
) -> float:
    """
    The Griewank function (R^D -> R).

    Ref: <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
    Ref: <https://en.wikipedia.org/wiki/Griewank_function>

    Parameters
    ----------
    input_array: np.array
        The vector input to the function. Input dimensionality is inferred from
        the array length.

    A: float, optional, default 4000
        A parameter of the Griewank function.

    Returns
    -------
    float
        The scalar output of the function.
    """

    #   1. infer dimensionality
    D = len(input_array)

    #   2. check dimensionality
    if D <= 0:
        error_string = "ERROR:  griewank()\t"
        error_string += "received an empty input array"

        raise RuntimeError(error_string)

    #   3. compute output
    input_array = np.array(input_array)
    index_array = np.linspace(1, D, D, dtype=int)

    array_sum = (1 / A) * np.sum(np.power(input_array, 2))

    array_product = np.prod(
        np.cos(
            np.divide(input_array, np.sqrt(index_array))
        )
    )

    return 1 + array_sum - array_product


def styblinskitang(
    input_array: np.array,
    A: float = 16,
    B: float = 5
) -> float:
    """
    The Styblinski–Tang function (R^D -> R).

    Ref: <https://en.wikipedia.org/wiki/Test_functions_for_optimization>

    Parameters
    ----------
    input_array: np.array
        The vector input to the function. Input dimensionality is inferred from
        the array length.

    A: float, optional, default 16
        A first parameter of the Styblinski–Tang function.

    B: float, optional, default 5
        A second parameter of the Styblinski–Tang function.

    Returns
    -------
    float
        The scalar output of the function.
    """

    #   1. infer dimensionality
    D = len(input_array)

    #   2. check dimensionality
    if D <= 0:
        error_string = "ERROR:  styblinskitang()\t"
        error_string += "received an empty input array"

        raise RuntimeError(error_string)

    #   3. compute output
    input_array = np.array(input_array)

    array_sum_1 = np.sum(np.power(input_array, 4))
    array_sum_2 = A * np.sum(np.power(input_array, 2))
    array_sum_3 = B * np.sum(input_array)

    return 0.5 * (array_sum_1 - array_sum_2 + array_sum_3)


def getDampedAPEPercentiles(
    prediction_array: np.array,
    truth_array: np.array,
    damping_threshold: float = 1,
    percentiles_list: list = [50]
) -> dict:
    """
    Function to compute and return a dictionary of damped absolute percentage
    error (d-APE) percentiles.

    Parameters
    ----------
    prediction_array: np.array
        An array of predicted values, parallel to truth_array.

    truth_array: np.array
        An array of true values, parallel to prediction_array.

    damping_threshold: float, optional, default 1
        A threshold used in computing damped absolute percentage error values.

    percentiles_list: list, optional, default [50]
        A list of percentiles for which to extract corresponding damped
        absolute percentage error values for. Defaults to [50], in which case
        the 50th percentile (i.e. mean damped absolute percentage error) is
        extracted.

    Returns
    -------
    dict
        A dictionary of damped asbolute percentage error percentiles.
        Dictionary structure is {percentile: d-APE}.
    """

    #   1. infer array length
    N = len(prediction_array)

    #   2. check for parallel arrays
    if len(truth_array) != N:
        error_string = "ERROR:  getDampedAPEPercentiles()\t"
        error_string += "input arrays are not of equal length"

        raise RuntimeError(error_string)

    #   3. check that the damping threshold is not too small
    if abs(damping_threshold) <= FLOAT_TOLERANCE:
        error_string = "ERROR:  getDampedAPEPercentiles()\t"
        error_string += "abs(damping_threshold) must be > {}".format(
            FLOAT_TOLERANCE
        )

        raise RuntimeError(error_string)

    #   4. check that the percentiles list is non-empty
    if len(percentiles_list) == 0:
        error_string = "ERROR:  getDampedAPEPercentiles()\t"
        error_string += "percentiles list is empty"

        raise RuntimeError(error_string)

    """
    #   5. construct d-APE array
    dAPE_array = np.zeros(N)

    for i in range(0, N):
        prediction = prediction_array[i]
        truth = truth_array[i]

        if abs(truth) >= damping_threshold:
            dAPE_array[i] = abs((prediction - truth) / truth)

        else:
            dAPE_array[i] = abs((prediction - truth) / damping_threshold)
    """

    #   5. construct d-APE array (vectorized)
    dAPE_array = np.abs(
        np.divide(
            np.subtract(prediction_array, truth_array),
            np.maximum(np.abs(truth_array), damping_threshold * np.ones(N))
        )
    )

    #   6. extract d-APE array percentiles
    dAPE_percentiles_array = np.percentile(dAPE_array, percentiles_list)

    #   7. construct return dict
    dAPE_percentiles_dict = {}

    for i in range(0, len(percentiles_list)):
        percentile = percentiles_list[i]
        dAPE_percentile = dAPE_percentiles_array[i]
        dAPE_percentiles_dict[percentile] = dAPE_percentile

    return dAPE_percentiles_dict


def getSurrogateEfficiency(
    mean_dAPE: float,
    IQR_dAPE: float,
    number_of_samples: int,
    dimensionality: int
) -> float:
    """
    Function to compute surrogate efficiency (logically, surrogate performance
    divided by surrogate cost).

    Parameters
    ----------
    mean_dAPE: float
        A mean damped absolute percentage error value for the surrogate.

    IQR_dAPE: float
        A value for the inter-quartile range of damped asbolute percentages for
        the surrogate.

    number_of_samples: int
        The number of samples taken in constructing the surrogate.

    dimensionality: int
        The dimensionality (input space) of the surrogate.

    Returns
    -------
    float
        A measure of surrogate efficiency.
    """

    return math.exp(
        -1 * math.pow(number_of_samples, 1 / dimensionality)
        * (mean_dAPE + IQR_dAPE)
    )


# ==== TESTS ================================================================ #

if __name__ == "__main__":
    import random
    random.seed()

    import matplotlib.pyplot as plt


    CONTOUR_DENSITY = 32
    PLOT_DENSITY = 256
    PLOT_RANGE = 5
    TRIAL_COUNT = 100


    #   1. Rastrigin function tests
    print("testing rastrigin() ... ", end="", flush=True)

    try:
        #   1.1. pass empty input (should raise)
        try:
            rastrigin([])

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   1.2. for any arbitrary input vector, output should be >= 0
        for trial in range(0, TRIAL_COUNT):
            test_array = (
                random.uniform(1, 10)
                * np.random.rand(random.randint(1, 100))
            )

            test_output = rastrigin(test_array)
            assert(test_output >= 0)

        #   1.3. for any arbitrary zero vector, output should be 0
        for trial in range(0, TRIAL_COUNT):
            test_array = np.zeros(random.randint(1, 100))
            test_output = rastrigin(test_array)
            assert(abs(test_output) <= FLOAT_TOLERANCE)

        #   1.4. make R^2 -> R plot
        x_array = np.linspace(-1 * PLOT_RANGE, PLOT_RANGE, PLOT_DENSITY)
        f_array = np.zeros((PLOT_DENSITY, PLOT_DENSITY))

        for i in range(0, PLOT_DENSITY):
            x_0 = x_array[i]
            for j in range(0, PLOT_DENSITY):
                x_1 = x_array[j]
                f_array[j, i] = rastrigin([x_0, x_1])

        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.contourf(
            x_array,
            x_array,
            f_array,
            levels=np.linspace(
                math.floor(np.min(f_array)),
                math.ceil(np.max(f_array)),
                CONTOUR_DENSITY
            ),
            cmap="viridis",
            extend="both",
            antialiased=False,
            alpha=0.95,
            zorder=2
        )
        plt.colorbar(label="Rastrigin Function")
        plt.scatter(
            [0],
            [0],
            marker="*",
            color="C3",
            zorder=3,
            label=r"Global Minimum:  $f(0,0)=0$"
        )
        plt.xticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.xlabel(r"$x_1$")
        plt.yticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "../tex/figures/test_Rastrigin.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\t\tFAIL")
        raise e

    else:
        print("\t\tPASS")

    #   2. Rosenbrock function tests
    print("testing rosenbrock() ... ", end="", flush=True)

    try:
        #   2.1. pass empty input (should raise)
        try:
            rosenbrock([])

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   2.2. for any arbitrary input vector, output should be >= 0
        for trial in range(0, TRIAL_COUNT):
            test_array = (
                random.uniform(1, 10)
                * np.random.rand(random.randint(1, 100))
            )

            test_output = rosenbrock(test_array)
            assert(test_output >= 0)

        #   2.3. for any arbitrary vector of ones, output should be 0
        for trial in range(0, TRIAL_COUNT):
            test_array = np.ones(random.randint(1, 100))
            test_output = rosenbrock(test_array)
            assert(abs(test_output) <= FLOAT_TOLERANCE)

        #   2.4. make R^2 -> R plot
        x_array = np.linspace(-1 * PLOT_RANGE, PLOT_RANGE, PLOT_DENSITY)
        f_array = np.zeros((PLOT_DENSITY, PLOT_DENSITY))

        for i in range(0, PLOT_DENSITY):
            x_0 = x_array[i]
            for j in range(0, PLOT_DENSITY):
                x_1 = x_array[j]
                f_array[j, i] = rosenbrock([x_0, x_1])

        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.contourf(
            x_array,
            x_array,
            f_array,
            levels=np.append(
                np.array(
                    [math.floor(np.min(f_array)), 2e-5 * np.max(f_array)]
                ),
                np.linspace(
                    3e-5 * np.max(f_array),
                    math.ceil(np.max(f_array)),
                    CONTOUR_DENSITY - 2
                )
            ),
            cmap="viridis",
            extend="both",
            antialiased=False,
            alpha=0.95,
            zorder=2
        )
        plt.colorbar(label="Rosenbrock Function")
        plt.scatter(
            [1],
            [1],
            marker="*",
            color="C3",
            zorder=3,
            label=r"Global Minimum:  $f(1,1)=0$"
        )
        plt.xticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.xlabel(r"$x_1$")
        plt.yticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "../tex/figures/test_Rosenbrock.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\t\tFAIL")
        raise e

    else:
        print("\t\tPASS")

    #   3. Griewank function tests
    print("testing griewank() ... ", end="", flush=True)

    try:
        #   3.1. pass empty input (should raise)
        try:
            griewank([])

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   3.2. for any arbitrary input vector, output should be >= 0
        for trial in range(0, TRIAL_COUNT):
            test_array = (
                random.uniform(1, 10)
                * np.random.rand(random.randint(1, 100))
            )

            test_output = griewank(test_array)
            assert(test_output >= 0)

        #   3.3. for any arbitrary zero vector, output should be 0
        for trial in range(0, TRIAL_COUNT):
            test_array = np.zeros(random.randint(1, 100))
            test_output = griewank(test_array)
            assert(abs(test_output) <= FLOAT_TOLERANCE)

        #   3.4. make R^2 -> R plot
        x_array = np.linspace(-1 * PLOT_RANGE, PLOT_RANGE, PLOT_DENSITY)
        f_array = np.zeros((PLOT_DENSITY, PLOT_DENSITY))

        for i in range(0, PLOT_DENSITY):
            x_0 = x_array[i]
            for j in range(0, PLOT_DENSITY):
                x_1 = x_array[j]
                f_array[j, i] = griewank([x_0, x_1])

        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.contourf(
            x_array,
            x_array,
            f_array,
            levels=np.linspace(
                math.floor(np.min(f_array)),
                math.ceil(np.max(f_array)),
                CONTOUR_DENSITY
            ),
            cmap="viridis",
            extend="both",
            antialiased=False,
            alpha=0.95,
            zorder=2
        )
        plt.colorbar(label="Griewank Function")
        plt.scatter(
            [0],
            [0],
            marker="*",
            color="C3",
            zorder=3,
            label=r"Global Minimum:  $f(0,0)=0$"
        )
        plt.xticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.xlabel(r"$x_1$")
        plt.yticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "../tex/figures/test_Griewank.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\t\t\tFAIL")
        raise e

    else:
        print("\t\t\tPASS")

    #   4. Styblinski–Tang function tests
    print("testing styblinskitang() ... ", end="", flush=True)

    try:
        #   4.1. pass empty input (should raise)
        try:
            styblinskitang([])

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   4.2. for any arbitrary input vector of dimension D,
        #        output should be > -39.16617 * D
        for trial in range(0, TRIAL_COUNT):
            test_array = (
                random.uniform(1, 10)
                * np.random.rand(random.randint(1, 100))
            )

            test_output = styblinskitang(test_array)

            D = len(test_array)
            assert(test_output > -39.16617 * D)

        #   4.3. for any optimal input vector of dimension D,
        #        output should be in (-39.16617 * D, -39.16616 * D)
        for trial in range(0, TRIAL_COUNT):
            test_array = -2.903534 * np.ones(random.randint(1, 100))
            test_output = styblinskitang(test_array)

            D = len(test_array)
            assert(test_output > -39.16617 * D)
            assert(test_output < -39.16616 * D)

        #   4.4. make R^2 -> R plot
        x_array = np.linspace(-1 * PLOT_RANGE, PLOT_RANGE, PLOT_DENSITY)
        f_array = np.zeros((PLOT_DENSITY, PLOT_DENSITY))

        for i in range(0, PLOT_DENSITY):
            x_0 = x_array[i]
            for j in range(0, PLOT_DENSITY):
                x_1 = x_array[j]
                f_array[j, i] = styblinskitang([x_0, x_1])

        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.contourf(
            x_array,
            x_array,
            f_array,
            levels=np.linspace(
                math.floor(np.min(f_array)),
                math.ceil(np.max(f_array)),
                CONTOUR_DENSITY
            ),
            cmap="viridis",
            extend="both",
            antialiased=False,
            alpha=0.95,
            zorder=2
        )
        plt.colorbar(label="Styblinski–Tang Function")
        plt.scatter(
            [-2.903534],
            [-2.903534],
            marker="*",
            color="C3",
            zorder=3,
            label=r"Global Minimum:  $-78.33234 < f(-2.903534,-2.903534) < -78.33232$"
        )
        plt.xticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.xlabel(r"$x_1$")
        plt.yticks([i for i in range(-1 * PLOT_RANGE, PLOT_RANGE + 1)])
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "../tex/figures/test_Styblinski–Tang.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\t\tFAIL")
        raise e

    else:
        print("\t\tPASS")


    #   5. d-APE percentiles function tests
    print("testing getDampedAPEPercentiles() ... ", end="", flush=True)

    try:
        #   5.1. pass arrays of different length (should raise)
        try:
            getDampedAPEPercentiles(
                np.random.rand(100),
                np.random.rand(200),
                damping_threshold=1,
                percentiles_list=[50]
            )

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   5.2. pass exceedingly small damping threshold (should raise)
        try:
            getDampedAPEPercentiles(
                np.random.rand(100),
                np.random.rand(100),
                damping_threshold=0,
                percentiles_list=[50]
            )

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   5.3. pass empty percentiles list (should raise)
        try:
            getDampedAPEPercentiles(
                np.random.rand(100),
                np.random.rand(100),
                damping_threshold=1,
                percentiles_list=[]
            )

        except RuntimeError as e:
            pass

        except Exception as e:
            raise e

        else:
            raise RuntimeError("TEST:  expected a RuntimeError here")

        #   5.4. test normal operation
        test_prediction_array = np.array(
            [
                2.7713772, 2.8543618, 2.6379359, 3.5680907,
                4.1981345, 3.7477980, 2.8567891, 1.1559992,
                0.8527500, 1.7579608, 0.8053023, 1.4561078,
                0.7113965, 3.0035538, 0.4478658, 1.9958152,
                0.0217980, 0.2521098, 0.2268945, 0.2894086
            ]
        )

        test_truth_array = np.array(
            [
                1.6160982, 3.7768894, 4.8624433, 1.2993779,
                4.0063128, 2.5712496, 2.8342452, 0.7757955,
                0.5429077, 1.1872136, 0.5479943, 3.4974758,
                1.6411618, 4.8819121, 3.9007399, 2.6261119,
                4.1244293, 2.6569416, 3.0827306, 3.7069864
            ]
        )

        expected_percentiles_dict = {
            0:   0.0079541,
            5:   0.0458836,
            10:  0.2207982,
            20:  0.2546976,
            40:  0.4283960,
            60:  0.5733848,
            80:  0.9084760,
            90:  0.9332299,
            95:  1.0322791,
            100: 1.7459992
        }

        test_percentiles_dict = getDampedAPEPercentiles(
            test_prediction_array,
            test_truth_array,
            damping_threshold=1,
            percentiles_list=[0, 5, 10, 20, 40, 60, 80, 90, 95, 100]
        )

        for percentile in expected_percentiles_dict.keys():
            expected_percentile = expected_percentiles_dict[percentile]
            test_percentile = test_percentiles_dict[percentile]

            assert(
                abs(test_percentile - expected_percentile) <= FLOAT_TOLERANCE
            )

    except Exception as e:
        print("\tFAIL")
        raise e

    else:
        print("\tPASS")

    #   6. surrogate efficiency function tests
    print("testing getSurrogateEfficiency() ... ", end="", flush=True)

    try:
        #   6.1. for any arbitrary positive inputs, efficiency should be
        #        between 0 and 1.
        for trial in range(0, TRIAL_COUNT):
            test_efficiency = getSurrogateEfficiency(
                random.uniform(0, 1),
                random.uniform(0, 1),
                random.randint(10, 10000),
                random.randint(2, 20)
            )

            assert(test_efficiency >= 0)
            assert(test_efficiency <= 1)

    except Exception as e:
        print("\tFAIL")
        raise e

    else:
        print("\tPASS")

    #plt.show()
