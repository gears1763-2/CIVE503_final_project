"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../../LICENSE)*** <--

Library of optimization test functions.
"""


import numpy as np


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


if __name__ == "__main__":
    import math
    import random
    random.seed()

    import matplotlib.pyplot as plt


    CONTOUR_DENSITY = 32
    FLOAT_TOLERANCE = 1e-6
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
            raise RuntimeError("expected a RuntimeError here")

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
            "../tex/figures/Rastrigin.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\tFAIL")
        raise e

    else:
        print("\tPASS")

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
            raise RuntimeError("expected a RuntimeError here")

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
            "../tex/figures/Rosenbrock.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\tFAIL")
        raise e

    else:
        print("\tPASS")

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
            raise RuntimeError("expected a RuntimeError here")

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
            "../tex/figures/Griewank.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\t\tFAIL")
        raise e

    else:
        print("\t\tPASS")

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
            raise RuntimeError("expected a RuntimeError here")

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
            "../tex/figures/Styblinski–Tang.png",
            format="png",
            dpi=128
        )

    except Exception as e:
        print("\tFAIL")
        raise e

    else:
        print("\tPASS")

    plt.show()
