import math
from functools import partial
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

from scipy import linalg
import scipy.signal as sig


def linear_ramp(xs, ys, x):
    assert len(xs) == len(ys)

    for i, x_curr in enumerate(xs):
        if x < x_curr:
            break

    x_prev = xs[i - 1] if i > 0 else 0.0
    y_curr = ys[i]
    y_prev = ys[i - 1] if i > 0 else 0.0

    slope = (y_curr - y_prev) / (x_curr - x_prev)
    return y_prev + ((x - x_prev) * slope)


def firdes(num_taps, bands, desired, antisymmetric=False, norm="inf", grid_density=32):
    order = num_taps - 1
    even_order = order % 2 == 0

    fs = []
    D = []

    desired = list(map(lambda x: np.pow(10, x / 20), desired))
    _f_resp = partial(linear_ramp, bands, desired)
    step = 0.5 / ((num_taps + 1) * grid_density)

    fs = np.arange(0.0, 0.5 + step, step)
    D = np.array([_f_resp(f) for f in fs])

    A = []

    for f in fs:
        if antisymmetric:
            if even_order:
                row = [
                    2 * math.sin(2 * math.pi * f * n)
                    for n in range(1, (order // 2) + 1)
                ]
            else:
                row = [
                    2 * math.sin(2 * math.pi * f * (n - 0.5))
                    for n in range(1, ((order + 1) // 2) + 1)
                ]
        else:  # symmetric
            if even_order:
                row = [1.0] + [
                    2 * math.cos(2 * math.pi * f * n)
                    for n in range(1, (order // 2) + 1)
                ]
            else:
                row = [
                    2 * math.cos(2 * math.pi * f * (n - 0.5))
                    for n in range(1, ((order + 1) // 2) + 1)
                ]
        A.append(row)

    A = np.array(A)
    x_dim = A.shape[1]

    # plt.pcolor(A, cmap='Greys')
    # plt.show()

    vx = cvx.Variable(x_dim)
    objective = cvx.Minimize(cvx.norm(A @ vx - D, norm))
    constraints = [vx <= 1, vx >= -1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(verbose=False)
    taps = np.array(vx.value)

    # glue together the calculated half-taps
    if antisymmetric:
        if even_order:
            taps = np.concatenate((np.flip(taps), np.zeros(1), -taps))
        else:
            taps = np.concatenate((np.flip(taps), -taps))
    else:
        if even_order:
            taps = np.concatenate((np.flip(taps[1:]), taps[0:1], taps[1:]))
        else:
            taps = np.concatenate((np.flip(taps), taps))

    assert len(taps) == num_taps

    return np.flip(taps), fs, D


if __name__ == "__main__":
    NUM_TAPS = 41
    taps, fs, mags = firdes(NUM_TAPS, [0, 0.24, 0.26, 0.5], [0, 0, -60, -60])
    plt.plot(fs, 20 * np.log10(abs(mags)), label="Desired response")

    w, h = sig.freqz(taps, fs=1)
    plt.plot(w, 20 * np.log10(abs(h)), label="All-in-one design function")

    taps = sig.remez(NUM_TAPS, [0, 0.24, 0.26, 0.5], [1, 10 ** (-60 / 20)], fs=1)
    w, h = sig.freqz(taps, fs=1)
    plt.plot(w, 20 * np.log10(abs(h)), label="SciPy remez")

    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 0.5)
    plt.show()

    plt.plot(fs, 20 * np.log10(abs(mags)), label="Desired response")
    for norm in [1, 2, 3, "inf"]:
        taps, fs, mags = firdes(
            NUM_TAPS, [0, 0.24, 0.26, 0.5], [0, 0, -60, -60], norm=norm
        )
        w, h = sig.freqz(taps, fs=1)
        plt.plot(w, 20 * np.log10(abs(h)), label=f"Norm: L{norm}")

    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 0.5)
    plt.show()
