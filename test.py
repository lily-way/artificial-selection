import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

GLOBAL_LOG = "all_runs.csv"


def initialize_global_log():
    if not os.path.exists(GLOBAL_LOG):
        with open(GLOBAL_LOG, "w") as f:
            f.write("run_id,filename,Du,Dv,F,k,region\n")


@njit
def initialize_grid(n, seed_size, noise_strength, rng_seed):
    np.random.seed(rng_seed)

    U = np.ones((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    center = n // 2
    half = seed_size // 2

    for i in range(center - half, center + half):
        for j in range(center - half, center + half):
            U[i, j] = 0.50
            V[i, j] = 0.25

    for i in range(n):
        for j in range(n):
            U[i, j] += noise_strength * np.random.rand()
            V[i, j] += noise_strength * np.random.rand()

            if U[i, j] < 0.0:
                U[i, j] = 0.0
            elif U[i, j] > 1.0:
                U[i, j] = 1.0

            if V[i, j] < 0.0:
                V[i, j] = 0.0
            elif V[i, j] > 1.0:
                V[i, j] = 1.0

    return U, V


@njit
def step_gray_scott(U, V, U_next, V_next, Du, Dv, F, k, dt):
    n = U.shape[0]

    for i in range(n):
        up = (i - 1) % n
        down = (i + 1) % n

        for j in range(n):
            left = (j - 1) % n
            right = (j + 1) % n

            Lu = U[up, j] + U[down, j] + U[i, left] + U[i, right] - 4.0 * U[i, j]
            Lv = V[up, j] + V[down, j] + V[i, left] + V[i, right] - 4.0 * V[i, j]

            uvv = U[i, j] * V[i, j] * V[i, j]

            u_new = U[i, j] + (Du * Lu - uvv + F * (1.0 - U[i, j])) * dt
            v_new = V[i, j] + (Dv * Lv + uvv - (F + k) * V[i, j]) * dt

            if u_new < 0.0:
                u_new = 0.0
            elif u_new > 1.0:
                u_new = 1.0

            if v_new < 0.0:
                v_new = 0.0
            elif v_new > 1.0:
                v_new = 1.0

            U_next[i, j] = u_new
            V_next[i, j] = v_new


@njit
def run_gray_scott(Du, Dv, F, k, n, steps, dt, seed_size, noise_strength, rng_seed):
    U, V = initialize_grid(n, seed_size, noise_strength, rng_seed)

    U_next = np.empty_like(U)
    V_next = np.empty_like(V)

    for _ in range(steps):
        step_gray_scott(U, V, U_next, V_next, Du, Dv, F, k, dt)

        temp = U
        U = U_next
        U_next = temp

        temp = V
        V = V_next
        V_next = temp

    return U, V


def save_image(V, filename, cmap="plasma"):
    plt.figure(figsize=(4, 4))
    plt.imshow(V, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def show_gallery(results, cols=3, cmap="plasma"):
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    flat_axes = axes.flatten()

    for ax, r in zip(flat_axes, results):
        ax.imshow(r["V"], cmap=cmap, interpolation="nearest")
        ax.set_title(
            f'{r["label"]}\nF={r["params"]["F"]:.5f}\nk={r["params"]["k"]:.5f}',
            fontsize=10,
        )
        ax.axis("off")

    for ax in flat_axes[len(results) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def sample_tight_local_neighbors(num_patterns=9):
    """
    Tighter stable box based on your observed good results.
    """
    params_list = []

    for _ in range(num_patterns):
        F = np.random.uniform(0.0294, 0.0312)
        k = np.random.uniform(0.0549, 0.0561)

        params_list.append(
            {
                "Du": 0.16,
                "Dv": 0.08,
                "F": F,
                "k": k,
                "region": "tight_local_test",
            }
        )

    return params_list


def main():
    initialize_global_log()

    n = 150
    steps = 5000
    dt = 1.0
    seed_size = 16
    noise_strength = 0.01
    num_patterns = 9

    run_id = np.random.randint(0, 1_000_000)
    print("Run ID:", run_id)

    folder = f"runs/run_{run_id}"
    os.makedirs(folder, exist_ok=True)

    # Same seed within one run for fair comparison
    base_seed = run_id

    params_list = sample_tight_local_neighbors(num_patterns=num_patterns)

    results = []

    for i, params in enumerate(params_list):
        _, V = run_gray_scott(
            Du=params["Du"],
            Dv=params["Dv"],
            F=params["F"],
            k=params["k"],
            n=n,
            steps=steps,
            dt=dt,
            seed_size=seed_size,
            noise_strength=noise_strength,
            rng_seed=base_seed,
        )

        filename = f"{folder}/tight_{i+1}_" f"F{params['F']:.5f}_k{params['k']:.5f}.png"

        save_image(V, filename)

        with open(GLOBAL_LOG, "a") as f:
            f.write(
                f"{run_id},"
                f"{filename},"
                f"{params['Du']},"
                f"{params['Dv']},"
                f"{params['F']:.5f},"
                f"{params['k']:.5f},"
                f"{params['region']}\n"
            )

        results.append(
            {
                "label": f"Pattern {i+1}",
                "params": params,
                "V": V,
            }
        )

    show_gallery(results, cols=3)


if __name__ == "__main__":
    main()
