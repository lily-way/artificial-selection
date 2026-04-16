import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# -----------------------------
# Fixed simulation settings
# -----------------------------
N = 150
DT = 1.0
STEPS = 3000  # fixed developmental stage
POP_SIZE = 6

DU = 0.16
DV = 0.08

# Stable-ish working region
F_MIN, F_MAX = 0.0288, 0.0320
K_MIN, K_MAX = 0.0546, 0.0562
SEED_SIZE_MIN, SEED_SIZE_MAX = 12, 24
NOISE_MIN, NOISE_MAX = 0.006, 0.018

# Initial parents from your good area
INITIAL_PARENT_A = {
    "Du": DU,
    "Dv": DV,
    "F": 0.02989,
    "k": 0.05528,
    "seed_size": 14,
    "noise_strength": 0.006,
}
INITIAL_PARENT_B = {
    "Du": DU,
    "Dv": DV,
    "F": 0.03064,
    "k": 0.05513,
    "seed_size": 22,
    "noise_strength": 0.018,
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# Gray-Scott core
# -----------------------------
@njit
def initialize_grid(n, seed_size, noise_strength, rng_seed):
    np.random.seed(rng_seed)

    U = np.ones((n, n), dtype=np.float64)
    V = np.zeros((n, n), dtype=np.float64)

    center = n // 2
    half = seed_size // 2

    r0 = max(0, center - half)
    r1 = min(n, center + half)
    c0 = max(0, center - half)
    c1 = min(n, center + half)

    for i in range(r0, r1):
        for j in range(c0, c1):
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


# -----------------------------
# Breeding
# -----------------------------
def blend(parent_a, parent_b, alpha):
    return {
        "Du": DU,
        "Dv": DV,
        "F": (1.0 - alpha) * parent_a["F"] + alpha * parent_b["F"],
        "k": (1.0 - alpha) * parent_a["k"] + alpha * parent_b["k"],
        "seed_size": int(
            round((1.0 - alpha) * parent_a["seed_size"] + alpha * parent_b["seed_size"])
        ),
        "noise_strength": (1.0 - alpha) * parent_a["noise_strength"]
        + alpha * parent_b["noise_strength"],
    }


def mutate(genotype, fk_scale=0.00035, seed_size_delta=2, noise_scale=0.002):
    child = dict(genotype)

    child["F"] = clamp(
        child["F"] + np.random.uniform(-fk_scale, fk_scale), F_MIN, F_MAX
    )
    child["k"] = clamp(
        child["k"] + np.random.uniform(-fk_scale, fk_scale), K_MIN, K_MAX
    )
    child["seed_size"] = int(
        round(
            clamp(
                child["seed_size"]
                + np.random.randint(-seed_size_delta, seed_size_delta + 1),
                SEED_SIZE_MIN,
                SEED_SIZE_MAX,
            )
        )
    )
    child["noise_strength"] = clamp(
        child["noise_strength"] + np.random.uniform(-noise_scale, noise_scale),
        NOISE_MIN,
        NOISE_MAX,
    )
    return child


def make_next_generation(parent_a, parent_b):
    # Diversity-preserving set of 6
    c1 = blend(parent_a, parent_b, 0.15)  # strongly A-leaning
    c2 = blend(parent_a, parent_b, 0.85)  # strongly B-leaning
    c3 = blend(parent_a, parent_b, 0.50)  # midpoint
    c4 = mutate(blend(parent_a, parent_b, 0.35))  # A-ish mutated
    c5 = mutate(blend(parent_a, parent_b, 0.65))  # B-ish mutated
    c6 = mutate(
        blend(parent_a, parent_b, np.random.uniform(0.2, 0.8)),
        fk_scale=0.0005,
        seed_size_delta=3,
        noise_scale=0.003,
    )  # exploratory child

    return [c1, c2, c3, c4, c5, c6]


def render_population(population, generation_num):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, genotype in enumerate(population):
        seed = generation_num * 1000 + i
        _, V = run_gray_scott(
            Du=genotype["Du"],
            Dv=genotype["Dv"],
            F=genotype["F"],
            k=genotype["k"],
            n=N,
            steps=STEPS,
            dt=DT,
            seed_size=genotype["seed_size"],
            noise_strength=genotype["noise_strength"],
            rng_seed=seed,
        )

        axes[i].imshow(V, cmap="plasma", interpolation="nearest")
        axes[i].set_title(
            f"{i+1}\n"
            f"F={genotype['F']:.5f}, k={genotype['k']:.5f}\n"
            f"ss={genotype['seed_size']}, nz={genotype['noise_strength']:.3f}",
            fontsize=10,
        )
        axes[i].axis("off")

    plt.suptitle(f"Generation {generation_num}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    # Initial generation from the two seed parents
    population = make_next_generation(INITIAL_PARENT_A, INITIAL_PARENT_B)

    num_generations = 5

    for generation in range(1, num_generations + 1):
        render_population(population, generation)

        print(f"\nGeneration {generation}")
        for i, g in enumerate(population, start=1):
            print(
                f"{i}: F={g['F']:.5f}, k={g['k']:.5f}, "
                f"seed_size={g['seed_size']}, noise={g['noise_strength']:.4f}"
            )

        if generation == num_generations:
            break

        try:
            raw = input("Choose two parents by index, like '2 5': ").strip().split()
            a_idx, b_idx = int(raw[0]) - 1, int(raw[1]) - 1
            assert 0 <= a_idx < POP_SIZE and 0 <= b_idx < POP_SIZE and a_idx != b_idx
        except Exception:
            print("Invalid choice, defaulting to parents 1 and 2.")
            a_idx, b_idx = 0, 1

        parent_a = population[a_idx]
        parent_b = population[b_idx]
        population = make_next_generation(parent_a, parent_b)


if __name__ == "__main__":
    main()
