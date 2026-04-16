import numpy as np
import matplotlib.pyplot as plt


def make_grid(n=300):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    return np.meshgrid(x, y)


def smooth_noise(X, Y, scale_x, scale_y, phase_x, phase_y):
    return (
        np.sin(scale_x * X + phase_x)
        + np.sin(scale_y * Y + phase_y)
        + np.sin((scale_x + scale_y) * (X + Y) + 0.5 * (phase_x + phase_y))
    ) / 3.0


def generate_pattern(params, n=300):
    X, Y = make_grid(n)

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)

    warp_x = params["warp_amp"] * np.sin(
        params["warp_freq_x"] * Y + params["warp_phase_x"]
    )
    warp_y = params["warp_amp"] * np.cos(
        params["warp_freq_y"] * X + params["warp_phase_y"]
    )

    Xw = X + warp_x
    Yw = Y + warp_y

    Rw = np.sqrt(Xw**2 + Yw**2)
    Thetaw = np.arctan2(Yw, Xw)

    field = (
        params["amp1"] * np.sin(params["freq1_x"] * Xw + params["phase1"])
        + params["amp2"] * np.sin(params["freq2_y"] * Yw + params["phase2"])
        + params["amp3"] * np.sin(params["radial_freq"] * Rw + params["phase3"])
        + params["amp4"] * np.sin(params["angular_freq"] * Thetaw + params["phase4"])
    )

    noise = smooth_noise(
        Xw,
        Yw,
        params["noise_freq_x"],
        params["noise_freq_y"],
        params["noise_phase_x"],
        params["noise_phase_y"],
    )

    field += params["noise_amp"] * noise
    field += params["radial_bias_amp"] * np.cos(params["radial_bias_freq"] * Rw)

    pattern = np.tanh(params["contrast"] * field)
    binaryish = pattern > params["threshold"]
    return 0.65 * pattern + 0.35 * binaryish.astype(float)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mutate_genotype(genotype, strength=0.08):
    child = dict(genotype)

    for key, value in child.items():
        if "phase" in key:
            child[key] = (value + np.random.normal(0, strength * np.pi)) % (2 * np.pi)
        elif key == "threshold":
            child[key] = np.clip(value + np.random.normal(0, 0.08), -0.8, 0.8)
        elif key == "contrast":
            child[key] = np.clip(value + np.random.normal(0, 0.3), 0.8, 6.0)
        elif key == "warp_amp":
            child[key] = np.clip(value + np.random.normal(0, 0.03), 0.0, 0.45)
        elif "amp" in key:
            child[key] = np.clip(value + np.random.normal(0, 0.10), -1.5, 1.5)
        elif "freq" in key:
            child[key] = np.clip(value + np.random.normal(0, 0.7), 1.0, 24.0)
        else:
            child[key] = value + np.random.normal(0, strength)

    return child


def blend_value(a, b, alpha):
    return (1 - alpha) * a + alpha * b


def blend_genotypes(parent_a, parent_b, alpha):
    return {k: blend_value(parent_a[k], parent_b[k], alpha) for k in parent_a}


# -----------------------------
# Hand-designed archetypes
# -----------------------------


def stripe_archetype():
    return {
        "freq1_x": 10.0,
        "freq2_y": 4.5,
        "radial_freq": 7.0,
        "angular_freq": 3.0,
        "phase1": 0.2,
        "phase2": 1.1,
        "phase3": 0.7,
        "phase4": 2.2,
        "amp1": 1.1,
        "amp2": 0.5,
        "amp3": 0.2,
        "amp4": 0.1,
        "warp_amp": 0.06,
        "warp_freq_x": 3.0,
        "warp_freq_y": 5.0,
        "warp_phase_x": 0.8,
        "warp_phase_y": 1.5,
        "noise_amp": 0.08,
        "noise_freq_x": 6.0,
        "noise_freq_y": 8.0,
        "noise_phase_x": 0.3,
        "noise_phase_y": 2.0,
        "radial_bias_amp": 0.05,
        "radial_bias_freq": 4.0,
        "contrast": 2.8,
        "threshold": 0.0,
    }


def radial_archetype():
    return {
        "freq1_x": 5.0,
        "freq2_y": 5.0,
        "radial_freq": 16.0,
        "angular_freq": 8.0,
        "phase1": 1.7,
        "phase2": 0.5,
        "phase3": 0.2,
        "phase4": 1.0,
        "amp1": 0.3,
        "amp2": 0.3,
        "amp3": 1.0,
        "amp4": 0.7,
        "warp_amp": 0.03,
        "warp_freq_x": 2.5,
        "warp_freq_y": 2.5,
        "warp_phase_x": 0.4,
        "warp_phase_y": 1.1,
        "noise_amp": 0.05,
        "noise_freq_x": 5.0,
        "noise_freq_y": 5.0,
        "noise_phase_x": 2.1,
        "noise_phase_y": 0.7,
        "radial_bias_amp": 0.45,
        "radial_bias_freq": 6.0,
        "contrast": 3.2,
        "threshold": 0.05,
    }


def marbled_archetype():
    return {
        "freq1_x": 7.5,
        "freq2_y": 9.5,
        "radial_freq": 9.0,
        "angular_freq": 4.5,
        "phase1": 2.0,
        "phase2": 0.8,
        "phase3": 2.7,
        "phase4": 1.8,
        "amp1": 0.8,
        "amp2": 0.9,
        "amp3": 0.4,
        "amp4": 0.3,
        "warp_amp": 0.22,
        "warp_freq_x": 7.0,
        "warp_freq_y": 6.0,
        "warp_phase_x": 0.9,
        "warp_phase_y": 2.5,
        "noise_amp": 0.35,
        "noise_freq_x": 9.0,
        "noise_freq_y": 7.0,
        "noise_phase_x": 1.2,
        "noise_phase_y": 2.7,
        "radial_bias_amp": -0.10,
        "radial_bias_freq": 3.0,
        "contrast": 2.5,
        "threshold": -0.05,
    }


def spotty_archetype():
    return {
        "freq1_x": 8.0,
        "freq2_y": 8.0,
        "radial_freq": 12.0,
        "angular_freq": 10.0,
        "phase1": 0.1,
        "phase2": 2.4,
        "phase3": 1.5,
        "phase4": 0.6,
        "amp1": 0.5,
        "amp2": 0.5,
        "amp3": 0.8,
        "amp4": 0.6,
        "warp_amp": 0.10,
        "warp_freq_x": 4.0,
        "warp_freq_y": 4.0,
        "warp_phase_x": 1.9,
        "warp_phase_y": 0.4,
        "noise_amp": 0.20,
        "noise_freq_x": 10.0,
        "noise_freq_y": 10.0,
        "noise_phase_x": 0.8,
        "noise_phase_y": 1.6,
        "radial_bias_amp": 0.15,
        "radial_bias_freq": 7.5,
        "contrast": 4.0,
        "threshold": 0.18,
    }


def make_initial_population():
    archetypes = [
        stripe_archetype(),
        radial_archetype(),
        marbled_archetype(),
        spotty_archetype(),
    ]

    pop = []
    for a in archetypes:
        pop.append(mutate_genotype(a, strength=0.06))

    pop.append(
        mutate_genotype(
            blend_genotypes(stripe_archetype(), radial_archetype(), 0.5), strength=0.08
        )
    )
    pop.append(
        mutate_genotype(
            blend_genotypes(marbled_archetype(), spotty_archetype(), 0.5), strength=0.08
        )
    )

    return pop


def make_next_generation(parent_a, parent_b):
    children = []
    alphas = [0.1, 0.9, 0.3, 0.7, 0.5, np.random.uniform(0.2, 0.8)]
    mutation_strengths = [0.03, 0.03, 0.05, 0.05, 0.08, 0.12]

    for alpha, m in zip(alphas, mutation_strengths):
        child = blend_genotypes(parent_a, parent_b, alpha)
        child = mutate_genotype(child, strength=m)
        children.append(child)

    return children


def show_population(population, generation_num, n=300):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, genotype in enumerate(population):
        img = generate_pattern(genotype, n=n)
        axes[i].imshow(img, cmap="plasma", interpolation="nearest")
        axes[i].set_title(f"{i+1}", fontsize=14)
        axes[i].axis("off")

    plt.suptitle(f"Generation {generation_num}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    population = make_initial_population()
    num_generations = 5

    for generation in range(1, num_generations + 1):
        show_population(population, generation, n=300)

        if generation == num_generations:
            break

        try:
            raw = input("Choose two parents by index, like '2 5': ").strip().split()
            a_idx, b_idx = int(raw[0]) - 1, int(raw[1]) - 1
            assert 0 <= a_idx < 6 and 0 <= b_idx < 6 and a_idx != b_idx
        except Exception:
            print("Invalid choice, defaulting to 1 and 2.")
            a_idx, b_idx = 0, 1

        parent_a = population[a_idx]
        parent_b = population[b_idx]
        population = make_next_generation(parent_a, parent_b)


if __name__ == "__main__":
    main()
