import math
import random
from pyscript import document, ffi


def make_grid(n=140):
    x_vals = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]
    y_vals = [-1.0 + 2.0 * j / (n - 1) for j in range(n)]
    return x_vals, y_vals


def smooth_noise(x, y, scale_x, scale_y, phase_x, phase_y):
    return (
        math.sin(scale_x * x + phase_x)
        + math.sin(scale_y * y + phase_y)
        + math.sin((scale_x + scale_y) * (x + y) + 0.5 * (phase_x + phase_y))
    ) / 3.0


def generate_pattern(params, n=140):
    x_vals, y_vals = make_grid(n)
    image = []

    for y in y_vals:
        row = []
        for x in x_vals:
            warp_x = params["warp_amp"] * math.sin(
                params["warp_freq_x"] * y + params["warp_phase_x"]
            )
            warp_y = params["warp_amp"] * math.cos(
                params["warp_freq_y"] * x + params["warp_phase_y"]
            )

            xw = x + warp_x
            yw = y + warp_y

            rw = math.sqrt(xw * xw + yw * yw)
            thetaw = math.atan2(yw, xw)

            field = (
                params["amp1"] * math.sin(params["freq1_x"] * xw + params["phase1"])
                + params["amp2"] * math.sin(params["freq2_y"] * yw + params["phase2"])
                + params["amp3"]
                * math.sin(params["radial_freq"] * rw + params["phase3"])
                + params["amp4"]
                * math.sin(params["angular_freq"] * thetaw + params["phase4"])
            )

            noise = smooth_noise(
                xw,
                yw,
                params["noise_freq_x"],
                params["noise_freq_y"],
                params["noise_phase_x"],
                params["noise_phase_y"],
            )

            field += params["noise_amp"] * noise
            field += params["radial_bias_amp"] * math.cos(
                params["radial_bias_freq"] * rw
            )

            pattern = math.tanh(params["contrast"] * field)
            binaryish = 1.0 if pattern > params["threshold"] else 0.0
            value = 0.65 * pattern + 0.35 * binaryish

            # map to 0..255 grayscale
            gray = int(max(0, min(255, (value + 1.0) * 127.5)))
            row.append(gray)

        image.append(row)

    return image


def mutate_genotype(genotype, strength=0.08):
    child = dict(genotype)

    for key, value in child.items():
        if "phase" in key:
            child[key] = (value + random.gauss(0, strength * math.pi)) % (2 * math.pi)
        elif key == "threshold":
            child[key] = max(-0.8, min(0.8, value + random.gauss(0, 0.08)))
        elif key == "contrast":
            child[key] = max(0.8, min(6.0, value + random.gauss(0, 0.3)))
        elif key == "warp_amp":
            child[key] = max(0.0, min(0.45, value + random.gauss(0, 0.03)))
        elif "amp" in key:
            child[key] = max(-1.5, min(1.5, value + random.gauss(0, 0.10)))
        elif "freq" in key:
            child[key] = max(1.0, min(24.0, value + random.gauss(0, 0.7)))
        else:
            child[key] = value + random.gauss(0, strength)

    return child


def blend_value(a, b, alpha):
    return (1 - alpha) * a + alpha * b


def blend_genotypes(parent_a, parent_b, alpha):
    return {k: blend_value(parent_a[k], parent_b[k], alpha) for k in parent_a}


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
            blend_genotypes(stripe_archetype(), radial_archetype(), 0.5),
            strength=0.08,
        )
    )
    pop.append(
        mutate_genotype(
            blend_genotypes(marbled_archetype(), spotty_archetype(), 0.5),
            strength=0.08,
        )
    )

    return pop


def make_next_generation(parent_a, parent_b):
    children = []
    alphas = [0.1, 0.9, 0.3, 0.7, 0.5, random.uniform(0.2, 0.8)]
    mutation_strengths = [0.03, 0.03, 0.05, 0.05, 0.08, 0.12]

    for alpha, m in zip(alphas, mutation_strengths):
        child = blend_genotypes(parent_a, parent_b, alpha)
        child = mutate_genotype(child, strength=m)
        children.append(child)

    return children


GRID_SIZE = 140

state = {
    "population": [],
    "generation": 1,
    "selected": [],
}

persistent_proxies = []
card_proxies = []


def image_to_data_url(image):
    height = len(image)
    width = len(image[0])

    canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height

    ctx = canvas.getContext("2d")
    image_data = ctx.createImageData(width, height)
    data = image_data.data

    idx = 0
    for row in image:
        for gray in row:
            data[idx] = gray
            data[idx + 1] = gray
            data[idx + 2] = gray
            data[idx + 3] = 255
            idx += 4

    ctx.putImageData(image_data, 0, 0)
    return canvas.toDataURL("image/png")


def toggle_select(index):
    selected = state["selected"]

    if index in selected:
        selected.remove(index)
    else:
        if len(selected) < 2:
            selected.append(index)

    render_population()
    update_status()


def make_card_click_handler(index):
    def handler(event=None):
        toggle_select(index)

    proxy = ffi.create_proxy(handler)
    card_proxies.append(proxy)
    return proxy


def render_population():
    global card_proxies
    card_proxies = []

    container = document.getElementById("population")
    container.innerHTML = ""

    for i, genotype in enumerate(state["population"]):
        img_matrix = generate_pattern(genotype, n=GRID_SIZE)
        img_url = image_to_data_url(img_matrix)

        card = document.createElement("button")
        card.setAttribute("type", "button")
        card.className = "pattern-card"
        if i in state["selected"]:
            card.classList.add("selected")

        click_proxy = make_card_click_handler(i)
        card.addEventListener("click", click_proxy)

        label = document.createElement("div")
        label.className = "pattern-label"
        label.innerText = f"Pattern {i + 1}"

        img = document.createElement("img")
        img.src = img_url
        img.className = "pattern-image"
        img.alt = f"Pattern {i + 1}"

        helper = document.createElement("div")
        helper.className = "pattern-helper"
        if i in state["selected"]:
            helper.innerText = "Selected"
        else:
            helper.innerText = "Click to select"

        card.appendChild(label)
        card.appendChild(img)
        card.appendChild(helper)
        container.appendChild(card)


def update_status():
    gen_el = document.getElementById("generation")
    status_el = document.getElementById("status")
    next_btn = document.getElementById("next-btn")

    gen_el.innerText = f"Generation {state['generation']}"

    if len(state["selected"]) == 0:
        status_el.innerText = "Select 2 patterns."
    elif len(state["selected"]) == 1:
        status_el.innerText = "Select 1 more pattern."
    else:
        a, b = state["selected"]
        status_el.innerText = f"Selected: Pattern {a + 1} and Pattern {b + 1}"

    next_btn.disabled = len(state["selected"]) != 2


def next_generation(event=None):
    if len(state["selected"]) != 2:
        return

    a_idx, b_idx = state["selected"]
    parent_a = state["population"][a_idx]
    parent_b = state["population"][b_idx]

    state["population"] = make_next_generation(parent_a, parent_b)
    state["generation"] += 1
    state["selected"] = []

    render_population()
    update_status()


def reset_population(event=None):
    state["population"] = make_initial_population()
    state["generation"] = 1
    state["selected"] = []

    render_population()
    update_status()


def setup():
    next_proxy = ffi.create_proxy(next_generation)
    reset_proxy = ffi.create_proxy(reset_population)
    persistent_proxies.append(next_proxy)
    persistent_proxies.append(reset_proxy)

    document.getElementById("next-btn").addEventListener("click", next_proxy)
    document.getElementById("reset-btn").addEventListener("click", reset_proxy)

    reset_population()


setup()
