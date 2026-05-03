import math
import random
from pyscript import document, ffi, window

DEBUG = False


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

            base = max(0.0, min(1.0, (value + 1.0) / 2.0))

            r = 255 * (
                params["color_r_dark"] * (1 - base) + params["color_r_light"] * base
            )
            g = 255 * (
                params["color_g_dark"] * (1 - base) + params["color_g_light"] * base
            )
            b = 255 * (
                params["color_b_dark"] * (1 - base) + params["color_b_light"] * base
            )

            accent = max(
                0.0,
                min(
                    1.0,
                    0.5
                    + 0.5
                    * math.sin(
                        params["accent_freq"] * (xw + yw) + params["accent_phase"]
                    ),
                ),
            )

            accent_strength = params["accent_strength"] * accent * abs(pattern)

            r = (
                r * (1 - accent_strength)
                + 255 * params["color_r_accent"] * accent_strength
            )
            g = (
                g * (1 - accent_strength)
                + 255 * params["color_g_accent"] * accent_strength
            )
            b = (
                b * (1 - accent_strength)
                + 255 * params["color_b_accent"] * accent_strength
            )

            row.append((r, g, b))

        image.append(row)

    return image


def mutate_genotype(genotype, strength=0.08):
    child = dict(genotype)

    for key, value in child.items():
        if key.startswith("color_"):
            child[key] = max(0.0, min(1.0, value + random.gauss(0, 0.06)))
        elif key == "accent_strength":
            child[key] = max(0.0, min(0.75, value + random.gauss(0, 0.05)))
        elif key == "accent_freq":
            child[key] = max(1.0, min(24.0, value + random.gauss(0, 0.7)))
        elif key == "accent_phase":
            child[key] = (value + random.gauss(0, strength * math.pi)) % (2 * math.pi)
        elif "phase" in key:
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


def random_genotype():
    return {
        "freq1_x": random.uniform(2.0, 18.0),
        "freq2_y": random.uniform(2.0, 18.0),
        "radial_freq": random.uniform(2.0, 20.0),
        "angular_freq": random.uniform(1.0, 14.0),
        "phase1": random.uniform(0, 2 * math.pi),
        "phase2": random.uniform(0, 2 * math.pi),
        "phase3": random.uniform(0, 2 * math.pi),
        "phase4": random.uniform(0, 2 * math.pi),
        "amp1": random.uniform(-1.2, 1.2),
        "amp2": random.uniform(-1.2, 1.2),
        "amp3": random.uniform(-1.2, 1.2),
        "amp4": random.uniform(-1.2, 1.2),
        "warp_amp": random.uniform(0.0, 0.35),
        "warp_freq_x": random.uniform(1.0, 12.0),
        "warp_freq_y": random.uniform(1.0, 12.0),
        "warp_phase_x": random.uniform(0, 2 * math.pi),
        "warp_phase_y": random.uniform(0, 2 * math.pi),
        "noise_amp": random.uniform(0.0, 0.55),
        "noise_freq_x": random.uniform(2.0, 16.0),
        "noise_freq_y": random.uniform(2.0, 16.0),
        "noise_phase_x": random.uniform(0, 2 * math.pi),
        "noise_phase_y": random.uniform(0, 2 * math.pi),
        "radial_bias_amp": random.uniform(-0.45, 0.45),
        "radial_bias_freq": random.uniform(1.0, 12.0),
        "contrast": random.uniform(1.6, 4.8),
        "threshold": random.uniform(-0.25, 0.25),
        "color_r_dark": random.uniform(0.0, 0.4),
        "color_g_dark": random.uniform(0.0, 0.4),
        "color_b_dark": random.uniform(0.0, 0.4),
        "color_r_light": random.uniform(0.55, 1.0),
        "color_g_light": random.uniform(0.55, 1.0),
        "color_b_light": random.uniform(0.55, 1.0),
        "color_r_accent": random.uniform(0.0, 1.0),
        "color_g_accent": random.uniform(0.0, 1.0),
        "color_b_accent": random.uniform(0.0, 1.0),
        "accent_freq": random.uniform(2.0, 16.0),
        "accent_phase": random.uniform(0, 2 * math.pi),
        "accent_strength": random.uniform(0.2, 0.65),
    }


def randomize_palette(genotype, amount=0.2):
    child = dict(genotype)

    for key in [k for k in child if k.startswith("color_")]:
        child[key] = max(0.0, min(1.0, child[key] + random.gauss(0, amount)))

    child["accent_strength"] = max(
        0.0,
        min(0.75, child["accent_strength"] + random.gauss(0, amount * 0.5)),
    )
    child["accent_freq"] = max(
        1.0,
        min(24.0, child["accent_freq"] + random.gauss(0, amount * 4.0)),
    )
    child["accent_phase"] = random.uniform(0, 2 * math.pi)

    return child


def make_initial_population():
    pop = []

    for _ in range(6):
        child = random_genotype()
        child = mutate_genotype(child, strength=random.uniform(0.05, 0.14))

        if random.random() < 0.45:
            child = randomize_palette(child, amount=random.uniform(0.08, 0.20))

        pop.append(child)

    return pop


def make_next_generation(parent_a, parent_b):
    children = []

    for _ in range(6):
        alpha = random.triangular(0.05, 0.95, 0.5)
        mutation_strength = random.uniform(0.04, 0.14)

        child = blend_genotypes(parent_a, parent_b, alpha)
        debug(f"alpha: {alpha}")
        if random.random() < 0.25:
            child = randomize_palette(child, amount=random.uniform(0.06, 0.16))
            debug("randomize pallete")

        if random.random() < 0.25:
            child = mutate_genotype(child, strength=mutation_strength)
            debug("mutate genotype")

        if random.random() < 0.15:
            child = blend_genotypes(
                child,
                random_genotype(),
                random.uniform(0.05, 0.18),
            )
            debug("fully random")

        children.append(child)

    return children


GRID_SIZE = 140

state = {
    "population": [],
    "generation": 1,
    "selected": [],
    "history": [],
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
        for r, g, b in row:
            data[idx] = int(max(0, min(255, r)))
            data[idx + 1] = int(max(0, min(255, g)))
            data[idx + 2] = int(max(0, min(255, b)))
            data[idx + 3] = 255
            idx += 4

    ctx.putImageData(image_data, 0, 0)
    return canvas.toDataURL("image/png")


def save_history_snapshot():
    snapshot = {
        "generation": state["generation"],
        "images": [],
        "selected": [],
    }

    for genotype in state["population"]:
        img_url = image_to_data_url(generate_pattern(genotype, GRID_SIZE))
        snapshot["images"].append(img_url)

    state["history"].append(snapshot)
    render_history()


def mark_current_generation_selection(selected_indices):
    if len(state["history"]) > 0:
        state["history"][-1]["selected"] = list(selected_indices)
        render_history()


def render_history():
    history_container = document.getElementById("history")
    history_container.innerHTML = ""

    for snapshot in state["history"]:
        generation_block = document.createElement("div")
        generation_block.className = "history-generation"

        title = document.createElement("div")
        title.className = "history-title"
        title.innerText = f"Generation {snapshot['generation']}"

        grid = document.createElement("div")
        grid.className = "history-grid"

        for i, img_url in enumerate(snapshot["images"]):
            img = document.createElement("img")
            img.src = img_url
            img.className = "history-image"

            if i in snapshot["selected"]:
                img.classList.add("history-selected")

            grid.appendChild(img)

        generation_block.appendChild(title)
        generation_block.appendChild(grid)
        history_container.appendChild(generation_block)


def show_current_view(event=None):
    document.getElementById("current-view").hidden = False
    document.getElementById("history-view").hidden = True

    document.getElementById("current-tab").classList.add("active")
    document.getElementById("history-tab").classList.remove("active")


def show_history_view(event=None):
    document.getElementById("current-view").hidden = True
    document.getElementById("history-view").hidden = False

    document.getElementById("current-tab").classList.remove("active")
    document.getElementById("history-tab").classList.add("active")


def toggle_select(index):
    if index in state["selected"]:
        state["selected"].remove(index)
    elif len(state["selected"]) < 2:
        state["selected"].append(index)

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
        img_url = image_to_data_url(generate_pattern(genotype, GRID_SIZE))

        card = document.createElement("button")
        card.setAttribute("type", "button")
        card.className = "pattern-card"

        if i in state["selected"]:
            card.classList.add("selected")

        card.addEventListener("click", make_card_click_handler(i))

        label = document.createElement("div")
        label.className = "pattern-label"
        label.innerText = f"Pattern {i + 1}"

        img = document.createElement("img")
        img.src = img_url
        img.className = "pattern-image"
        img.alt = f"Pattern {i + 1}"

        helper = document.createElement("div")
        helper.className = "pattern-helper"
        helper.innerText = "Selected" if i in state["selected"] else "Click to select"

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

    selected_indices = list(state["selected"])
    a_idx, b_idx = selected_indices

    mark_current_generation_selection(selected_indices)

    parent_a = state["population"][a_idx]
    parent_b = state["population"][b_idx]

    state["population"] = make_next_generation(parent_a, parent_b)
    state["generation"] += 1
    state["selected"] = []

    render_population()
    update_status()
    save_history_snapshot()


def reset_population(event=None):
    state["population"] = make_initial_population()
    state["generation"] = 1
    state["selected"] = []
    state["history"] = []

    render_population()
    update_status()
    save_history_snapshot()
    show_current_view()


def setup():
    debug_log = document.getElementById("debug-log")

    if debug_log is not None:
        if DEBUG:
            debug_log.hidden = False
            debug_log.innerText = "Debug log:\n"
        else:
            debug_log.hidden = True
            debug_log.innerText = ""

    debug("setup started")

    next_proxy = ffi.create_proxy(next_generation)
    reset_proxy = ffi.create_proxy(reset_population)
    current_tab_proxy = ffi.create_proxy(show_current_view)
    history_tab_proxy = ffi.create_proxy(show_history_view)

    persistent_proxies.extend(
        [next_proxy, reset_proxy, current_tab_proxy, history_tab_proxy]
    )

    document.getElementById("next-btn").addEventListener("click", next_proxy)
    document.getElementById("reset-btn").addEventListener("click", reset_proxy)
    document.getElementById("current-tab").addEventListener("click", current_tab_proxy)
    document.getElementById("history-tab").addEventListener("click", history_tab_proxy)

    reset_population()


def debug(message):
    if not DEBUG:
        return

    log = document.getElementById("debug-log")
    if log is not None:
        log.innerText += str(message) + "\n"


setup()
debug("main.py loaded")
