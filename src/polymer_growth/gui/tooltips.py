"""Parameter descriptions and tooltip content for the GUI.

Chemistry-accurate explanations of simulation parameters with
HTML-formatted tooltips including formulas.
"""

# PEtOx chemistry constants (from simulation.py)
MONOMER_NAME = "2-ethyl-2-oxazoline (EtOx)"
MONOMER_MASS_STR = "99.13 g/mol"
INITIATOR_NAME = "methyl tosylate"
INITIATOR_MASS_STR = "180.0 g/mol"


PARAM_INFO = {
    "time_sim": {
        "title": "Simulation Time",
        "tooltip": (
            "<b>Simulation Time (timesteps)</b><br><br>"
            "Number of discrete timesteps in the simulation. Each timestep, "
            "every living chain has a chance to grow, terminate, or be attacked "
            "by a dead chain.<br><br>"
            "<b>Effect:</b> Longer simulations allow higher conversions and "
            "longer chains. Too few timesteps may not reach steady-state."
            "<br><b>Typical range:</b> 100 - 10,000"
        ),
    },
    "number_of_molecules": {
        "title": "Number of Molecules",
        "tooltip": (
            "<b>Initial Number of Polymer Chains (Initiator Count)</b><br><br>"
            "The number of living chains at t=0. All start with degree of "
            "polymerization DP=1 (one monomer unit attached to the initiator)."
            "<br><br>"
            "<b>Chemistry:</b> Corresponds to the [Initiator] concentration "
            f"({INITIATOR_NAME}). The target degree of polymerization is "
            "approximately:<br>"
            "<b>DP_target = monomer_pool / number_of_molecules</b>"
            "<br><b>Effect:</b> More chains = better statistics but slower. "
            "Increasing this (with fixed monomer pool) lowers average chain length."
            "<br><b>Typical range:</b> 1,000 - 100,000"
        ),
    },
    "monomer_pool": {
        "title": "Monomer Pool",
        "tooltip": (
            "<b>Initial Monomer Supply</b><br><br>"
            f"Total number of {MONOMER_NAME} monomer units available. "
            "As chains grow, they consume monomers from this pool, reducing "
            "the effective growth and death probabilities.<br><br>"
            "<b>Formula:</b> monomer_ratio = current_pool / initial_pool"
            "<br><b>Special:</b> Set to -1 for infinite monomer (ratio always 1.0)."
            "<br><b>Chemistry:</b> Controls the [M]/[M]<sub>0</sub> ratio, "
            "analogous to monomer concentration depletion in batch polymerization."
            "<br><b>Typical range:</b> 10,000 - 100,000,000"
        ),
    },
    "p_growth": {
        "title": "P(Growth)",
        "tooltip": (
            "<b>Base Growth Probability</b><br><br>"
            "Probability that a living chain adds one monomer unit per timestep, "
            "scaled by monomer availability.<br><br>"
            "<b>Formula:</b> P(growth) = p_growth x monomer_ratio"
            "<br><b>Chemistry:</b> Represents the propagation rate constant "
            "k<sub>p</sub> in a normalized form. Higher values = faster chain growth."
            "<br><b>Effect:</b> Directly controls the average chain length and "
            "molecular weight."
            "<br><b>Typical range:</b> 0.10 - 0.99"
        ),
    },
    "p_death": {
        "title": "P(Death)",
        "tooltip": (
            "<b>Base Termination Probability</b><br><br>"
            "Probability that a living chain terminates (becomes dead) per timestep."
            "<br><br>"
            "<b>Formula:</b> P(death) = p_death x monomer_ratio"
            "<br><b>Chemistry:</b> Represents the termination rate constant "
            "k<sub>t</sub>. In controlled radical polymerization (CRP), this is "
            "intentionally kept very low to maintain living character."
            "<br><b>Effect:</b> Higher = more dead chains, broader distribution (higher PDI)."
            "<br><b>Typical range:</b> 0.00001 - 0.01"
        ),
    },
    "p_dead_react": {
        "title": "P(Dead React)",
        "tooltip": (
            "<b>Base Coupling Reactivity</b><br><br>"
            "Base probability for a dead chain to successfully couple with a "
            "living chain (vampiric coupling). Actual probability depends on "
            "chain lengths.<br><br>"
            "<b>Formula:</b> P(success) = p_dead_react / "
            "(L<sub>living</sub><sup>f<sub>1</sub></sup> x "
            "L<sub>dead</sub><sup>f<sub>2</sub></sup>)"
            "<br><b>Chemistry:</b> Models termination by combination, where "
            "a terminated chain reacts with a living chain to form a coupled product."
            "<br><b>Effect:</b> Higher = more coupling events, creates high-MW tail "
            "in distribution."
            "<br><b>Typical range:</b> 0.10 - 0.99"
        ),
    },
    "l_exponent": {
        "title": "Living Exponent",
        "tooltip": (
            "<b>Living Chain Length Exponent</b><br><br>"
            "Controls how much a living chain's length reduces its coupling "
            "probability. Longer living chains are harder for dead chains to attack."
            "<br><br>"
            "<b>Formula:</b> f<sub>living</sub> = min(L x l_exp / l_naked, l_exp)"
            "<br><b>Chemistry:</b> Models diffusion-controlled termination: "
            "longer chains diffuse more slowly and have less accessible reactive "
            "end-groups (chain-end shielding)."
            "<br><b>Effect:</b> Higher = stronger length penalty on coupling."
            "<br><b>Typical range:</b> 0.10 - 0.99"
        ),
    },
    "d_exponent": {
        "title": "Death Exponent",
        "tooltip": (
            "<b>Dead Chain Length Exponent</b><br><br>"
            "Controls how much a dead chain's length reduces its coupling "
            "probability. Longer dead chains are less reactive.<br><br>"
            "<b>Formula:</b> f<sub>dead</sub> = min(L x d_exp / l_naked, d_exp)"
            "<br><b>Chemistry:</b> Dead chains also experience diffusion limitations. "
            "Their mobility decreases with length, reducing encounter frequency "
            "with living chain ends."
            "<br><b>Effect:</b> Higher = dead chains couple less easily."
            "<br><b>Typical range:</b> 0.10 - 0.99"
        ),
    },
    "l_naked": {
        "title": "Accessible Surface Ratio",
        "tooltip": (
            "<b>Accessible Surface Ratio (l_naked)</b><br><br>"
            "Controls the saturation point of the length exponents. Determines "
            "at what chain length the exponent function reaches its maximum.<br><br>"
            "<b>Formula:</b> Exponent saturates when L x exp / l_naked >= exp, "
            "i.e., at chain length L = l_naked."
            "<br><b>Chemistry:</b> Represents the fraction of chain surface that "
            "remains accessible for reaction. Lower values mean the exponent "
            "saturates at shorter chains (smaller chains already behave like long ones)."
            "<br><b>Effect:</b> Lower = coupling probability drops off faster with length."
            "<br><b>Typical range:</b> 0.10 - 0.99"
        ),
    },
    "kill_spawns_new": {
        "title": "Kill Spawns New",
        "tooltip": (
            "<b>Chain Re-initiation on Termination</b><br><br>"
            "When a living chain terminates (dies), this controls what happens next."
            "<br><br>"
            "<b>Enabled (True):</b> The terminated chain becomes dead, and a new "
            "living chain of length 1 is spawned. This keeps the total number of "
            "active species constant."
            "<br><b>Disabled (False):</b> The terminated chain is simply removed "
            "from the living pool. Total chain count decreases over time."
            "<br><br>"
            "<b>Chemistry:</b> Re-initiation models systems where initiator "
            "is in excess or where new chains continuously start (e.g., slow "
            "initiator decomposition)."
        ),
    },
    "seed": {
        "title": "Random Seed",
        "tooltip": (
            "<b>Random Number Generator Seed</b><br><br>"
            "Sets the seed for the pseudo-random number generator. Using the "
            "same seed with the same parameters produces identical results."
            "<br><br>"
            "<b>Purpose:</b> Reproducibility. Different seeds give different "
            "stochastic realizations of the same underlying model."
        ),
    },
}


METRIC_INFO = {
    "mn": {
        "title": "Number-Average Molecular Weight (Mn)",
        "tooltip": (
            "<b>M<sub>n</sub> - Number-Average Molecular Weight</b><br><br>"
            "The average molecular weight where each chain counts equally, "
            "regardless of its size.<br><br>"
            "<b>Formula:</b> M<sub>n</sub> = sum(N<sub>i</sub> x M<sub>i</sub>) "
            "/ sum(N<sub>i</sub>)<br>"
            "= mean(DP) x 99.13 + 180.0 g/mol"
            "<br><br>"
            "<b>Measured by:</b> Osmometry, end-group analysis, GPC/SEC"
        ),
    },
    "mw": {
        "title": "Weight-Average Molecular Weight (Mw)",
        "tooltip": (
            "<b>M<sub>w</sub> - Weight-Average Molecular Weight</b><br><br>"
            "The average molecular weight weighted by mass. Heavier chains "
            "contribute more than lighter ones.<br><br>"
            "<b>Formula:</b> M<sub>w</sub> = sum(N<sub>i</sub> x M<sub>i</sub>"
            "<sup>2</sup>) / sum(N<sub>i</sub> x M<sub>i</sub>)"
            "<br><br>"
            "<b>Measured by:</b> Light scattering, GPC/SEC"
            "<br><b>Always:</b> M<sub>w</sub> >= M<sub>n</sub>"
        ),
    },
    "pdi": {
        "title": "Polydispersity Index (PDI)",
        "tooltip": (
            "<b>PDI (D) - Polydispersity Index</b><br><br>"
            "Measures the breadth of the molecular weight distribution."
            "<br><br>"
            "<b>Formula:</b> PDI = M<sub>w</sub> / M<sub>n</sub>"
            "<br><br>"
            "<b>Interpretation:</b><br>"
            "- PDI = 1.0: perfectly uniform (monodisperse)<br>"
            "- PDI = 1.0 - 1.2: well-controlled polymerization (CRP)<br>"
            "- PDI = 1.5 - 2.0: conventional free radical polymerization<br>"
            "- PDI > 2.0: poorly controlled or branched system"
        ),
    },
}