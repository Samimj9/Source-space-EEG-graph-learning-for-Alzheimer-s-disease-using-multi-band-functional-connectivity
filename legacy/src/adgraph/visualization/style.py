"""Central figure style: journal sizes, fonts, line widths, colour-blind-safe palette.

Colours are the Okabe-Ito palette (Okabe & Ito 2008), safe for the common colour-vision
deficiencies. Every categorical encoding also differs in marker shape and fill so that
figures stay readable in grayscale. Sequential maps use ``cividis`` (perceptually
uniform, CVD-safe, monotonic in lightness); diverging maps use ``RdBu_r`` with
limits symmetric around zero (the sign of a diverging map is *not* recoverable in
grayscale; difference maps therefore also report values in their data files).
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import yaml

MM_PER_INCH = 25.4

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F",
}

DEFAULT_STYLE = {
    "widths_mm": {"single": 85.0, "one_half": 120.0, "double": 178.0},   # journal column widths
    "max_height_mm": 230.0,
    "font_family": "sans-serif",
    "font_size": 7.0, "title_size": 8.0, "label_size": 7.0, "tick_size": 6.0, "legend_size": 6.0,
    "line_width": 0.8, "axes_line_width": 0.6, "marker_size": 3.5, "point_size": 2.0, "point_alpha": 0.55,
    "errorbar_capsize": 1.5, "jitter": 0.12,
    "dpi": 600, "formats": ["pdf", "svg", "png"],
    "sequential_cmap": "cividis", "diverging_cmap": "RdBu_r",
}

# Categorical encodings (colour + marker + fill), grayscale-distinguishable.
CONDITION_STYLE = {
    "plv": {"color": OKABE_ITO["blue"], "marker": "o", "fill": True, "linestyle": "-", "label": "PLV"},
    "ciplv": {"color": OKABE_ITO["vermillion"], "marker": "s", "fill": False, "linestyle": "--", "label": "|ciPLV|"},
    "none": {"color": OKABE_ITO["black"], "marker": "D", "fill": True, "linestyle": ":", "label": "no connectivity"},
}
GROUP_STYLE = {
    "AD": {"color": OKABE_ITO["vermillion"], "marker": "^", "fill": True, "label": "AD"},
    "HC": {"color": OKABE_ITO["blue"], "marker": "o", "fill": False, "label": "HC"},
}
DATASET_MARKER = {"D1": "o", "D2": "s"}
BAND_STYLE = {
    "theta": {"color": OKABE_ITO["orange"], "hatch": "", "marker": "o"},
    "alpha1": {"color": OKABE_ITO["sky_blue"], "hatch": "//", "marker": "s"},
    "alpha2": {"color": OKABE_ITO["green"], "hatch": "..", "marker": "^"},
    "beta": {"color": OKABE_ITO["purple"], "hatch": "xx", "marker": "D"},
}
SETTING_LABEL = {"within_D1": "Within D1", "within_D2": "Within D2", "pooled": "Pooled D1+D2",
                 "cross_D1_to_D2": "D1 → D2", "cross_D2_to_D1": "D2 → D1", "within_D3": "Within D3 (ds004504)"}
METRIC_LABEL = {"balanced_accuracy": "Balanced accuracy", "roc_auc": "ROC-AUC", "f1": "F1 (AD)",
                "sensitivity": "Sensitivity", "specificity": "Specificity", "accuracy": "Accuracy", "mcc": "MCC"}


def load_style(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    style = copy.deepcopy(DEFAULT_STYLE)
    if path:
        style.update(yaml.safe_load(Path(path).read_text()) or {})
    style.update(overrides or {})
    return style


def rc_params(style: dict) -> dict:
    return {
        "font.family": style["font_family"], "font.size": style["font_size"],
        "axes.titlesize": style["title_size"], "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"], "ytick.labelsize": style["tick_size"],
        "legend.fontsize": style["legend_size"], "legend.frameon": False,
        "axes.linewidth": style["axes_line_width"], "lines.linewidth": style["line_width"],
        "lines.markersize": style["marker_size"], "xtick.major.width": style["axes_line_width"],
        "ytick.major.width": style["axes_line_width"], "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # editable text in vector output
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100,
    }


@contextmanager
def styled(style: dict):
    with mpl.rc_context(rc_params(style)):
        yield


def figure_size(style: dict, width: str = "single", height_ratio: float = 0.75,
                height_mm: float | None = None) -> tuple[float, float]:
    """(width, height) in inches for a journal column width key."""
    w_mm = style["widths_mm"][width]
    h_mm = min(height_mm if height_mm is not None else w_mm * height_ratio, style["max_height_mm"])
    return w_mm / MM_PER_INCH, h_mm / MM_PER_INCH


def marker_kwargs(spec: dict, size: float | None = None) -> dict:
    """Marker style with open/filled markers for grayscale separation."""
    return {"marker": spec["marker"], "markersize": size or mpl.rcParams["lines.markersize"],
            "markeredgecolor": spec["color"], "markerfacecolor": spec["color"] if spec.get("fill", True) else "white",
            "markeredgewidth": 0.8, "color": spec["color"]}


def condition_style(condition: str) -> dict:
    return CONDITION_STYLE.get(condition, {"color": OKABE_ITO["grey"], "marker": "o", "fill": True,
                                           "linestyle": "-", "label": condition})
