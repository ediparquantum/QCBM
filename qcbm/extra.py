"""Rigetti-specific theming, etc."""
from typing import Optional

import matplotlib as mpl
import numpy as np
import seaborn as sns

TEAL = "#00b5ad"
DARK_BLUE = "#0d0d36"
WHITE = "#ffffff"
MAGENTA = "#ef476f"
YELLOW = "#ffc504"
BLUE = "#3D47D9"
GRAY = "#8A8B92"

DIVERGING_PALETTE = [
    "#0d0d36",
    "#3d3d5e",
    "#6e6e86",
    "#9e9eaf",
    "#cfcfd7",
    "#ffffff",
    "#ccf0ef",
    "#99e1de",
    "#66d3ce",
    "#33c4bd",
    "#00b5ad",
]


def set_theme(context: str = "talk"):
    """Set the plotting theme."""
    sns.set_theme(
        context=context,
        style="white",
        rc={
            "font.sans-serif": ["Open Sans", "Arial"],
            "axes.spines.top": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "axes.spines.right": False,
        },
    )


def bitstring_plot(data: np.ndarray, ax: Optional[mpl.axes.Axes] = None) -> mpl.figure.Figure:
    """Visualize bitstrings."""
    return sns.heatmap(
        data=data,
        yticklabels=False,
        xticklabels=[str(i) for i in reversed(range(data.shape[1]))],
        cbar=False,
        cmap=[DARK_BLUE, WHITE],
        ax=ax,
    )
