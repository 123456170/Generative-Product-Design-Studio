import io
import os
import json
import time
import math
import random
import struct
import hashlib
import zipfile
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import imageio


st.set_page_config(
    page_title="Generative Product Design Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------------------
# Constants and design knowledge base
# --------------------------------------------------------------------------------------

CATEGORIES = [
    "Consumer electronics housings",
    "Furniture components",
    "Household product enclosures",
    "Lighting fixtures",
    "Desk accessories",
]

STYLES = [
    "Minimalist",
    "Organic",
    "Technical",
    "Playful",
    "Luxury",
    "Sustainable",
]

FUNCTIONAL_LIBRARY = {
    "Consumer electronics housings": [
        "Acoustic transparency",
        "Thermal dissipation",
        "Button access",
        "Cable routing",
        "Stability",
        "Easy cleaning",
    ],
    "Furniture components": [
        "Stability",
        "Comfortable grip",
        "Modular attachment",
        "Easy cleaning",
        "Tool-free assembly",
    ],
    "Household product enclosures": [
        "Cable routing",
        "Easy cleaning",
        "Modular attachment",
        "Stability",
        "Tool-free assembly",
    ],
    "Lighting fixtures": [
        "Diffused light",
        "Thermal dissipation",
        "Cable routing",
        "Stability",
        "Easy cleaning",
    ],
    "Desk accessories": [
        "Cable routing",
        "Stability",
        "Modular attachment",
        "Easy cleaning",
        "Button access",
    ],
}

DEFAULT_WEIGHTS = {
    "requirement": 0.25,
    "constraint": 0.20,
    "manufacturability": 0.15,
    "sustainability": 0.10,
    "aesthetics": 0.15,
    "diversity": 0.15,
}

DEFAULT_PARAMS = {
    "width": 100.0,
    "height": 100.0,
    "depth": 100.0,
    "wall_thickness": 2.0,
    "roundness": 4.0,
    "feature_intensity": 0.25,
    "taper": 0.0,
    "asymmetry": 0.0,
    "vent_rows": 0,
    "vent_cols": 0,
    "rib_count": 0,
    "wave_freq": 4.0,
    "wave_amp": 0.0,
    "basin_depth": 0.0,
    "leg_count": 0,
    "subdivision_level": 3,
    "seed": 0,
}

PARAM_BOUNDS = {
    "width": (20.0, 500.0),
    "height": (20.0, 500.0),
    "depth": (20.0, 500.0),
    "wall_thickness": (0.8, 15.0),
    "roundness": (2.0, 12.0),
    "feature_intensity": (0.0, 1.0),
    "taper": (0.0, 0.8),
    "asymmetry": (-0.8, 0.8),
    "vent_rows": (0, 12),
    "vent_cols": (0, 20),
    "rib_count": (0, 32),
    "wave_freq": (1.0, 12.0),
    "wave_amp": (0.0, 1.0),
    "basin_depth": (0.0, 1.0),
    "leg_count": (0, 8),
    "subdivision_level": (2, 4),
}

INT_KEYS = {
    "vent_rows",
    "vent_cols",
    "rib_count",
    "leg_count",
    "subdivision_level",
}

EDITOR_KEYS = [
    "width",
    "height",
    "depth",
    "wall_thickness",
    "roundness",
    "feature_intensity",
    "taper",
    "asymmetry",
    "vent_rows",
    "vent_cols",
    "rib_count",
    "wave_freq",
    "wave_amp",
    "basin_depth",
    "leg_count",
    "subdivision_level",
]

BASE_RULES = {
    "rounded_shell": {
        "roundness": 7.0,
        "feature_intensity": 0.08,
        "vent_rows": 0,
        "vent_cols": 0,
        "rib_count": 0,
        "wave_amp": 0.05,
        "basin_depth": 0.0,
        "leg_count": 0,
        "taper": 0.05,
        "asymmetry": 0.0,
    },
    "ventilated_block": {
        "roundness": 4.0,
        "feature_intensity": 0.60,
        "vent_rows": 4,
        "vent_cols": 8,
        "rib_count": 0,
        "wave_amp": 0.0,
        "basin_depth": 0.05,
        "leg_count": 0,
        "taper": 0.08,
        "asymmetry": 0.0,
    },
    "organic_capsule": {
        "roundness": 2.3,
        "feature_intensity": 0.18,
        "vent_rows": 0,
        "vent_cols": 0,
        "rib_count": 0,
        "wave_amp": 0.35,
        "basin_depth": 0.08,
        "leg_count": 0,
        "taper": 0.30,
        "asymmetry": 0.10,
    },
    "ribbed_modular": {
        "roundness": 4.8,
        "feature_intensity": 0.42,
        "vent_rows": 2,
        "vent_cols": 4,
        "rib_count": 14,
        "wave_amp": 0.08,
        "basin_depth": 0.0,
        "leg_count": 0,
        "taper": 0.10,
        "asymmetry": 0.0,
    },
    "sculpted_diffuser": {
        "roundness": 3.0,
        "feature_intensity": 0.25,
        "vent_rows": 0,
        "vent_cols": 0,
        "rib_count": 0,
        "wave_freq": 6.0,
        "wave_amp": 0.55,
        "basin_depth": 0.35,
        "leg_count": 0,
        "taper": 0.18,
        "asymmetry": 0.05,
    },
    "compact_tray": {
        "roundness": 5.5,
        "feature_intensity": 0.20,
        "vent_rows": 0,
        "vent_cols": 0,
        "rib_count": 0,
        "wave_amp": 0.08,
        "basin_depth": 0.65,
        "leg_count": 0,
        "taper": 0.12,
        "asymmetry": 0.0,
    },
    "legged_component": {
        "roundness": 3.6,
        "feature_intensity": 0.28,
        "vent_rows": 0,
        "vent_cols": 0,
        "rib_count": 4,
        "wave_amp": 0.12,
        "basin_depth": 0.05,
        "leg_count": 4,
        "taper": 0.20,
        "asymmetry": 0.05,
    },
}

CATEGORY_RULES = {
    "Consumer electronics housings": [
        "rounded_shell",
        "ventilated_block",
        "organic_capsule",
        "ribbed_modular",
        "sculpted_diffuser",
    ],
    "Furniture components": [
        "organic_capsule",
        "legged_component",
        "rounded_shell",
        "ribbed_modular",
        "compact_tray",
    ],
    "Household product enclosures": [
        "rounded_shell",
        "compact_tray",
        "ventilated_block",
        "ribbed_modular",
        "organic_capsule",
    ],
    "Lighting fixtures": [
        "sculpted_diffuser",
        "organic_capsule",
        "rounded_shell",
        "compact_tray",
        "ribbed_modular",
    ],
    "Desk accessories": [
        "compact_tray",
        "rounded_shell",
        "ribbed_modular",
        "organic_capsule",
        "legged_component",
    ],
}

STYLE_TARGETS = {
    "Minimalist": {
        "feature_intensity": 0.12,
        "roundness": 7.5,
        "wave_amp": 0.05,
        "asymmetry": 0.0,
    },
    "Organic": {
        "roundness": 2.4,
        "taper": 0.30,
        "wave_amp": 0.35,
        "feature_intensity": 0.20,
    },
    "Technical": {
        "roundness": 4.2,
        "vent_rows": 5,
        "rib_count": 10,
        "feature_intensity": 0.55,
    },
    "Playful": {
        "asymmetry": 0.35,
        "wave_amp": 0.50,
        "leg_count": 4,
        "feature_intensity": 0.45,
    },
    "Luxury": {
        "roundness": 8.5,
        "feature_intensity": 0.18,
        "basin_depth": 0.25,
        "wave_amp": 0.15,
    },
    "Sustainable": {
        "feature_intensity": 0.20,
        "roundness": 5.5,
        "taper": 0.10,
    },
}

FUNCTION_TO_PARAM = {
    "Acoustic transparency": ("param", "vent_rows", 2),
    "Thermal dissipation": ("param", "vent_rows", 3),
    "Button access": ("param", "feature_intensity", 0.20),
    "Cable routing": ("param", "basin_depth", 0.15),
    "Stability": ("metric", "stability_ratio", 0.55),
    "Diffused light": ("param", "wave_amp", 0.25),
    "Modular attachment": ("param", "rib_count", 4),
    "Easy cleaning": ("param", "roundness", 5.0),
    "Tool-free assembly": ("param", "feature_intensity", 0.30),
    "Comfortable grip": ("param", "taper", 0.15),
}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    h = hashlib.sha256(
        json.dumps(jsonable(payload), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{prefix}-{h}"


def minmax(series: pd.Series) -> pd.Series:
    mn = float(series.min())
    mx = float(series.max())
    if mx - mn < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 1e-12:
        return {k: 1.0 / len(clean) for k in clean}
    return {k: v / total for k, v in clean.items()}


def pc_range(values: List[float]) -> List[float]:
    vals = [float(v) for v in values]
    mn = min(vals)
    mx = max(vals)
    if mx - mn < 1e-9:
        return [mn - 1.0, mx + 1.0]
    return [mn, mx]


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------

@dataclass
class DesignSpec:
    category: str
    product_name: str
    width: float
    height: float
    depth: float
    wall_thickness: float
    fill_fraction_min: float
    fill_fraction_max: float
    style: str
    complexity: float
    functional_requirements: List[str]
    max_aspect_ratio: float = 8.0
    stability_required: bool = False
    notes: str = ""
    target_volume_min: float = 0.0
    target_volume_max: float = 0.0


@dataclass
class Candidate:
    id: str
    name: str
    category: str
    grammar_rule: str
    params: Dict[str, Any]
    vertices: np.ndarray
    faces: np.ndarray
    metrics: Dict[str, Any]
    validation: Dict[str, Any]
    score: float = 0.0
    rank: int = 99
    parent_id: Optional[str] = None
    generation: int = 0
    created_at: float = field(default_factory=time.time)


def finalize_spec(spec: DesignSpec) -> DesignSpec:
    spec.category = str(spec.category)
    spec.product_name = str(spec.product_name)
    spec.width = float(np.clip(spec.width, 20.0, 500.0))
    spec.height = float(np.clip(spec.height, 20.0, 500.0))
    spec.depth = float(np.clip(spec.depth, 20.0, 500.0))
    spec.wall_thickness = float(np.clip(spec.wall_thickness, 0.8, 15.0))
    spec.fill_fraction_min = float(np.clip(spec.fill_fraction_min, 0.05, 0.93))
    spec.fill_fraction_max = float(np.clip(spec.fill_fraction_max, 0.07, 0.95))
    if spec.fill_fraction_max < spec.fill_fraction_min + 0.02:
        spec.fill_fraction_max = min(0.95, spec.fill_fraction_min + 0.02)
    spec.style = str(spec.style)
    spec.complexity = float(np.clip(spec.complexity, 0.0, 1.0))
    spec.functional_requirements = [str(x) for x in spec.functional_requirements]
    spec.max_aspect_ratio = float(np.clip(spec.max_aspect_ratio, 2.0, 12.0))

    bbox_volume = spec.width * spec.height * spec.depth
    spec.target_volume_min = bbox_volume * spec.fill_fraction_min
    spec.target_volume_max = bbox_volume * spec.fill_fraction_max

    spec.stability_required = bool(
        spec.stability_required
        or spec.category in {
            "Consumer electronics housings",
            "Furniture components",
            "Desk accessories",
        }
    )
    return spec


DEFAULT_SPEC = finalize_spec(
    DesignSpec(
        category="Consumer electronics housings",
        product_name="Compact Smart Speaker Housing",
        width=92.0,
        height=118.0,
        depth=92.0,
        wall_thickness=2.0,
        fill_fraction_min=0.35,
        fill_fraction_max=0.70,
        style="Organic",
        complexity=0.55,
        functional_requirements=[
            "Acoustic transparency",
            "Stability",
            "Button access",
            "Cable routing",
        ],
        max_aspect_ratio=6.0,
        notes="M1 demonstration specification.",
    )
)

BENCHMARK_SPECS = [
    {
        "category": "Consumer electronics housings",
        "product_name": "Benchmark smart speaker shell",
        "width": 92.0,
        "height": 118.0,
        "depth": 92.0,
        "wall_thickness": 2.0,
        "fill_fraction_min": 0.35,
        "fill_fraction_max": 0.70,
        "style": "Organic",
        "complexity": 0.55,
        "functional_requirements": [
            "Acoustic transparency",
            "Stability",
            "Button access",
            "Cable routing",
        ],
        "max_aspect_ratio": 6.0,
    },
    {
        "category": "Furniture components",
        "product_name": "Benchmark shelf bracket",
        "width": 280.0,
        "height": 220.0,
        "depth": 170.0,
        "wall_thickness": 3.0,
        "fill_fraction_min": 0.16,
        "fill_fraction_max": 0.42,
        "style": "Minimalist",
        "complexity": 0.35,
        "functional_requirements": [
            "Stability",
            "Tool-free assembly",
            "Modular attachment",
        ],
        "max_aspect_ratio": 5.0,
    },
    {
        "category": "Household product enclosures",
        "product_name": "Benchmark storage enclosure",
        "width": 180.0,
        "height": 120.0,
        "depth": 130.0,
        "wall_thickness": 2.2,
        "fill_fraction_min": 0.25,
        "fill_fraction_max": 0.60,
        "style": "Sustainable",
        "complexity": 0.40,
        "functional_requirements": [
            "Easy cleaning",
            "Modular attachment",
            "Stability",
        ],
        "max_aspect_ratio": 6.0,
    },
    {
        "category": "Lighting fixtures",
        "product_name": "Benchmark pendant diffuser",
        "width": 140.0,
        "height": 180.0,
        "depth": 140.0,
        "wall_thickness": 1.8,
        "fill_fraction_min": 0.12,
        "fill_fraction_max": 0.38,
        "style": "Luxury",
        "complexity": 0.60,
        "functional_requirements": [
            "Diffused light",
            "Thermal dissipation",
            "Cable routing",
        ],
        "max_aspect_ratio": 5.0,
    },
    {
        "category": "Desk accessories",
        "product_name": "Benchmark cable organizer",
        "width": 160.0,
        "height": 48.0,
        "depth": 90.0,
        "wall_thickness": 2.0,
        "fill_fraction_min": 0.20,
        "fill_fraction_max": 0.55,
        "style": "Technical",
        "complexity": 0.50,
        "functional_requirements": [
            "Cable routing",
            "Stability",
            "Easy cleaning",
        ],
        "max_aspect_ratio": 7.0,
    },
]


def benchmark_to_spec(d: Dict[str, Any]) -> DesignSpec:
    return finalize_spec(DesignSpec(**d))


# --------------------------------------------------------------------------------------
# Procedural geometry: editable closed triangle meshes from parametric shape grammars
# --------------------------------------------------------------------------------------

def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroids = tri.mean(axis=1)
    dots = np.einsum("ij,ij->i", normals, centroids)
    flip = dots < 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


@st.cache_data(show_spinner=False)
def icosphere(level: int) -> Tuple[np.ndarray, np.ndarray]:
    level = int(np.clip(level, 1, 4))

    t = (1.0 + math.sqrt(5.0)) / 2.0
    base_verts = np.array(
        [
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ],
        dtype=float,
    )
    base_verts /= np.linalg.norm(base_verts, axis=1, keepdims=True)

    base_faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )

    verts_list = [v for v in base_verts]
    faces_list = [tuple(int(x) for x in f) for f in base_faces]

    for _ in range(level):
        edge_map: Dict[Tuple[int, int], int] = {}
        new_faces: List[Tuple[int, int, int]] = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in edge_map:
                mid = (verts_list[i] + verts_list[j]) * 0.5
                norm = np.linalg.norm(mid)
                if norm < 1e-12:
                    mid = verts_list[i]
                else:
                    mid = mid / norm
                verts_list.append(mid)
                edge_map[key] = len(verts_list) - 1
            return edge_map[key]

        for a, b, c in faces_list:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend(
                [
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ]
            )
        faces_list = new_faces

    verts = np.array(verts_list, dtype=float)
    faces = np.array(faces_list, dtype=np.int32)
    faces = orient_faces_outward(verts, faces)
    return verts, faces


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.power(np.abs(values), exponent)


def angular_distances(directions: np.ndarray, center: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    n = float(np.linalg.norm(center))
    if n < 1e-12:
        return np.zeros(len(directions), dtype=float)
    center = center / n
    dots = np.clip(np.sum(directions * center, axis=1), -1.0, 1.0)
    return np.arccos(dots)


def apply_feature_modifiers(
    unit_dirs: np.ndarray,
    points: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    pts = points.copy()
    amp = float(params.get("feature_intensity", 0.0))
    min_dim = min(float(params["width"]), float(params["height"]), float(params["depth"]))

    # Ventilation / acoustic perforation pattern on the +Y face.
    rows = int(params.get("vent_rows", 0))
    cols = int(params.get("vent_cols", 0))
    if rows > 0 and cols > 0 and amp > 0.0:
        vent_depth = 0.08 * min_dim * amp
        radius = 0.16
        for r in range(rows):
            zoff = 0.0 if rows == 1 else (r / (rows - 1) - 0.5) * 0.70
            for c in range(cols):
                xoff = 0.0 if cols == 1 else (c / (cols - 1) - 0.5) * 0.90
                center = np.array([xoff, 1.0, zoff], dtype=float)
                d = angular_distances(unit_dirs, center)
                mask = d < radius
                if np.any(mask):
                    w = (1.0 - (d[mask] / radius) ** 2) ** 2
                    w = np.clip(w, 0.0, 1.0)
                    pts[mask] -= unit_dirs[mask] * (vent_depth * w[:, None])

    # Structural ribs around the vertical axis.
    rib_count = int(params.get("rib_count", 0))
    if rib_count > 0 and amp > 0.0:
        theta = np.arctan2(unit_dirs[:, 1], unit_dirs[:, 0])
        band = np.exp(-((unit_dirs[:, 2] / 0.65) ** 2))
        disp = 0.05 * min_dim * amp * np.sin(rib_count * theta) * band
        pts += unit_dirs * disp[:, None]

    # Organic wave / diffuser ripple.
    wave_amp = float(params.get("wave_amp", 0.0))
    if wave_amp > 0.0:
        freq = float(params.get("wave_freq", 4.0))
        lat = np.arccos(np.clip(unit_dirs[:, 2], -1.0, 1.0))
        disp = (
            0.07
            * min_dim
            * wave_amp
            * np.sin(freq * lat)
            * (1.0 - np.abs(unit_dirs[:, 2]))
        )
        pts += unit_dirs * disp[:, None]

    # Basin / tray / cable channel depression from the top.
    basin = float(params.get("basin_depth", 0.0))
    if basin > 0.0:
        d = angular_distances(unit_dirs, np.array([0.0, 0.0, 1.0]))
        radius = 0.85
        mask = d < radius
        if np.any(mask):
            w = np.clip(1.0 - (d[mask] / radius) ** 2, 0.0, 1.0) ** 2
            pts[mask] -= unit_dirs[mask] * (0.35 * float(params["height"]) * basin * w[:, None])

    # Legs / feet for furniture and stable desk objects.
    legs = int(params.get("leg_count", 0))
    if legs > 0:
        leg_lift = 0.20 * float(params["height"]) * (0.25 + 0.75 * amp)
        for i in range(legs):
            ang = 2.0 * math.pi * i / max(legs, 1)
            center = np.array(
                [math.cos(ang) * 0.55, math.sin(ang) * 0.55, -1.0],
                dtype=float,
            )
            d = angular_distances(unit_dirs, center)
            radius = 0.45
            mask = d < radius
            if np.any(mask):
                w = np.clip(1.0 - (d[mask] / radius) ** 2, 0.0, 1.0) ** 2
                pts[mask] += unit_dirs[mask] * (leg_lift * w[:, None])

    return pts


def deform_unit_mesh(unit_verts: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    a = float(params["width"]) / 2.0
    b = float(params["depth"]) / 2.0
    c = float(params["height"]) / 2.0
    p = max(2.0, float(params.get("roundness", 4.0)))
    exponent = 2.0 / p

    x, y, z = unit_verts.T
    vertical = np.clip((z + 1.0) / 2.0, 0.0, 1.0)

    taper = float(params.get("taper", 0.0))
    xy_scale = 1.0 - taper * vertical

    sx = signed_power(x, exponent) * xy_scale * a
    sy = signed_power(y, exponent) * xy_scale * b
    sz = signed_power(z, exponent) * c

    asym = float(params.get("asymmetry", 0.0))
    sx += asym * 0.22 * float(params["width"]) * vertical

    pts = np.column_stack([sx, sy, sz])
    pts = apply_feature_modifiers(unit_verts, pts, params)
    return pts.astype(float)


def generate_mesh(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    params = clamp_params(params)
    level = int(np.clip(params.get("subdivision_level", 3), 2, 4))
    unit_verts, faces = icosphere(level)
    verts = deform_unit_mesh(unit_verts, params)
    return verts, faces.copy()


# --------------------------------------------------------------------------------------
# Geometry metrics and validity verification
# --------------------------------------------------------------------------------------

def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.array([], dtype=float)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        return 0.0
    tri = vertices[faces]
    return float(
        np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )


def compute_mesh_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    areas = triangle_areas(vertices, faces)
    volume = abs(signed_mesh_volume(vertices, faces))
    surface = float(np.sum(areas)) if areas.size else 0.0

    bbox_max = vertices.max(axis=0)
    bbox_min = vertices.min(axis=0)
    bbox = bbox_max - bbox_min

    compactness = 0.0
    if surface > 1e-9 and volume > 1e-9:
        compactness = (36.0 * math.pi * volume**2) / (surface**3)

    aspect = float(np.max(bbox) / max(np.min(bbox), 1e-6))
    footprint = float(bbox[0] * bbox[1])
    stability_ratio = footprint / max(bbox[2] ** 2, 1e-6)

    if areas.size:
        area_mean = float(np.mean(areas))
        area_roughness = float(np.std(areas) / area_mean) if area_mean > 1e-9 else 1.0
    else:
        area_mean = 0.0
        area_roughness = 1.0

    feature_count = int(
        int(params.get("vent_rows", 0)) * int(params.get("vent_cols", 0))
        + int(params.get("rib_count", 0))
        + int(params.get("leg_count", 0))
        + (1 if float(params.get("basin_depth", 0.0)) > 0.0 else 0)
    )

    bounding_volume = float(params["width"] * params["height"] * params["depth"])
    fill_fraction = volume / max(bounding_volume, 1e-9)

    return {
        "volume_mm3": float(volume),
        "surface_area_mm2": float(surface),
        "bounding_volume_mm3": float(bounding_volume),
        "fill_fraction": float(fill_fraction),
        "compactness": float(compactness),
        "aspect_ratio": float(aspect),
        "footprint_mm2": float(footprint),
        "stability_ratio": float(stability_ratio),
        "width_size": float(bbox[0]),
        "depth_size": float(bbox[1]),
        "height_size": float(bbox[2]),
        "triangle_count": int(len(faces)),
        "vertex_count": int(len(vertices)),
        "mean_triangle_area": float(area_mean),
        "area_roughness": float(area_roughness),
        "feature_count": int(feature_count),
    }


def validate_geometry(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, Any]:
    errors: List[str] = []

    if vertices.size == 0:
        errors.append("empty vertices")
    if faces.size == 0:
        errors.append("empty faces")
    if not np.all(np.isfinite(vertices)):
        errors.append("non-finite vertices")

    euler = 0
    degenerate_ratio = 1.0
    volume = 0.0
    closed_manifold = False

    if (
        faces.size > 0
        and int(faces.min()) >= 0
        and int(faces.max()) < len(vertices)
    ):
        areas = triangle_areas(vertices, faces)
        degenerate_ratio = float(np.mean(areas < 1e-8)) if areas.size else 1.0
        volume = abs(signed_mesh_volume(vertices, faces))

        edges = np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        unique_edges = np.unique(edges, axis=0)

        V = len(vertices)
        E = len(unique_edges)
        F = len(faces)
        euler = int(V - E + F)
        closed_manifold = euler == 2
    else:
        if faces.size > 0:
            errors.append("invalid face index")

    valid = (
        len(errors) == 0
        and closed_manifold
        and degenerate_ratio < 0.01
        and volume > 1e-6
    )

    return {
        "valid": bool(valid),
        "errors": errors,
        "euler_characteristic": int(euler),
        "closed_manifold": bool(closed_manifold),
        "degenerate_ratio": float(degenerate_ratio),
        "volume": float(volume),
    }


# --------------------------------------------------------------------------------------
# Constraint, requirement, and multi-objective scoring
# --------------------------------------------------------------------------------------

def requirement_match_score(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    spec: DesignSpec,
) -> float:
    scores: List[float] = []

    targets = STYLE_TARGETS.get(spec.style, {})
    for k, target in targets.items():
        val = float(params.get(k, DEFAULT_PARAMS.get(k, 0.0)))
        sigma = max(abs(float(target)) * 0.35, 0.15)
        scores.append(math.exp(-0.5 * ((val - float(target)) / sigma) ** 2))

    for req in spec.functional_requirements:
        mapping = FUNCTION_TO_PARAM.get(req)
        if not mapping:
            continue
        kind, key, threshold = mapping
        threshold = float(threshold)
        if threshold <= 0.0:
            scores.append(1.0)
            continue

        if kind == "param":
            val = float(params.get(key, DEFAULT_PARAMS.get(key, 0.0)))
        else:
            val = float(metrics.get(key, 0.0))

        scores.append(float(np.clip(val / threshold, 0.0, 1.0)))

    return float(np.mean(scores)) if scores else 0.5


def manufacturability_score(
    metrics: Dict[str, Any],
    validation: Dict[str, Any],
    params: Dict[str, Any],
) -> float:
    aspect = float(metrics.get("aspect_ratio", 1.0))
    aspect_score = math.exp(-max(0.0, aspect - 5.0) / 4.0)

    feature_score = 1.0 / (1.0 + float(metrics.get("feature_count", 0)) / 40.0)
    deg_score = 1.0 - min(1.0, float(validation.get("degenerate_ratio", 1.0)) * 10.0)

    wall = float(params.get("wall_thickness", 2.0))
    wall_score = 1.0 if wall >= 1.2 else wall / 1.2

    round_score = 1.0 if float(params.get("roundness", 4.0)) >= 2.0 else 0.0

    return float(
        np.mean([aspect_score, feature_score, deg_score, wall_score, round_score])
    )


def sustainability_score(metrics: Dict[str, Any], spec: DesignSpec) -> float:
    fill = float(metrics.get("fill_fraction", 0.5))
    target_fill = float((spec.fill_fraction_min + spec.fill_fraction_max) / 2.0)
    sigma = max(float((spec.fill_fraction_max - spec.fill_fraction_min) / 2.0), 0.08)
    fill_score = math.exp(-0.5 * ((fill - target_fill) / sigma) ** 2)

    compact = float(np.clip(metrics.get("compactness", 0.5), 0.0, 1.0))
    return float(0.70 * fill_score + 0.30 * compact)


def aesthetic_score(
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    spec: DesignSpec,
) -> float:
    rough = float(metrics.get("area_roughness", 0.25))
    continuity = math.exp(-rough * 4.0)
    style_align = requirement_match_score(params, metrics, spec)
    curvature = 1.0 / (1.0 + abs(float(params.get("roundness", 4.0)) - 5.0) / 8.0)
    return float(0.40 * continuity + 0.35 * style_align + 0.25 * curvature)


def constraint_report(candidate: Candidate, spec: DesignSpec) -> Dict[str, Any]:
    p = candidate.params
    m = candidate.metrics
    v = candidate.validation

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, score: float, detail: str = "") -> None:
        checks.append(
            {
                "check": name,
                "score": float(np.clip(score, 0.0, 1.0)),
                "detail": detail,
            }
        )

    for label, actual_key, requested in [
        ("Width dimension", "width_size", spec.width),
        ("Height dimension", "height_size", spec.height),
        ("Depth dimension", "depth_size", spec.depth),
    ]:
        actual = float(m.get(actual_key, requested))
        rel = abs(actual - requested) / max(requested, 1e-6)
        add_check(label, math.exp(-6.0 * rel), f"{actual:.1f} mm vs {requested:.1f} mm")

    vol = float(m.get("volume_mm3", 0.0))
    lo = float(spec.target_volume_min)
    hi = float(spec.target_volume_max)
    if hi <= lo:
        hi = lo + 1.0
    if lo <= vol <= hi:
        add_check("Target volume", 1.0, f"{vol:.0f} mm³ inside target")
    else:
        dist = (lo - vol) if vol < lo else (vol - hi)
        range_ = max(hi - lo, 1e-6)
        add_check(
            "Target volume",
            math.exp(-2.0 * dist / range_),
            f"{vol:.0f} mm³ outside [{lo:.0f}, {hi:.0f}]",
        )

    wall = float(p.get("wall_thickness", 2.0))
    add_check(
        "Wall thickness",
        1.0 if wall >= spec.wall_thickness - 1e-6 else wall / max(spec.wall_thickness, 1e-6),
        f"{wall:.2f} mm",
    )

    aspect = float(m.get("aspect_ratio", 1.0))
    add_check(
        "Aspect ratio",
        1.0 if aspect <= spec.max_aspect_ratio else spec.max_aspect_ratio / max(aspect, 1e-6),
        f"{aspect:.2f} <= {spec.max_aspect_ratio:.2f}",
    )

    if spec.stability_required:
        stability = float(m.get("stability_ratio", 0.0))
        add_check(
            "Stability footprint",
            min(1.0, stability / 0.55),
            f"ratio {stability:.2f}",
        )

    vent_cols = int(p.get("vent_cols", 0))
    if vent_cols > 0:
        spacing = float(m.get("depth_size", p.get("depth", 100.0))) / max(vent_cols + 1, 1)
        add_check("Vent spacing", min(1.0, spacing / 3.0), f"{spacing:.2f} mm")

    add_check(
        "Closed valid mesh",
        1.0 if v.get("valid", False) else 0.0,
        ", ".join(v.get("errors", [])) or f"Euler={v.get('euler_characteristic', 0)}",
    )

    df = pd.DataFrame(checks)
    check_weights = {
        "Closed valid mesh": 2.0,
        "Target volume": 1.5,
        "Width dimension": 1.0,
        "Height dimension": 1.0,
        "Depth dimension": 1.0,
        "Wall thickness": 1.0,
        "Aspect ratio": 1.0,
        "Stability footprint": 1.0,
        "Vent spacing": 0.5,
    }
    weights = df["check"].map(lambda x: check_weights.get(x, 1.0)).astype(float)
    satisfaction = float(np.average(df["score"], weights=weights)) if len(df) else 0.0

    return {
        "satisfaction": satisfaction,
        "checks": df,
    }


def rank_population(
    candidates: List[Candidate],
    spec: DesignSpec,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()

    weights = normalize_weights(weights)
    param_keys = [k for k in EDITOR_KEYS if k != "subdivision_level"]

    rows: List[Dict[str, Any]] = []
    matrix: List[List[float]] = []

    for c in candidates:
        cons = constraint_report(c, spec)["satisfaction"]
        req = requirement_match_score(c.params, c.metrics, spec)
        man = manufacturability_score(c.metrics, c.validation, c.params)
        sus = sustainability_score(c.metrics, spec)
        aes = aesthetic_score(c.metrics, c.params, spec)

        rows.append(
            {
                "id": c.id,
                "constraint": float(cons),
                "requirement": float(req),
                "manufacturability": float(man),
                "sustainability": float(sus),
                "aesthetics": float(aes),
            }
        )
        matrix.append([float(c.params.get(k, DEFAULT_PARAMS.get(k, 0.0))) for k in param_keys])

    df = pd.DataFrame(rows)
    mat = np.array(matrix, dtype=float)

    if len(mat) > 1:
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0)
        sd[sd < 1e-9] = 1.0
        mat_z = (mat - mu) / sd
        dist = squareform(pdist(mat_z, metric="euclidean"))
        np.fill_diagonal(dist, np.nan)
        div_raw = np.nanmean(dist, axis=1)
        if np.nanmax(div_raw) - np.nanmin(div_raw) < 1e-9:
            div = np.full(len(mat), 0.5)
        else:
            div = (div_raw - np.nanmin(div_raw)) / (np.nanmax(div_raw) - np.nanmin(div_raw))
    else:
        div = np.array([0.5])

    df["diversity"] = div

    for col in [
        "constraint",
        "requirement",
        "manufacturability",
        "sustainability",
        "aesthetics",
        "diversity",
    ]:
        df[col] = minmax(df[col])

    df["total_score"] = (
        weights.get("requirement", 0.0) * df["requirement"]
        + weights.get("constraint", 0.0) * df["constraint"]
        + weights.get("manufacturability", 0.0) * df["manufacturability"]
        + weights.get("sustainability", 0.0) * df["sustainability"]
        + weights.get("aesthetics", 0.0) * df["aesthetics"]
        + weights.get("diversity", 0.0) * df["diversity"]
    )

    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    score_map = dict(zip(df["id"], df["total_score"]))
    rank_map = dict(zip(df["id"], df["rank"]))

    for c in candidates:
        c.score = float(score_map.get(c.id, 0.0))
        c.rank = int(rank_map.get(c.id, 99))

    return df


# --------------------------------------------------------------------------------------
# Parametric specification to editable geometry parameters
# --------------------------------------------------------------------------------------

def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_PARAMS)
    out.update(jsonable(params))

    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k not in out:
            continue
        val = out[k]
        if k in INT_KEYS:
            val = int(round(float(val)))
        else:
            val = float(val)
        out[k] = float(np.clip(val, lo, hi)) if k not in INT_KEYS else int(np.clip(val, lo, hi))

    out["seed"] = int(out.get("seed", 0))
    return out


def apply_style(params: Dict[str, Any], style: str, complexity: float) -> Dict[str, Any]:
    p = dict(params)
    complexity = float(np.clip(complexity, 0.0, 1.0))

    if style == "Minimalist":
        p["feature_intensity"] *= 0.55
        p["roundness"] = max(float(p.get("roundness", 4.0)), 6.0)
        p["wave_amp"] *= 0.40
        p["asymmetry"] *= 0.30
        p["vent_rows"] = int(min(float(p.get("vent_rows", 0)), 2))
        p["vent_cols"] = int(min(float(p.get("vent_cols", 0)), 4))
    elif style == "Organic":
        p["roundness"] = min(float(p.get("roundness", 4.0)), 3.0)
        p["taper"] = max(float(p.get("taper", 0.0)), 0.22)
        p["wave_amp"] = max(float(p.get("wave_amp", 0.0)), 0.25)
        p["feature_intensity"] *= 0.85
    elif style == "Technical":
        p["roundness"] = float(np.clip(p.get("roundness", 4.0), 3.0, 6.0))
        p["vent_rows"] = max(int(p.get("vent_rows", 0)), 3)
        p["vent_cols"] = max(int(p.get("vent_cols", 0)), 6)
        p["rib_count"] = max(int(p.get("rib_count", 0)), 8)
        p["feature_intensity"] = max(float(p.get("feature_intensity", 0.0)), 0.45)
    elif style == "Playful":
        p["asymmetry"] = float(np.clip(p.get("asymmetry", 0.0) + 0.20, -0.8, 0.8))
        p["wave_amp"] = max(float(p.get("wave_amp", 0.0)), 0.35)
        p["leg_count"] = max(int(p.get("leg_count", 0)), 2)
        p["feature_intensity"] = max(float(p.get("feature_intensity", 0.0)), 0.35)
    elif style == "Luxury":
        p["roundness"] = max(float(p.get("roundness", 4.0)), 6.5)
        p["feature_intensity"] *= 0.75
        p["basin_depth"] = max(float(p.get("basin_depth", 0.0)), 0.18)
        p["wave_amp"] = float(np.clip(p.get("wave_amp", 0.0) * 0.75, 0.0, 0.45))
    elif style == "Sustainable":
        p["feature_intensity"] *= 0.75
        p["roundness"] = max(float(p.get("roundness", 4.0)), 5.0)
        p["taper"] = float(np.clip(p.get("taper", 0.0) * 0.80, 0.0, 0.35))

    p["feature_intensity"] *= 0.55 + 0.90 * complexity
    p["vent_rows"] = int(round(float(p.get("vent_rows", 0)) * (0.50 + complexity)))
    p["vent_cols"] = int(round(float(p.get("vent_cols", 0)) * (0.50 + complexity)))
    p["rib_count"] = int(round(float(p.get("rib_count", 0)) * (0.50 + complexity)))
    p["wave_amp"] *= 0.60 + 0.80 * complexity
    p["basin_depth"] *= 0.60 + 0.80 * complexity

    return p


def params_from_spec_rule(
    spec: DesignSpec,
    rule: str,
    overrides: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params.update(BASE_RULES.get(rule, {}))

    params.update(
        {
            "width": float(spec.width),
            "height": float(spec.height),
            "depth": float(spec.depth),
            "wall_thickness": float(spec.wall_thickness),
        }
    )

    params = apply_style(params, spec.style, spec.complexity)

    if spec.category == "Lighting fixtures":
        params["wave_amp"] = max(float(params.get("wave_amp", 0.0)), 0.30)
        params["basin_depth"] = max(float(params.get("basin_depth", 0.0)), 0.20)
    elif spec.category == "Furniture components":
        if rule in {"organic_capsule", "rounded_shell"}:
            params["leg_count"] = max(int(params.get("leg_count", 0)), 2)
        params["taper"] = max(float(params.get("taper", 0.0)), 0.12)
    elif spec.category == "Desk accessories":
        if rule in {"rounded_shell", "organic_capsule"}:
            params["basin_depth"] = max(float(params.get("basin_depth", 0.0)), 0.25)
    elif spec.category == "Household product enclosures":
        params["roundness"] = max(float(params.get("roundness", 4.0)), 4.5)

    rng = random.Random(int(seed))
    jitter_keys = [
        "roundness",
        "feature_intensity",
        "taper",
        "asymmetry",
        "wave_amp",
        "basin_depth",
    ]
    for k in jitter_keys:
        if k in params:
            lo, hi = PARAM_BOUNDS[k]
            params[k] = float(params[k]) + rng.gauss(0.0, (hi - lo) * 0.025)

    if overrides:
        params.update(jsonable(overrides))

    params = clamp_params(params)
    params["seed"] = int(seed)
    return params


def make_candidate(
    spec: DesignSpec,
    rule: str,
    params: Dict[str, Any],
    parent_id: Optional[str] = None,
    generation: int = 0,
    name_suffix: str = "",
) -> Candidate:
    params = clamp_params(params)
    params["subdivision_level"] = int(np.clip(params.get("subdivision_level", 3), 2, 4))

    vertices, faces = generate_mesh(params)
    metrics = compute_mesh_metrics(vertices, faces, params)
    validation = validate_geometry(vertices, faces)

    nonce = f"{time.time_ns()}-{random.randint(0, 999999)}"
    cid = stable_id("CAND", {"rule": rule, "params": params, "nonce": nonce})

    title = rule.replace("_", " ").title()
    suffix = f" {name_suffix}" if name_suffix else ""
    name = f"{title} G{generation}{suffix}"

    return Candidate(
        id=cid,
        name=name,
        category=spec.category,
        grammar_rule=rule,
        params=params,
        vertices=vertices,
        faces=faces,
        metrics=metrics,
        validation=validation,
        parent_id=parent_id,
        generation=int(generation),
    )


def generate_initial_concepts(
    spec: DesignSpec,
    seed: int = 2026,
    subdivision: int = 3,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Candidate], pd.DataFrame, float]:
    t0 = time.time()
    weights = weights or DEFAULT_WEIGHTS

    rules = CATEGORY_RULES.get(spec.category, [])[:]
    for r in BASE_RULES.keys():
        if len(rules) >= 5:
            break
        if r not in rules:
            rules.append(r)
    rules = rules[:5]

    candidates: List[Candidate] = []
    for i, rule in enumerate(rules):
        params = params_from_spec_rule(
            spec,
            rule,
            overrides={"subdivision_level": int(subdivision)},
            seed=int(seed) + i * 101,
        )
        candidates.append(
            make_candidate(
                spec,
                rule,
                params,
                generation=0,
                name_suffix=str(i + 1),
            )
        )

    df = rank_population(candidates, spec, weights)
    elapsed = time.time() - t0
    return candidates, df, elapsed


# --------------------------------------------------------------------------------------
# Variation engine and evolutionary search
# --------------------------------------------------------------------------------------

def mutate_params(
    params: Dict[str, Any],
    strength: float,
    rng: random.Random,
) -> Dict[str, Any]:
    strength = float(np.clip(strength, 0.0, 1.0))
    new = clamp_params(params)

    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k not in new:
            continue
        if k in INT_KEYS:
            if rng.random() < 0.35 * strength:
                delta = rng.choice([-2, -1, 1, 2])
                new[k] = int(new[k]) + delta
        else:
            span = hi - lo
            new[k] = float(new[k]) + rng.gauss(0.0, span * 0.08 * strength)

    if rng.random() < 0.20 * strength:
        new["feature_intensity"] = float(
            np.clip(float(new.get("feature_intensity", 0.2)) + rng.gauss(0.0, 0.15), 0.0, 1.0)
        )

    return clamp_params(new)


def crossover_params(
    a: Dict[str, Any],
    b: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    aa = clamp_params(a)
    bb = clamp_params(b)
    child: Dict[str, Any] = {}

    all_keys = set(list(aa.keys()) + list(bb.keys()) + list(DEFAULT_PARAMS.keys()))
    fixed_keys = {"width", "height", "depth", "wall_thickness", "subdivision_level", "seed"}

    for k in all_keys:
        va = aa.get(k, DEFAULT_PARAMS.get(k))
        vb = bb.get(k, DEFAULT_PARAMS.get(k))

        if k in fixed_keys:
            child[k] = va
        elif isinstance(va, (int, float, np.integer, np.floating)) and isinstance(
            vb, (int, float, np.integer, np.floating)
        ):
            if k in INT_KEYS:
                child[k] = int(round(rng.choice([float(va), float(vb)])))
            else:
                child[k] = float(rng.uniform(min(float(va), float(vb)), max(float(va), float(vb))))
        else:
            child[k] = va

    return clamp_params(child)


def generate_variations(
    spec: DesignSpec,
    population: List[Candidate],
    n_children: int,
    mutation_strength: float,
    seed: int,
    weights: Dict[str, float],
) -> List[Candidate]:
    if not population or n_children <= 0:
        return []

    rng = random.Random(int(seed))
    ranked = rank_population(population, spec, weights)
    id_to_c = {c.id: c for c in population}

    parent_ids = list(ranked["id"])
    parent_weights = [1.0 / ((i + 1) ** 2) for i in range(len(parent_ids))]

    children: List[Candidate] = []
    for i in range(int(n_children)):
        parent_id = rng.choices(parent_ids, weights=parent_weights, k=1)[0]
        parent = id_to_c[parent_id]

        if len(population) > 1 and rng.random() < 0.30:
            other = rng.choice(population)
            params = crossover_params(parent.params, other.params, rng)
            rule = parent.grammar_rule if rng.random() < 0.70 else other.grammar_rule
        else:
            params = mutate_params(parent.params, mutation_strength, rng)
            rule = parent.grammar_rule

        params["subdivision_level"] = int(
            np.clip(params.get("subdivision_level", 3), 2, 4)
        )

        children.append(
            make_candidate(
                spec,
                rule,
                params,
                parent_id=parent.id,
                generation=parent.generation + 1,
                name_suffix=f"var{i + 1}",
            )
        )

    return children


def evolve_population(
    spec: DesignSpec,
    population: List[Candidate],
    generations: int,
    pop_size: int,
    mutation_strength: float,
    seed: int,
    weights: Dict[str, float],
) -> List[Candidate]:
    if not population:
        return []

    rng = random.Random(int(seed))
    pop = list(population)

    for g in range(int(generations)):
        rank_population(pop, spec, weights)

        children = generate_variations(
            spec,
            pop,
            max(2, int(pop_size) // 2),
            mutation_strength,
            int(seed) + g * 17,
            weights,
        )

        merged: Dict[str, Candidate] = {c.id: c for c in pop}
        for c in children:
            merged[c.id] = c
        pop = list(merged.values())

        df = rank_population(pop, spec, weights)
        keep_n = min(int(pop_size), len(pop))
        top_n = max(1, int(keep_n * 0.70))
        keep_ids = list(df.head(top_n)["id"])

        remaining = [c for c in pop if c.id not in set(keep_ids)]
        rng.shuffle(remaining)
        extra = remaining[: max(0, keep_n - len(keep_ids))]
        keep_ids.extend([c.id for c in extra])

        id_map = {c.id: c for c in pop}
        pop = [id_map[i] for i in keep_ids if i in id_map]

    rank_population(pop, spec, weights)
    return pop


# --------------------------------------------------------------------------------------
# Serialization, history, and export
# --------------------------------------------------------------------------------------

def candidate_to_dict(c: Candidate) -> Dict[str, Any]:
    return {
        "id": c.id,
        "name": c.name,
        "category": c.category,
        "grammar_rule": c.grammar_rule,
        "params": jsonable(c.params),
        "metrics": jsonable(c.metrics),
        "validation": jsonable(c.validation),
        "score": float(c.score),
        "rank": int(c.rank),
        "parent_id": c.parent_id,
        "generation": int(c.generation),
        "created_at": float(c.created_at),
    }


def history_entry(c: Candidate) -> Dict[str, Any]:
    entry = candidate_to_dict(c)
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return entry


def obj_bytes(vertices: np.ndarray, faces: np.ndarray, name: str = "candidate") -> bytes:
    safe = safe_filename(name)
    lines = [f"o {safe}"]
    for v in vertices:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def stl_binary(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    buf = io.BytesIO()
    header = b"Generative Product Design Studio".ljust(80, b"\x00")
    buf.write(header)
    buf.write(struct.pack("<I", len(faces)))

    for f in faces:
        a = vertices[int(f[0])]
        b = vertices[int(f[1])]
        c = vertices[int(f[2])]

        n = np.cross(b - a, c - a)
        ln = float(np.linalg.norm(n))
        if ln > 1e-12:
            n = n / ln
        else:
            n = np.zeros(3, dtype=float)

        buf.write(struct.pack("<3f", float(n[0]), float(n[1]), float(n[2])))
        buf.write(struct.pack("<3f", float(a[0]), float(a[1]), float(a[2])))
        buf.write(struct.pack("<3f", float(b[0]), float(b[1]), float(b[2])))
        buf.write(struct.pack("<3f", float(c[0]), float(c[1]), float(c[2])))
        buf.write(struct.pack("<H", 0))

    return buf.getvalue()


def make_zip_bytes(
    spec: DesignSpec,
    candidates: List[Candidate],
    comparison_df: pd.DataFrame,
    demo_video: Optional[Tuple[Optional[bytes], str]] = None,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "README.txt",
            "Generative Product Design Studio export\n"
            "Includes editable OBJ/STL meshes, parameter JSON, validation reports, "
            "comparison metrics, and optional live demo video.\n"
            "No API keys required. Local-first prototype.\n",
        )
        zf.writestr(
            "design_specification.json",
            json.dumps(jsonable(asdict(spec)), indent=2),
        )
        zf.writestr(
            "comparison.csv",
            comparison_df.to_csv(index=False),
        )

        for c in candidates:
            base = f"candidates/{safe_filename(c.id)}"
            zf.writestr(base + ".obj", obj_bytes(c.vertices, c.faces, c.name))
            zf.writestr(base + ".stl", stl_binary(c.vertices, c.faces))
            zf.writestr(
                base + "_params.json",
                json.dumps(jsonable(c.params), indent=2),
            )
            zf.writestr(
                base + "_validation.json",
                json.dumps(jsonable(c.validation), indent=2),
            )
            zf.writestr(
                base + "_metrics.json",
                json.dumps(jsonable(c.metrics), indent=2),
            )

        if demo_video and demo_video[0] is not None:
            data, mime = demo_video
            ext = "mp4" if mime.startswith("video/") else "gif"
            zf.writestr(f"live_demo.{ext}", data)

    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Live demo video generation
# --------------------------------------------------------------------------------------

def render_mesh_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
    frames: int = 24,
) -> List[np.ndarray]:
    v = vertices - np.mean(vertices, axis=0)
    maxr = float(np.max(np.linalg.norm(v, axis=1)))
    if maxr < 1e-9:
        maxr = 1.0
    v = v / maxr

    imgs: List[np.ndarray] = []
    dpi = 80
    figsize = (320.0 / dpi, 240.0 / dpi)

    light = np.array([0.35, 0.25, 0.90], dtype=float)
    light = light / np.linalg.norm(light)

    for fi in range(int(frames)):
        angle = 2.0 * math.pi * fi / max(int(frames), 1)
        c = math.cos(angle)
        s = math.sin(angle)
        R = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        vr = v @ R.T

        tri = vr[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        norms[norms < 1e-12] = 1.0
        normals = normals / norms[:, None]

        intensity = np.clip(normals @ light, 0.0, 1.0)
        colors = np.column_stack(
            [
                0.55 + 0.35 * intensity,
                0.70 + 0.25 * intensity,
                0.92 + 0.08 * intensity,
            ]
        )

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(
            tri,
            facecolors=colors,
            edgecolors=(0.05, 0.08, 0.12, 0.18),
            linewidths=0.12,
        )
        ax.add_collection3d(pc)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        ax.view_init(elev=18, azim=0)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        imgs.append(img)
        plt.close(fig)

    return imgs


def encode_video(frames: List[np.ndarray], fps: int = 12) -> Tuple[Optional[bytes], str]:
    if not frames:
        return None, "No frames generated."

    try:
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        imageio.mimsave(
            path,
            frames,
            fps=int(fps),
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return data, "video/mp4"
    except Exception as e1:
        try:
            fd, path = tempfile.mkstemp(suffix=".gif")
            os.close(fd)
            imageio.mimsave(
                path,
                frames,
                format="GIF",
                duration=1.0 / max(float(fps), 1.0),
                loop=0,
            )
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
            return data, "image/gif"
        except Exception as e2:
            return None, f"MP4 failed: {e1}. GIF failed: {e2}."


@st.cache_data(show_spinner=False)
def generate_demo_video_cached(
    params_json: str,
    grammar_rule: str,
    frames: int,
    fps: int,
) -> Tuple[Optional[bytes], str]:
    try:
        params = json.loads(params_json)
        params = clamp_params(params)
        params["subdivision_level"] = 2

        vertices, faces = generate_mesh(params)
        imgs = render_mesh_frames(vertices, faces, frames=frames)
        return encode_video(imgs, fps=fps)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# Visualization helpers
# --------------------------------------------------------------------------------------

def mesh_plotly(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str = "",
    height: int = 420,
) -> go.Figure:
    x, y, z = vertices.T
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color="#74b9ff",
            opacity=0.96,
            flatshading=True,
            lighting=dict(
                ambient=0.35,
                diffuse=0.70,
                specular=0.25,
                roughness=0.55,
                fresnel=0.10,
            ),
            name=title,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
        showlegend=False,
        scene=dict(
            xaxis_title="X / width (mm)",
            yaxis_title="Y / depth (mm)",
            zaxis_title="Z / height (mm)",
            camera=dict(eye={"x": 1.6, "y": 1.6, "z": 1.15}),
        ),
    )
    return fig


def get_candidate(
    cid: str,
    candidates: Optional[List[Candidate]] = None,
) -> Candidate:
    if candidates is None:
        candidates = st.session_state.candidates
    for c in candidates:
        if c.id == cid:
            return c
    return candidates[0]


def add_to_population(new_candidates: List[Candidate]) -> Tuple[List[Candidate], pd.DataFrame]:
    existing = {c.id: c for c in st.session_state.candidates}
    for c in new_candidates:
        existing[c.id] = c
    pop = list(existing.values())
    df = rank_population(pop, st.session_state.spec, st.session_state.weights)
    st.session_state.candidates = pop
    st.session_state.comparison_df = df
    return pop, df


# --------------------------------------------------------------------------------------
# Session initialization: automatic realistic demo data and M1 concepts
# --------------------------------------------------------------------------------------

if "weights" not in st.session_state:
    st.session_state.weights = normalize_weights(DEFAULT_WEIGHTS)

if "spec" not in st.session_state:
    st.session_state.spec = DEFAULT_SPEC

if "history" not in st.session_state:
    st.session_state.history = []

if "candidates" not in st.session_state:
    with st.spinner("Loading realistic demo specification and generating five editable concepts..."):
        cands, comp_df, gen_time = generate_initial_concepts(
            st.session_state.spec,
            seed=2026,
            subdivision=3,
            weights=st.session_state.weights,
        )
    st.session_state.candidates = cands
    st.session_state.comparison_df = comp_df
    st.session_state.generation_time = gen_time
    st.session_state.selected_id = comp_df.iloc[0]["id"] if not comp_df.empty else cands[0].id
    st.session_state.initial_demo_done = False
    st.session_state.demo_video = None
    st.session_state.demo_candidate_id = st.session_state.selected_id


# --------------------------------------------------------------------------------------
# Sidebar: design specification, generation controls, ranking weights
# --------------------------------------------------------------------------------------

current = st.session_state.spec

st.sidebar.title("Design Specification")
st.sidebar.caption("Local generative product design studio — M1 live demonstration.")

category = st.sidebar.selectbox(
    "Product category",
    CATEGORIES,
    index=CATEGORIES.index(current.category) if current.category in CATEGORIES else 0,
)

product_name = st.sidebar.text_input(
    "Product name",
    value=current.product_name,
)

cc1, cc2, cc3 = st.sidebar.columns(3)
width = cc1.number_input(
    "Width (mm)",
    min_value=20.0,
    max_value=500.0,
    value=float(current.width),
    step=1.0,
)
height = cc2.number_input(
    "Height (mm)",
    min_value=20.0,
    max_value=500.0,
    value=float(current.height),
    step=1.0,
)
depth = cc3.number_input(
    "Depth (mm)",
    min_value=20.0,
    max_value=500.0,
    value=float(current.depth),
    step=1.0,
)

wall_thickness = st.sidebar.number_input(
    "Wall thickness (mm)",
    min_value=0.8,
    max_value=15.0,
    value=float(current.wall_thickness),
    step=0.1,
)

fill_min = st.sidebar.slider(
    "Minimum fill fraction",
    min_value=0.05,
    max_value=0.93,
    value=float(np.clip(current.fill_fraction_min, 0.05, 0.93)),
    step=0.01,
)
fill_max_value = float(np.clip(max(current.fill_fraction_max, fill_min + 0.02), 0.07, 0.95))
fill_max = st.sidebar.slider(
    "Maximum fill fraction",
    min_value=0.07,
    max_value=0.95,
    value=fill_max_value,
    step=0.01,
)
fill_max = max(fill_min + 0.02, fill_max)

style = st.sidebar.selectbox(
    "Design style",
    STYLES,
    index=STYLES.index(current.style) if current.style in STYLES else 0,
)

complexity = st.sidebar.slider(
    "Functional complexity",
    min_value=0.0,
    max_value=1.0,
    value=float(current.complexity),
    step=0.01,
)

functional_options = FUNCTIONAL_LIBRARY.get(category, [])
default_functional = [x for x in current.functional_requirements if x in functional_options]
if not default_functional and functional_options:
    default_functional = functional_options[:2]

functional_requirements = st.sidebar.multiselect(
    "Functional requirements",
    options=functional_options,
    default=default_functional,
)

max_aspect_ratio = st.sidebar.slider(
    "Max aspect ratio",
    min_value=2.0,
    max_value=12.0,
    value=float(np.clip(current.max_aspect_ratio, 2.0, 12.0)),
    step=0.5,
)

st.sidebar.divider()
st.sidebar.subheader("Generation engine")

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=2026,
    step=1,
)

subdivision = st.sidebar.select_slider(
    "Mesh subdivision level",
    options=[2, 3, 4],
    value=int(st.session_state.get("subdivision_level", 3)),
)
st.session_state.subdivision_level = subdivision

population_size = st.sidebar.slider(
    "Target population size",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

mutation_strength = st.sidebar.slider(
    "Mutation strength",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
)

st.sidebar.divider()
st.sidebar.subheader("Multi-objective ranking weights")

cw = st.session_state.get("weights", DEFAULT_WEIGHTS)

w_requirement = st.sidebar.slider(
    "Requirement match",
    0.0,
    1.0,
    value=float(cw.get("requirement", 0.25)),
    step=0.01,
)
w_constraint = st.sidebar.slider(
    "Constraint satisfaction",
    0.0,
    1.0,
    value=float(cw.get("constraint", 0.20)),
    step=0.01,
)
w_manufacturability = st.sidebar.slider(
    "Manufacturability",
    0.0,
    1.0,
    value=float(cw.get("manufacturability", 0.15)),
    step=0.01,
)
w_sustainability = st.sidebar.slider(
    "Sustainability",
    0.0,
    1.0,
    value=float(cw.get("sustainability", 0.10)),
    step=0.01,
)
w_aesthetics = st.sidebar.slider(
    "Aesthetics",
    0.0,
    1.0,
    value=float(cw.get("aesthetics", 0.15)),
    step=0.01,
)
w_diversity = st.sidebar.slider(
    "Diversity",
    0.0,
    1.0,
    value=float(cw.get("diversity", 0.15)),
    step=0.01,
)

raw_weights = {
    "requirement": w_requirement,
    "constraint": w_constraint,
    "manufacturability": w_manufacturability,
    "sustainability": w_sustainability,
    "aesthetics": w_aesthetics,
    "diversity": w_diversity,
}
st.session_state.weights = normalize_weights(raw_weights)

st.sidebar.divider()

generate_clicked = st.sidebar.button("Generate 5 concepts")
rerank_clicked = st.sidebar.button("Re-rank current population")
evolve_clicked = st.sidebar.button("Evolve population")
reset_clicked = st.sidebar.button("Reset demo")

if generate_clicked:
    new_spec = finalize_spec(
        DesignSpec(
            category=category,
            product_name=product_name,
            width=width,
            height=height,
            depth=depth,
            wall_thickness=wall_thickness,
            fill_fraction_min=fill_min,
            fill_fraction_max=fill_max,
            style=style,
            complexity=complexity,
            functional_requirements=functional_requirements,
            max_aspect_ratio=max_aspect_ratio,
            notes="User-generated specification.",
        )
    )
    st.session_state.spec = new_spec

    with st.spinner("Generating five editable parametric concepts..."):
        cands, comp_df, gen_time = generate_initial_concepts(
            new_spec,
            seed=int(seed),
            subdivision=int(subdivision),
            weights=st.session_state.weights,
        )

    st.session_state.candidates = cands
    st.session_state.comparison_df = comp_df
    st.session_state.generation_time = gen_time
    st.session_state.selected_id = comp_df.iloc[0]["id"] if not comp_df.empty else cands[0].id
    st.session_state.initial_demo_done = False
    st.success(f"Generated {len(cands)} candidates in {gen_time:.2f} seconds.")

if rerank_clicked:
    st.session_state.comparison_df = rank_population(
        st.session_state.candidates,
        st.session_state.spec,
        st.session_state.weights,
    )
    st.success("Population re-ranked with current weights.")

if evolve_clicked:
    with st.spinner("Running evolutionary variation and multi-objective selection..."):
        pop = evolve_population(
            st.session_state.spec,
            st.session_state.candidates,
            generations=2,
            pop_size=int(population_size),
            mutation_strength=float(mutation_strength),
            seed=int(seed) + len(st.session_state.candidates),
            weights=st.session_state.weights,
        )
    st.session_state.candidates = pop
    st.session_state.comparison_df = rank_population(
        pop,
        st.session_state.spec,
        st.session_state.weights,
    )
    st.session_state.selected_id = (
        st.session_state.comparison_df.iloc[0]["id"]
        if not st.session_state.comparison_df.empty
        else pop[0].id
    )
    st.session_state.initial_demo_done = False
    st.success(f"Evolved population now contains {len(pop)} editable candidates.")

if reset_clicked:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# --------------------------------------------------------------------------------------
# Main application state
# --------------------------------------------------------------------------------------

spec = st.session_state.spec
candidates = st.session_state.candidates

if not candidates:
    st.warning("No candidates available. Use the sidebar to generate concepts.")
    st.stop()

if "comparison_df" not in st.session_state or st.session_state.comparison_df is None or st.session_state.comparison_df.empty:
    st.session_state.comparison_df = rank_population(candidates, spec, st.session_state.weights)

ids = [c.id for c in sorted(candidates, key=lambda c: c.rank)]
if st.session_state.get("selected_id") not in ids:
    st.session_state.selected_id = ids[0]

selected = get_candidate(st.session_state.selected_id, candidates)
sorted_cands = sorted(candidates, key=lambda c: c.rank)
comp_df = st.session_state.comparison_df


# Automatic live demo video on first run.
if not st.session_state.get("initial_demo_done", False):
    with st.spinner("Automatically rendering live demo video for the top-ranked concept..."):
        params_json = json.dumps(jsonable(selected.params))
        video_bytes, mime = generate_demo_video_cached(
            params_json,
            selected.grammar_rule,
            frames=24,
            fps=12,
        )
    st.session_state.demo_video = (video_bytes, mime)
    st.session_state.demo_candidate_id = selected.id
    st.session_state.initial_demo_done = True


st.title("Generative Product Design Studio")
st.caption(
    "M1 executable local demonstration: five editable 3D product concepts from one specification, "
    "with parametric shape grammars, constraint checking, multi-objective ranking, interactive editing, "
    "and automatic live demo video generation. No API keys required."
)

k1, k2, k3, k4, k5 = st.columns(5)
valid_count = sum(1 for c in candidates if c.validation.get("valid", False))
k1.metric("Candidates", len(candidates))
k2.metric("Valid geometries", f"{valid_count}/{len(candidates)}")
k3.metric(
    "Mean constraint score",
    f"{float(comp_df['constraint'].mean()) if 'constraint' in comp_df else 0.0:.2f}",
)
k4.metric(
    "Population diversity",
    f"{float(comp_df['diversity'].mean()) if 'diversity' in comp_df else 0.0:.2f}",
)
k5.metric(
    "Last generation time",
    f"{float(st.session_state.get('generation_time', 0.0)):.2f} s",
)


tab_demo, tab_concepts, tab_compare, tab_editor, tab_history, tab_export, tab_eval = st.tabs(
    [
        "Live Demo",
        "Concepts",
        "Compare & Rank",
        "Interactive Editor",
        "History",
        "Export",
        "Evaluation",
    ]
)


# --------------------------------------------------------------------------------------
# Live Demo tab
# --------------------------------------------------------------------------------------

with tab_demo:
    st.subheader("Working live demonstration")
    st.markdown(
        "This view automatically loads a realistic consumer-electronics housing specification, "
        "generates five editable parametric concepts, verifies each mesh, and renders a short "
        "rotating demo video locally. You can interact with the 3D viewer and edit parameters."
    )

    video_data = st.session_state.get("demo_video")
    if video_data:
        data, mime = video_data
        if data is not None:
            if mime.startswith("video/"):
                st.video(data, format=mime)
            else:
                st.image(data, use_container_width=True)
            st.caption(
                f"Live demo rendered for candidate {st.session_state.get('demo_candidate_id', selected.id)}."
            )
        else:
            st.warning(f"Local video encoding unavailable: {mime}")

    if st.button("Render live demo for selected candidate"):
        with st.spinner("Rendering local demo video..."):
            params_json = json.dumps(jsonable(selected.params))
            data, mime = generate_demo_video_cached(
                params_json,
                selected.grammar_rule,
                frames=24,
                fps=12,
            )
        st.session_state.demo_video = (data, mime)
        st.session_state.demo_candidate_id = selected.id
        st.rerun()

    st.divider()

    left, right = st.columns([2.0, 1.0])
    with left:
        st.plotly_chart(
            mesh_plotly(selected.vertices, selected.faces, selected.name, height=520),
            use_container_width=True,
            config={"responsive": True},
        )
    with right:
        st.markdown(f"### {selected.name}")
        st.markdown(f"**Grammar rule:** `{selected.grammar_rule}`")
        st.markdown(f"**Rank:** {selected.rank}")
        st.markdown(f"**Composite score:** {selected.score:.3f}")
        st.markdown(f"**Valid closed mesh:** {'Yes' if selected.validation.get('valid') else 'No'}")
        st.markdown(f"**Triangles:** {selected.metrics.get('triangle_count', 0)}")
        st.markdown(f"**Vertices:** {selected.metrics.get('vertex_count', 0)}")
        st.markdown(f"**Volume:** {selected.metrics.get('volume_mm3', 0):,.0f} mm³")
        st.markdown(f"**Surface area:** {selected.metrics.get('surface_area_mm2', 0):,.0f} mm²")
        st.markdown(f"**Compactness:** {selected.metrics.get('compactness', 0):.3f}")
        st.markdown(f"**Constraint satisfaction:** {constraint_report(selected, spec)['satisfaction']:.2%}")

        st.info(
            "Drag to rotate, scroll to zoom, and right-drag to pan. "
            "Use the Interactive Editor tab to create editable parameter descendants."
        )


# --------------------------------------------------------------------------------------
# Concepts tab
# --------------------------------------------------------------------------------------

with tab_concepts:
    st.subheader("Generated editable design candidates")

    label_map = {
        c.id: f"{c.rank}. {c.name} — score {c.score:.3f} — {'valid' if c.validation.get('valid') else 'check'}"
        for c in sorted_cands
    }

    selected_id = st.radio(
        "Select candidate",
        options=ids,
        format_func=lambda x: label_map.get(x, x),
        index=ids.index(st.session_state.selected_id),
        horizontal=True,
    )
    st.session_state.selected_id = selected_id
    selected = get_candidate(selected_id, candidates)

    display_cards = sorted_cands[:5]
    card_cols = st.columns(min(5, len(display_cards)))
    for col, c in zip(card_cols, display_cards):
        col.markdown(f"**{c.rank}. {c.name}**")
        col.caption(
            f"Rule: `{c.grammar_rule}` | Score: {c.score:.3f} | "
            f"Valid: {'✅' if c.validation.get('valid') else '⚠️'}"
        )
        col.plotly_chart(
            mesh_plotly(c.vertices, c.faces, c.name, height=260),
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### Selected candidate parameter snapshot")
    param_rows = [
        {"parameter": k, "value": selected.params.get(k, DEFAULT_PARAMS.get(k, ""))}
        for k in EDITOR_KEYS
    ]
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Compare & Rank tab
# --------------------------------------------------------------------------------------

with tab_compare:
    st.subheader("Multi-objective comparison")

    meta_map = {
        c.id: {
            "name": c.name,
            "rule": c.grammar_rule,
            "valid": c.validation.get("valid", False),
        }
        for c in candidates
    }

    comp_display = comp_df.copy()
    comp_display["name"] = comp_display["id"].map(lambda x: meta_map.get(x, {}).get("name", x))
    comp_display["grammar_rule"] = comp_display["id"].map(lambda x: meta_map.get(x, {}).get("rule", ""))
    comp_display["valid"] = comp_display["id"].map(lambda x: meta_map.get(x, {}).get("valid", False))

    st.dataframe(
        comp_display[
            [
                "rank",
                "name",
                "grammar_rule",
                "total_score",
                "requirement",
                "constraint",
                "manufacturability",
                "sustainability",
                "aesthetics",
                "diversity",
                "valid",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Geometry and performance metrics")
    metric_rows = []
    for c in sorted_cands:
        metric_rows.append(
            {
                "rank": c.rank,
                "name": c.name,
                "rule": c.grammar_rule,
                "score": c.score,
                "volume_mm3": c.metrics.get("volume_mm3", 0.0),
                "surface_mm2": c.metrics.get("surface_area_mm2", 0.0),
                "fill_fraction": c.metrics.get("fill_fraction", 0.0),
                "compactness": c.metrics.get("compactness", 0.0),
                "aspect_ratio": c.metrics.get("aspect_ratio", 0.0),
                "triangles": c.metrics.get("triangle_count", 0),
                "features": c.metrics.get("feature_count", 0),
                "valid": c.validation.get("valid", False),
                "euler": c.validation.get("euler_characteristic", 0),
                "degenerate_pct": c.validation.get("degenerate_ratio", 1.0) * 100.0,
            }
        )
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Candidate diversity matrix")
    param_keys = [k for k in EDITOR_KEYS if k != "subdivision_level"]
    mat = np.array(
        [[float(c.params.get(k, DEFAULT_PARAMS.get(k, 0.0))) for k in param_keys] for c in sorted_cands],
        dtype=float,
    )
    if len(mat) > 1:
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0)
        sd[sd < 1e-9] = 1.0
        mat_z = (mat - mu) / sd
        dist = squareform(pdist(mat_z, metric="euclidean"))
    else:
        dist = np.zeros((len(sorted_cands), len(sorted_cands)), dtype=float)

    labels = [c.name for c in sorted_cands]
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=dist,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hoverongaps=False,
        )
    )
    fig_heat.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=80),
        title="Normalized parameter-space distance",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Parallel coordinates")
    dims = []
    for k in param_keys[:10]:
        vals = [float(c.params.get(k, DEFAULT_PARAMS.get(k, 0.0))) for c in sorted_cands]
        dims.append(dict(range=pc_range(vals), label=k, values=vals))
    score_vals = [float(c.score) for c in sorted_cands]
    dims.append(dict(range=pc_range(score_vals), label="score", values=score_vals))

    fig_pc = go.Figure(data=go.Parcoords(dimensions=dims))
    fig_pc.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_pc, use_container_width=True)

    st.markdown("#### Constraint checks for selected candidate")
    cons = constraint_report(selected, spec)
    st.metric("Constraint satisfaction", f"{cons['satisfaction']:.2%}")
    st.dataframe(cons["checks"], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Interactive Editor tab
# --------------------------------------------------------------------------------------

with tab_editor:
    st.subheader("Interactive parametric editing")
    st.markdown(
        "Every edit creates a new descendant candidate with preserved parameter history. "
        "Geometry remains editable because each shape is regenerated from parametric grammar rules."
    )

    edited: Dict[str, Any] = {}
    editor_cols = st.columns(4)

    for idx, key in enumerate(EDITOR_KEYS):
        col = editor_cols[idx % 4]
        lo, hi = PARAM_BOUNDS[key]
        val = selected.params.get(key, DEFAULT_PARAMS.get(key, 0.0))

        label = key.replace("_", " ").title()

        if key in INT_KEYS:
            edited[key] = col.number_input(
                label,
                min_value=int(lo),
                max_value=int(hi),
                value=int(val),
                step=1,
                key=f"edit_{selected.id}_{key}",
            )
        elif key in {"width", "height", "depth"}:
            edited[key] = col.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(val),
                step=0.5,
                format="%.1f",
                key=f"edit_{selected.id}_{key}",
            )
        elif key == "wall_thickness":
            edited[key] = col.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(val),
                step=0.1,
                format="%.2f",
                key=f"edit_{selected.id}_{key}",
            )
        else:
            edited[key] = col.slider(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(val),
                step=0.01,
                key=f"edit_{selected.id}_{key}",
            )

    b1, b2, b3, b4 = st.columns(4)

    if b1.button("Apply edits as new version"):
        new_params = clamp_params({**selected.params, **edited})
        child = make_candidate(
            spec,
            selected.grammar_rule,
            new_params,
            parent_id=selected.id,
            generation=selected.generation + 1,
            name_suffix="edit",
        )
        add_to_population([child])
        st.session_state.selected_id = child.id
        st.session_state.history.append(history_entry(child))
        st.session_state.initial_demo_done = False
        st.rerun()

    if b2.button("Mutate selected"):
        rng = random.Random(int(seed) + len(candidates) + 17)
        mutated = mutate_params(selected.params, max(0.15, float(mutation_strength)), rng)
        child = make_candidate(
            spec,
            selected.grammar_rule,
            mutated,
            parent_id=selected.id,
            generation=selected.generation + 1,
            name_suffix="mutant",
        )
        add_to_population([child])
        st.session_state.selected_id = child.id
        st.session_state.history.append(history_entry(child))
        st.session_state.initial_demo_done = False
        st.rerun()

    if b3.button("Generate 3 variations from selected"):
        rng = random.Random(int(seed) + len(candidates) + 31)
        children = []
        for i in range(3):
            params = mutate_params(selected.params, max(0.20, float(mutation_strength)), rng)
            children.append(
                make_candidate(
                    spec,
                    selected.grammar_rule,
                    params,
                    parent_id=selected.id,
                    generation=selected.generation + 1,
                    name_suffix=f"var{i + 1}",
                )
            )
        add_to_population(children)
        st.session_state.selected_id = children[0].id
        for ch in children:
            st.session_state.history.append(history_entry(ch))
        st.session_state.initial_demo_done = False
        st.rerun()

    if b4.button("Save selected to history"):
        st.session_state.history.append(history_entry(selected))
        st.success("Saved selected candidate to design history.")

    st.divider()
    st.markdown("#### Live edited preview")
    st.plotly_chart(
        mesh_plotly(selected.vertices, selected.faces, f"{selected.name} — editor preview", height=480),
        use_container_width=True,
        config={"responsive": True},
    )


# --------------------------------------------------------------------------------------
# History tab
# --------------------------------------------------------------------------------------

with tab_history:
    st.subheader("Design history and version tree")

    if not st.session_state.history:
        st.info("No history yet. Apply an edit, mutate a candidate, or restore a benchmark variant.")
    else:
        history_df = pd.DataFrame(
            [
                {
                    "timestamp": h.get("timestamp", ""),
                    "name": h.get("name", ""),
                    "rule": h.get("grammar_rule", ""),
                    "generation": h.get("generation", 0),
                    "score": h.get("score", 0.0),
                    "parent_id": h.get("parent_id", ""),
                    "id": h.get("id", ""),
                }
                for h in st.session_state.history
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        labels = [
            f"{h.get('timestamp', '')} — {h.get('name', '')} ({h.get('grammar_rule', '')})"
            for h in st.session_state.history
        ]
        choice = st.selectbox("Choose history entry to restore", labels)
        idx = labels.index(choice)
        entry = st.session_state.history[idx]

        rc1, rc2 = st.columns(2)
        if rc1.button("Restore selected history entry"):
            child = make_candidate(
                spec,
                entry["grammar_rule"],
                entry["params"],
                parent_id=entry.get("id"),
                generation=int(entry.get("generation", 0)) + 1,
                name_suffix="restored",
            )
            add_to_population([child])
            st.session_state.selected_id = child.id
            st.session_state.history.append(history_entry(child))
            st.session_state.initial_demo_done = False
            st.rerun()

        if rc2.button("Clear history"):
            st.session_state.history = []
            st.rerun()

        st.download_button(
            "Download history JSON",
            json.dumps(jsonable(st.session_state.history), indent=2).encode("utf-8"),
            file_name="design_history.json",
            mime="application/json",
        )


# --------------------------------------------------------------------------------------
# Export tab
# --------------------------------------------------------------------------------------

with tab_export:
    st.subheader("Export editable 3D and parameter data")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button(
            "Download selected STL",
            stl_binary(selected.vertices, selected.faces),
            file_name=f"{safe_filename(selected.id)}.stl",
            mime="model/stl",
        )
    with e2:
        st.download_button(
            "Download selected OBJ",
            obj_bytes(selected.vertices, selected.faces, selected.name),
            file_name=f"{safe_filename(selected.id)}.obj",
            mime="text/plain",
        )
    with e3:
        st.download_button(
            "Download selected parameters",
            json.dumps(jsonable(selected.params), indent=2).encode("utf-8"),
            file_name=f"{safe_filename(selected.id)}_params.json",
            mime="application/json",
        )
    with e4:
        st.download_button(
            "Download comparison CSV",
            comp_df.to_csv(index=False).encode("utf-8"),
            file_name="candidate_comparison.csv",
            mime="text/csv",
        )

    st.download_button(
        "Download all concepts as ZIP",
        make_zip_bytes(spec, candidates, comp_df, st.session_state.get("demo_video")),
        file_name="generative_product_design_export.zip",
        mime="application/zip",
    )

    demo_video = st.session_state.get("demo_video")
    if demo_video and demo_video[0] is not None:
        data, mime = demo_video
        ext = "mp4" if mime.startswith("video/") else "gif"
        st.download_button(
            "Download live demo video",
            data,
            file_name=f"live_demo_{safe_filename(selected.id)}.{ext}",
            mime=mime,
        )


# --------------------------------------------------------------------------------------
# Evaluation tab
# --------------------------------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluation and M1 verification")

    st.markdown("#### Current population geometry verification")
    val_rows = []
    for c in sorted_cands:
        val_rows.append(
            {
                "rank": c.rank,
                "name": c.name,
                "rule": c.grammar_rule,
                "valid": c.validation.get("valid", False),
                "closed_manifold": c.validation.get("closed_manifold", False),
                "euler": c.validation.get("euler_characteristic", 0),
                "degenerate_pct": c.validation.get("degenerate_ratio", 1.0) * 100.0,
                "volume_mm3": c.validation.get("volume", 0.0),
                "errors": ", ".join(c.validation.get("errors", [])),
            }
        )
    st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

    all_valid = all(c.validation.get("valid", False) for c in candidates)
    if all_valid:
        st.success(
            "M1 verification passed: every generated model is a finite, closed, non-degenerate "
            "editable triangle mesh with preserved parameter history."
        )
    else:
        st.warning(
            "M1 verification detected invalid geometry. Inspect the errors column and reduce "
            "mutation strength or feature intensity."
        )

    st.divider()
    st.markdown("#### Fixed benchmark of design specifications")
    st.caption(
        "Runs five benchmark specifications across the initial product categories and measures "
        "geometry validity, constraint satisfaction, diversity, and generation time."
    )

    if st.button("Run fixed benchmark"):
        rows = []
        with st.spinner("Running fixed benchmark..."):
            for bd in BENCHMARK_SPECS:
                bspec = benchmark_to_spec(bd)
                t0 = time.time()
                bcands, bdf, _ = generate_initial_concepts(
                    bspec,
                    seed=17,
                    subdivision=2,
                    weights=st.session_state.weights,
                )
                elapsed = time.time() - t0

                valid_rate = sum(1 for c in bcands if c.validation.get("valid", False)) / max(len(bcands), 1)
                mean_constraint = float(bdf["constraint"].mean()) if "constraint" in bdf else 0.0
                mean_diversity = float(bdf["diversity"].mean()) if "diversity" in bdf else 0.0
                best_score = float(bdf["total_score"].max()) if "total_score" in bdf else 0.0

                rows.append(
                    {
                        "category": bspec.category,
                        "product": bspec.product_name,
                        "candidates": len(bcands),
                        "valid_rate": valid_rate,
                        "mean_constraint": mean_constraint,
                        "mean_diversity": mean_diversity,
                        "best_score": best_score,
                        "generation_time_s": elapsed,
                    }
                )

        st.session_state.benchmark_df = pd.DataFrame(rows)

    if "benchmark_df" in st.session_state:
        st.dataframe(st.session_state.benchmark_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download benchmark CSV",
            st.session_state.benchmark_df.to_csv(index=False).encode("utf-8"),
            file_name="fixed_benchmark_results.csv",
            mime="text/csv",
        )