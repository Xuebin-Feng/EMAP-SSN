# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Colors the Viewer's commands share, so that one meaning looks the same whichever command paints it.

This module is not shadowed by the VR viewer, which re-exports Command_Engine's names one by one, so
the commands the two viewers share import such constants from here.
"""

import copy
import math
import string
import threading

import numpy as np

from utilities.Localization import Message, translate

# RGBA of a node with no value to color it by or no group to belong to: translucent light gray.
NOISE_COLOR = (0.8, 0.8, 0.8, 0.4)


# ---------------------------------------------------------------------
# Categorical palettes for clusters and subclusters
# ---------------------------------------------------------------------
# A fixed prefix sequence so that the largest clusters (numbered 1..N by size)
# keep stable, maximally distinguishable colors as parameters change.
#
# - First 7 cluster colors: hand-tuned CVD-safe core (min CIEDE2000 >= 24.6 in
#   normal vision, >= 14.0 under deuteranopia, >= 13.7 under protanopia, and
#   >= 12.1 under tritanopia), all clear of the white canvas, black outlines,
#   and gray NOISE_COLOR.
# - Colors 8..64 (and dynamic extensions past 64): greedy farthest-point
#   sampling in CIEDE2000 (Glasbey method) over a gamut bounded away from white
#   (WCAG contrast >= 2.25:1), dark backgrounds (>= 2.25:1 against #18181b),
#   black (L* >= 30), and gray noise (CIEDE2000 >= 18 from #ccc/#ebebeb/#fff/#000),
#   with tiered OKLCH chroma floors so the top 32 colors stay crisp and vivid.
# - Subclusters use a bold, high-chroma sequence whose top 8 colors also stay
#   separated (CIEDE2000 >= 12) from the top 8 main cluster colors so subclustering
#   one cluster stands out cleanly against surrounding clusters.

CLUSTER_PALETTE = (
    # CVD-universal publication core (1..7)
    "#1d5bbf", "#d94816", "#009e73", "#dfa000", "#d962a8", "#29b6e6", "#882266",
    # Tiered-chroma CIEDE2000 farthest-point sequence (8..64)
    "#95b520", "#9f95ff", "#206a00", "#9f0bff", "#ff8080", "#aa002a", "#008aff",
    "#ff8a40", "#f40b60", "#807500", "#955500", "#8a6ab5", "#00bfbf", "#009f00",
    "#d460ff", "#60409f", "#ca00aa", "#9f2a00", "#bf7500", "#6a6aff", "#aa4a6a",
    "#6abf6a", "#0b20ff", "#d495df", "#008ab5", "#006060", "#758a55", "#aa6a60",
    "#aa9560", "#208a8a", "#357555", "#555515", "#d495aa", "#bf8a6a", "#006080",
    "#805580", "#ff3540", "#8a4a40", "#9f9500", "#6a80b5", "#80aaf4", "#aa80aa",
    "#75b595", "#608000", "#d46a80", "#9500aa", "#957540", "#d40b20", "#aa6a40",
    "#ea7555", "#9faa6a", "#606095", "#008a40", "#ff00d4", "#df0080", "#df9f60",
    "#009fb5",
)

SUBCLUSTER_PALETTE = (
    # Bold high-chroma CVD-aware core, distinct from top main clusters (1..5)
    "#dc143c", "#7c2ae8", "#007a33", "#ff8c1a", "#00a6b4",
    # High-chroma CIEDE2000 farthest-point sequence (6..64)
    "#808aff", "#df8aff", "#ff7580", "#aab500", "#15bf75", "#9f0b6a", "#0060bf",
    "#ff00aa", "#00aaff", "#ca4a00", "#559f0b", "#aa8000", "#9f60aa", "#607500",
    "#d40bff", "#953520", "#b55575", "#f48abf", "#ff6a40", "#00bfaa", "#952a40",
    "#6a60b5", "#75408a", "#00956a", "#0080bf", "#df9f00", "#bf752a", "#754a00",
    "#007575", "#b56a60", "#959555", "#d49575", "#9f8aca", "#806a2a", "#b500b5",
    "#8a60ff", "#8ab56a", "#00604a", "#156075", "#bf9f6a", "#956040", "#b5809f",
    "#55550b", "#ff006a", "#804a60", "#1540d4", "#358095", "#ff0015", "#40602a",
    "#608a4a", "#4a958a", "#d49595", "#9faadf", "#2075ff", "#ff4aea", "#60b5d4",
    "#6a80b5", "#6aaa8a", "#955555",
)


def _hex_to_rgb(hex_color):
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16) / 255.0, int(value[2:4], 16) / 255.0, int(value[4:6], 16) / 255.0)


_CLUSTER_BASE_RGB = tuple(_hex_to_rgb(code) for code in CLUSTER_PALETTE)
_SUBCLUSTER_BASE_RGB = tuple(_hex_to_rgb(code) for code in SUBCLUSTER_PALETTE)
_PALETTE_LOCK = threading.Lock()
_EXTENDED_CACHE = {}
_CANDIDATE_GAMUT = None


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb):
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / 1.00000
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    eps = (6.0 / 29.0) ** 3
    kap = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
    fx = np.where(x > eps, np.cbrt(np.maximum(x, 0.0)), kap * x + 4.0 / 29.0)
    fy = np.where(y > eps, np.cbrt(np.maximum(y, 0.0)), kap * y + 4.0 / 29.0)
    fz = np.where(z > eps, np.cbrt(np.maximum(z, 0.0)), kap * z + 4.0 / 29.0)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _rgb_to_oklab_chroma(rgb):
    lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = np.cbrt(np.maximum(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b, 0.0))
    m_ = np.cbrt(np.maximum(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b, 0.0))
    s_ = np.cbrt(np.maximum(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b, 0.0))
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.hypot(a, b_)


def _ciede2000_to_many(lab1, lab2):
    """Exact CIEDE2000 color difference from one CIELAB point ``lab1`` to an ``(M, 3)`` array."""
    l1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    avg_l = 0.5 * (l1 + l2)
    c1 = math.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    avg_c7 = (0.5 * (c1 + c2)) ** 7
    g = 0.5 * (1.0 - np.sqrt(avg_c7 / (avg_c7 + 25.0 ** 7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    avg_cp = 0.5 * (c1p + c2p)
    h1p = np.where(c1p > 1e-12, np.degrees(np.arctan2(b1, a1p)) % 360.0, 0.0)
    h2p = np.where(c2p > 1e-12, np.degrees(np.arctan2(b2, a2p)) % 360.0, 0.0)
    d_lp = l2 - l1
    d_cp = c2p - c1p
    diff = h2p - h1p
    d_hp_deg = np.where(
        np.abs(diff) <= 180.0, diff,
        np.where(diff > 180.0, diff - 360.0, diff + 360.0),
    )
    nonzero = (c1p * c2p) > 0.0
    d_hp_deg = np.where(nonzero, d_hp_deg, 0.0)
    d_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(0.5 * d_hp_deg))
    h_sum = h1p + h2p
    avg_hp = np.where(
        ~nonzero, h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180.0, 0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )
    t_term = (
        1.0
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )
    d_ro = 30.0 * np.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    avg_cp7 = avg_cp ** 7
    r_c = 2.0 * np.sqrt(avg_cp7 / (avg_cp7 + 25.0 ** 7))
    lm50_2 = (avg_l - 50.0) ** 2
    s_l = 1.0 + (0.015 * lm50_2) / np.sqrt(20.0 + lm50_2)
    s_c = 1.0 + 0.045 * avg_cp
    s_h = 1.0 + 0.015 * avg_cp * t_term
    r_t = -np.sin(np.radians(2.0 * d_ro)) * r_c
    return np.sqrt(
        np.maximum(
            0.0,
            (d_lp / s_l) ** 2 + (d_cp / s_c) ** 2 + (d_hp / s_h) ** 2 + r_t * (d_cp / s_c) * (d_hp / s_h),
        )
    )


def _candidate_gamut():
    """Lazily build the bounded sRGB candidate pool for dynamic extensions past 64 colors."""
    global _CANDIDATE_GAMUT
    if _CANDIDATE_GAMUT is not None:
        return _CANDIDATE_GAMUT
    levels = np.linspace(0.0, 1.0, 17)
    grid = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    lin = _srgb_to_linear(grid)
    lum = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    cr_white = 1.05 / (lum + 0.05)
    cr_dark = (lum + 0.05) / (0.009 + 0.05)
    lab = _rgb_to_lab(grid)
    chroma = _rgb_to_oklab_chroma(grid)
    mask = (cr_white >= 2.25) & (cr_dark >= 2.25) & (lab[:, 0] >= 30.0) & (lab[:, 0] <= 76.0) & (chroma >= 0.075)
    cand_rgb = grid[mask]
    cand_lab = lab[mask]
    forbidden = _rgb_to_lab(np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.8, 0.8, 0.8],
        [0.92, 0.92, 0.92],
        [0.5, 0.5, 0.5],
    ]))
    min_forb = np.min(np.stack([_ciede2000_to_many(anchor, cand_lab) for anchor in forbidden], axis=0), axis=0)
    keep = min_forb >= 18.0
    _CANDIDATE_GAMUT = (cand_rgb[keep], cand_lab[keep])
    return _CANDIDATE_GAMUT


def _palette_colors(key, base_rgb, count):
    """Return ``count`` distinguishable (r, g, b) tuples, extending ``base_rgb`` in CIEDE2000 as needed."""
    count = int(count)
    if count <= 0:
        return []
    if count <= len(base_rgb):
        return list(base_rgb[:count])

    with _PALETTE_LOCK:
        cached = _EXTENDED_CACHE.get(key)
        if cached is None:
            cand_rgb, cand_lab = _candidate_gamut()
            chosen = list(base_rgb)
            min_d = np.full(len(cand_rgb), np.inf, dtype=np.float64)
            for lab_point in _rgb_to_lab(np.asarray(base_rgb, dtype=np.float64)):
                min_d = np.minimum(min_d, _ciede2000_to_many(lab_point, cand_lab))
            cached = [chosen, min_d]
            _EXTENDED_CACHE[key] = cached

        chosen, min_d = cached
        cand_rgb, cand_lab = _candidate_gamut()
        max_unique = len(base_rgb) + len(cand_rgb)
        target = min(count, max_unique)
        while len(chosen) < target:
            idx = int(np.argmax(min_d))
            if min_d[idx] <= 0.0:
                break
            chosen.append(tuple(float(channel) for channel in cand_rgb[idx]))
            min_d = np.minimum(min_d, _ciede2000_to_many(cand_lab[idx], cand_lab))
        cached[1] = min_d

        if count <= len(chosen):
            return list(chosen[:count])
        return [chosen[idx % len(chosen)] for idx in range(count)]


def cluster_colors(n_clusters):
    """Return ``n_clusters`` perceptually distinguishable (r, g, b) tuples for main clusters."""
    return _palette_colors("cluster", _CLUSTER_BASE_RGB, n_clusters)


def subcluster_colors(n_subclusters):
    """Return ``n_subclusters`` bold, distinguishable (r, g, b) tuples for subclusters."""
    return _palette_colors("subcluster", _SUBCLUSTER_BASE_RGB, n_subclusters)


# ---------------------------------------------------------------------
# The color key: what the node colors mean
# ---------------------------------------------------------------------
# A command that paints nodes also declares what its colors mean, right after it writes
# viewer.current_colors and inside the same undo step. The record lives in cacheable Viewer
# attributes (EMAPSSN_Viewer.CUSTOM_ATTRIBUTES_INIT), so undo, redo, save and load carry it:
#
#   color_key_layers    list of JSON-safe dicts, oldest first. A categories layer is {"id", "kind":
#                       "categories", "title", "categories": [{"label", "rgba"}, ...]}, and may carry
#                       "extend": True (see declare_categories); a gradient layer is
#                       {"id", "kind": "gradient", "title", "scheme", "min", "max", "stops": 16 [r, g, b]
#                       of the colormap, "extras": [{"label", "rgba"}, ...]}. A title or label is plain
#                       text, or a text() dict that key_state shows in the current language.
#   color_key_owner     int32 per node: the id of the layer that owns it, -1 for none.
#   color_key_category  int32 per node: its category, or its gradient extra; -1 for a gradient value.
#   color_key_rgba      uint8 (n, 4): the color the key expects the node to have.
#   color_key_values    float32 per node: a gradient-owned node's value, NaN otherwise.
#
# A node counts for its owner only while its color still is the expected one, so a command that paints
# without declaring never leaves a stale name on a node.
#
# A layout cache is a file people share, and it can set all of these attributes to anything. Whatever
# reads the key goes through _checked_key, so malformed data reads as no key or a trimmed one.

KEY_ATTRIBUTES = ("color_key_layers", "color_key_owner", "color_key_category", "color_key_rgba", "color_key_values")
GRADIENT_STOPS = 16


def text(message):
    """A title or label as the key stores it: a Message as {"template", "values"}, anything else as str.

    The values of a stored Message must be JSON-safe, as the cache keeps them; others are stored as text.
    """
    if isinstance(message, Message):
        values = {name: value if isinstance(value, (str, int, float, bool)) else str(value)
                  for name, value in message.values.items()}
        return {"template": message.template, "values": values}
    return str(message)


_FORMATTER = string.Formatter()


def _filled(template, values):
    """``template`` with its {name} fields filled from ``values``, or None when anything else is in it.

    A field must be a plain identifier that ``values`` holds: no attribute or index access, no conversion and
    no format spec, so a stored template can read nothing but the values stored beside it. Values show as str.
    """
    pieces = []
    try:
        for literal, field, spec, conversion in _FORMATTER.parse(template):
            pieces.append(literal)
            if field is None:
                continue
            if not field.isidentifier() or spec or conversion is not None or field not in values:
                return None
            pieces.append(str(values[field]))
    except ValueError:
        return None
    return "".join(pieces)


def shown(stored):
    """A stored title or label as read now: a stored Message in the current language, text as it is.

    Never raises. A template that its values don't fit, in the translation and then in English, shows as written.
    """
    if not isinstance(stored, dict):
        return str(stored)
    template = stored.get("template")
    if not isinstance(template, str):
        return ""
    values = stored.get("values")
    values = values if isinstance(values, dict) else {}
    try:
        translated = translate(template)
    except Exception:
        translated = template
    for candidate in (translated, template):
        filled = _filled(candidate, values) if isinstance(candidate, str) else None
        if filled is not None:
            return filled
    return template


def _node_indices(viewer, nodes):
    """``nodes`` (None for all, a bool mask or node numbers) as an int array of node numbers."""
    if nodes is None:
        return np.arange(viewer.n_nodes)
    nodes = np.atleast_1d(nodes)
    return np.flatnonzero(nodes) if nodes.dtype == bool else nodes.astype(np.intp, copy=False)


def _rounded(colors):
    """Colors as tree_backend.colors_snapshot sends them: float RGBA to 4 uint8."""
    return np.clip(np.rint(np.asarray(colors, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


# --- Reading the key: nothing of it is trusted ---

def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text_ok(stored):
    return isinstance(stored, str) or (isinstance(stored, dict) and isinstance(stored.get("template"), str))


def _entries_ok(entries):
    return isinstance(entries, list) and all(
        isinstance(entry, dict) and _text_ok(entry.get("label")) and isinstance(entry.get("rgba"), list)
        and len(entry["rgba"]) == 4 and all(_number(channel) for channel in entry["rgba"])
        for entry in entries)


def _layer_ok(layer):
    """Whether ``layer`` is well formed: one bad category or extra drops the layer, as its indexes would shift."""
    if not isinstance(layer, dict) or isinstance(layer.get("id"), bool) or not isinstance(layer.get("id"), int):
        return False
    if not _text_ok(layer.get("title")):
        return False
    if layer.get("kind") == "categories":
        return _entries_ok(layer.get("categories")) and isinstance(layer.get("extend", False), bool)
    if layer.get("kind") == "gradient":
        stops = layer.get("stops")
        return (_entries_ok(layer.get("extras")) and isinstance(layer.get("scheme"), str)
                and all(layer.get(name) is None or _number(layer.get(name)) for name in ("min", "max"))
                and isinstance(stops, list)
                and all(isinstance(stop, list) and len(stop) == 3 and all(_number(c) for c in stop) for stop in stops))
    return False


def _key_array(viewer, name, shape, dtype, kinds):
    """The viewer's key array ``name`` as ``dtype`` and ``shape``, or None when it isn't one or doesn't fit.

    A compatible integer or float array of another dtype is converted when no value is lost.
    """
    array = getattr(viewer, name, None)
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.kind not in kinds:
        return None
    if array.dtype == dtype:
        return array
    if np.dtype(dtype).kind == "f":
        return array.astype(dtype)
    info = np.iinfo(dtype)
    if array.size and (array.min() < info.min or array.max() > info.max):
        return None
    return array.astype(dtype)


def _checked_key(viewer):
    """(layers, arrays) of the key as it can be trusted; arrays is (owner, category, rgba, values), or None.

    Layers that are not well formed (or repeat an id) are left out. One warning line is printed, once for the
    Viewer, when a key that was there had to be dropped or trimmed.
    """
    stored = getattr(viewer, "color_key_layers", None)
    candidates = stored if isinstance(stored, list) else []
    layers, seen = [], set()
    for layer in candidates:
        if _layer_ok(layer) and layer["id"] not in seen:
            seen.add(layer["id"])
            layers.append(layer)
    n = getattr(viewer, "n_nodes", 0)
    arrays = tuple(_key_array(viewer, name, shape, dtype, kinds) for name, shape, dtype, kinds in (
        ("color_key_owner", (n,), np.int32, "iu"), ("color_key_category", (n,), np.int32, "iu"),
        ("color_key_rgba", (n, 4), np.uint8, "iu"), ("color_key_values", (n,), np.float32, "iuf")))
    present = [getattr(viewer, name, None) is not None for name in KEY_ATTRIBUTES]
    arrays_ok = all(array is not None for array in arrays)
    damaged = (stored is not None and not isinstance(stored, list)) or len(layers) < len(candidates)
    damaged = damaged or (any(present[1:]) and not arrays_ok) or (arrays_ok and not present[0])
    if damaged and not getattr(viewer, "_color_key_warned", False):
        viewer._color_key_warned = True
        print("Warning: the color key of the loaded layout was damaged; what could not be read was dropped.")
    return layers, (arrays if arrays_ok else None)


def _begin(viewer):
    """Make the viewer's key attributes sound, so that a declaration can add to them.

    A key that cannot be read starts afresh; the layers that can be read stay, and a node owned by a layer
    that is gone (or by none) owns nothing. Returns the layers.
    """
    layers, arrays = _checked_key(viewer)
    n = viewer.n_nodes
    if arrays is None:
        layers = []
        arrays = (np.full(n, -1, np.int32), np.full(n, -1, np.int32), np.zeros((n, 4), np.uint8),
                  np.full(n, np.nan, np.float32))
    owner, category, rgba, values = (array.copy() for array in arrays)
    owner[~np.isin(owner, [layer["id"] for layer in layers])] = -1
    for name, array in zip(KEY_ATTRIBUTES[1:], (owner, category, rgba, values)):
        setattr(viewer, name, array)
    viewer.color_key_layers = layers
    return layers


def _new_layer(viewer, kind, title, **fields):
    """A layer dict with an id above every id this session has used: undo may bring an old key back, never an old id."""
    last = max([getattr(viewer, "_color_key_last_id", 0)] + [layer["id"] for layer in viewer.color_key_layers])
    viewer._color_key_last_id = last + 1
    return {"id": last + 1, "kind": kind, "title": text(title), **fields}


def _paint(viewer, layer, nodes, category, values=None):
    """Let ``layer`` own ``nodes``, as they are painted now."""
    viewer.color_key_owner[nodes] = layer["id"]
    viewer.color_key_category[nodes] = category
    viewer.color_key_rgba[nodes] = _rounded(viewer.current_colors[nodes])
    viewer.color_key_values[nodes] = np.nan if values is None else np.asarray(values, dtype=np.float32)


def _prune(viewer, layers):
    """Keep ``layers`` that own a node."""
    owner = viewer.color_key_owner
    owning = set(np.unique(owner[owner >= 0]).tolist())
    viewer.color_key_layers = [layer for layer in layers if layer["id"] in owning]


def _entry(label, rgba):
    return {"label": text(label), "rgba": [float(channel) for channel in rgba]}


def declare_categories(viewer, title, categories, slots, nodes=None, extend=False):
    """Declare that ``nodes`` (None for all; a mask or node numbers) now show the colors of a categories layer.

    ``categories`` is a list of (label, RGBA) pairs, ``slots`` the category of each of the nodes (indexes into
    it). Title and labels are Message or str. Call it after current_colors holds the new colors.

    With ``extend`` the layer is an extendable one: a later declaration with ``extend`` and the same title adds
    its categories to it, in the order declared, instead of starting a layer. A category with the label and
    color of one the layer has is that one. The layer keeps its id and its place among the layers, and a
    category that no node holds any more is dropped.
    """
    layers = _begin(viewer)
    nodes = _node_indices(viewer, nodes)
    entries = [_entry(label, rgba) for label, rgba in categories]
    slots = np.asarray(slots, dtype=np.int32)
    if extend:
        stored = text(title)
        same = [at for at, layer in enumerate(layers)
                if layer["kind"] == "categories" and layer.get("extend") is True and layer["title"] == stored]
        if same:
            _extend(viewer, layers, same[-1], entries, slots, nodes)
            return
    layer = _new_layer(viewer, "categories", title, categories=entries, **({"extend": True} if extend else {}))
    _paint(viewer, layer, nodes, slots)
    _prune(viewer, layers + [layer])


def _extend(viewer, layers, at, entries, slots, nodes):
    """Paint ``nodes`` as categories ``entries`` of the extendable layer ``layers[at]``, in a new layer dict.

    Nothing the old layer dict or the list of layers holds is changed: an undo step may hold them.
    """
    old = layers[at]
    merged = list(old["categories"])
    place = []
    for entry in entries:
        if entry not in merged:
            merged.append(entry)
        place.append(merged.index(entry))
    _paint(viewer, old, nodes, np.asarray(place, dtype=np.int32)[slots])
    # A category no node holds (a node holds it while it is the layer's and still shows its color) goes, and
    # a node that no longer shows the color its category names is released, so the layer cannot grow.
    owner, category = viewer.color_key_owner, viewer.color_key_category
    mine = (owner == old["id"]) & (category >= 0) & (category < len(merged))
    holding = mine & np.all(viewer.color_key_rgba[:, :3] == _rounded(viewer.current_colors)[:, :3], axis=1)
    released = (owner == old["id"]) & ~holding
    owner[released] = -1
    category[released] = -1
    viewer.color_key_rgba[released] = 0
    viewer.color_key_values[released] = np.nan
    used = np.unique(category[holding])
    remap = np.full(len(merged), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    category[holding] = remap[category[holding]]
    updated = dict(old, categories=[merged[slot] for slot in used.tolist()])
    layers = layers[:at] + [updated] + layers[at + 1:]
    _prune(viewer, layers)


def declare_gradient(viewer, title, scheme, colormap, low, high, nodes, values, extras=()):
    """Declare that ``nodes`` now show a gradient of ``colormap`` for ``values``, which ran from ``low`` to ``high``.

    ``extras`` is a list of (label, RGBA, nodes) for nodes the gradient leaves out, such as those with no
    value; ``nodes`` there is a mask or node numbers. A gradient with no value has low and high None.
    """
    layers = _begin(viewer)
    nodes = _node_indices(viewer, nodes)
    stops = np.asarray(colormap(np.linspace(0.0, 1.0, GRADIENT_STOPS)))[:, :3]
    layer = _new_layer(
        viewer, "gradient", title, scheme=str(scheme),
        min=None if low is None else float(low), max=None if high is None else float(high),
        stops=[[float(channel) for channel in stop] for stop in stops],
        extras=[_entry(label, rgba) for label, rgba, _ in extras],
    )
    _paint(viewer, layer, nodes, -1, values)
    for slot, (_, _, extra_nodes) in enumerate(extras):
        _paint(viewer, layer, _node_indices(viewer, extra_nodes), slot)
    _prune(viewer, layers + [layer])


def clear_key(viewer, nodes=None):
    """Forget what the colors of ``nodes`` (None for all; a mask or node numbers) mean, as a reset of them does."""
    if nodes is None or getattr(viewer, "color_key_owner", None) is None:
        for name in KEY_ATTRIBUTES:
            setattr(viewer, name, None)
        return
    layers = _begin(viewer)
    nodes = _node_indices(viewer, nodes)
    viewer.color_key_owner[nodes] = -1
    viewer.color_key_category[nodes] = -1
    viewer.color_key_rgba[nodes] = 0
    viewer.color_key_values[nodes] = np.nan
    _prune(viewer, layers)


def key_state(viewer):
    """The key as it reads now, as copies; titles and labels are shown in the current language.

    Returns None when the Viewer has no key (or one that cannot be read); otherwise a dict: "layers" (the
    layer dicts, with title and labels as text), "owner", "category", "rgba", "values" (the node arrays), and
    "labelled", a bool per node that is True while the node's current color (its alpha aside) still is the
    one its owner expects. Never raises, whatever a loaded cache put in the key's attributes.
    """
    layers, arrays = _checked_key(viewer)
    colors = getattr(viewer, "current_colors", None)
    n = getattr(viewer, "n_nodes", 0)
    if not layers or arrays is None or not isinstance(colors, np.ndarray) or colors.shape != (n, 4):
        return None
    owner, category, rgba, values = (array.copy() for array in arrays)
    # Alpha is left out: a node that only got fainter or stronger still is the color its row names.
    labelled = np.all(rgba[:, :3] == _rounded(colors)[:, :3], axis=1)
    shown_layers, owned = [], np.zeros(n, dtype=bool)
    for layer in layers:
        layer = copy.deepcopy(layer)
        layer["title"] = shown(layer["title"])
        slots = layer["categories" if layer["kind"] == "categories" else "extras"]
        for entry in slots:
            entry["label"] = shown(entry["label"])
        # A node counts for its layer when its category is one the layer has (a gradient value has none).
        low = 0 if layer["kind"] == "categories" else -1
        owned |= (owner == layer["id"]) & (category >= low) & (category < len(slots))
        shown_layers.append(layer)
    return {"layers": shown_layers, "owner": owner, "category": category, "rgba": rgba, "values": values,
            "labelled": labelled & owned}
