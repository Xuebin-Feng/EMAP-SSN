# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Colors the Viewer's commands share, so that one meaning looks the same whichever command paints it.

This module is not shadowed by the VR viewer, which re-exports Command_Engine's names one by one, so
the commands the two viewers share import such constants from here.
"""

import numpy as np

from utilities.Localization import Message, translate

# RGBA of a node with no value to color it by or no group to belong to: translucent light gray.
NOISE_COLOR = (0.8, 0.8, 0.8, 0.4)


# ---------------------------------------------------------------------
# The color key: what the node colors mean
# ---------------------------------------------------------------------
# A command that paints nodes also declares what its colors mean, right after it writes
# viewer.current_colors and inside the same undo step. The record lives in cacheable Viewer
# attributes (EMAPSSN_Viewer.CUSTOM_ATTRIBUTES_INIT), so undo, redo, save and load carry it:
#
#   color_key_layers    list of JSON-safe dicts, oldest first. A categories layer is {"id", "kind":
#                       "categories", "title", "categories": [{"label", "rgba"}, ...]}; a gradient layer is
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


def shown(stored):
    """A stored title or label as read now: a stored Message in the current language, text as it is."""
    if isinstance(stored, dict):
        template, values = stored["template"], stored.get("values", {})
        try:
            return translate(template).format(**values)
        except (KeyError, IndexError, ValueError):
            return template.format(**values)  # a translation whose fields don't match: show the English
    return str(stored)


def _node_indices(viewer, nodes):
    """``nodes`` (None for all, a bool mask or node numbers) as an int array of node numbers."""
    if nodes is None:
        return np.arange(viewer.n_nodes)
    nodes = np.atleast_1d(nodes)
    return np.flatnonzero(nodes) if nodes.dtype == bool else nodes.astype(np.intp, copy=False)


def _rounded(colors):
    """Colors as tree_backend.colors_snapshot sends them: float RGBA to 4 uint8."""
    return np.clip(np.rint(np.asarray(colors, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _writable(viewer, name, shape, dtype, fill):
    """The viewer's key array ``name``, made fresh when it is missing, of another size, or read-only."""
    array = getattr(viewer, name, None)
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
        array = np.full(shape, fill, dtype=dtype)
    elif not array.flags.writeable:
        array = array.copy()
    setattr(viewer, name, array)
    return array


def _key_layers(viewer):
    layers = getattr(viewer, "color_key_layers", None)
    return layers if isinstance(layers, list) else []


def _new_layer(viewer, kind, title, **fields):
    """A layer dict with an id above every id this session has used: undo may bring an old key back, never an old id."""
    last = max([getattr(viewer, "_color_key_last_id", 0)] + [layer["id"] for layer in _key_layers(viewer)])
    viewer._color_key_last_id = last + 1
    return {"id": last + 1, "kind": kind, "title": text(title), **fields}


def _paint(viewer, layer, nodes, category, values=None):
    """Let ``layer`` (the key's newest from now on) own ``nodes``, as they are painted now."""
    n = viewer.n_nodes
    owner = _writable(viewer, "color_key_owner", (n,), np.int32, -1)
    category_of = _writable(viewer, "color_key_category", (n,), np.int32, -1)
    rgba = _writable(viewer, "color_key_rgba", (n, 4), np.uint8, 0)
    held = _writable(viewer, "color_key_values", (n,), np.float32, np.nan)
    owner[nodes] = layer["id"]
    category_of[nodes] = category
    rgba[nodes] = _rounded(viewer.current_colors[nodes])
    held[nodes] = np.nan if values is None else np.asarray(values, dtype=np.float32)


def _prune(viewer, layers):
    """Keep ``layers`` that own a node."""
    owner = viewer.color_key_owner
    owning = set(np.unique(owner[owner >= 0]).tolist())
    viewer.color_key_layers = [layer for layer in layers if layer["id"] in owning]


def _entry(label, rgba):
    return {"label": text(label), "rgba": [float(channel) for channel in rgba]}


def declare_categories(viewer, title, categories, slots, nodes=None):
    """Declare that ``nodes`` (None for all; a mask or node numbers) now show the colors of a categories layer.

    ``categories`` is a list of (label, RGBA) pairs, ``slots`` the category of each of the nodes (indexes into
    it). Title and labels are Message or str. Call it after current_colors holds the new colors.
    """
    nodes = _node_indices(viewer, nodes)
    layer = _new_layer(viewer, "categories", title, categories=[_entry(label, rgba) for label, rgba in categories])
    _paint(viewer, layer, nodes, np.asarray(slots, dtype=np.int32))
    _prune(viewer, _key_layers(viewer) + [layer])


def declare_gradient(viewer, title, scheme, colormap, low, high, nodes, values, extras=()):
    """Declare that ``nodes`` now show a gradient of ``colormap`` for ``values``, which ran from ``low`` to ``high``.

    ``extras`` is a list of (label, RGBA, nodes) for nodes the gradient leaves out, such as those with no
    value; ``nodes`` there is a mask or node numbers. A gradient with no value has low and high None.
    """
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
    _prune(viewer, _key_layers(viewer) + [layer])


def clear_key(viewer, nodes=None):
    """Forget what the colors of ``nodes`` (None for all; a mask or node numbers) mean, as a reset of them does."""
    if nodes is None or not isinstance(getattr(viewer, "color_key_owner", None), np.ndarray):
        for name in KEY_ATTRIBUTES:
            setattr(viewer, name, None)
        return
    nodes = _node_indices(viewer, nodes)
    n = viewer.n_nodes
    owner = _writable(viewer, "color_key_owner", (n,), np.int32, -1)
    owner[nodes] = -1
    _writable(viewer, "color_key_category", (n,), np.int32, -1)[nodes] = -1
    _writable(viewer, "color_key_rgba", (n, 4), np.uint8, 0)[nodes] = 0
    _writable(viewer, "color_key_values", (n,), np.float32, np.nan)[nodes] = np.nan
    _prune(viewer, _key_layers(viewer))


def key_state(viewer):
    """The key as it reads now, as copies; titles and labels are shown in the current language.

    Returns None when the Viewer has no key; otherwise a dict: "layers" (the layer dicts, with title and
    labels as text), "owner", "category", "rgba", "values" (the node arrays), and "labelled", a bool per node
    that is True while the node's current color still is the one its owner expects.
    """
    layers, colors = _key_layers(viewer), getattr(viewer, "current_colors", None)
    n = getattr(viewer, "n_nodes", 0)
    arrays = [getattr(viewer, name, None) for name in KEY_ATTRIBUTES[1:]]
    if not layers or colors is None or any(not isinstance(a, np.ndarray) or a.shape[0] != n for a in arrays):
        return None
    owner, category, rgba, values = (a.copy() for a in arrays)
    ids = [layer["id"] for layer in layers]
    labelled = np.isin(owner, ids) & np.all(rgba == _rounded(colors), axis=1)
    shown_layers = []
    for layer in layers:
        layer = {**layer, "title": shown(layer["title"])}
        for name in ("categories", "extras"):
            if name in layer:
                layer[name] = [{**entry, "label": shown(entry["label"])} for entry in layer[name]]
        shown_layers.append(layer)
    return {"layers": shown_layers, "owner": owner, "category": category, "rgba": rgba, "values": values,
            "labelled": labelled}
