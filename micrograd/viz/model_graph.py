"""Layered view of one MLP sample: weights on edges, values and grads on nodes."""

from __future__ import annotations

import random
from pathlib import Path

import anywidget
import traitlets

from micrograd.engine import Value
from micrograd.nn import MLP, Neuron

_ROW = 46
_COL = 210


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _edge_color(weight: float) -> str:
    # Init is uniform(-1, 1); keep later, larger weights saturated.
    clipped = max(-1.5, min(1.5, weight)) / 1.5
    if clipped >= 0:
        alpha = 0.30 + 0.70 * clipped
        return f"rgba(37, 99, 235, {alpha:.3f})"
    alpha = 0.30 + 0.70 * (-clipped)
    return f"rgba(220, 38, 38, {alpha:.3f})"


def _edge_width(weight: float) -> float:
    return 1.0 + 4.0 * min(abs(weight), 2.0) / 2.0


def _border_color(grad: float) -> str:
    if grad > 1e-8:
        return "#15803d"
    if grad < -1e-8:
        return "#c2410c"
    return "#a8a29e"


def _border_width(grad: float) -> float:
    return 1.5 + 2.5 * min(abs(grad), 1.0)


def _label_num(value: float) -> str:
    if value == 0:
        return "0.00"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


def _grad_scale(values: list[float]) -> float:
    peak = max((abs(value) for value in values), default=0.0)
    return peak if peak > 1e-12 else 1.0


def _grad_paint(grad: float, scale: float) -> tuple[str, float]:
    # Zero stays thin and gray so a closed ReLU reads as "no gradient".
    magnitude = abs(grad)
    if magnitude <= 1e-8:
        return "#a8a29e", 1.0
    ratio = min(magnitude / scale, 1.0)
    width = 1.4 + 4.6 * ratio
    alpha = 0.55 + 0.45 * ratio
    if grad > 0:
        return f"rgba(21, 128, 61, {alpha:.3f})", width
    return f"rgba(194, 65, 12, {alpha:.3f})", width


def clone_model(model: MLP) -> MLP:
    """Copy architecture and parameter data without touching the live net.

    ``MLP`` draws fresh random weights. The RNG is restored so a later
    ``random.seed`` in the notebook still sees the same stream.
    """
    sizes = [len(layer.neurons) for layer in model.layers]
    nin = len(model.layers[0].neurons[0].w)
    state = random.getstate()
    try:
        cloned = MLP(nin, sizes)
    finally:
        random.setstate(state)
    for src, dst in zip(model.parameters(), cloned.parameters(), strict=True):
        dst.data = src.data
        dst.grad = 0.0
    return cloned


def _forward_neuron(neuron: Neuron, inputs: list[Value]):
    # Same left fold as Neuron.__call__: b + w0*x0 + w1*x1 + ...
    terms: list[tuple[Value, Value, Value]] = []
    total = neuron.b
    for weight, incoming in zip(neuron.w, inputs, strict=True):
        product = weight * incoming
        terms.append((weight, incoming, product))
        total = total + product
    output = total.relu() if neuron.nonlin else total
    return output, total, terms


def _column_ys(count: int) -> list[float]:
    span = (count - 1) * _ROW
    return [row * _ROW - span / 2 for row in range(count)]


def _node(node_id: str, kind: str, label: str, x: float, y: float, grad: float, detail: dict) -> dict:
    return {
        "data": {
            "id": node_id,
            "kind": kind,
            "label": label,
            "border_color": _border_color(grad),
            "border_width": _border_width(grad),
            "detail": detail,
        },
        "position": {"x": x, "y": y},
    }


def _weight_edge(source: str, target: str, weight: Value, product: Value, source_name: str, target_name: str) -> dict:
    return {
        "data": {
            "id": f"{source}>{target}",
            "source": source,
            "target": target,
            "kind": "weight",
            "weight": float(weight.data),
            "grad": float(weight.grad),
            "color": _edge_color(weight.data),
            "width": _edge_width(weight.data),
            "detail": {
                "title": f"{source_name} → {target_name}",
                "rows": [
                    ["权重 w", _fmt(weight.data)],
                    ["∂w", _fmt(weight.grad)],
                    ["w·x", _fmt(product.data)],
                    ["∂(w·x)", _fmt(product.grad)],
                ],
                "terms": [],
                "note": "∂w = x · ∂(w·x)。这条边的梯度只来自当前样本的 hinge。",
            },
        }
    }


def view_height(model: MLP) -> int:
    rows = max(
        len(model.layers[0].neurons[0].w),
        *(len(layer.neurons) for layer in model.layers),
    )
    return min(780, max(440, rows * _ROW + 130))


def trace_sample(model: MLP, features, label: float, *, view: str = "forward") -> tuple[list[dict], dict]:
    """Forward one row, backward its hinge, and return Cytoscape elements.

    ``view="forward"`` puts the activation in each node and the weight on each
    edge. ``view="backward"`` puts the gradient in each node and draws
    ``d(hinge)/dw`` on edges that point back toward the inputs.

    Grads are ``d(hinge)/d(parameter)`` for this row only. The caller's model
    is left unchanged.
    """
    if view not in {"forward", "backward"}:
        raise ValueError(f"view must be 'forward' or 'backward', got {view!r}")
    cloned = clone_model(model)
    inputs = [Value(float(value)) for value in features]
    layer_traces = []
    current = inputs
    for layer in cloned.layers:
        outputs = []
        traced = []
        for neuron in layer.neurons:
            output, pre, terms = _forward_neuron(neuron, current)
            outputs.append(output)
            traced.append((neuron, output, pre, terms))
        layer_traces.append(traced)
        current = outputs

    score = current[0]
    hinge = (1 + -float(label) * score).relu()
    cloned.zero_grad()
    hinge.backward()

    elements: list[dict] = []
    input_ids = [f"x{index}" for index in range(len(inputs))]
    input_names = input_ids[:]
    x_pos = 0.0
    for node_id, value, y in zip(input_ids, inputs, _column_ys(len(inputs)), strict=True):
        elements.append(
            _node(
                node_id,
                "input",
                f"{node_id}\n{_fmt(value.data, 2)}",
                x_pos,
                y,
                value.grad,
                {
                    "title": node_id,
                    "rows": [
                        ["值", _fmt(value.data)],
                        ["梯度", _fmt(value.grad)],
                    ],
                    "terms": [],
                    "note": "输入没有参数。梯度是 hinge 对这个坐标的导数。",
                },
            )
        )

    previous_ids = input_ids
    previous_names = input_names
    for layer_index, traced in enumerate(layer_traces):
        x_pos += _COL
        last = layer_index == len(layer_traces) - 1
        ids: list[str] = []
        names: list[str] = []
        for neuron_index, (neuron, output, pre, terms) in enumerate(traced):
            if last and len(traced) == 1:
                name = "score"
                node_id = "score"
                kind = "linear"
            else:
                name = f"h{layer_index}.{neuron_index}"
                node_id = name
                kind = "relu" if neuron.nonlin else "linear"
            ids.append(node_id)
            names.append(name)
            op = "ReLU(b + Σ w·x)" if neuron.nonlin else "b + Σ w·x"
            note = "∂w = x · ∂(w·x)。"
            if neuron.nonlin:
                note += " ReLU 在预激活 ≤ 0 时关掉，后面的梯度就是 0。"
            elements.append(
                _node(
                    node_id,
                    kind,
                    f"{name}\n{_fmt(output.data, 2)}",
                    x_pos,
                    _column_ys(len(traced))[neuron_index],
                    output.grad,
                    {
                        "title": f"{name} · {op}",
                        "rows": [
                            ["输出", _fmt(output.data)],
                            ["∂输出", _fmt(output.grad)],
                            ["预激活", _fmt(pre.data)],
                            ["∂预激活", _fmt(pre.grad)],
                            ["偏置 b", _fmt(neuron.b.data)],
                            ["∂b", _fmt(neuron.b.grad)],
                        ],
                        "terms": [
                            {
                                "i": str(term_index),
                                "src": previous_names[term_index],
                                "x": _fmt(incoming.data),
                                "w": _fmt(weight.data),
                                "prod": _fmt(product.data),
                                "w_grad": _fmt(weight.grad),
                            }
                            for term_index, (weight, incoming, product) in enumerate(terms)
                        ],
                        "note": note,
                    },
                )
            )
            for term_index, (weight, _incoming, product) in enumerate(terms):
                elements.append(
                    _weight_edge(
                        previous_ids[term_index],
                        node_id,
                        weight,
                        product,
                        previous_names[term_index],
                        name,
                    )
                )
        previous_ids = ids
        previous_names = names

    x_pos += _COL
    elements.append(
        _node(
            "hinge",
            "loss",
            f"hinge\n{_fmt(hinge.data, 2)}",
            x_pos,
            0.0,
            hinge.grad,
            {
                "title": "hinge · ReLU(1 − y · score)",
                "rows": [
                    ["y", f"{float(label):+.0f}"],
                    ["score", _fmt(score.data)],
                    ["hinge", _fmt(hinge.data)],
                    ["∂hinge", _fmt(hinge.grad)],
                ],
                "terms": [],
                "note": "反向从这个标量开始。它不是网络参数。",
            },
        )
    )
    elements.append(
        {
            "data": {
                "id": "score>hinge",
                "source": "score",
                "target": "hinge",
                "kind": "flow",
                "weight": 0.0,
                "grad": 0.0,
                "color": "#78716c",
                "width": 2.0,
                "detail": {
                    "title": "score → hinge",
                    "rows": [
                        ["y", f"{float(label):+.0f}"],
                        ["score", _fmt(score.data)],
                        ["hinge", _fmt(hinge.data)],
                    ],
                    "terms": [],
                    "note": "hinge = ReLU(1 − y · score)。虚线不是权重。",
                },
            }
        }
    )

    summary = {
        "x": ", ".join(_fmt(value.data, 2) for value in inputs),
        "y": f"{float(label):+.0f}",
        "score": _fmt(score.data, 3),
        "hinge": _fmt(hinge.data, 3),
    }
    if view == "backward":
        elements = _backward_elements(layer_traces, inputs, score, hinge, float(label))
        summary["score_grad"] = _fmt(score.grad, 3)
    return elements, summary


def _backward_elements(layer_traces, inputs, score: Value, hinge: Value, label: float) -> list[dict]:
    """Same columns as the forward view, with arrows running loss → inputs."""
    weight_grads = [
        weight.grad
        for traced in layer_traces
        for _neuron, _output, _pre, terms in traced
        for weight, _incoming, _product in terms
    ]
    scale = _grad_scale([*weight_grads, score.grad])
    elements: list[dict] = []
    input_ids = [f"x{index}" for index in range(len(inputs))]
    input_names = input_ids[:]
    x_pos = 0.0
    for node_id, value, y in zip(input_ids, inputs, _column_ys(len(inputs)), strict=True):
        elements.append(
            _node(
                node_id,
                "input",
                f"{node_id}\n{_label_num(value.grad)}",
                x_pos,
                y,
                value.grad,
                {
                    "title": f"{node_id} 的梯度",
                    "rows": [
                        ["梯度", _fmt(value.grad)],
                        ["值", _fmt(value.data)],
                    ],
                    "terms": [],
                    "note": "输入没有参数。这个梯度是后面每个神经元贡献的 w·∂(w·x) 之和。",
                },
            )
        )

    previous_ids = input_ids
    previous_names = input_names
    for layer_index, traced in enumerate(layer_traces):
        x_pos += _COL
        last = layer_index == len(layer_traces) - 1
        ids: list[str] = []
        names: list[str] = []
        for neuron_index, (neuron, output, pre, terms) in enumerate(traced):
            if last and len(traced) == 1:
                name = "score"
                node_id = "score"
                kind = "linear"
            else:
                name = f"h{layer_index}.{neuron_index}"
                node_id = name
                kind = "relu" if neuron.nonlin else "linear"
            ids.append(node_id)
            names.append(name)
            elements.append(
                _node(
                    node_id,
                    kind,
                    f"{name}\n{_label_num(output.grad)}",
                    x_pos,
                    _column_ys(len(traced))[neuron_index],
                    output.grad,
                    {
                        "title": f"{name} 的梯度",
                        "rows": [
                            ["∂输出", _fmt(output.grad)],
                            ["输出", _fmt(output.data)],
                            ["∂预激活", _fmt(pre.grad)],
                            ["预激活", _fmt(pre.data)],
                            ["∂b", _fmt(neuron.b.grad)],
                            ["b", _fmt(neuron.b.data)],
                        ],
                        "columns": [
                            ["src", "来自"],
                            ["x", "x"],
                            ["d_prod", "∂(w·x)"],
                            ["w_grad", "∂w"],
                            ["d_x", "贡献 ∂x"],
                        ],
                        "terms": [
                            {
                                "src": previous_names[term_index],
                                "x": _fmt(incoming.data),
                                "d_prod": _fmt(product.grad),
                                "w_grad": _fmt(weight.grad),
                                "d_x": _fmt(weight.data * product.grad),
                            }
                            for term_index, (weight, incoming, product) in enumerate(terms)
                        ],
                        "note": _neuron_backward_note(neuron, pre),
                    },
                )
            )
            for term_index, (weight, incoming, product) in enumerate(terms):
                elements.append(
                    _grad_edge(
                        node_id,
                        previous_ids[term_index],
                        weight.grad,
                        scale,
                        {
                            "title": f"{name} ← {previous_names[term_index]}",
                            "rows": [
                                ["x", _fmt(incoming.data)],
                                ["w", _fmt(weight.data)],
                                ["∂(w·x)", _fmt(product.grad)],
                                ["∂w", _fmt(weight.grad)],
                                ["这条边贡献的 ∂x", _fmt(weight.data * product.grad)],
                            ],
                            "terms": [],
                            "note": _edge_backward_note(neuron, pre),
                        },
                    )
                )
        previous_ids = ids
        previous_names = names

    x_pos += _COL
    elements.append(
        _node(
            "hinge",
            "loss",
            f"hinge\n{_label_num(hinge.grad)}",
            x_pos,
            0.0,
            hinge.grad,
            {
                "title": "hinge 的梯度",
                "rows": [
                    ["∂hinge", _fmt(hinge.grad)],
                    ["hinge", _fmt(hinge.data)],
                    ["y", f"{label:+.0f}"],
                    ["score", _fmt(score.data)],
                    ["∂score", _fmt(score.grad)],
                ],
                "terms": [],
                "note": "backward() 从这里开始，先把 ∂hinge 设成 1。hinge = ReLU(1 − y·score)。打开时 ∂score = −y，关掉时 ∂score = 0。",
            },
        )
    )
    color, width = _grad_paint(score.grad, scale)
    elements.append(
        {
            "data": {
                "id": "hinge<score",
                "source": "hinge",
                "target": "score",
                "kind": "flow",
                "weight": 0.0,
                "grad": float(score.grad),
                "color": color,
                "width": width,
                "detail": {
                    "title": "hinge → score",
                    "rows": [
                        ["y", f"{label:+.0f}"],
                        ["score", _fmt(score.data)],
                        ["∂score", _fmt(score.grad)],
                        ["hinge", _fmt(hinge.data)],
                    ],
                    "terms": [],
                    "note": "虚线不是权重，是损失对 score 的梯度。打开时 ∂score = −y。",
                },
            }
        }
    )
    return elements


def _neuron_backward_note(neuron: Neuron, pre: Value) -> str:
    note = "∂b = ∂预激活。每一行 ∂w = x · ∂(w·x)，∂(w·x) 等于这一层的 ∂预激活。"
    if neuron.nonlin and pre.data <= 0:
        return note + " 这条 ReLU 的预激活 ≤ 0，局部导数是 0，梯度在这里被截断。"
    if neuron.nonlin:
        return note + " 这条 ReLU 开着，∂预激活 = ∂输出。"
    return note + " 这一层是线性的，没有 ReLU 门。"


def _edge_backward_note(neuron: Neuron, pre: Value) -> str:
    note = "∂w = x · ∂(w·x)。这条边贡献的 ∂x = w · ∂(w·x)。箭头指向梯度流回去的那边。"
    if neuron.nonlin and pre.data <= 0:
        return note + " 上游 ReLU 关着，所以这些都是 0。"
    return note


def _grad_edge(source: str, target: str, grad: float, scale: float, detail: dict) -> dict:
    color, width = _grad_paint(grad, scale)
    return {
        "data": {
            "id": f"{source}<{target}",
            "source": source,
            "target": target,
            "kind": "grad",
            "weight": 0.0,
            "grad": float(grad),
            "color": color,
            "width": width,
            "detail": detail,
        }
    }


class ModelGraph(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "model_graph.js"
    _css = Path(__file__).parent / "model_graph.css"

    elements = traitlets.List().tag(sync=True)
    summary = traitlets.Dict().tag(sync=True)
    height = traitlets.Int(640).tag(sync=True)
    # "forward" draws weights. "backward" draws d(hinge)/dw pointing upstream.
    mode = traitlets.Unicode("forward").tag(sync=True)
