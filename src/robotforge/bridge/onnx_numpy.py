"""Run small ONNX policies (MLPs) with numpy alone.

Learned walkers are tiny feed-forward networks, so robotforge reads the ONNX file itself (a protobuf,
decoded here without the protobuf or onnx packages) and evaluates the handful of operators an MLP uses.
onnxruntime stays an optional extra for anything bigger; `NumpyOnnx` says clearly when a model needs it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ONNX TensorProto data types we can hold
_DTYPES = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}


class UnsupportedModel(ValueError):
    pass


# ---- protobuf wire format ---------------------------------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes) -> list[tuple[int, int, Any]]:
    """(field number, wire type, value) for every field of one message; nested messages stay bytes."""
    out = []
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value = buf[i : i + 8]
            i += 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value = buf[i : i + length]
            i += length
        elif wire == 5:
            value = buf[i : i + 4]
            i += 4
        else:
            raise UnsupportedModel(f"unexpected protobuf wire type {wire}")
        out.append((number, wire, value))
    return out


def _packed_varints(value: bytes) -> list[int]:
    out, i = [], 0
    while i < len(value):
        v, i = _varint(value, i)
        out.append(v)
    return out


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


# ---- ONNX messages ------------------------------------------------------------------------------


def _tensor(buf: bytes) -> tuple[str, np.ndarray]:
    dims: list[int] = []
    dtype = 1
    name = ""
    raw = None
    floats: list[float] = []
    ints: list[int] = []
    for number, wire, value in _fields(buf):
        if number == 1:
            dims.extend(_packed_varints(value) if wire == 2 else [value])
        elif number == 2:
            dtype = value
        elif number == 4:  # float_data
            floats.extend(struct.unpack(f"<{len(value) // 4}f", value) if wire == 2 else struct.unpack("<f", value))
        elif number in (5, 7):  # int32_data, int64_data
            ints.extend(_signed(v) for v in (_packed_varints(value) if wire == 2 else [value]))
        elif number == 8:
            name = value.decode()
        elif number == 9:
            raw = value
    if dtype not in _DTYPES:
        raise UnsupportedModel(f"tensor {name!r} has ONNX data type {dtype}")
    np_type = _DTYPES[dtype]
    if raw is not None:
        array = np.frombuffer(raw, dtype=np.dtype(np_type).newbyteorder("<")).astype(np_type)
    elif floats:
        array = np.array(floats, dtype=np_type)
    else:
        array = np.array(ints, dtype=np_type)
    return name, array.reshape(dims) if dims else array.reshape(())


def _attribute(buf: bytes) -> tuple[str, Any]:
    name, value = "", None
    floats: list[float] = []
    ints: list[int] = []
    for number, wire, v in _fields(buf):
        if number == 1:
            name = v.decode()
        elif number == 2:  # f
            value = struct.unpack("<f", v)[0]
        elif number == 3:  # i
            value = _signed(v)
        elif number == 4:  # s
            value = v.decode(errors="replace")
        elif number == 5:  # t
            value = _tensor(v)[1]
        elif number == 7:  # floats
            floats.extend(struct.unpack(f"<{len(v) // 4}f", v) if wire == 2 else struct.unpack("<f", v))
        elif number == 8:  # ints
            ints.extend(_signed(x) for x in (_packed_varints(v) if wire == 2 else [v]))
    if value is None:
        value = floats or ints
    return name, value


@dataclass
class Node:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


def _node(buf: bytes) -> Node:
    node = Node(op="", inputs=[], outputs=[])
    for number, _, value in _fields(buf):
        if number == 1:
            node.inputs.append(value.decode())
        elif number == 2:
            node.outputs.append(value.decode())
        elif number == 4:
            node.op = value.decode()
        elif number == 5:
            key, attr = _attribute(value)
            node.attrs[key] = attr
    return node


def _value_info(buf: bytes) -> tuple[str, list[int | None]]:
    """Name and shape of a graph input or output (None for symbolic sizes like the batch)."""
    name, dims = "", []
    for number, _, value in _fields(buf):
        if number == 1:
            name = value.decode()
        elif number == 2:  # TypeProto
            tensor = next((v for n, _, v in _fields(value) if n == 1), b"")
            shape = next((v for n, _, v in _fields(tensor) if n == 2), b"")
            for n, _, dim in _fields(shape):
                if n == 1:
                    size = next((v for k, w, v in _fields(dim) if k == 1 and w == 0), None)
                    dims.append(size)
    return name, dims


# ---- evaluation ---------------------------------------------------------------------------------


def _gemm(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0):  # noqa: N803 (ONNX attribute names)
    a = a.T if transA else a
    b = b.T if transB else b
    out = alpha * (a @ b)
    return out if c is None else out + beta * c


OPS = {
    "Gemm": lambda x, a: _gemm(*x, **a),
    "MatMul": lambda x, a: x[0] @ x[1],
    "Add": lambda x, a: x[0] + x[1],
    "Sub": lambda x, a: x[0] - x[1],
    "Mul": lambda x, a: x[0] * x[1],
    "Div": lambda x, a: x[0] / x[1],
    "Relu": lambda x, a: np.maximum(x[0], 0),
    "LeakyRelu": lambda x, a: np.where(x[0] > 0, x[0], a.get("alpha", 0.01) * x[0]),
    "Elu": lambda x, a: np.where(x[0] > 0, x[0], a.get("alpha", 1.0) * np.expm1(np.minimum(x[0], 0))),
    "Selu": lambda x, a: (
        a.get("gamma", 1.0507) * np.where(x[0] > 0, x[0], a.get("alpha", 1.67326) * np.expm1(np.minimum(x[0], 0)))
    ),
    "Tanh": lambda x, a: np.tanh(x[0]),
    "Sigmoid": lambda x, a: 1 / (1 + np.exp(-x[0])),
    "Softplus": lambda x, a: np.logaddexp(0, x[0]),
    "Identity": lambda x, a: x[0],
    "Neg": lambda x, a: -x[0],
    "Clip": lambda x, a: np.clip(
        x[0],
        x[1] if len(x) > 1 and x[1] is not None else a.get("min", -np.inf),
        x[2] if len(x) > 2 and x[2] is not None else a.get("max", np.inf),
    ),
    "Flatten": lambda x, a: x[0].reshape(x[0].shape[0] if a.get("axis", 1) else 1, -1),
    "Reshape": lambda x, a: x[0].reshape([int(d) for d in x[1]]),
    "Constant": lambda x, a: a["value"],
}


class NumpyOnnx:
    """A feed-forward ONNX graph evaluated with numpy. Raises UnsupportedModel for anything else."""

    def __init__(self, path: str | Path) -> None:
        data = Path(path).read_bytes()
        graph = next((v for n, _, v in _fields(data) if n == 7), None)
        if graph is None:
            raise UnsupportedModel(f"{path} has no graph")
        self.nodes: list[Node] = []
        self.weights: dict[str, np.ndarray] = {}
        inputs: list[tuple[str, list[int | None]]] = []
        outputs: list[tuple[str, list[int | None]]] = []
        for number, _, value in _fields(graph):
            if number == 1:
                self.nodes.append(_node(value))
            elif number == 5:
                name, array = _tensor(value)
                self.weights[name] = array
            elif number == 11:
                inputs.append(_value_info(value))
            elif number == 12:
                outputs.append(_value_info(value))
        self.shapes = {name: dims for name, dims in inputs + outputs}
        self.inputs = [name for name, _ in inputs if name not in self.weights]
        self.outputs = [name for name, _ in outputs]
        unknown = sorted({node.op for node in self.nodes} - OPS.keys())
        if unknown:
            raise UnsupportedModel(f"operators {', '.join(unknown)} need onnxruntime")
        if len(self.inputs) != 1:
            raise UnsupportedModel(f"expected one input, found {self.inputs}")

    @property
    def ops(self) -> list[str]:
        return [node.op for node in self.nodes]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        values: dict[str, Any] = dict(self.weights)
        values[self.inputs[0]] = np.asarray(obs, dtype=np.float32)
        for node in self.nodes:
            args = [values.get(name) if name else None for name in node.inputs]
            result = OPS[node.op](args, node.attrs)
            values[node.outputs[0]] = np.asarray(result, dtype=np.float32)
        return values[self.outputs[0]]
