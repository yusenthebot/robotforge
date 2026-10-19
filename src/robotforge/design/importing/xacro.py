"""A small xacro expander, so beginners do not need ROS (or the `xacro` package) to open robots.

Supported: properties (lazy, with block properties), args, macros (with defaults, `*block`
parameters and `insert_block`), include, if/unless, `${...}` formulas (a safe arithmetic
subset with math functions) and the `$(find|arg|env|optenv|dirname|eval ...)` substitutions.
Not supported: other Python helpers, dynamic macro names and `xacro:element`. Unsupported
constructs raise XacroError saying which one, so the user knows what to expand by hand.

xacro files come from anywhere, so expansion is bounded: formulas cannot build huge numbers,
strings or lists, and one expansion has a budget of elements, output size and time.
"""

from __future__ import annotations

import ast
import copy
import functools
import math
import operator
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robotforge.design.importing.resolve import MeshResolver

_XACRO_NS = "http://www.ros.org/wiki/xacro"
_MAX_DEPTH = 80
_MAX_FILE_BYTES = 5_000_000
# One expansion may not exceed these, however a file tries to multiply itself.
_MAX_NODES = 100_000  # elements and macro calls processed (the largest real robots I tried need ~1,200)
_MAX_OUTPUT_CHARS = 5_000_000
_MAX_SECONDS = 15.0
# What a single `${...}` formula may produce.
_MAX_SEQ = 100_000  # characters in a string, items in a list
_MAX_INT_BITS = 8192
_MAX_FORMULA_NODES = 2000
_MAX_YAML_NODES = 500_000
_KNOWN_DIRECTIVES = frozenset({"element", "attribute", "call", "include_all", "eval", "arg_scope"})
_GROUP = "_group"  # holds the children passed to a `**name` macro parameter
_SUBST = re.compile(r"\$\$|\$\{([^}]*)\}|\$\(([^)]*)\)")
_SAFE_FUNCTIONS: dict[str, Any] = {
    **{n: getattr(math, n) for n in ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "exp", "log")},
    **{
        n: getattr(math, n)
        for n in ("floor", "ceil", "fabs", "hypot", "radians", "degrees", "copysign", "sinh", "cosh")
    },
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "sum": sum,
    "pow": lambda a, b: _binop(ast.Pow(), a, b, "pow"),
}
_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "True": True,
    "False": False,
}
_STR_METHODS = frozenset(
    {
        "split",
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "replace",
        "startswith",
        "endswith",
        "find",
        "count",
        "join",
    }
)
_DICT_METHODS = frozenset({"get", "keys", "values", "items"})
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class XacroError(ValueError):
    """The xacro file uses something the built-in expander cannot handle."""


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float)


def _bits(x: int) -> int:
    return abs(x).bit_length()


def _binop(op: ast.operator, left: Any, right: Any, expr: str) -> Any:
    """An arithmetic operator that refuses to build huge numbers, strings or lists."""
    too_big = XacroError(f"the formula '${{{expr}}}' would build something far too large; formulas are kept small")
    both_numbers = _is_number(left) and _is_number(right)
    both_ints = isinstance(left, int) and isinstance(right, int)
    if isinstance(op, ast.Pow):
        if not both_numbers:
            raise XacroError(f"'**' works on numbers only, in '${{{expr}}}'")
        if both_ints and (abs(right) > 10_000 or (right > 0 and _bits(left) * right > _MAX_INT_BITS)):
            raise too_big
        result = operator.pow(left, right)
        if isinstance(result, complex):
            raise XacroError(f"'${{{expr}}}' has no real value")
        return result
    if isinstance(op, ast.Mult):
        if both_ints and _bits(left) + _bits(right) > _MAX_INT_BITS:
            raise too_big
        for seq, count in ((left, right), (right, left)):
            if isinstance(seq, str | list | tuple) and isinstance(count, int) and len(seq) * max(count, 0) > _MAX_SEQ:
                raise too_big
        return operator.mul(left, right)
    if isinstance(op, ast.Add):
        both_sequences = isinstance(left, str | list | tuple) and isinstance(right, str | list | tuple)
        if both_sequences and len(left) + len(right) > _MAX_SEQ:
            raise too_big
        return operator.add(left, right)
    if isinstance(op, ast.Mod):
        if not both_numbers:  # `'%99999999s' % 1` would build a huge string
            raise XacroError(f"'%' works on numbers only, in '${{{expr}}}'")
        return operator.mod(left, right)
    return _BINOPS[type(op)](left, right)


def _safe_replace(text: str, old: str, new: str, count: int = -1) -> str:
    if not old or text.count(old) * len(new) + len(text) > _MAX_SEQ:
        raise XacroError("str.replace in a formula would build a string that is far too large")
    return text.replace(old, new, count)


def parse_text(text: str, label: str) -> ET.Element:
    """Parse URDF text, tolerating prefixed tags that the file never declared (raises ValueError)."""
    root = _parse(text, label)
    assert root is not None
    return root


def needs_expansion(text: str) -> bool:
    """True when a URDF still contains xacro constructs in the parts robotforge reads."""
    root = _parse(text, "this file", tolerant=True)
    if root is None:  # not well-formed: let the expander produce the error message
        return True
    if any(_is_xacro(el.tag) for el in root.iter()):
        return True
    for tag in ("link", "joint", "material"):
        for top in root.findall(tag):
            for el in top.iter():
                if any("${" in v or "$(" in v for v in el.attrib.values()):
                    return True
    return False


@dataclass
class Expansion:
    """The expanded URDF text, every file that was read, and macros that were called but never defined."""

    text: str
    files: list[Path]
    undefined_macros: list[str]


def expand(path: Path, args: dict[str, str] | None = None) -> Expansion:
    """Expand `path` into plain URDF text."""
    ex = _Expander(path, args or {})
    try:
        root = ex.run()
        text = ET.tostring(root, encoding="unicode")
    except RecursionError as exc:
        raise XacroError(f"{path.name}: macros or elements are nested too deeply") from exc
    if len(text) > _MAX_OUTPUT_CHARS:
        raise XacroError(f"{path.name}: the expanded robot description is far too large")
    return Expansion(text, ex.files, sorted(ex.undefined))


# ---- xml helpers -----------------------------------------------------------------------------


def _is_xacro(tag: str) -> bool:
    return tag.startswith("{") and "xacro" in tag.split("}", 1)[0].lower()


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse(text: str, label: str, tolerant: bool = False) -> ET.Element | None:
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        if "unbound prefix" in str(exc):  # prefixed tags (xacro:, sensor:, ...) used without declaring them
            patched = _declare_prefixes(text)
            try:
                return ET.fromstring(patched)
            except ET.ParseError as exc2:
                exc = exc2
        if tolerant:
            return None
        raise XacroError(f"{label} is not valid XML ({exc})") from exc


def _declare_prefixes(text: str) -> str:
    """Add an xmlns declaration to <robot> for every namespace prefix the file uses but never declares."""
    used = set(re.findall(r"</?([A-Za-z_][\w.-]*):[\w.-]", text)) - {"xml", "xmlns"}
    declared = set(re.findall(r"xmlns:([\w.-]+)=", text))
    extra = " ".join(
        f'xmlns:{p}="{_XACRO_NS if p == "xacro" else "urn:robotforge:undeclared:" + p}"'
        for p in sorted(used - declared)
    )
    return re.sub(r"<robot\b", f"<robot {extra}", text, count=1) if extra else text


# ---- safe expressions ------------------------------------------------------------------------


def _evaluate(expr: str, lookup, extra: dict[str, Any] | None = None) -> Any:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise XacroError(f"cannot read the formula '${{{expr}}}'") from exc
    except (RecursionError, MemoryError, ValueError) as exc:
        raise XacroError(f"the formula '${{{expr.strip()[:60]}...}}' is too deeply nested or too large") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_FORMULA_NODES:
        raise XacroError(f"the formula '${{{expr.strip()[:60]}...}}' is too long")
    try:
        return _eval_node(tree.body, lookup, expr, extra or {})
    except XacroError:
        raise
    except RecursionError as exc:
        raise XacroError(f"the formula '${{{expr.strip()[:60]}...}}' is too deeply nested") from exc
    except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError, AttributeError, OverflowError) as exc:
        raise XacroError(f"the formula '${{{expr}}}' failed ({type(exc).__name__}: {exc})") from exc


class _XacroYamlLoader(yaml.SafeLoader):
    """Safe YAML with xacro's `!degrees` and `!radians` number tags."""


def _angle_constructor(to_radians: bool):
    def construct(loader: yaml.SafeLoader, node: yaml.Node) -> float:
        text = loader.construct_scalar(node)  # type: ignore[arg-type]
        value = float(_evaluate(str(text), lambda n: _SAFE_NAMES[n]))
        return math.radians(value) if to_radians else value

    return construct


_XacroYamlLoader.add_constructor("!degrees", _angle_constructor(True))
_XacroYamlLoader.add_constructor("!radians", _angle_constructor(False))


class _Yaml(dict):
    """YAML data loaded by load_yaml; keys are also readable as attributes, like xacro's wrapper."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value: Any, budget: list[int] | None = None) -> Any:
    """Wrap loaded YAML for attribute access. YAML aliases can make a tiny file describe an enormous tree, so
    the number of values visited is bounded."""
    budget = budget if budget is not None else [_MAX_YAML_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise XacroError("load_yaml: the YAML expands into far too many values (aliases that multiply themselves?)")
    if isinstance(value, dict):
        return _Yaml({k: _wrap(v, budget) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v, budget) for v in value]
    return value


def _eval_node(node: ast.AST, lookup, expr: str, extra: dict[str, Any]) -> Any:
    def ev(n: ast.AST) -> Any:
        return _eval_node(n, lookup, expr, extra)

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return lookup(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _binop(node.op, ev(node.left), ev(node.right), expr)
    if isinstance(node, ast.UnaryOp):
        value = ev(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not value
    if isinstance(node, ast.BoolOp):
        values = (ev(v) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = ev(node.left)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = ev(right_node)
            if type(op) in _CMPOPS:
                ok = _CMPOPS[type(op)](left, right)
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise XacroError(f"unsupported comparison in '${{{expr}}}'")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return ev(node.body) if ev(node.test) else ev(node.orelse)
    if isinstance(node, ast.Tuple | ast.List):
        return [ev(v) for v in node.elts]
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        return {ev(k): ev(v) for k, v in zip(node.keys, node.values, strict=True)}  # type: ignore[arg-type]
    if isinstance(node, ast.ListComp | ast.GeneratorExp) and len(node.generators) == 1:
        return _comprehension(node, lookup, expr, extra)
    if isinstance(node, ast.Subscript):
        return ev(node.value)[ev(node.slice)]
    if isinstance(node, ast.Attribute):
        is_math = isinstance(node.value, ast.Name) and node.value.id == "math"
        if is_math and (node.attr in _SAFE_FUNCTIONS or node.attr in ("pi", "e", "inf", "nan")):
            return _SAFE_FUNCTIONS.get(node.attr, getattr(math, node.attr))
        if isinstance(node.value, ast.Name) and node.value.id == "xacro" and node.attr in extra:
            return extra[node.attr]
        owner = ev(node.value)
        if isinstance(owner, _Yaml) and node.attr in owner:
            return owner[node.attr]
        if isinstance(owner, str) and node.attr == "replace":
            return functools.partial(_safe_replace, owner)
        if isinstance(owner, str) and node.attr in _STR_METHODS:
            return getattr(owner, node.attr)
        if isinstance(owner, dict) and node.attr in _DICT_METHODS:
            return getattr(owner, node.attr)
        raise XacroError(f"cannot read '.{node.attr}' in '${{{expr}}}'")
    if isinstance(node, ast.Call) and not node.keywords:
        name = getattr(node.func, "id", "")
        func = ev(node.func) if isinstance(node.func, ast.Attribute) else extra.get(name) or _SAFE_FUNCTIONS.get(name)
        if func is None or not callable(func):
            name = name or "that function"
            raise XacroError(
                f"'{name}' in '${{{expr}}}' is a Python helper the built-in xacro reader does not support; "
                "expand this file with ROS `xacro` once, or replace the formula by its value"
            )
        return func(*[ev(a) for a in node.args])
    raise XacroError(f"cannot evaluate '${{{expr}}}' (only arithmetic and math functions are supported)")


def _comprehension(node: ast.ListComp | ast.GeneratorExp, lookup, expr: str, extra: dict[str, Any]) -> list[Any]:
    """`[f(x) for x in items if g(x)]`, as ROS packages write them in their formulas: one loop, names bound by it."""
    gen = node.generators[0]
    if gen.is_async or not isinstance(gen.target, ast.Name):
        raise XacroError(f"cannot evaluate '${{{expr}}}' (only a simple loop is supported)")
    bound: dict[str, Any] = {}

    def inner(name: str) -> Any:
        return bound[name] if name in bound else lookup(name)

    out = []
    for item in list(_eval_node(gen.iter, lookup, expr, extra))[:_MAX_SEQ]:
        bound[gen.target.id] = item
        if all(_eval_node(c, inner, expr, extra) for c in gen.ifs):
            out.append(_eval_node(node.elt, inner, expr, extra))
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return " ".join(_fmt(v) for v in value)
    return str(value)


# ---- expander --------------------------------------------------------------------------------


class _Scope:
    """Properties of one macro call (or the file), falling back to the caller's scope."""

    def __init__(self, parent: _Scope | None = None):
        self.parent = parent
        self.values: dict[str, Any] = {}
        self.lazy: dict[str, str] = {}
        self.blocks: dict[str, ET.Element] = {}
        self.block_owner: dict[str, _Scope] = {}  # where each block was written: the caller of the macro

    def find(self, name: str) -> _Scope | None:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.values or name in scope.lazy or name in scope.blocks:
                return scope
            scope = scope.parent
        return None


class _Expander:
    def __init__(self, path: Path, args: dict[str, str]):
        self.path = path
        self.args = dict(args)
        self.macros: dict[str, ET.Element] = {}
        self.files: list[Path] = [path]
        self.resolver = MeshResolver(path)
        self.depth = 0
        self.package_roots: list[Path] = []  # folders of packages named by $(find ...)
        self.nodes = 0  # elements and macro calls handled so far
        self.chars = 0  # characters of text produced so far
        self.deadline = time.monotonic() + _MAX_SECONDS
        self.undefined: set[str] = set()  # macros that were called but never defined
        # xacro's own diagnostics are accepted and ignored.
        quiet = lambda *args: ""  # noqa: E731
        self.functions: dict[str, Any] = {
            "load_yaml": self._load_yaml,
            "warning": quiet,
            "message": quiet,
            "print_location": quiet,
        }

    def _load_yaml(self, filename: str) -> Any:
        target = Path(filename)
        if not target.is_absolute():
            target = self.path.parent / target
        if not target.is_file():
            raise XacroError(f"load_yaml: the file '{filename}' does not exist ({target})")
        self._check_inside_project(target, "load_yaml")
        self.files.append(target)
        try:
            return _wrap(yaml.load(target.read_text(encoding="utf-8"), Loader=_XacroYamlLoader))  # noqa: S506 - safe subclass
        except yaml.YAMLError as exc:
            raise XacroError(f"load_yaml: '{target.name}' is not valid YAML ({exc})") from exc

    def _spend(self, nodes: int = 0, chars: int = 0) -> None:
        """Count work against the expansion budget (a macro that doubles at every level would never finish)."""
        self.nodes += nodes
        self.chars += chars
        if self.nodes > _MAX_NODES:
            raise XacroError(
                f"{self.path.name} expands into more than {_MAX_NODES:,} elements; a macro probably calls "
                "itself or doubles at every level"
            )
        if self.chars > _MAX_OUTPUT_CHARS:
            raise XacroError(f"{self.path.name} expands into far too much text (over {_MAX_OUTPUT_CHARS // 10**6} MB)")
        if nodes and self.nodes % 256 == 0 and time.monotonic() > self.deadline:
            raise XacroError(f"{self.path.name} takes more than {_MAX_SECONDS:.0f} s to expand")

    def _project_root(self) -> Path:
        """What a xacro may read: its package (the nearest folder with a package.xml above it), else its own folder."""
        for folder in [self.path.parent, *list(self.path.parent.parents)[:5]]:
            if (folder / "package.xml").is_file():
                return folder
        return self.path.parent

    def _check_inside_project(self, target: Path, what: str) -> None:
        """Files named inside a xacro (which may come from anywhere) are read only from its own project."""
        resolved = target.resolve()
        roots = [self._project_root(), *self.package_roots]
        if not any(resolved == r or r.resolve() in resolved.parents for r in roots):
            raise XacroError(
                f"{what} '{target}' is outside the project folder; xacro files may only read their own project"
            )
        if target.stat().st_size > _MAX_FILE_BYTES:
            raise XacroError(f"{what} '{target.name}' is too large to read")

    def run(self) -> ET.Element:
        if self.path.stat().st_size > _MAX_FILE_BYTES * 4:
            raise XacroError(f"{self.path.name} is too large to read")
        root = _parse(self.path.read_text(encoding="utf-8-sig", errors="replace"), self.path.name)
        assert root is not None
        if _local(root.tag) != "robot":
            raise XacroError(f"{self.path.name}: the top element is <{_local(root.tag)}>, expected <robot>")
        out = ET.Element("robot", {k: v for k, v in root.attrib.items() if not k.startswith("{")})
        scope = _Scope()
        self._hoist_macros(root)
        self._children(root, out, scope, self.path)
        if "name" in out.attrib:
            out.set("name", self._text(out.get("name", ""), scope, self.path))
        return out

    # ---- tree walk ----

    def _children(self, src: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
        for child in src:
            self._node(child, out, scope, file)

    def _node(self, el: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
        self._spend(nodes=1)
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise XacroError("xacro macros call each other too deeply (an infinite loop?)")
        try:
            if _is_xacro(el.tag):
                self._directive(_local(el.tag), el, out, scope, file)
            else:
                new = ET.SubElement(out, el.tag, {k: self._text(v, scope, file) for k, v in el.attrib.items()})
                if el.text and el.text.strip():
                    new.text = self._text(el.text, scope, file)
                self._children(el, new, scope, file)
        finally:
            self.depth -= 1

    def _directive(self, name: str, el: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
        if name == "property":
            self._property(el, scope, file)
        elif name == "arg":
            arg = el.get("name", "")
            if arg not in self.args and el.get("default") is not None:
                self.args[arg] = self._text(el.get("default", ""), scope, file)
        elif name == "macro":
            self.macros[el.get("name", "")] = el
        elif name == "include":
            self._include(el, out, scope, file)
        elif name in ("if", "unless"):
            raw = self._value(el.get("value", ""), scope, file)
            truth = _truthy(raw)
            if truth == (name == "if"):
                self._children(el, out, scope, file)
        elif name == "insert_block":
            block, owner = self._lookup_block(el.get("name", ""), scope)
            # A block means what it meant where it was written: a macro that is given `<xacro:insert_block
            # name="origin"/>` as its own origin block would otherwise insert itself without end (the HSR's).
            if block.tag == _GROUP:
                self._children(block, out, owner, file)
            else:
                out.append(self._copy_expanded(block, owner, file))
        elif name in self.macros:
            self._call(self.macros[name], el, out, scope, file)
        elif name in _KNOWN_DIRECTIVES:
            raise XacroError(f"xacro:{name} is not supported by the built-in xacro reader")
        else:  # a call to a macro nobody defined (autogenerated files sometimes keep such leftovers)
            self.undefined.add(name)

    def _property(self, el: ET.Element, scope: _Scope, file: Path) -> None:
        name = el.get("name", "")
        target = scope
        if el.get("scope") == "parent" and scope.parent is not None:
            target = scope.parent
        elif el.get("scope") == "global":
            while target.parent is not None:
                target = target.parent
        if el.get("value") is not None:
            eager = target is not scope or el.get("lazy_eval") == "false"
            if eager:  # evaluate now, while the local names the value uses still exist
                target.values[name] = self._value(el.get("value", ""), scope, file)
                target.lazy.pop(name, None)
            else:
                target.lazy[name] = el.get("value", "")
                target.values.pop(name, None)
        elif len(el):
            target.blocks[name] = el
        elif el.get("default") is not None and scope.find(name) is None:
            target.lazy[name] = el.get("default", "")

    def _include(self, el: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
        filename = self._text(el.get("filename", ""), scope, file)
        target = Path(filename)
        if not target.is_absolute():
            target = file.parent / target
        if not target.is_file():
            raise XacroError(f"xacro:include of '{filename}' failed: the file does not exist ({target})")
        self._check_inside_project(target, "xacro:include")
        self.files.append(target)
        root = _parse(target.read_text(encoding="utf-8-sig", errors="replace"), target.name)
        assert root is not None
        self._hoist_macros(root)
        self._children(root, out, scope, target)

    def _hoist_macros(self, root: ET.Element) -> None:
        """Macros can be used before the line that defines them, as in xacro itself."""
        for child in root:
            if _is_xacro(child.tag) and _local(child.tag) == "macro":
                self.macros.setdefault(child.get("name", ""), child)

    def _call(self, macro: ET.Element, call: ET.Element, out: ET.Element, caller: _Scope, file: Path) -> None:
        scope = _Scope(caller)
        blocks = [c for c in call if isinstance(c.tag, str)]
        index = 0
        for token in _split_params(macro.get("params") or ""):
            if token.startswith("**"):  # all remaining child elements, as one group
                group = ET.Element(_GROUP)
                group.extend(blocks[index:])
                scope.blocks[token[2:]] = group
                scope.block_owner[token[2:]] = caller
                index = len(blocks)
                continue
            if token.startswith("*"):
                if index >= len(blocks):
                    raise XacroError(f"macro '{macro.get('name')}' needs a block for parameter {token}")
                scope.blocks[token[1:]] = blocks[index]
                scope.block_owner[token[1:]] = caller
                index += 1
                continue
            name, has_default, default = _split_param(token)
            default = _unquote(default)
            inherit = name.startswith("^")
            name = name.lstrip("^")
            if name in call.attrib:
                scope.values[name] = self._value(call.attrib[name], caller, file)
            elif default.startswith("^|"):
                found = caller.find(name)
                scope.values[name] = self._lookup(name, caller) if found else self._value(default[2:], scope, file)
            elif inherit or default == "^":
                scope.values[name] = self._lookup(name, caller)
            elif has_default:
                scope.values[name] = self._value(default, scope, file)
            else:
                raise XacroError(f"macro '{macro.get('name')}' was called without its parameter '{name}'")
        declared = {_split_param(t)[0].lstrip("^") for t in _split_params(macro.get("params") or "")}
        extra = set(call.attrib) - declared
        if extra:
            raise XacroError(f"macro '{macro.get('name')}' has no parameter {', '.join(sorted(extra))}")
        self._children(macro, out, scope, file)

    # ---- lookups and substitution ----

    def _lookup(self, name: str, scope: _Scope) -> Any:
        holder = scope.find(name)
        if holder is None and name in _SAFE_NAMES:
            return _SAFE_NAMES[name]
        if holder is None:
            raise XacroError(f"the name '{name}' is used in a formula but never defined")
        if name in holder.values:
            return holder.values[name]
        if name in holder.lazy:
            raw = holder.lazy.pop(name)
            holder.values[name] = None
            holder.values[name] = self._value(raw, holder, self.path)
            return holder.values[name]
        raise XacroError(f"'{name}' is a block property and cannot be used inside a formula")

    def _lookup_block(self, name: str, scope: _Scope) -> tuple[ET.Element, _Scope]:
        holder = scope.find(name)
        if holder is None or name not in holder.blocks:
            raise XacroError(f"xacro:insert_block refers to '{name}', which is not a block")
        return holder.blocks[name], holder.block_owner.get(name, scope)

    def _copy_expanded(self, block: ET.Element, scope: _Scope, file: Path) -> ET.Element:
        holder = ET.Element("holder")
        self._node(block, holder, scope, file)
        return holder[0] if len(holder) else copy.deepcopy(block)

    def _value(self, text: str, scope: _Scope, file: Path) -> Any:
        """Evaluate a property/attribute value; a lone `${expr}` keeps its Python type."""
        stripped = text.strip()
        match = re.fullmatch(r"\$\{([^}]*)\}", stripped)
        if match:
            expr = self._resolve_commands(match.group(1), scope, file)
            value = _evaluate(expr, lambda n: self._lookup(n, scope), self.functions)
            return _convert(value) if isinstance(value, str) else value
        return _convert(self._text(text, scope, file))

    def _text(self, text: str, scope: _Scope, file: Path) -> str:
        def sub(m: re.Match) -> str:
            if m.group(0) == "$$":
                return "$"
            if m.group(1) is not None:
                expr = self._resolve_commands(m.group(1), scope, file)
                return _fmt(_evaluate(expr, lambda n: self._lookup(n, scope), self.functions))
            return self._command(m.group(2).strip(), scope, file)

        result = _SUBST.sub(sub, text)
        self._spend(chars=len(result))
        return result

    def _resolve_commands(self, expr: str, scope: _Scope, file: Path) -> str:
        """`$(arg x)` and friends inside a formula are replaced by their value before the formula runs."""
        return re.sub(r"\$\(([^)]*)\)", lambda m: self._command(m.group(1).strip(), scope, file), expr)

    def _command(self, command: str, scope: _Scope, file: Path) -> str:
        verb, _, rest = command.partition(" ")
        rest = rest.strip()
        if verb == "find":
            root = self.resolver.find_package(rest)
            if root is None:
                raise XacroError(f"$(find {rest}): package '{rest}' was not found next to the xacro file")
            self.package_roots.append(root)
            return str(root)
        if verb == "arg":
            if rest not in self.args:
                raise XacroError(f"$(arg {rest}) has no value; pass xacro_args={{'{rest}': '...'}}")
            return self.args[rest]
        if verb == "env":
            raise XacroError(
                f"$(env {rest}) is not allowed: robotforge does not read environment variables while expanding xacro. "
                f"Pass the value with xacro_args instead"
            )
        if verb == "optenv":  # always the default, so the result never depends on the machine it runs on
            return rest.partition(" ")[2].strip()
        if verb == "dirname":
            return str(file.parent)
        if verb == "eval":
            return _fmt(_evaluate(rest, lambda n: self._lookup(n, scope), self.functions))
        raise XacroError(f"$({command}) is not supported by the built-in xacro reader")


def _convert(text: str) -> Any:
    """Numbers and booleans written as text become numbers and booleans, as in xacro."""
    if text in ("true", "false"):
        return text == "true"
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            continue
    return text


def _split_params(text: str) -> list[str]:
    """Split a macro `params` string on spaces, but not inside quotes, ${...} or $(...)."""
    tokens, current, depth, quote = [], "", 0, ""
    for i, ch in enumerate(text):
        if quote:
            current += ch
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
            current += ch
        elif ch in "{(" and i > 0 and text[i - 1] == "$":
            depth += 1
            current += ch
        elif ch in "})" and depth:
            depth -= 1
            current += ch
        elif ch.isspace() and not depth:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def _split_param(token: str) -> tuple[str, str, str]:
    """`name:=default` (also the sloppy `name=default` that some files contain) -> (name, separator, default)."""
    name, sep, default = token.partition(":=")
    if not sep and "=" in token.split("'")[0].split('"')[0]:
        name, sep, default = token.partition("=")
    return name, sep, default


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("false", "0", "no", ""):
            return False
        if low in ("true", "1", "yes"):
            return True
        raise XacroError(f"xacro:if/unless needs true or false, got '{value}'")
    return bool(value)
