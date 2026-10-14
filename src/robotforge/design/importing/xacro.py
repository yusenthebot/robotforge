"""A small xacro expander, so beginners do not need ROS (or the `xacro` package) to open robots.

Supported: properties (lazy, with block properties), args, macros (with defaults, `*block`
parameters and `insert_block`), include, if/unless, `${...}` formulas (a safe arithmetic
subset with math functions) and the `$(find|arg|env|optenv|dirname|eval ...)` substitutions.
Not supported: Python helpers such as load_yaml, dynamic macro names and `xacro:element`.
Unsupported constructs raise XacroError saying which one, so the user knows what to expand
by hand.
"""

from __future__ import annotations

import ast
import copy
import math
import operator
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from robotforge.design.importing.resolve import MeshResolver

_XACRO_NS = "http://www.ros.org/wiki/xacro"
_MAX_DEPTH = 80
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
    "pow": pow,
}
_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    "True": True,
    "False": False,
}
_STR_METHODS = frozenset({"split", "strip", "lstrip", "rstrip", "lower", "upper", "replace", "startswith", "endswith"})
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


def expand(path: Path, args: dict[str, str] | None = None) -> tuple[str, list[Path]]:
    """Expand `path` into plain URDF text. Also returns every file that was read."""
    ex = _Expander(path, args or {})
    root = ex.run()
    return ET.tostring(root, encoding="unicode"), ex.files


# ---- xml helpers -----------------------------------------------------------------------------


def _is_xacro(tag: str) -> bool:
    return tag.startswith("{") and "xacro" in tag.split("}", 1)[0].lower()


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse(text: str, label: str, tolerant: bool = False) -> ET.Element | None:
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        if "unbound prefix" in str(exc):  # xacro: elements used without declaring the namespace
            patched = re.sub(r"<robot\b", f'<robot xmlns:xacro="{_XACRO_NS}"', text, count=1)
            try:
                return ET.fromstring(patched)
            except ET.ParseError as exc2:
                exc = exc2
        if tolerant:
            return None
        raise XacroError(f"{label} is not valid XML ({exc})") from exc


# ---- safe expressions ------------------------------------------------------------------------


def _evaluate(expr: str, lookup, extra: dict[str, Any] | None = None) -> Any:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise XacroError(f"cannot read the formula '${{{expr}}}'") from exc
    try:
        return _eval_node(tree.body, lookup, expr, extra or {})
    except XacroError:
        raise
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


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return _Yaml({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _eval_node(node: ast.AST, lookup, expr: str, extra: dict[str, Any]) -> Any:
    def ev(n: ast.AST) -> Any:
        return _eval_node(n, lookup, expr, extra)

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return lookup(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
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
        self.files.append(target)
        try:
            return _wrap(yaml.load(target.read_text(encoding="utf-8"), Loader=_XacroYamlLoader))  # noqa: S506 - safe subclass
        except yaml.YAMLError as exc:
            raise XacroError(f"load_yaml: '{target.name}' is not valid YAML ({exc})") from exc

    def run(self) -> ET.Element:
        root = _parse(self.path.read_text(encoding="utf-8-sig", errors="replace"), self.path.name)
        assert root is not None
        if _local(root.tag) != "robot":
            raise XacroError(f"{self.path.name}: the top element is <{_local(root.tag)}>, expected <robot>")
        out = ET.Element("robot", {k: v for k, v in root.attrib.items() if not k.startswith("{")})
        scope = _Scope()
        self._children(root, out, scope, self.path)
        if "name" in out.attrib:
            out.set("name", self._text(out.get("name", ""), scope, self.path))
        return out

    # ---- tree walk ----

    def _children(self, src: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
        for child in src:
            self._node(child, out, scope, file)

    def _node(self, el: ET.Element, out: ET.Element, scope: _Scope, file: Path) -> None:
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
            block = self._lookup_block(el.get("name", ""), scope)
            if block.tag == _GROUP:
                self._children(block, out, scope, file)
            else:
                out.append(self._copy_expanded(block, scope, file))
        elif name in self.macros:
            self._call(self.macros[name], el, out, scope, file)
        else:
            raise XacroError(f"xacro:{name} is not supported by the built-in xacro reader")

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
        self.files.append(target)
        root = _parse(target.read_text(encoding="utf-8-sig", errors="replace"), target.name)
        assert root is not None
        self._children(root, out, scope, target)

    def _call(self, macro: ET.Element, call: ET.Element, out: ET.Element, caller: _Scope, file: Path) -> None:
        scope = _Scope(caller)
        blocks = [c for c in call if isinstance(c.tag, str)]
        index = 0
        for token in _split_params(macro.get("params") or ""):
            if token.startswith("**"):  # all remaining child elements, as one group
                group = ET.Element(_GROUP)
                group.extend(blocks[index:])
                scope.blocks[token[2:]] = group
                index = len(blocks)
                continue
            if token.startswith("*"):
                if index >= len(blocks):
                    raise XacroError(f"macro '{macro.get('name')}' needs a block for parameter {token}")
                scope.blocks[token[1:]] = blocks[index]
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

    def _lookup_block(self, name: str, scope: _Scope) -> ET.Element:
        holder = scope.find(name)
        if holder is None or name not in holder.blocks:
            raise XacroError(f"xacro:insert_block refers to '{name}', which is not a block")
        return holder.blocks[name]

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

        return _SUBST.sub(sub, text)

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
            return str(root)
        if verb == "arg":
            if rest not in self.args:
                raise XacroError(f"$(arg {rest}) has no value; pass xacro_args={{'{rest}': '...'}}")
            return self.args[rest]
        if verb == "env":
            if rest not in os.environ:
                raise XacroError(f"$(env {rest}): the environment variable is not set")
            return os.environ[rest]
        if verb == "optenv":
            name, _, default = rest.partition(" ")
            return os.environ.get(name, default)
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
