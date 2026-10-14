"""The doctor report: what was wrong with an imported file, what was fixed, what was not.

Wording is written for beginners: plain sentences, no jargon without a hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["fixed", "unfixed", "info"]
_ITEM_LIMIT = 6


@dataclass(frozen=True)
class Finding:
    """One thing the doctor noticed.

    `status` is "fixed" (found and repaired), "unfixed" (found but could not be repaired,
    `action` says what happens instead) or "info" (worth knowing, nothing to repair).
    """

    topic: str
    status: Status
    problem: str
    action: str = ""
    items: tuple[str, ...] = ()

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """JSON-ready form; long item lists are shortened unless `full` (used by the on-disk cache)."""
        out: dict[str, Any] = {"topic": self.topic, "status": self.status, "problem": self.problem}
        if self.action:
            out["action"] = self.action
        if self.items:
            shown = list(self.items if full else self.items[:_ITEM_LIMIT])
            if not full and len(self.items) > _ITEM_LIMIT:
                shown.append(f"... and {len(self.items) - _ITEM_LIMIT} more")
            out["items"] = shown
        return out


@dataclass
class DoctorReport:
    """Everything the importer found and did for one file."""

    source: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        topic: str,
        status: Status,
        problem: str,
        action: str = "",
        items: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.findings.append(Finding(topic, status, problem, action, tuple(items)))

    def fixed(self, topic: str, problem: str, action: str, items: list[str] | tuple[str, ...] = ()) -> None:
        self.add(topic, "fixed", problem, action, items)

    def unfixed(self, topic: str, problem: str, action: str = "", items: list[str] | tuple[str, ...] = ()) -> None:
        self.add(topic, "unfixed", problem, action, items)

    def info(self, topic: str, problem: str, items: list[str] | tuple[str, ...] = ()) -> None:
        self.add(topic, "info", problem, "", items)

    def count(self, status: Status) -> int:
        return sum(1 for f in self.findings if f.status == status)

    @property
    def summary(self) -> str:
        fixed, unfixed = self.count("fixed"), self.count("unfixed")
        if not fixed and not unfixed:
            return "No problems found."
        parts = []
        if fixed:
            parts.append(f"fixed {fixed} problem{'s' if fixed != 1 else ''}")
        if unfixed:
            parts.append(f"{unfixed} thing{'s' if unfixed != 1 else ''} could not be fixed")
        return "The doctor " + " and ".join(parts) + "."

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary,
            "fixed": self.count("fixed"),
            "unfixed": self.count("unfixed"),
            "findings": [f.to_dict(full) for f in self.findings],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DoctorReport:
        report = cls(source=data.get("source", ""), stats=dict(data.get("stats", {})))
        for f in data.get("findings", []):
            report.add(f["topic"], f["status"], f["problem"], f.get("action", ""), f.get("items", []))
        return report

    def to_text(self) -> str:
        """A short console rendering for `robotforge open`."""
        lines = [f"Import doctor: {self.summary}"]
        marks = {"fixed": "fixed", "unfixed": "NOT FIXED", "info": "note"}
        for f in self.findings:
            lines.append(f"  [{marks[f.status]}] {f.problem}")
            if f.action:
                lines.append(f"      -> {f.action}")
            d = f.to_dict()
            if "items" in d:
                lines.append(f"      ({', '.join(d['items'])})")
        return "\n".join(lines)
