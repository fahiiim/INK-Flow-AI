"""Validated studio routing rules with a small, safe condition language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .pricing import parse_size_cm
from .schemas import (
    STYLE_TAG_OPTIONS,
    AIExtractionOutput,
    TattooExtractionDraft,
)

DEFAULT_ROUTING_RULES_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "routing_rules.yaml"
)
_ACTION_PATTERN = re.compile(
    r"^suggest_artist:\s*([a-z0-9][a-z0-9_-]{0,63})$",
    flags=re.IGNORECASE,
)
_TEXT_CONDITION_PATTERN = re.compile(
    r"^(style_tags|placement|color_preference)\s+"
    r"(contains|==|!=)\s+(['\"])(.+)\3$",
    flags=re.IGNORECASE,
)
_SIZE_CONDITION_PATTERN = re.compile(
    r"^size\s*(<=|>=|==|<|>)\s*(\d+(?:\.\d+)?)\s*cm$",
    flags=re.IGNORECASE,
)
_AND_PATTERN = re.compile(r"\s+AND\s+", flags=re.IGNORECASE)

RuleField = Literal["style_tags", "placement", "color_preference", "size"]
RuleOperator = Literal["contains", "==", "!=", "<", "<=", ">", ">="]
RuleAnalysis = AIExtractionOutput | TattooExtractionDraft


class RoutingRuleConfigError(ValueError):
    """Raised when routing-rule configuration is invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class _ConditionClause:
    """One compiled, non-executable routing condition clause."""

    field: RuleField
    operator: RuleOperator
    expected_text: str | None = None
    expected_size_cm: Decimal | None = None


def _parse_condition(condition: str) -> tuple[_ConditionClause, ...]:
    """Compile a validated `AND` expression into safe condition clauses."""
    normalized = condition.strip()
    if not normalized:
        raise RoutingRuleConfigError("Routing rule condition cannot be empty.")

    raw_clauses = _AND_PATTERN.split(normalized)
    clauses = tuple(_parse_clause(clause.strip()) for clause in raw_clauses)
    if not clauses:
        raise RoutingRuleConfigError("Routing rule requires a condition.")
    return clauses


def _parse_clause(clause: str) -> _ConditionClause:
    """Parse one supported text or numeric condition clause."""
    size_match = _SIZE_CONDITION_PATTERN.fullmatch(clause)
    if size_match:
        return _ConditionClause(
            field="size",
            operator=size_match.group(1),
            expected_size_cm=Decimal(size_match.group(2)),
        )

    text_match = _TEXT_CONDITION_PATTERN.fullmatch(clause)
    if not text_match:
        raise RoutingRuleConfigError(
            "Unsupported routing condition clause: "
            f"{clause}. Only AND expressions are supported."
        )

    field = text_match.group(1).casefold()
    operator = text_match.group(2).casefold()
    expected = text_match.group(4).strip().casefold()
    if field == "style_tags":
        if operator != "contains":
            raise RoutingRuleConfigError(
                "style_tags conditions must use the contains operator."
            )
        if expected not in STYLE_TAG_OPTIONS or expected == "unknown":
            raise RoutingRuleConfigError(
                f"Unsupported style tag in routing rule: {expected}."
            )
    return _ConditionClause(
        field=field,
        operator=operator,
        expected_text=expected,
    )


def _parse_artist_action(action: str) -> str:
    """Return the normalized artist key from a supported action."""
    match = _ACTION_PATTERN.fullmatch(action.strip())
    if not match:
        raise RoutingRuleConfigError(
            "Routing action must use 'suggest_artist: artist_key'."
        )
    return match.group(1).casefold()


class RoutingRule(BaseModel):
    """One validated condition and artist suggestion action."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    condition: str = Field(min_length=1, max_length=500)
    action: str = Field(min_length=1, max_length=100)
    priority: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_expression(self) -> Self:
        """Compile condition and action once during schema validation."""
        _parse_condition(self.condition)
        _parse_artist_action(self.action)
        return self

    @property
    def artist_key(self) -> str:
        """Return the normalized artist key selected by this rule."""
        return _parse_artist_action(self.action)


class StudioRoutingRulesConfig(BaseModel):
    """Versioned collection of studio-specific routing rules."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    config_version: int = Field(default=1, ge=1)
    rules: list[RoutingRule] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_unique_names(self) -> Self:
        """Reject duplicate rule names to keep audit messages unambiguous."""
        names = [rule.name.casefold() for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Routing rule names must be unique.")
        return self


class RoutingRuleEngine:
    """Load and evaluate safe studio-specific artist routing rules.

    Larger numeric priority values take precedence. Rules with the same
    priority retain their source-file order.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: StudioRoutingRulesConfig | None = None,
    ) -> None:
        """Initialize rules from injected data or the default YAML file."""
        self._lock = RLock()
        if config is not None:
            self._config_path = self._optional_path(config_path)
            self._config = config.model_copy(deep=True)
            return

        resolved_path = Path(
            config_path or DEFAULT_ROUTING_RULES_PATH
        ).expanduser().resolve()
        self._config_path = resolved_path
        self._config = self._load_config(resolved_path)

    @property
    def rules(self) -> tuple[RoutingRule, ...]:
        """Return an immutable snapshot of configured rules."""
        with self._lock:
            return tuple(
                rule.model_copy(deep=True) for rule in self._config.rules
            )

    def evaluate(self, analysis: RuleAnalysis) -> list[RoutingRule]:
        """Return matching rules ordered from highest to lowest priority."""
        with self._lock:
            configured_rules = [
                rule.model_copy(deep=True) for rule in self._config.rules
            ]
        applicable = [
            rule
            for rule in configured_rules
            if self._condition_matches(rule.condition, analysis)
        ]
        applicable.sort(key=lambda rule: rule.priority, reverse=True)
        return applicable

    def reload(self) -> StudioRoutingRulesConfig:
        """Reload and return routing rules from the configured YAML path."""
        if self._config_path is None:
            raise RoutingRuleConfigError(
                "Cannot reload injected rules without a source path."
            )
        loaded = self._load_config(self._config_path)
        with self._lock:
            self._config = loaded
            return self._config.model_copy(deep=True)

    def _condition_matches(
        self,
        condition: str,
        analysis: RuleAnalysis,
    ) -> bool:
        """Return whether every clause matches the extracted tattoo facts."""
        return all(
            self._clause_matches(clause, analysis)
            for clause in _parse_condition(condition)
        )

    def _clause_matches(
        self,
        clause: _ConditionClause,
        analysis: RuleAnalysis,
    ) -> bool:
        """Evaluate one compiled clause without executing arbitrary code."""
        if clause.field == "size":
            return self._size_matches(clause, analysis.size_estimate_cm)
        if clause.field == "style_tags":
            expected = clause.expected_text or ""
            return expected in {tag.casefold() for tag in analysis.style_tags}

        actual = getattr(analysis, clause.field).casefold()
        expected = clause.expected_text or ""
        if clause.operator == "contains":
            return expected in actual
        if clause.operator == "==":
            return actual == expected
        if clause.operator == "!=":
            return actual != expected
        return False

    def _size_matches(
        self,
        clause: _ConditionClause,
        size_estimate_cm: str,
    ) -> bool:
        """Evaluate one numeric size comparison in centimeters."""
        actual = parse_size_cm(size_estimate_cm)
        expected = clause.expected_size_cm
        if actual is None or expected is None:
            return False
        comparisons = {
            "<": actual < expected,
            "<=": actual <= expected,
            "==": actual == expected,
            ">": actual > expected,
            ">=": actual >= expected,
        }
        return comparisons.get(clause.operator, False)

    def _load_config(self, path: Path) -> StudioRoutingRulesConfig:
        """Read and validate a safe YAML routing-rules file."""
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise RoutingRuleConfigError(
                "Routing rules must use a .yaml or .yml file."
            )
        if not path.is_file():
            raise RoutingRuleConfigError(
                f"Routing rules file not found: {path}"
            )
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            return StudioRoutingRulesConfig.model_validate(payload)
        except (OSError, ValidationError, yaml.YAMLError) as exc:
            raise RoutingRuleConfigError(
                f"Invalid routing rules configuration: {path}"
            ) from exc

    def _optional_path(self, path: str | Path | None) -> Path | None:
        """Resolve an optional source path for injected configurations."""
        if path is None:
            return None
        return Path(path).expanduser().resolve()
