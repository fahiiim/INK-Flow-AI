"""Studio-specific artist configuration loading and runtime management."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .schemas import StyleTag

DEFAULT_ARTIST_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "studio_artists.yaml"
)
_ARTIST_KEY_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_SUPPORTED_CONFIG_SUFFIXES = {".json", ".yaml", ".yml"}


class ArtistConfigError(ValueError):
    """Raised when artist configuration cannot be loaded or persisted."""


class ArtistProfile(BaseModel):
    """Validated routing profile for one studio artist."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    artist_key: str = Field(
        min_length=1,
        max_length=64,
        pattern=_ARTIST_KEY_PATTERN,
    )
    display_name: str = Field(min_length=1, max_length=80)
    specialties: list[StyleTag] = Field(min_length=1, max_length=10)
    min_size_cm: float | None = Field(default=None, ge=0)
    max_size_cm: float | None = Field(default=None, ge=0)
    is_active: bool = True

    @field_validator("artist_key", mode="before")
    @classmethod
    def normalize_artist_key(cls, value: Any) -> Any:
        """Normalize artist keys for stable lookup and persistence."""
        if isinstance(value, str):
            return value.strip().casefold()
        return value

    @field_validator("specialties")
    @classmethod
    def validate_specialties(
        cls,
        value: list[StyleTag],
    ) -> list[StyleTag]:
        """Require unique, meaningful specialties for routing."""
        if "unknown" in value:
            raise ValueError("Artist specialties cannot contain 'unknown'.")
        if len(value) != len(set(value)):
            raise ValueError("Artist specialties must be unique.")
        return value

    @model_validator(mode="after")
    def validate_size_range(self) -> Self:
        """Require maximum accepted size to be no smaller than minimum."""
        if (
            self.min_size_cm is not None
            and self.max_size_cm is not None
            and self.max_size_cm < self.min_size_cm
        ):
            raise ValueError(
                "max_size_cm must be greater than or equal to min_size_cm."
            )
        return self


class StudioArtistConfig(BaseModel):
    """Complete versioned artist catalog for one studio deployment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    config_version: int = Field(default=1, ge=1)
    artists: list[ArtistProfile] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_artists(self) -> Self:
        """Reject duplicate artist keys and display names."""
        artist_keys = [artist.artist_key for artist in self.artists]
        if len(artist_keys) != len(set(artist_keys)):
            raise ValueError("Artist keys must be unique.")

        display_names = [
            artist.display_name.casefold() for artist in self.artists
        ]
        if len(display_names) != len(set(display_names)):
            raise ValueError("Artist display names must be unique.")
        return self


class ArtistConfigManager:
    """Load, query, update, and persist studio artist configuration.

    A caller may inject a validated ``StudioArtistConfig`` for tests or custom
    application composition. Without an injected model, the manager loads the
    default YAML file or the caller-supplied JSON/YAML path.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: StudioArtistConfig | None = None,
    ) -> None:
        """Initialize the manager from a file or injected configuration.

        Args:
            config_path: Optional JSON or YAML source and persistence path.
            config: Optional prevalidated configuration for dependency
                injection. When supplied without a path, ``save`` requires an
                explicit destination and ``reload`` is unavailable.
        """
        self._lock = RLock()
        if config is not None:
            self._config_path = self._optional_path(config_path)
            self._config = config.model_copy(deep=True)
            return

        resolved_path = self._resolve_path(
            config_path or DEFAULT_ARTIST_CONFIG_PATH
        )
        self._config_path = resolved_path
        self._config = self._load_config(resolved_path)

    @property
    def config(self) -> StudioArtistConfig:
        """Return a deep copy of the current validated configuration."""
        with self._lock:
            return self._config.model_copy(deep=True)

    @property
    def source_path(self) -> Path | None:
        """Return the configured source path, if the manager has one."""
        return self._config_path

    def get_active_artists(self) -> list[ArtistProfile]:
        """Return independent copies of artists accepting new work."""
        with self._lock:
            return [
                artist.model_copy(deep=True)
                for artist in self._config.artists
                if artist.is_active
            ]

    def get_artist_by_key(self, artist_key: str) -> ArtistProfile | None:
        """Return an artist by normalized key, including inactive artists."""
        normalized_key = artist_key.strip().casefold()
        with self._lock:
            for artist in self._config.artists:
                if artist.artist_key == normalized_key:
                    return artist.model_copy(deep=True)
        return None

    def validate_artist_assignment(
        self,
        artist_key: str,
        style_tags: Sequence[StyleTag] | None = None,
    ) -> bool:
        """Return whether an artist is active and supports detected styles.

        Args:
            artist_key: Stable, case-insensitive artist identifier.
            style_tags: Optional detected styles. When provided, at least one
                meaningful style must match an artist specialty.
        """
        artist = self.get_artist_by_key(artist_key)
        if artist is None or not artist.is_active:
            return False
        if style_tags is None:
            return True

        detected_styles = set(style_tags).difference({"unknown"})
        if not detected_styles:
            return False
        return bool(detected_styles.intersection(artist.specialties))

    def upsert_artist(self, artist: ArtistProfile) -> ArtistProfile:
        """Add a new artist or replace an existing profile by stable key."""
        candidate = artist.model_copy(deep=True)
        with self._lock:
            artists = [
                existing.model_copy(deep=True)
                for existing in self._config.artists
                if existing.artist_key != candidate.artist_key
            ]
            artists.append(candidate)
            self._config = StudioArtistConfig(
                config_version=self._config.config_version,
                artists=artists,
            )
        return candidate.model_copy(deep=True)

    def set_artist_active(
        self,
        artist_key: str,
        is_active: bool,
    ) -> ArtistProfile:
        """Change whether an existing artist can receive suggestions."""
        normalized_key = artist_key.strip().casefold()
        with self._lock:
            current = self._find_artist(normalized_key)
            updated = ArtistProfile.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "is_active": is_active,
                }
            )
            self._replace_artist(updated)
            return updated.model_copy(deep=True)

    def replace_config(self, config: StudioArtistConfig) -> None:
        """Atomically replace the in-memory catalog with validated data."""
        with self._lock:
            self._config = config.model_copy(deep=True)

    def reload(self) -> StudioArtistConfig:
        """Reload and return the configuration from its original file."""
        if self._config_path is None:
            raise ArtistConfigError(
                "Cannot reload an injected config without a source path."
            )
        loaded = self._load_config(self._config_path)
        with self._lock:
            self._config = loaded
            return self._config.model_copy(deep=True)

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the current configuration as JSON or YAML atomically."""
        target = self._save_target(path)
        with self._lock:
            payload = self._config.model_dump(mode="json")
        serialized = self._serialize_payload(payload, target.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = target.with_name(f"{target.name}.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(target)
        self._config_path = target
        return target

    def _load_config(self, path: Path) -> StudioArtistConfig:
        """Read and validate one supported artist configuration file."""
        if not path.is_file():
            raise ArtistConfigError(f"Artist config file not found: {path}")
        suffix = path.suffix.lower()
        self._validate_suffix(suffix)
        try:
            raw_text = path.read_text(encoding="utf-8")
            payload = self._parse_payload(raw_text, suffix)
            return StudioArtistConfig.model_validate(payload)
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise ArtistConfigError(
                f"Invalid artist configuration: {path}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ArtistConfigError(
                f"Invalid YAML artist configuration: {path}"
            ) from exc

    def _parse_payload(self, raw_text: str, suffix: str) -> Any:
        """Parse JSON or safe YAML text into a validation payload."""
        if suffix == ".json":
            return json.loads(raw_text)
        return yaml.safe_load(raw_text)

    def _serialize_payload(
        self,
        payload: dict[str, Any],
        suffix: str,
    ) -> str:
        """Serialize validated configuration for the selected file type."""
        self._validate_suffix(suffix)
        if suffix == ".json":
            return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
        return yaml.safe_dump(
            payload,
            allow_unicode=False,
            sort_keys=False,
        )

    def _find_artist(self, artist_key: str) -> ArtistProfile:
        """Return an internal artist model or raise for an unknown key."""
        for artist in self._config.artists:
            if artist.artist_key == artist_key:
                return artist
        raise KeyError(f"Unknown artist key: {artist_key}")

    def _replace_artist(self, updated: ArtistProfile) -> None:
        """Replace one artist while revalidating the complete catalog."""
        artists = [
            updated if artist.artist_key == updated.artist_key else artist
            for artist in self._config.artists
        ]
        self._config = StudioArtistConfig(
            config_version=self._config.config_version,
            artists=artists,
        )

    def _save_target(self, path: str | Path | None) -> Path:
        """Resolve an explicit or previously configured save target."""
        if path is not None:
            return self._resolve_path(path)
        if self._config_path is None:
            raise ArtistConfigError(
                "A save path is required for an injected configuration."
            )
        return self._config_path

    def _optional_path(self, path: str | Path | None) -> Path | None:
        """Resolve an optional path without applying the default."""
        return self._resolve_path(path) if path is not None else None

    def _resolve_path(self, path: str | Path) -> Path:
        """Return an absolute path and validate its supported suffix."""
        resolved = Path(path).expanduser().resolve()
        self._validate_suffix(resolved.suffix.lower())
        return resolved

    def _validate_suffix(self, suffix: str) -> None:
        """Reject unsupported artist configuration file formats."""
        if suffix not in _SUPPORTED_CONFIG_SUFFIXES:
            supported = ", ".join(sorted(_SUPPORTED_CONFIG_SUFFIXES))
            raise ArtistConfigError(
                f"Artist config must use one of these formats: {supported}."
            )
