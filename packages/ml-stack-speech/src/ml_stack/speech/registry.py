"""Choosing a provider, once, for all three modalities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ml_stack.speech.protocols import NoProviderAvailable, ProviderHealth

P = TypeVar("P")

Factory = Callable[[], P]


@dataclass
class Registry(Generic[P]):
    """Named provider factories, an order to try them in, and one cached instance."""

    kind: str
    """``"asr"`` / ``"tts"`` / ``"vad"``. Only used in messages."""

    factories: dict[str, Factory] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    _cached: P | None = field(default=None, repr=False)
    _cached_name: str | None = field(default=None, repr=False)

    def register(self, name: str, factory: Factory, *, prefer: bool = False) -> None:
        """Add a candidate. ``prefer=True`` puts it at the front of the auto order."""
        self.factories[name] = factory
        if name in self.order:
            self.order.remove(name)
        self.order.insert(0, name) if prefer else self.order.append(name)

    def names(self) -> list[str]:
        return list(self.order)

    def probe_all(self) -> dict[str, ProviderHealth]:
        """Ask every candidate whether it could work, without starting any of them."""
        results: dict[str, ProviderHealth] = {}
        for name in self.order:
            try:
                results[name] = self.factories[name]().probe()
            except Exception as exc:
                results[name] = ProviderHealth.missing(f"construction failed: {exc}")
        return results

    def create(self, name: str) -> P:
        """Build and start one named provider. Raises if it will not start."""
        if name not in self.factories:
            raise NoProviderAvailable(
                f"unknown {self.kind} provider {name!r}; registered: {sorted(self.factories)}"
            )
        provider = self.factories[name]()
        provider.start()
        return provider

    def auto(self) -> P:
        """Start the first candidate that actually works."""
        failures: list[str] = []
        for name in self.order:
            try:
                return self.create(name)
            except Exception as exc:
                failures.append(f"  {name}: {type(exc).__name__}: {exc}")

        detail = "\n".join(failures) if failures else "  (nothing registered)"
        raise NoProviderAvailable(f"no {self.kind} provider could be started:\n{detail}")

    def resolve(self, name: str | None = None, *, refresh: bool = False) -> P:
        """The cached provider, starting one if there is not one yet."""
        if not refresh and self._cached is not None:
            if name is None or name == self._cached_name:
                return self._cached

        self.reset()
        provider = self.create(name) if name else self.auto()
        self._cached = provider
        self._cached_name = getattr(provider, "name", name)
        return provider

    def reset(self) -> None:
        """Stop and drop the cached provider, releasing its model."""
        if self._cached is not None:
            try:
                self._cached.stop()
            except Exception:
                pass
        self._cached = None
        self._cached_name = None
