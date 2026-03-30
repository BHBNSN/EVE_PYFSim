from __future__ import annotations

import hashlib
import re

from ..user_errors import UserFacingError
from .models import ParsedEftFit, ParsedModuleSpec


class EftFitParser:
    _header_re = re.compile(r"^\[(?P<ship>[^,\]]+)\s*,\s*(?P<name>[^\]]+)\]$")
    _offline_suffixes = ("/offline", "/OFFLINE")

    def parse(self, fit_text: str) -> ParsedEftFit:
        lines = [line.strip() for line in fit_text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise UserFacingError("Fit text is empty.")

        match = self._header_re.match(lines[0])
        if not match:
            raise UserFacingError("EFT header is invalid; expected [Ship, Fit Name].")

        ship_name = match.group("ship").strip()
        fit_name = match.group("name").strip()

        modules: list[str] = []
        module_specs: list[ParsedModuleSpec] = []
        cargo_item_names: list[str] = []

        for raw in lines[1:]:
            if raw.lower().startswith("dna:"):
                continue
            if raw.lower().startswith("x-"):
                continue
            line = raw
            if " x" in line:
                qty_name = line.split(" x", 1)[0].strip()
                if qty_name:
                    cargo_item_names.append(qty_name)
                continue
            offline = False
            for suffix in self._offline_suffixes:
                if line.endswith(suffix):
                    offline = True
                    line = line[: -len(suffix)].strip()
                    break
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            charge_name: str | None = None
            if "," in line:
                mod, charge = line.split(",", 1)
                line = mod.strip()
                charge_name = charge.strip() or None
            if line.startswith("[Empty"):
                continue
            modules.append(line)
            module_specs.append(ParsedModuleSpec(module_name=line, charge_name=charge_name, offline=offline))

        fit_key = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        return ParsedEftFit(
            ship_name=ship_name,
            fit_name=fit_name,
            module_names=modules,
            module_specs=module_specs,
            cargo_item_names=cargo_item_names,
            fit_key=f"eft-{fit_key}",
        )


__all__ = ["EftFitParser"]
