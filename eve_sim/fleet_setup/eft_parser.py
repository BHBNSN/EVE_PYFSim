from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass

from ..config import resolve_pyfa_source_dir
from ..user_errors import UserFacingError
from .models import ParsedCargoSpec, ParsedEftFit, ParsedModuleSpec, ParsedMutationSpec


@dataclass(slots=True)
class _ItemKind:
    canonical_name: str
    group_name: str
    category_name: str


class _PyfaEveDbItemClassifier:
    def __init__(self) -> None:
        self._db_path = resolve_pyfa_source_dir() / "eve.db"
        self._has_type_name_zh: bool | None = None
        self._cache: dict[str, _ItemKind | None] = {}

    def kind_for(self, type_name: str) -> _ItemKind | None:
        name = str(type_name or "").strip()
        if not name or not self._db_path.exists():
            return None
        cache_key = name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        kind: _ItemKind | None = None
        try:
            conn = sqlite3.connect(str(self._db_path))
            cur = conn.cursor()
            if self._has_type_name_zh is None:
                cur.execute("PRAGMA table_info(invtypes)")
                cols = {str(row[1]).lower() for row in cur.fetchall()}
                self._has_type_name_zh = "typename_zh" in cols

            where = (
                "LOWER(t.typeName)=LOWER(?) "
                "OR LOWER(REPLACE(t.typeName,' ',''))=LOWER(REPLACE(?, ' ', ''))"
            )
            params: tuple[str, ...] = (name, name)
            if self._has_type_name_zh:
                where = (
                    f"{where} "
                    "OR LOWER(t.typeName_zh)=LOWER(?) "
                    "OR LOWER(REPLACE(t.typeName_zh,' ',''))=LOWER(REPLACE(?, ' ', ''))"
                )
                params = (name, name, name, name)

            cur.execute(
                "SELECT t.typeName, g.name, c.name "
                "FROM invtypes t "
                "JOIN invgroups g ON g.groupID=t.groupID "
                "JOIN invcategories c ON c.categoryID=g.categoryID "
                f"WHERE {where} "
                "LIMIT 1",
                params,
            )
            row = cur.fetchone()
            conn.close()
            if row:
                kind = _ItemKind(
                    canonical_name=str(row[0] or name),
                    group_name=str(row[1] or ""),
                    category_name=str(row[2] or ""),
                )
        except Exception:
            kind = None

        self._cache[cache_key] = kind
        return kind


class EftFitParser:
    _header_re = re.compile(r"^\[(?P<ship>[^,\]]+)\s*,\s*(?P<name>[^\]]+)\]$")
    _offline_suffixes = ("/offline", "/OFFLINE")
    _stack_re = re.compile(r"^(?P<name>.+?)\s+x(?P<qty>\d+)$", re.IGNORECASE)
    _mutation_suffix_re = re.compile(r"\s+\[(?P<ref>\d+)\]$")
    _mutation_detail_header_re = re.compile(r"^\[(?P<ref>\d+)\](?:\s+.*)?$")

    _classifier = _PyfaEveDbItemClassifier()

    @classmethod
    def _strip_mutation_suffix_with_ref(cls, line: str) -> tuple[str, int | None]:
        match = cls._mutation_suffix_re.search(line)
        if not match:
            return line.strip(), None
        return cls._mutation_suffix_re.sub("", line).strip(), int(match.group("ref"))

    @staticmethod
    def _parse_mutation_attributes(line: str) -> dict[str, float]:
        attributes: dict[str, float] = {}
        for pair in str(line or "").split(","):
            text = pair.strip()
            if not text:
                continue
            parts = text.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            attr_name, raw_value = parts[0].strip(), parts[1].strip()
            if not attr_name:
                continue
            try:
                attributes[attr_name] = float(raw_value)
            except Exception:
                continue
        return attributes

    @staticmethod
    def _contains_non_ascii(text: str) -> bool:
        return any(ord(ch) > 127 for ch in str(text or ""))

    @classmethod
    def _extract_mutation_specs(cls, source_lines: list[str], start_index: int) -> tuple[dict[int, ParsedMutationSpec], set[int]]:
        mutation_specs: dict[int, ParsedMutationSpec] = {}
        consumed_indices: set[int] = set()
        current_ref: int | None = None
        current_lines: list[str] = []
        current_indices: list[int] = []

        def complete_current() -> None:
            if current_ref is None:
                return
            consumed_indices.update(current_indices)
            if len(current_lines) < 2:
                return
            if cls._contains_non_ascii(current_lines[0]) or cls._contains_non_ascii(current_lines[1]):
                raise UserFacingError(
                    "Mutation block contains non-English item names. This is a known old pyfa bug; export EFT from English pyfa/client or update pyfa to the latest version."
                )
            attributes = cls._parse_mutation_attributes(current_lines[2]) if len(current_lines) >= 3 else {}
            mutation_specs[current_ref] = ParsedMutationSpec(
                base_item_name=current_lines[0].strip(),
                mutaplasmid_name=current_lines[1].strip(),
                attributes=attributes,
            )

        for idx in range(start_index, len(source_lines)):
            line = source_lines[idx]
            match = cls._mutation_detail_header_re.match(line) if line else None
            if match:
                complete_current()
                current_ref = int(match.group("ref"))
                tail = match.group(0).split("]", 1)[1].strip()
                current_lines = [tail] if tail else []
                current_indices = [idx]
            elif not line:
                complete_current()
                current_ref = None
                current_lines = []
                current_indices = []
            elif current_ref is not None:
                current_lines.append(line)
                current_indices.append(idx)
        complete_current()
        return mutation_specs, consumed_indices

    @staticmethod
    def _is_booster(kind: _ItemKind | None) -> bool:
        return str(getattr(kind, "category_name", "") or "").lower() == "implant" and str(
            getattr(kind, "group_name", "") or ""
        ).lower() == "booster"

    @staticmethod
    def _is_implant(kind: _ItemKind | None) -> bool:
        if kind is None:
            return False
        category = str(kind.category_name or "").lower()
        group = str(kind.group_name or "").lower()
        return category == "implant" and group != "booster"

    def parse(self, fit_text: str) -> ParsedEftFit:
        source_lines = [line.strip() for line in fit_text.splitlines()]
        nonblank_lines = [line for line in source_lines if line]
        if not nonblank_lines:
            raise UserFacingError("Fit text is empty.")

        header_index = next((idx for idx, line in enumerate(source_lines) if line), -1)
        match = self._header_re.match(nonblank_lines[0])
        if not match:
            raise UserFacingError("EFT header is invalid; expected [Ship, Fit Name].")

        ship_name = match.group("ship").strip()
        fit_name = match.group("name").strip()
        header_seen = False

        modules: list[str] = []
        module_specs: list[ParsedModuleSpec] = []
        cargo_item_names: list[str] = []
        cargo_specs: list[ParsedCargoSpec] = []
        implant_names: list[str] = []
        booster_names: list[str] = []

        mutation_specs, mutation_line_indices = self._extract_mutation_specs(source_lines, header_index + 1)
        normalized_lines_for_key: list[str] = [nonblank_lines[0]]

        for idx, raw in enumerate(source_lines):
            if not header_seen:
                if raw:
                    header_seen = True
                continue
            if idx in mutation_line_indices:
                continue
            if not raw:
                continue
            if raw.lower().startswith("dna:"):
                continue
            if raw.lower().startswith("x-"):
                continue
            line, mutation_ref = self._strip_mutation_suffix_with_ref(raw)
            stack_match = self._stack_re.match(line)
            if stack_match:
                qty_name = stack_match.group("name").strip()
                qty = max(1, int(stack_match.group("qty")))
                if qty_name:
                    cargo_item_names.append(qty_name)
                    cargo_specs.append(ParsedCargoSpec(item_name=qty_name, quantity=qty))
                    normalized_lines_for_key.append(f"{qty_name} x{qty}")
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
            kind = self._classifier.kind_for(line)
            if self._is_booster(kind):
                booster_names.append(line)
                normalized_lines_for_key.append(line)
                continue
            if self._is_implant(kind):
                implant_names.append(line)
                normalized_lines_for_key.append(line)
                continue
            modules.append(line)
            module_specs.append(
                ParsedModuleSpec(
                    module_name=line,
                    charge_name=charge_name,
                    offline=offline,
                    mutation_ref=mutation_ref,
                )
            )
            normalized_line = line
            if charge_name:
                normalized_line = f"{normalized_line}, {charge_name}"
            if offline:
                normalized_line = f"{normalized_line} /offline"
            if mutation_ref is not None:
                normalized_line = f"{normalized_line} [{mutation_ref}]"
            normalized_lines_for_key.append(normalized_line)

        for ref, mutation_spec in sorted(mutation_specs.items()):
            normalized_lines_for_key.append(f"[{ref}] {mutation_spec.base_item_name}")
            normalized_lines_for_key.append(mutation_spec.mutaplasmid_name)
            for attr_name, value in sorted(mutation_spec.attributes.items()):
                normalized_lines_for_key.append(f"{attr_name} {value:g}")

        fit_key = hashlib.sha1("\n".join(normalized_lines_for_key).encode("utf-8")).hexdigest()[:16]
        return ParsedEftFit(
            ship_name=ship_name,
            fit_name=fit_name,
            module_names=modules,
            module_specs=module_specs,
            cargo_item_names=cargo_item_names,
            fit_key=f"eft-{fit_key}",
            cargo_specs=cargo_specs,
            implant_names=implant_names,
            booster_names=booster_names,
            mutation_specs=mutation_specs,
        )


__all__ = ["EftFitParser"]
