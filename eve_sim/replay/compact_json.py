from __future__ import annotations

from typing import Any


KEY_TO_SHORT = {
    "schema_version": "v",
    "scenario_id": "sc",
    "rng_seed": "rs",
    "metadata": "md",
    "events": "e",
    "frames": "f",
    "tick": "t",
    "at": "a",
    "now": "n",
    "kind": "k",
    "source_id": "si",
    "target_id": "ti",
    "module_id": "mi",
    "rng_counter": "rc",
    "payload": "p",
    "world": "w",
    "patch": "d",
    "removed": "rm",
    "ships": "sh",
    "drones": "dr",
    "fighters": "fi",
    "projectiles": "pr",
    "projectile_blasts": "pb",
    "bubble_fields": "bf",
    "intents": "in",
    "squad_focus_queues": "fq",
    "squad_focus_updated_at": "fu",
    "movement_mode": "mm",
    "ship_id": "id",
    "team": "tm",
    "squad_id": "sq",
    "owner_squad_id": "osq",
    "owner_ship_id": "osi",
    "ship_name": "sn",
    "alive": "al",
    "connected": "cn",
    "state": "sts",
    "type_name": "tn",
    "group_name": "gn",
    "slot_kind": "sk",
    "squadron_size": "ssz",
    "is_sentry": "ise",
    "target_command_at": "tca",
    "position": "pos",
    "velocity": "vel",
    "facing_deg": "fd",
    "system_id": "sys",
    "command_mode": "cm",
    "command_target_ship_id": "cts",
    "command_target_structure_id": "ctu",
    "command_range_m": "crm",
    "command_orbit_clockwise": "coc",
    "gate_target_structure_id": "gt",
    "gate_cloak_active": "gca",
    "gate_cloak_expires_at": "gce",
    "gate_cloak_source": "gcs",
    "follow_hold_active": "fha",
    "follow_hold_leader_id": "fhl",
    "shield": "sl",
    "armor": "ar",
    "structure": "st",
    "shield_max": "slm",
    "armor_max": "arm",
    "structure_max": "stm",
    "cap": "cp",
    "cap_max": "cpm",
    "target": "tg",
    "projected_targets": "pt",
    "module_cycle_timers": "mct",
    "ecm_jam_sources": "ejs",
    "ecm_last_attempt_target": "ejt",
    "ecm_last_attempt_module": "ejm",
    "ecm_last_attempt_success": "ejsu",
    "ecm_last_attempt_chance": "ejc",
    "ecm_last_attempt_at": "eja",
    "ecm_last_attempt_target_by_module": "ejtb",
    "ecm_last_attempt_success_by_module": "ejsb",
    "ecm_last_attempt_at_by_module": "ejab",
    "module_states": "mst",
    "cycle_timer": "ct",
    "ewar_cycle_timer": "ect",
    "ability_cycle_timers": "act",
    "ability_ammo_remaining": "aar",
    "ability_reload_timers": "art",
    "pending_manual_abilities": "pma",
    "mwd_active_timer": "mat",
    "mwd_cooldown_timer": "mctd",
    "projectile_id": "pid",
    "source_ship_id": "ssi",
    "source_module_id": "smi",
    "target_ship_id": "tsi",
    "target_structure_id": "tsu",
    "target_range_m": "trm",
    "speed": "spd",
    "max_speed": "msp",
    "distance_traveled": "dst",
    "flight_time": "flt",
    "age": "ag",
    "blast_radius": "br",
    "blast_id": "bid",
    "radius_m": "rad",
    "expires_at": "exp",
    "field_id": "fid",
    "interdiction_kind": "ik",
    "blocks_warp": "bw",
    "speed_factor_mult": "sf",
    "anchor_ship_id": "asi",
    "started": "sta",
    "countdown_left": "cd",
    "tidi_factor": "tf",
    "engine_config": "ec",
    "map": "mp",
    "partial": "pa",
    "removed_ship_ids": "rsi",
    "total_damage": "td",
    "applied_damage": "ad",
    "shield_repaired": "sr",
    "armor_repaired": "rr",
    "duration_s": "dur",
    "cycle_time": "cy",
    "effects": "fx",
    "group": "grp",
    "target_team": "tt",
    "em": "em",
    "thermal": "th",
    "kinetic": "ki",
    "explosive": "ex",
}

SHORT_TO_KEY = {value: key for key, value in KEY_TO_SHORT.items()}

KIND_TO_SHORT = {
    "keyframe": "K",
    "delta": "D",
}
SHORT_TO_KIND = {value: key for key, value in KIND_TO_SHORT.items()}


def _encode_key(key: str) -> str:
    if key in KEY_TO_SHORT:
        return KEY_TO_SHORT[key]
    if key in SHORT_TO_KEY or key.startswith("~"):
        return f"~{key}"
    return key


def _decode_key(key: str) -> str:
    if key in SHORT_TO_KEY:
        return SHORT_TO_KEY[key]
    if key.startswith("~"):
        return key[1:]
    return key


def compact_replay_data(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            compacted[_encode_key(key)] = compact_replay_data(raw_value, parent_key=key)
        return compacted
    if isinstance(value, list):
        return [compact_replay_data(item, parent_key=parent_key) for item in value]
    if parent_key == "kind" and isinstance(value, str):
        return KIND_TO_SHORT.get(value, value)
    return value


def expand_replay_data(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        expanded: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _decode_key(str(raw_key))
            expanded[key] = expand_replay_data(raw_value, parent_key=key)
        return expanded
    if isinstance(value, list):
        return [expand_replay_data(item, parent_key=parent_key) for item in value]
    if parent_key == "kind" and isinstance(value, str):
        return SHORT_TO_KIND.get(value, value)
    return value


def expand_replay_document(data: dict[str, Any]) -> dict[str, Any]:
    if any(key in data for key in ("v", "sc", "rs", "md", "e", "f", "ss")):
        expanded = expand_replay_data(data)
        return expanded if isinstance(expanded, dict) else data
    return data
