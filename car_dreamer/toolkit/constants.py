import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@lru_cache(maxsize=1)
def _load_constants() -> Dict[str, Any]:
    constants_path = Path(__file__).resolve().parents[1] / "configs" / "constants.yaml"
    if not constants_path.exists():
        return {}
    with constants_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _extract_town_name(map_name: str) -> str:
    match = re.search(r"Town\d+", map_name or "")
    return match.group(0) if match else ""


def get_stop_sign_ids_for_map(map_name: str) -> Tuple[int, ...]:
    constants = _load_constants()
    default_ids = (
        constants.get("defaults", {}).get("traffic_signs", {}).get("stop_sign_ids", ())
    )
    town_name = _extract_town_name(map_name)
    map_ids = (
        constants.get("maps", {})
        .get(town_name, {})
        .get("traffic_signs", {})
        .get("stop_sign_ids", default_ids)
    )
    return tuple(int(x) for x in map_ids)
