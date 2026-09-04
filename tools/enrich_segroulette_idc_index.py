#!/usr/bin/env python3
"""Enrich every SegRoulette row with idc-index metadata into a *new* JSON file.

Does not modify the original segroulette.json.

Requires: pip install idc-index

    python SlicerLive/tools/enrich_segroulette_idc_index.py
    python SlicerLive/tools/enrich_segroulette_idc_index.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "render" / "demos" / "segroulette.json"
DEFAULT_OUT = REPO_ROOT / "render" / "demos" / "segroulette-idc.json"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [json_safe(v) for v in value.tolist()]
    except ImportError:
        pass
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            return json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    text = str(value)
    if text.lower() in ("nan", "nat", "none"):
        return None
    return text


def row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        data = row.to_dict()
    else:
        data = dict(row)
    return {str(k): json_safe(v) for k, v in data.items()}


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def sql_in_list(uuids: list[str]) -> str:
    escaped = []
    for u in uuids:
        s = str(u).replace("'", "")
        if not s:
            continue
        escaped.append(f"'{s}'")
    return ",".join(escaped)


def fetch_index_rows(client: Any, uuids: list[str], chunk_size: int) -> dict[str, dict[str, Any]]:
    by_uuid: dict[str, dict[str, Any]] = {}
    unique = sorted(set(u for u in uuids if u))
    print(f"Looking up {len(unique)} crdc_series_uuid values in idc-index …", flush=True)
    for i, chunk in enumerate(chunked(unique, chunk_size), start=1):
        in_list = sql_in_list(chunk)
        if not in_list:
            continue
        df = client.sql_query(
            f"SELECT * FROM index WHERE crdc_series_uuid IN ({in_list})"
        )
        for _, rec in df.iterrows():
            uid = str(rec.get("crdc_series_uuid") or "")
            if uid:
                by_uuid[uid] = row_to_dict(rec)
        print(f"  chunk {i}: {len(chunk)} queried, {len(by_uuid)} unique hits so far", flush=True)
    return by_uuid


def num_mb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(n) or math.isinf(n):
        return None
    return round(n, 3)


def modalities_for(source_mod: str, has_seg: bool) -> list[str]:
    out: list[str] = []
    m = (source_mod or "").strip().upper()
    if m:
        if m == "PT":
            out.append("PT")
        else:
            out.append(m)
    if has_seg and "SEG" not in out:
        out.append("SEG")
    return out


def enrich_row(row: dict[str, Any], by_uuid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(row)
    ct_id = str(row.get("c") or "").strip()
    seg_id = str(row.get("s") or "").strip()
    ct_idx = by_uuid.get(ct_id)
    seg_idx = by_uuid.get(seg_id) if seg_id else None
    if ct_idx:
        out["ctIdx"] = ct_idx
    if seg_idx:
        out["segIdx"] = seg_idx

    src_mb = num_mb((ct_idx or {}).get("series_size_MB"))
    if src_mb is None:
        src_mb = num_mb(row.get("sz"))
    seg_mb = num_mb((seg_idx or {}).get("series_size_MB")) if seg_idx else None

    total = (src_mb or 0.0) + (seg_mb or 0.0)
    out["sizeSourceMB"] = src_mb
    out["sizeSegMB"] = seg_mb
    out["sizeMB"] = round(total, 3) if (src_mb is not None or seg_mb is not None) else None

    source_mod = str((ct_idx or {}).get("Modality") or row.get("m") or "").strip()
    out["modalities"] = modalities_for(source_mod, bool(seg_id))
    return out


def assert_safe_out_path(path: Path) -> None:
    if path.name.lower() == "segroulette.json":
        raise SystemExit(
            f"Refusing to write {path} — original segroulette.json must stay unchanged. "
            "Use a new filename such as segroulette-idc.json."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Optional directory to copy the new file into (not segroulette.json).",
    )
    args = parser.parse_args()

    in_path = args.in_path.resolve()
    out_path = args.out_path.resolve()
    assert_safe_out_path(out_path)
    if args.copy_to:
        copy_dest = (args.copy_to / out_path.name).resolve()
        assert_safe_out_path(copy_dest)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise SystemExit(f"Expected {{stats, rows}} in {in_path}")
    rows: list[dict[str, Any]] = data["rows"]
    print(f"Loaded {len(rows)} rows from {in_path}", flush=True)

    uuids: list[str] = []
    for row in rows:
        if row.get("c"):
            uuids.append(str(row["c"]))
        if row.get("s"):
            uuids.append(str(row["s"]))

    from idc_index import IDCClient

    client = IDCClient()
    if args.refresh_index:
        print("Refreshing idc-index …", flush=True)
        client.fetch_index()

    by_uuid = fetch_index_rows(client, uuids, max(1, args.chunk_size))

    missing_ct: list[str] = []
    missing_seg: list[str] = []
    enriched: list[dict[str, Any]] = []
    for row in rows:
        ct_id = str(row.get("c") or "").strip()
        seg_id = str(row.get("s") or "").strip()
        if ct_id and ct_id not in by_uuid:
            missing_ct.append(ct_id)
        if seg_id and seg_id not in by_uuid:
            missing_seg.append(seg_id)
        enriched.append(enrich_row(row, by_uuid))

    stats = dict(data.get("stats") or {})
    stats["idcVersion"] = getattr(client, "idc_version", stats.get("idcVersion"))
    stats["enrichedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats["source"] = in_path.name
    stats["rowCount"] = len(enriched)

    payload = {"stats": stats, "rows": enriched}

    print(
        f"Missing CT UUIDs: {len(set(missing_ct))}  Missing SEG UUIDs: {len(set(missing_seg))}",
        flush=True,
    )
    sample = next((r for r in enriched if r.get("col") == "c4kc_kits"), enriched[0])
    print(
        "Sample:",
        sample.get("col"),
        "sizeMB=",
        sample.get("sizeMB"),
        "modalities=",
        sample.get("modalities"),
        flush=True,
    )

    if args.dry_run:
        print("Dry run — not writing.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    tmp.replace(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(enriched)} rows)")

    if args.copy_to:
        dest = Path(args.copy_to).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        dest_file = dest / out_path.name
        assert_safe_out_path(dest_file)
        shutil.copyfile(out_path, dest_file)
        print(f"Copied to {dest_file}")


if __name__ == "__main__":
    main()
