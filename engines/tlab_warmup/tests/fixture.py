"""Deterministic small Parquet fixture with known periods, ties and repeats."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tlab_warmup.data import sha256


def make_fixture(directory):
    rows = []
    for u in range(4):
        for j in range(120):
            kind = ("like" if j % 17 == 0 else "dislike" if j % 29 == 0 else
                    "unlike" if j % 31 == 0 else "undislike" if j % 37 == 0 else "listen")
            rows.append({"uid": u, "item_id": (j * 3 + u * 7) % 35,
                         "timestamp": j * 50 + u * 5, "event_type": kind,
                         "is_organic": j % 2, "played_ratio_pct": 80 if kind == "listen" else None,
                         "track_length_seconds": 200 if kind == "listen" else None})
    rows.append(dict(rows[80]))
    rows.sort(key=lambda r: (r["uid"], r["timestamp"]))
    path = Path(directory) / "fixture.parquet"
    schema = pa.schema([("uid", pa.uint32()), ("item_id", pa.uint32()),
                        ("timestamp", pa.uint32()), ("event_type", pa.string()),
                        ("is_organic", pa.uint8()), ("played_ratio_pct", pa.uint16()),
                        ("track_length_seconds", pa.uint32())])
    table = pa.Table.from_pylist(rows, schema=schema)
    # Match the actual release's Arrow enum representation, including uint32 indices.
    dictionary = table["event_type"].dictionary_encode().cast(pa.dictionary(pa.uint32(), pa.string()))
    table = table.set_column(table.schema.get_field_index("event_type"), "event_type", dictionary)
    pq.write_table(table, path, row_group_size=37)
    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "configs/warmup.json").read_text())
    c["source"].update(synthetic=True, revision="synthetic-fixture-v1", bytes=path.stat().st_size, sha256=sha256(path))
    c.update(users=3, training_queries=30, development_queries=12, window_seconds=500)
    c["model"].update(max_iter=5, min_samples_leaf=3)
    return path, c
