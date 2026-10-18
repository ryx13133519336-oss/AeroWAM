"""Read selected primitive ULog topics without installing extra packages.

Diagnostic only. Does not modify flight logs or treat publication as receipt.
Unsupported nested field types fail explicitly for the selected topics.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct

TYPES = {"int8_t": "b", "uint8_t": "B", "int16_t": "h", "uint16_t": "H",
         "int32_t": "i", "uint32_t": "I", "int64_t": "q", "uint64_t": "Q",
         "float": "f", "double": "d", "bool": "?", "char": "c"}
SELECTED = {"vehicle_status", "vehicle_command", "vehicle_command_ack", "offboard_control_mode",
            "trajectory_setpoint", "vehicle_local_position", "failsafe_flags"}


def decode(definition, data):
    offset, result = 0, {}
    for field in definition.split(";"):
        if not field:
            continue
        kind, name = field.split()
        match = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", kind)
        if not match or match[1] not in TYPES:
            raise ValueError(f"Unsupported diagnostic field: {field}")
        count = int(match[2] or 1)
        fmt = "<" + TYPES[match[1]] * count
        size = struct.calcsize(fmt)
        # Trailing padding may be omitted from serialized data.
        if offset + size > len(data) and name.startswith("_padding"):
            break
        values = struct.unpack_from(fmt, data, offset)
        offset += size
        if not name.startswith("_padding"):
            result[name] = values[0] if count == 1 else values
    return result


def inspect(path):
    formats, topics = {}, {}
    counts, samples, distributions = Counter(), defaultdict(list), defaultdict(Counter)
    with path.open("rb") as source:
        if source.read(16)[:7] != b"ULog\x01\x12\x35":
            raise ValueError("ULog header required")
        while header := source.read(3):
            if len(header) != 3:
                break  # A killed simulator may have an incomplete final record.
            size, kind = struct.unpack("<HB", header)
            payload = source.read(size)
            if len(payload) != size:
                break
            if kind == ord("F"):
                name, definition = payload.decode().split(":", 1)
                formats[name] = definition
            elif kind == ord("A"):
                topics[struct.unpack_from("<H", payload, 1)[0]] = payload[3:].decode()
            elif kind == ord("D"):
                name = topics.get(struct.unpack_from("<H", payload)[0])
                if name not in SELECTED:
                    continue
                row = decode(formats[name], payload[2:])
                counts[name] += 1
                samples[name].append(row)
                samples[name] = samples[name][-3:]
                for field in ("nav_state", "arming_state", "command", "result", "target_system"):
                    if field in row:
                        distributions[f"{name}.{field}"][row[field]] += 1
    return {"file": str(path), "counts": dict(counts), "distributions": {k: dict(v) for k, v in distributions.items()},
            "last": dict(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    result = inspect(parser.parse_args().log)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
