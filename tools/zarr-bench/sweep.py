#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
#
# [tool.uv]
# # As generate.py: never the system interpreter, so that the sweep runs the same everywhere.
# python-preference = "only-managed"
# ///

"""Find the layout, and the carta-backend settings, that read fastest on this storage.

Driven by a TOML file (see example-sweep.toml), in four stages that each write into one results.csv:

  stage1    Every layout, written by generate.py in turn, read by carta-zarr-bench at the baseline
            settings with one user and with the target number, then deleted. The source's own
            layout, "current", is always one of them when there is a source.
  stage2    The best layouts of stage1, and current, read at every setting of the reader grid with
            the target number of users.
  confirm   The recommended settings and their nearest rivals, and the baseline, with one user and
            the typical number: how much the choice for the target costs everyone else.
  validate  The recommended layout again, larger than RAM, at the recommended settings and the
            baseline: whether what a 50-100 GB copy with its caches emptied said still holds.
  warm      Only when [warm] lists builds of carta-zarr-bench whose library replaces tuning
            constants: each of them and the default build on the recommended layout and settings,
            warm, for the reductions those constants divide. Reported, never recommended.

summary.md says what to set and why. Every stage resumes: a run already in sweep-state.json is not
repeated, and a bench interrupted part-way resumes from the CSV.

With [[source.shape]] tables, each shape -- a crop of the source, or a synthetic shape -- is a sweep of
its own in a directory of its own, since the best chunk depth depends on the size of the plane, and
the top-level summary.md compares them and says what one layout and one setting would serve them all.

  sweep.py CONFIG.toml [--output DIR] [--set KEY=VALUE ...] [--dry-run] [--report-only]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODES = ("plane", "animation", "spectrum", "region", "cube-histogram", "open")
# The modes ranked on first touches alone: the operations that read no chunk another operation of the
# trial read before them or alongside them, which carta-zarr-bench marks shares_chunks = false.
FIRST_TOUCH_MODES = ("plane", "spectrum", "region")
# Fewer first touches than this in a group, when some of its operations were not, and it is ranked on
# every operation, with a warning -- and so is every group it is compared with, the same stage, number
# of users, mode and method on other layouts and settings, so that no layout is ranked on cache hits
# its rivals were spared.
MIN_FIRST_TOUCHES = 5
# carta-zarr-bench's operations per user per trial when measure.ops does not say.
BENCH_DEFAULT_OPS = {"plane": 16, "animation": 2, "spectrum": 32, "region": 1, "cube-histogram": 1, "open": 8}
# The modes of the trade-off between deep chunks and shallow ones, in the order the table shows them.
TRADEOFF_MODES = ("plane", "animation", "spectrum", "region", "cube-histogram")
AXES = ("time", "frequency", "polarization", "l", "m")

# How much slower than its own best a mode may be at the recommended settings before the report says
# that mode was given up for the others.
SACRIFICE = 1.25
# How much faster a read budget must make some mode before the report suggests a backend setting for it.
READ_BUDGET_GAIN = 0.10
# How far validation may move a median before the report says the smaller copy misled.
VALIDATE_DRIFT = 0.20
# How much faster than the default build a tuning variant must make a mode before the report says so.
TUNING_GAIN = 0.10

# carta-backend's own defaults for the settings it has: --zarr_file_io_threads 2,
# --zarr_data_copy_threads the OpenMP thread count (every logical core), --zarr_cache_size 1024 MiB.
# There is no backend setting for the read budget, so the library's own (0) is the only one it uses.
BACKEND_DEFAULTS = {"io_threads": 2, "decode_threads": "cores", "cache_mib": 1024, "read_budget": "0"}

DEFAULTS: dict[str, Any] = {
    "paths": {"bench": "", "work": "", "generator": str(HERE / "generate.py"), "keep_datasets": False},
    "source": {"path": "", "image": "", "crop": "", "validate_crop": "",
               "synthetic": "", "validate_synthetic": "", "seed": 1, "shape": []},
    "users": {"target": 1, "typical": 0},
    "measure": {"modes": list(MODES), "trials": 5, "trial_timeout": 600, "seed": 1, "cold": "auto",
                "drop_cache_cmd": "", "ops": {}, "region_fraction": 0.05, "animation_frames": 32,
                "histogram_reference": [],
                "generator_workers": 0},
    "baseline": dict(BACKEND_DEFAULTS),
    "weights": {mode: 1.0 for mode in MODES},
    "stage1": {"chunk": [], "shard": [""], "codec": ["zstd:3"], "consolidated": [True], "stripe": [""],
               "layout": []},
    "stage2": {"top": 2, "layouts": [], "trials": 0, "io_threads": [], "decode_threads": [], "cache_mib": [],
               "read_budget": ["0"]},
    "checks": {"ram": True, "space": True},
    "warm": {"variants": [], "modes": ["region", "cube-histogram"], "trials": 0},
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# -- Configuration --------------------------------------------------------------------------------


def merge(defaults: dict[str, Any], given: dict[str, Any], where: str = "") -> dict[str, Any]:
    result = json.loads(json.dumps(defaults))
    for key, value in given.items():
        if key not in defaults:
            raise SystemExit(f"unknown setting {where}{key}")
        if isinstance(defaults[key], dict) and key not in ("ops", "weights"):
            if not isinstance(value, dict):
                raise SystemExit(f"{where}{key} must be a table")
            result[key] = merge(defaults[key], value, f"{where}{key}.")
        else:
            result[key] = value
    return result


def apply_set(config: dict[str, Any], assignment: str) -> None:
    """--set a.b=value, the value read as TOML when it is a TOML value and as a string otherwise."""
    path, sep, text = assignment.partition("=")
    if not sep:
        raise SystemExit(f"--set {assignment!r} is not KEY=VALUE")
    try:
        value = tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        value = text
    keys = path.strip().split(".")
    table = config
    for key in keys[:-1]:
        table = table.setdefault(key, {})
        if not isinstance(table, dict):
            raise SystemExit(f"--set {path}: {key} is not a table")
    table[keys[-1]] = value


def load_config(path: Path, assignments: list[str]) -> dict[str, Any]:
    try:
        given = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from None
    for assignment in assignments:
        apply_set(given, assignment)
    config = merge(DEFAULTS, given)

    if not config["paths"]["bench"] or not config["paths"]["work"]:
        raise SystemExit("paths.bench and paths.work are required")
    source = config["source"]
    if source["shape"]:
        check_shapes(source)
    elif bool(source["path"]) == bool(source["synthetic"]):
        raise SystemExit("source needs exactly one of path (a cube to rewrite) and synthetic (a shape)")
    for mode in config["measure"]["modes"]:
        if mode not in MODES:
            raise SystemExit(f"measure.modes: unknown mode {mode!r}")
    for mode, weight in config["weights"].items():
        if mode not in MODES or not isinstance(weight, (int, float)) or weight < 0:
            raise SystemExit(f"weights.{mode}: a mode's weight is a number of at least zero")
    users = config["users"]
    if users["target"] < 1:
        raise SystemExit("users.target must be at least one")
    if not config["stage1"]["chunk"] and not config["stage1"]["layout"] and not source["path"]:
        raise SystemExit("stage1 has no layouts: give chunk values or [[stage1.layout]] entries")
    names = set()
    for variant in config["warm"]["variants"]:
        if not isinstance(variant, dict) or set(variant) != {"name", "bench"}:
            raise SystemExit("warm.variants: each is a table of name and bench, and nothing else")
        if variant["name"] in names or variant["name"] == "default" or not re.fullmatch(r"[\w.-]+", variant["name"]):
            raise SystemExit(f"warm.variants: {variant['name']!r} is not a name of its own (letters, digits, "
                             "._-, and not \"default\")")
        names.add(variant["name"])
    for mode in config["warm"]["modes"]:
        if mode not in MODES:
            raise SystemExit(f"warm.modes: unknown mode {mode!r}")
    return config


SHAPE_KEYS = {"name", "crop", "synthetic", "validate_crop", "validate_synthetic", "channels", "weight"}


def check_shapes(source: dict[str, Any]) -> None:
    """[[source.shape]]: each named, each a crop of source.path or, without one, a synthetic shape."""
    if source["synthetic"] or source["crop"] or source["validate_crop"] or source["validate_synthetic"]:
        raise SystemExit("with [[source.shape]], crop and synthetic go in each shape, not in [source]")
    names = set()
    for shape in source["shape"]:
        if not isinstance(shape, dict) or not set(shape) <= SHAPE_KEYS or "name" not in shape:
            raise SystemExit(f"[[source.shape]]: each is a table of name and {', '.join(sorted(SHAPE_KEYS - {'name'}))}")
        name = shape["name"]
        if name in names or not re.fullmatch(r"[\w.-]+", str(name)):
            raise SystemExit(f"[[source.shape]]: {name!r} is not a name of its own (letters, digits, ._-)")
        names.add(name)
        if source["path"] and ("synthetic" in shape or "validate_synthetic" in shape):
            raise SystemExit(f"[[source.shape]] {name}: a crop of source.path takes crop, not synthetic")
        if not source["path"] and not shape.get("synthetic"):
            raise SystemExit(f"[[source.shape]] {name}: without source.path, each shape needs synthetic")
        if "channels" in shape and (not isinstance(shape["channels"], int) or shape["channels"] < 1):
            raise SystemExit(f"[[source.shape]] {name}: channels is the cube's full depth, a positive whole number")
        if "weight" in shape and (not isinstance(shape["weight"], (int, float)) or shape["weight"] < 0):
            raise SystemExit(f"[[source.shape]] {name}: weight is a number of at least zero")


def shape_config(config: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    """The configuration of one shape's sweep: the source cropped or synthesized as the shape says."""
    result = json.loads(json.dumps(config))
    source = result["source"]
    for key in ("crop", "synthetic", "validate_crop", "validate_synthetic"):
        source[key] = shape.get(key, "")
    source["shape"] = []
    return result


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(value)


def dump_config(config: dict[str, Any]) -> str:
    """The configuration as TOML, with every default filled in: what the sweep actually ran."""
    lines = []
    for section, table in config.items():
        lines.append(f"[{section}]")
        for key, value in table.items():
            if key == "shape" and isinstance(value, list):
                continue
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
        if section == "source":
            for shape in table.get("shape", []):
                lines.append("[[source.shape]]")
                lines += [f"{key} = {toml_value(value)}" for key, value in shape.items()]
                lines.append("")
    return "\n".join(lines)


# -- Layouts --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Layout:
    name: str
    chunk: str = ""
    shard: str = ""
    codec: str = "zstd:3"
    consolidated: bool = True
    stripe: str = ""
    current: bool = False

    @property
    def key(self) -> str:
        if self.current:
            return "current"
        parts = [f"chunk={self.chunk}", f"shard={self.shard or '-'}", f"codec={self.codec}",
                 "consolidated" if self.consolidated else "unconsolidated"]
        if self.stripe:
            parts.append(f"stripe={self.stripe}")
        return "|".join(parts)

    def describe(self) -> str:
        if self.current:
            return "the source's own layout"
        text = f"chunk {self.chunk}"
        if self.shard:
            text += f", shard {self.shard}"
        text += f", {self.codec}"
        if not self.consolidated:
            text += ", unconsolidated"
        if self.stripe:
            text += f", stripe {self.stripe}"
        return text

    def generator_args(self) -> list[str]:
        if self.current:
            return ["--layout-from-source"]
        args = ["--chunk", self.chunk, "--codec", self.codec]
        if self.shard:
            args += ["--shard", self.shard]
        if not self.consolidated:
            args.append("--no-consolidate")
        if self.stripe:
            args += ["--stripe", self.stripe]
        return args


def axis_map(text: str) -> dict[str, int]:
    result = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, sep, value = item.partition("=")
        if not sep or name not in AXES or not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{item!r} is not axis=positive-integer over {', '.join(AXES)}")
        result[name] = int(value)
    return result


CODEC = re.compile(r"^(none|zstd(:\d+)?|gzip(:\d+)?|blosc(:\w+(:\d+(:(noshuffle|shuffle|bitshuffle))?)?)?)$")
STRIPE = re.compile(r"^(lustre|beegfs):count=\d+,size=\d+[kKmMgG]?(i?B)?$")


def resolve_shard(chunk_text: str, shard_text: str) -> str:
    """A shard shape with every entry absolute: "frequency*8" is eight of the chunk's own along the
    spectral axis, and "frequency=256" is 256 whatever the chunk. "" when it comes to the chunk
    itself, which is no shard. Raises ValueError for an entry that is neither."""
    chunk = axis_map(chunk_text)
    shard: dict[str, int] = {}
    for item in filter(None, (part.strip() for part in shard_text.split(","))):
        name, times, factor = item.partition("*")
        if times:
            if name not in AXES or not factor.isdigit() or int(factor) <= 0:
                raise ValueError(f"{item!r} is not axis*positive-integer over {', '.join(AXES)}")
            shard[name] = chunk.get(name, 1) * int(factor)
        else:
            shard.update(axis_map(item))
    if all(length == chunk.get(name, 1) for name, length in shard.items()):
        return ""
    return ",".join(f"{name}={shard[name]}" for name in AXES if name in shard)


def invalid_reason(layout: Layout, lengths: dict[str, int] | None = None) -> str | None:
    """Why the generator would refuse this layout, or why it would misrepresent a cube of `lengths`
    (its axes' lengths, when they are known), found before anything is written."""
    if layout.current:
        return None
    try:
        chunk = axis_map(layout.chunk)
        shard = axis_map(layout.shard)
    except ValueError as error:
        return str(error)
    if not chunk:
        return "no chunk shape"
    for name, length in chunk.items():
        # A chunk the cube cannot fill is not the chunk a cube of the full size would have: a
        # spectrum would read one part-empty chunk where the real one reads several full ones.
        if lengths and name in lengths and length > lengths[name]:
            return f"chunk {name}={length} is larger than the cube's {lengths[name]}"
    for name, length in shard.items():
        inner = chunk.get(name, 1)
        if length % inner:
            return f"shard {name}={length} is not a multiple of its chunk, {inner}"
    if not CODEC.match(layout.codec):
        return f"codec {layout.codec!r} is not none, zstd[:level], gzip[:level] or blosc[:cname[:level[:shuffle]]]"
    if layout.stripe and not STRIPE.match(layout.stripe):
        return f"stripe {layout.stripe!r} is not lustre:count=N,size=S or beegfs:count=N,size=S"
    return None


def stage1_layouts(config: dict[str, Any]) -> tuple[list[Layout], list[tuple[Layout, str]]]:
    """Every layout stage1 measures, current first, and those it cannot, with why."""
    stage = config["stage1"]
    candidates: list[Layout] = []
    if config["source"]["path"]:
        candidates.append(Layout("current", current=True))
    index = 0
    refused: dict[str, str] = {}

    def resolved(name: str, chunk: str, shard: str, *rest: Any) -> Layout:
        try:
            shard = resolve_shard(chunk, shard)
        except ValueError as error:
            refused[name] = str(error)
        return Layout(name, chunk, shard, *rest)

    for chunk in stage["chunk"]:
        for shard in stage["shard"]:
            for codec in stage["codec"]:
                for consolidated in stage["consolidated"]:
                    for stripe in stage["stripe"]:
                        index += 1
                        candidates.append(resolved(f"L{index:02d}", chunk, shard, codec, consolidated, stripe))
    for entry in stage["layout"]:
        index += 1
        unknown = set(entry) - {"name", "chunk", "shard", "codec", "consolidated", "stripe"}
        if unknown:
            raise SystemExit(f"[[stage1.layout]] has unknown keys {sorted(unknown)}")
        candidates.append(resolved(entry.get("name", f"L{index:02d}"), entry.get("chunk", ""), entry.get("shard", ""),
                                   entry.get("codec", "zstd:3"), entry.get("consolidated", True),
                                   entry.get("stripe", "")))
    lengths = cube_lengths(config, False)
    valid, invalid, seen = [], [], set()
    for layout in candidates:
        if layout.key in seen:
            continue
        seen.add(layout.key)
        reason = refused.get(layout.name) or invalid_reason(layout, lengths)
        if reason:
            invalid.append((layout, reason))
        else:
            valid.append(layout)
    names = [layout.name for layout in valid]
    if len(set(names)) != len(names):
        raise SystemExit("two stage1 layouts share a name")
    return valid, invalid


# -- Reader settings ------------------------------------------------------------------------------


def parse_size(text: str | int) -> int:
    if isinstance(text, int):
        return text
    match = re.fullmatch(r"(\d+)\s*([KMGT]?)(i?B)?", str(text).strip())
    if not match:
        raise SystemExit(f"{text!r} is not a size")
    return int(match.group(1)) << {"": 0, "K": 10, "M": 20, "G": 30, "T": 40}[match.group(2)]


def resolve_threads(value: Any, cores: int) -> int:
    """A thread count, or "cores" or "cores/N", resolved on this machine."""
    if isinstance(value, int) and value >= 0:
        return value
    match = re.fullmatch(r"cores(?:/(\d+))?", str(value).strip())
    if not match:
        raise SystemExit(f"{value!r} is not a thread count, cores, or cores/N")
    return max(1, cores // int(match.group(1) or 1))


@dataclasses.dataclass(frozen=True)
class Setting:
    io_threads: int
    decode_threads: int
    cache_mib: int
    read_budget: int  # bytes; zero is the library's own

    @property
    def key(self) -> tuple[str, str, str, str]:
        """As the CSV spells it, which is how a row is matched back to its setting."""
        return (str(self.io_threads), str(self.decode_threads), str(self.cache_mib << 20), str(self.read_budget))

    def describe(self) -> str:
        budget = "library" if self.read_budget == 0 else f"{self.read_budget >> 20} MiB"
        return f"io {self.io_threads}, decode {self.decode_threads}, cache {self.cache_mib} MiB, budget {budget}"

    def bench_args(self) -> list[str]:
        return ["--io-threads", str(self.io_threads), "--decode-threads", str(self.decode_threads),
                "--cache-bytes", str(self.cache_mib << 20), "--read-budget-bytes", str(self.read_budget)]

    def backend_flags(self) -> str:
        return (f"--zarr_file_io_threads {self.io_threads} --zarr_data_copy_threads {self.decode_threads} "
                f"--zarr_cache_size {self.cache_mib}")

    def settings_json(self) -> str:
        return json.dumps({"zarr_file_io_threads": self.io_threads, "zarr_data_copy_threads": self.decode_threads,
                           "zarr_cache_size": self.cache_mib})


def setting_of(table: dict[str, Any], cores: int) -> Setting:
    return Setting(resolve_threads(table["io_threads"], cores), resolve_threads(table["decode_threads"], cores),
                   int(table["cache_mib"]), parse_size(table["read_budget"]))


def reader_grid(config: dict[str, Any], cores: int) -> list[Setting]:
    stage, base = config["stage2"], config["baseline"]
    axes = {name: stage[name] or [base[name]] for name in ("io_threads", "decode_threads", "cache_mib", "read_budget")}
    grid = []
    for io in axes["io_threads"]:
        for decode in axes["decode_threads"]:
            for cache in axes["cache_mib"]:
                for budget in axes["read_budget"]:
                    setting = setting_of({"io_threads": io, "decode_threads": decode, "cache_mib": cache,
                                          "read_budget": budget}, cores)
                    if setting not in grid:
                        grid.append(setting)
    return grid


# -- The machine and the source -------------------------------------------------------------------


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return int(command_output(["sysctl", "-n", "hw.memsize"]) or 0)


def filesystem_of(path: Path) -> str:
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return "unknown"
    resolved = str(path.resolve())
    best, kind = "", "unknown"
    for line in mounts.read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mountpoint = fields[1].replace("\\040", " ")
        if (resolved == mountpoint or resolved.startswith(mountpoint.rstrip("/") + "/")) and len(mountpoint) > len(best):
            best, kind = mountpoint, fields[2]
    return kind


def machine_info(work: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cores": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "work": str(work),
        "work_filesystem": filesystem_of(work if work.exists() else work.parent),
        "python": platform.python_version(),
    }
    lscpu = command_output(["lscpu"])
    if lscpu:
        fields = dict(line.split(":", 1) for line in lscpu.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        info["cpu"] = fields.get("Model name", "")
        for key in ("Socket(s)", "Core(s) per socket", "Thread(s) per core", "L2 cache", "L3 cache", "NUMA node(s)"):
            if key in fields:
                info[key.lower().replace("(s)", "s").replace(" ", "_")] = fields[key]
    else:
        info["cpu"] = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        for key, name in (("hw.physicalcpu", "physical_cores"), ("hw.l2cachesize", "l2_cache"),
                          ("hw.l3cachesize", "l3_cache")):
            if value := command_output(["sysctl", "-n", key]):
                info[name] = value
    return info


def default_image(root: dict[str, Any]) -> str:
    groups = root.get("attributes", {}).get("data_groups", {})
    base = groups.get("base", {}) if isinstance(groups, dict) else {}
    return base.get("sky", "SKY") if isinstance(base, dict) else "SKY"


def cube_lengths(config: dict[str, Any], validate: bool) -> dict[str, int]:
    """The length of each axis of the cube a dataset of this sweep holds, by XRADIO's axis name. Empty
    when the source cannot be read, as in a configuration checked before its source exists."""
    source = config["source"]
    if source["synthetic"]:
        return axis_map(source["validate_synthetic"] if validate else source["synthetic"])
    root = Path(source["path"])
    try:
        image = source["image"] or default_image(json.loads((root / "zarr.json").read_text()))
        metadata = json.loads((root / image / "zarr.json").read_text())
    except (OSError, ValueError):
        return {}
    lengths = dict(zip(metadata.get("dimension_names") or [], metadata["shape"]))
    crop = source["validate_crop"] if validate else source["crop"]
    for item in filter(None, crop.split(",")):
        name, _, window = item.partition("=")
        start, _, stop = window.partition(":")
        if name in lengths:
            lengths[name] = int(stop) - int(start)
    return lengths


def item_size(config: dict[str, Any]) -> int:
    source = config["source"]
    if source["synthetic"]:
        return 4
    root = Path(source["path"])
    image = source["image"] or default_image(json.loads((root / "zarr.json").read_text()))
    data_type = json.loads((root / image / "zarr.json").read_text())["data_type"]
    return {"float32": 4, "float64": 8, "int16": 2, "int32": 4, "uint8": 1}.get(data_type, 4)


def uncompressed_bytes(config: dict[str, Any], validate: bool) -> int:
    """What one dataset of this sweep holds before compression: its size on disk at worst."""
    return math.prod(cube_lengths(config, validate).values()) * item_size(config)


def has_validation(config: dict[str, Any]) -> bool:
    source = config["source"]
    return bool(source["validate_synthetic"] if source["synthetic"] else source["validate_crop"])


def cold_method(config: dict[str, Any]) -> str:
    """The method carta-zarr-bench's --cold auto settles on here, worked out the same way."""
    measure = config["measure"]
    if measure["cold"] != "auto":
        return measure["cold"]
    if measure["drop_cache_cmd"]:
        return "command"
    if hasattr(os, "geteuid") and os.geteuid() == 0 and os.access("/proc/sys/vm/drop_caches", os.W_OK):
        return "drop-caches"
    if sys.platform.startswith("linux"):
        return "fadvise"
    return "off"


# -- Running --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Run:
    """One invocation of carta-zarr-bench."""

    stage: str
    setting: Setting
    users: int
    modes: tuple[str, ...]
    trials: int
    histogram: str = "exact"
    # For the warm stage: caches left as they are, and another build of the bench. Empty for the
    # sweep's own.
    cold: str = ""
    bench: str = ""

    def key(self, dataset: str) -> str:
        key = (f"{self.stage}|{dataset}|{','.join(self.setting.key)}|users={self.users}|{self.histogram}|"
               f"{','.join(self.modes)}|trials={self.trials}")
        if self.cold:
            key += f"|cold={self.cold}"
        if self.bench:
            key += f"|bench={self.bench}"
        return key


class Sweep:
    def __init__(self, config: dict[str, Any], output: Path):
        self.config = config
        self.output = output
        self.csv = output / "results.csv"
        self.state_path = output / "sweep-state.json"
        self.logs = output / "logs"
        self.work = Path(config["paths"]["work"]).resolve()
        self.bench = str(Path(config["paths"]["bench"]).resolve())
        self.cores = os.cpu_count() or 1
        self.baseline = setting_of(config["baseline"], self.cores)
        self.state: dict[str, Any] = {"runs": {}, "datasets": {}, "failures": {}}
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text())

    def save(self) -> None:
        temporary = self.state_path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.state_path)

    def fail(self, what: str, reason: str) -> None:
        log(f"failed: {what}: {reason}")
        self.state["failures"][what] = reason
        self.save()

    # -- Datasets

    def generate(self, layout: Layout, validate: bool) -> Path | None:
        source = self.config["source"]
        dataset = dataset_key(layout, validate)
        if self.config["checks"]["space"]:
            needed = int(uncompressed_bytes(self.config, validate) * 1.1)
            self.work.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.work).free
            if free < needed:
                self.fail(dataset, f"{needed >> 30} GiB may be needed in {self.work} and {free >> 30} GiB is free")
                return None
        command = ["uv", "run", "--quiet", "--script", self.config["paths"]["generator"],
                   "--output-root", str(self.work)]
        if source["synthetic"]:
            command += ["--synthetic", "--shape", source["validate_synthetic"] if validate else source["synthetic"],
                        "--seed", str(source["seed"])]
        else:
            command += ["--source", source["path"]]
            crop = source["validate_crop"] if validate else source["crop"]
            if crop:
                command += ["--crop", crop]
            if source["image"]:
                command += ["--image", source["image"]]
        if self.config["measure"]["generator_workers"]:
            command += ["--workers", str(self.config["measure"]["generator_workers"])]
        command += layout.generator_args()
        log(f"generating {layout.name} ({layout.describe()}){' for validation' if validate else ''}")
        started = time.monotonic()
        with open(self.logs / f"generate-{'validate-' if validate else ''}{layout.name}.log", "a") as stderr:
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=stderr, text=True)
        if completed.returncode != 0 or not completed.stdout.strip():
            self.fail(dataset, f"generate.py failed ({completed.returncode}); see logs/")
            return None
        path = Path(completed.stdout.strip().splitlines()[-1])
        manifest = json.loads((path / "bench-manifest.json").read_text())
        probe = subprocess.run([self.bench, "probe", str(path)], capture_output=True, text=True)
        if probe.returncode != 0:
            self.fail(dataset, f"carta-zarr-bench probe refused it: {probe.stdout.strip() or probe.stderr.strip()}")
            shutil.rmtree(path, ignore_errors=True)
            return None
        self.state["datasets"][dataset] = {
            "layout": layout.name,
            "describe": layout.describe(),
            "validate": validate,
            "path": str(path),
            "identity_hash": manifest["identity_hash"],
            "layout_manifest": manifest["layout"],
            "stripe_effective": manifest["storage"]["stripe_effective"],
            "filesystem": manifest["storage"]["filesystem"],
            "compression_ratio": manifest["result"]["compression_ratio"],
            "files": manifest["result"]["files"],
            "stored_bytes": manifest["result"]["stored_bytes"],
            "write_seconds": round(time.monotonic() - started, 1),
        }
        self.save()
        return path

    def bench_command(self, path: Path, run: Run, csv_path: Path) -> list[str]:
        measure = self.config["measure"]
        cold = run.cold or measure["cold"]
        ops = ",".join(f"{mode}={count}" for mode, count in measure["ops"].items())
        command = [run.bench or self.bench, "run", str(path), "--csv", str(csv_path), "--resume", "--label",
                   run.stage, "--mode", ",".join(run.modes), "--processes", str(run.users), "--trials",
                   str(run.trials), "--seed", str(measure["seed"]), "--trial-timeout", str(measure["trial_timeout"]),
                   "--cold", cold, "--region-fraction", str(measure["region_fraction"]),
                   "--animation-frames", str(measure["animation_frames"]),
                   "--histogram-method", run.histogram, *run.setting.bench_args()]
        if ops:
            command += ["--ops", ops]
        if measure["drop_cache_cmd"] and cold != "off":
            command += ["--drop-cache-cmd", measure["drop_cache_cmd"]]
        return command

    def warm_up(self, path: Path, layout: Layout, runs: list[Run]) -> None:
        """Reads what the warm runs will, once and untimed, so that they find it in the page cache. Into
        a CSV of its own under logs/, which no report reads, and again every time the dataset is
        written again."""
        warm = [run for run in runs if run.cold == "off"]
        if not warm:
            return
        modes = tuple(dict.fromkeys(mode for run in warm for mode in run.modes))
        run = Run("warm-up", warm[0].setting, max(run.users for run in warm), modes, 1, cold="off")
        log(f"  warming {layout.name} up")
        csv_path = self.logs / f"warm-up-{layout.name}-{int(time.time())}.csv"
        with open(self.logs / f"warm-{layout.name}.log", "a") as stderr:
            subprocess.run(self.bench_command(path, run, csv_path), stdout=subprocess.DEVNULL, stderr=stderr)

    def bench_run(self, path: Path, layout: Layout, run: Run, validate: bool) -> None:
        key = run.key(dataset_key(layout, validate))
        command = self.bench_command(path, run, self.csv)
        log(f"  {run.stage}: {layout.name}, {run.users} user{'s' if run.users > 1 else ''}, "
            f"{run.setting.describe()}{'' if run.histogram == 'exact' else ', ' + run.histogram}")
        started = time.monotonic()
        name = f"{run.stage}-{layout.name}{'-validate' if validate else ''}.log"
        with open(self.logs / name, "a") as stderr:
            stderr.write(f"\n$ {' '.join(command)}\n")
            stderr.flush()
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr)
        seconds = round(time.monotonic() - started, 1)
        if completed.returncode == 2:
            # A command line it refused: every run after this one would be refused the same way.
            raise SystemExit(f"carta-zarr-bench refused its command line; see logs/{name}. Is "
                             f"{run.bench or self.bench} built from the same commit as this script?")
        if completed.returncode == 0:
            self.state["runs"][key] = {"seconds": seconds}
            self.save()
        else:
            self.fail(key, f"carta-zarr-bench exited {completed.returncode}; see logs/{name}")

    def measure(self, layout: Layout, runs: list[Run], validate: bool = False) -> None:
        """Generate the layout, run every run that is not done on it, and delete it again."""
        dataset = dataset_key(layout, validate)
        if dataset in self.state["failures"]:
            return
        pending = [run for run in runs if run.key(dataset) not in self.state["runs"]
                   and run.key(dataset) not in self.state["failures"]]
        if not pending:
            return
        path = self.generate(layout, validate)
        if path is None:
            return
        try:
            self.warm_up(path, layout, pending)
            for run in pending:
                self.bench_run(path, layout, run, validate)
        finally:
            if not self.config["paths"]["keep_datasets"]:
                shutil.rmtree(path, ignore_errors=True)

    # -- Stages

    def user_counts(self, stage: str) -> list[int]:
        users = self.config["users"]
        if stage == "stage1":
            return sorted({1, users["target"]})
        typical = users["typical"] or 1
        return sorted({1, typical} - {users["target"]})

    def stage1_runs(self) -> list[Run]:
        measure = self.config["measure"]
        modes = tuple(measure["modes"])
        runs = [Run("stage1", self.baseline, users, modes, measure["trials"]) for users in self.user_counts("stage1")]
        if "cube-histogram" in modes:
            for method in measure["histogram_reference"]:
                runs += [Run("stage1", self.baseline, users, ("cube-histogram",), measure["trials"], method)
                         for users in self.user_counts("stage1")]
        return runs

    def warm_variants(self) -> list[tuple[str, str]]:
        """Each build the warm stage compares, the sweep's own first as "default"."""
        variants = [("default", self.bench)]
        return variants + [(entry["name"], str(Path(entry["bench"]).resolve()))
                           for entry in self.config["warm"]["variants"]]

    def warm_methods(self) -> list[str]:
        """The cube histograms the warm stage times: the exact one, and the one-pass references, since
        kCubeAccumulatorCacheBytes divides only those."""
        if "cube-histogram" not in self.config["warm"]["modes"]:
            return []
        return ["exact"] + list(self.config["measure"]["histogram_reference"])

    def warm_runs(self, setting: Setting) -> list[Run]:
        warm = self.config["warm"]
        trials = warm["trials"] or self.config["measure"]["trials"]
        users = sorted({1, self.config["users"]["target"]})
        runs = []
        for name, bench in self.warm_variants():
            for count in users:
                runs.append(Run(f"warm-{name}", setting, count, tuple(warm["modes"]), trials, cold="off",
                                bench="" if name == "default" else bench))
                runs += [Run(f"warm-{name}", setting, count, ("cube-histogram",), trials, method, cold="off",
                             bench="" if name == "default" else bench) for method in self.warm_methods()[1:]]
        return runs

    def stage2_runs(self) -> list[Run]:
        trials = self.config["stage2"]["trials"] or self.config["measure"]["trials"]
        modes = tuple(self.config["measure"]["modes"])
        return [Run("stage2", setting, self.config["users"]["target"], modes, trials)
                for setting in reader_grid(self.config, self.cores)]


def dataset_key(layout: Layout, validate: bool) -> str:
    return f"{'validate' if validate else 'main'}|{layout.key}"


# -- Results --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Op:
    """One operation as the ranking sees it."""

    seconds: float  # inf for a timeout; per frame for an animation
    logical_bytes: int  # per frame for an animation
    shares_chunks: bool
    reads_pixels: bool


@dataclasses.dataclass
class Stats:
    """One group of operations: the same stage, dataset, setting, number of users, mode and method.

    Its figures are over the operations it is ranked on: for plane, spectrum and region the first
    touches, while there are at least MIN_FIRST_TOUCHES of them, and every operation otherwise."""

    mode: str
    ops: list[Op]
    errors: int
    makespans: list[float]
    # Ranked on every operation because a group it is compared with is short.
    every_operation: bool = False

    @property
    def first_touches(self) -> int:
        return sum(1 for op in self.ops if not op.shares_chunks)

    @property
    def short(self) -> bool:
        """Too few first touches to rank on them alone. A group of fewer operations than that, every one
        of them a first touch, is what was asked for, and not short."""
        return (self.mode in FIRST_TOUCH_MODES and
                self.first_touches < min(MIN_FIRST_TOUCHES, len(self.ops)))

    @property
    def sample(self) -> list[Op]:
        if self.mode in FIRST_TOUCH_MODES and not self.short and not self.every_operation:
            return [op for op in self.ops if not op.shares_chunks]
        return self.ops

    @property
    def seconds(self) -> list[float]:
        return [op.seconds for op in self.sample]

    @property
    def rates(self) -> list[float]:
        """MiB/s of each operation that read pixels; timeouts as 0."""
        rates = []
        for op in self.sample:
            if math.isinf(op.seconds):
                if op.reads_pixels:
                    rates.append(0.0)
            elif op.logical_bytes and op.seconds > 0:
                rates.append(op.logical_bytes / op.seconds / (1 << 20))
        return rates

    @property
    def logical_bytes(self) -> int:
        return sum(op.logical_bytes for op in self.sample)

    @property
    def median(self) -> float:
        return statistics.median(self.seconds) if self.seconds else math.inf

    @property
    def p90(self) -> float:
        if not self.seconds:
            return math.inf
        ordered = sorted(self.seconds)
        return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]

    @property
    def minimum(self) -> float:
        return min(self.seconds) if self.seconds else math.inf

    @property
    def timeouts(self) -> int:
        return sum(1 for value in self.seconds if math.isinf(value))

    @property
    def mib_per_s(self) -> float:
        finite = sum(op.seconds for op in self.sample if math.isfinite(op.seconds))
        logical = sum(op.logical_bytes for op in self.sample if math.isfinite(op.seconds))
        return logical / finite / (1 << 20) if finite > 0 else 0.0

    @property
    def median_rate(self) -> float:
        """The median operation's MiB/s: what compares between copies of different sizes, where a
        median time does not -- a spectrum of more channels takes longer and is no slower."""
        return statistics.median(self.rates) if self.rates else 0.0

    @property
    def makespan(self) -> float:
        return statistics.median(self.makespans) if self.makespans else math.inf


GroupKey = tuple[str, str, tuple[str, str, str, str], int, str, str]  # stage, dataset, setting, users, mode, method


class Results:
    def __init__(self, sweep: Sweep):
        self.sweep = sweep
        by_identity = {entry["identity_hash"]: key for key, entry in sweep.state["datasets"].items()}
        self.groups: dict[GroupKey, Stats] = {}
        self.cold_methods: set[str] = set()
        self.cold_failures = 0
        self.mismatches: list[str] = []
        self.build_types: set[str] = set()
        self.commits: set[str] = set()
        # Each dataset's chunk shape, as the bench reports it: "time=1;frequency=16;...".
        self.chunk_shapes: dict[str, str] = {}
        # The tuning overrides each label's rows were measured with, as carta-zarr-bench reports them.
        self.tunings: dict[str, set[str]] = {}
        self.shapes: dict[str, str] = {}
        rows = []
        if sweep.csv.is_file():
            with open(sweep.csv, newline="") as file:
                rows = [row for row in csv.DictReader(file) if row["dataset_identity_hash"] in by_identity]
        trial_ends: dict[tuple[GroupKey, str], float] = {}
        checksums: dict[tuple[Any, ...], dict[str, str]] = {}
        for row in rows:
            dataset = by_identity[row["dataset_identity_hash"]]
            setting = (row["io_threads"], row["decode_threads"], row["cache_bytes"], row["read_budget_bytes"])
            method = row["histogram_method"] or "exact"
            key = (row["label"], dataset, setting, int(row["processes"]), row["mode"], method)
            stats = self.groups.setdefault(key, Stats(row["mode"], [], 0, []))
            if row["chunk_shape"]:
                self.chunk_shapes[dataset] = row["chunk_shape"]
                self.shapes[dataset] = row["shape"]
            # An animation is timed per frame, which is what playing one feels like.
            frames = int(row.get("animation_frames") or 0) if row["mode"] == "animation" else 0
            shares = row.get("shares_chunks") == "true"
            self.cold_methods.add(row["cold_method"])
            self.cold_failures += row["cold_ok"] != "true" and row["cold_method"] != "off"
            self.build_types.add(row["build_type"])
            self.tunings.setdefault(row["label"], set()).add(row.get("tuning") or "unknown")
            self.commits.add(row["bench_commit"])
            if row["status"] == "ok":
                seconds, logical = float(row["seconds"]), int(row["logical_bytes"])
                if frames:
                    seconds, logical = seconds / frames, logical // frames
                stats.ops.append(Op(seconds, logical, shares, row["mode"] != "open"))
            elif row["status"] == "timeout":
                stats.ops.append(Op(math.inf, 0, shares, bool(int(row["elements"] or 0)) or row["mode"] != "open"))
            else:
                stats.errors += 1
            if row["t_end_s"]:
                trial = (key, row["trial"])
                trial_ends[trial] = max(trial_ends.get(trial, 0.0), float(row["t_end_s"]))
            # The same position of the same shape must read the same, whichever layout or setting.
            if row["status"] == "ok" and row["checksum"] and not dataset.startswith("validate|"):
                where = (row["mode"], method, row["processes"], row["trial"], row["process_index"], row["op_index"],
                         row["position"], row["shape"])
                # Tuning variants read one dataset, and must read it alike too.
                source = f"{dataset} {row['label']}" if row["label"].startswith("warm-") else dataset
                checksums.setdefault(where, {})[source] = row["checksum"]
        for (key, _), end in trial_ends.items():
            self.groups[key].makespans.append(end)
        # A comparison is ranked one way throughout: on every operation, if any group in it has to be.
        short = {(stage, users, mode, method) for (stage, _, _, users, mode, method), stats in self.groups.items()
                 if stats.short}
        for (stage, _, _, users, mode, method), stats in self.groups.items():
            stats.every_operation = (stage, users, mode, method) in short
        for where, seen in checksums.items():
            if len(set(seen.values())) > 1:
                self.mismatches.append(f"{where[0]} at {where[6]} ({', '.join(sorted(seen))})")

    def get(self, stage: str, dataset: str, setting: Setting, users: int, mode: str,
            method: str = "exact") -> Stats | None:
        return self.groups.get((stage, dataset, setting.key, users, mode, method))

    def medians(self, stage: str, dataset: str, setting: Setting, users: int) -> dict[str, float]:
        result = {}
        for mode in self.sweep.config["measure"]["modes"]:
            stats = self.get(stage, dataset, setting, users, mode)
            result[mode] = stats.median if stats else math.inf
        return result


def score(medians: dict[str, float], best: dict[str, float], weights: dict[str, float]) -> tuple[float, float]:
    """The weighted geometric mean of each mode's slowdown against that mode's best, and the largest
    slowdown of any mode: the two ways of saying how good one choice is across modes."""
    total = weighted = 0.0
    worst = 1.0
    for mode, median in medians.items():
        weight = weights.get(mode, 0.0)
        if weight <= 0:
            continue
        if math.isinf(median) or math.isinf(best[mode]) or best[mode] <= 0:
            return math.inf, math.inf
        slowdown = median / best[mode]
        worst = max(worst, slowdown)
        total += weight * math.log(slowdown)
        weighted += weight
    return (math.exp(total / weighted) if weighted else math.inf), worst


def worst_mode(medians: dict[str, float], best: dict[str, float], weights: dict[str, float]) -> str:
    """The weighted mode furthest from its own best: the one a warning about a slowdown names."""
    slowdowns = {mode: median / best[mode] for mode, median in medians.items()
                 if weights.get(mode, 0.0) > 0 and math.isfinite(median) and best[mode] > 0}
    return max(slowdowns, key=slowdowns.get) if slowdowns else "–"


def bests(candidates: list[dict[str, float]], modes: list[str]) -> dict[str, float]:
    return {mode: min((medians[mode] for medians in candidates), default=math.inf) for mode in modes}


@dataclasses.dataclass
class Choice:
    layout: Layout
    setting: Setting
    score: float
    worst: float
    medians: dict[str, float]


class Analysis:
    """What the results say, as far as the sweep has got."""

    def __init__(self, sweep: Sweep, layouts: list[Layout]):
        self.sweep = sweep
        self.results = Results(sweep)
        self.layouts = layouts
        config = sweep.config
        self.modes = list(config["measure"]["modes"])
        self.weights = {mode: float(config["weights"].get(mode, 1.0)) for mode in self.modes}
        self.target = config["users"]["target"]
        self.grid = reader_grid(config, sweep.cores)

    def done(self, layout: Layout, validate: bool = False) -> bool:
        return dataset_key(layout, validate) in self.sweep.state["datasets"]

    # -- stage1

    def stage1_ranking(self, users: int, weights: dict[str, float] | None = None) -> list[tuple[Layout, float, float]]:
        measured = [layout for layout in self.layouts if self.done(layout)]
        medians = {layout.name: self.results.medians("stage1", dataset_key(layout, False), self.sweep.baseline, users)
                   for layout in measured}
        best = bests(list(medians.values()), self.modes)
        ranked = [(layout, *score(medians[layout.name], best, weights or self.weights)) for layout in measured]
        return sorted(ranked, key=lambda entry: (entry[1], entry[0].name))

    def stage2_layouts(self) -> list[Layout]:
        chosen = self.sweep.config["stage2"]["layouts"]
        by_name = {layout.name: layout for layout in self.layouts}
        if chosen:
            unknown = [name for name in chosen if name not in by_name]
            if unknown:
                raise SystemExit(f"stage2.layouts names layouts stage1 does not have: {unknown}")
            picked = [by_name[name] for name in chosen]
        else:
            ranked = [entry for entry in self.stage1_ranking(self.target)
                      if not entry[0].current and math.isfinite(entry[1])]
            picked = [layout for layout, _, _ in ranked[: self.sweep.config["stage2"]["top"]]]
        current = by_name.get("current")
        if current and self.done(current) and current not in picked:
            picked.append(current)
        return picked

    # -- stage2

    def stage2_choices(self) -> list[Choice]:
        choices = []
        for layout in self.stage2_layouts():
            for setting in self.grid:
                medians = self.results.medians("stage2", dataset_key(layout, False), setting, self.target)
                if any(math.isfinite(value) for value in medians.values()):
                    choices.append(Choice(layout, setting, math.inf, math.inf, medians))
        best = bests([choice.medians for choice in choices], self.modes)
        for choice in choices:
            choice.score, choice.worst = score(choice.medians, best, self.weights)
        return sorted(choices, key=lambda choice: (choice.score, choice.layout.name))

    def recommendations(self) -> tuple[Choice | None, Choice | None]:
        """The best choice for the data as it is, and the best one once it is in the best layout. A
        setting the backend cannot be given -- any read budget but the library's -- is not one."""
        applicable = [choice for choice in self.stage2_choices()
                      if choice.setting.read_budget == 0 and math.isfinite(choice.score)]
        after = applicable[0] if applicable else None
        now = next((choice for choice in applicable if choice.layout.current), None)
        return now, after

    def ranked_settings(self, layout: Layout) -> list[Choice]:
        """The applicable settings on one layout, best first, each against that layout's own best."""
        choices = [choice for choice in self.stage2_choices()
                   if choice.layout == layout and choice.setting.read_budget == 0]
        best = bests([choice.medians for choice in choices], self.modes)
        for choice in choices:
            choice.score, choice.worst = score(choice.medians, best, self.weights)
        return sorted(choices, key=lambda choice: choice.score)

    def confirm_plan(self) -> list[tuple[Layout, list[Setting]]]:
        now, after = self.recommendations()
        plan = []
        for choice in (now, after):
            if choice is None or any(layout == choice.layout for layout, _ in plan):
                continue
            settings = [entry.setting for entry in self.ranked_settings(choice.layout)[:3]]
            if choice.setting not in settings:
                settings.insert(0, choice.setting)
            if self.sweep.baseline not in settings:
                settings.append(self.sweep.baseline)
            plan.append((choice.layout, settings))
        return plan


# -- The sweep ------------------------------------------------------------------------------------


def run_sweep(sweep: Sweep, layouts: list[Layout]) -> None:
    for layout in layouts:
        sweep.measure(layout, sweep.stage1_runs())

    analysis = Analysis(sweep, layouts)
    candidates = analysis.stage2_layouts()
    log(f"stage2 layouts: {', '.join(layout.name for layout in candidates) or 'none'}")
    for layout in candidates:
        sweep.measure(layout, sweep.stage2_runs())

    analysis = Analysis(sweep, layouts)
    measure = sweep.config["measure"]
    for layout, settings in analysis.confirm_plan():
        runs = [Run("confirm", setting, users, tuple(measure["modes"]), measure["trials"])
                for users in sweep.user_counts("confirm") for setting in settings]
        sweep.measure(layout, runs)

    _, after = analysis.recommendations()
    if after and validation_possible(sweep.config)[0]:
        runs = [Run("validate", setting, sweep.config["users"]["target"], tuple(measure["modes"]), measure["trials"])
                for setting in dict.fromkeys([after.setting, sweep.baseline])]
        sweep.measure(after.layout, runs, validate=True)

    if after and sweep.config["warm"]["variants"]:
        sweep.measure(after.layout, sweep.warm_runs(after.setting))


def validation_possible(config: dict[str, Any]) -> tuple[bool, str]:
    if not has_validation(config):
        return False, "no validate_crop (or validate_synthetic) was given"
    # Without a way to empty caches the stages already had to read more than RAM -- unless that check
    # is off, and then nothing has been validated yet.
    if cold_method(config) == "off" and config["checks"]["ram"]:
        return False, "caches cannot be emptied here, so the stages already had to read more than RAM"
    if config["checks"]["ram"] and uncompressed_bytes(config, True) <= ram_bytes():
        return False, "the validation copy is not larger than RAM"
    return True, ""


def preflight(config: dict[str, Any]) -> list[str]:
    """What stops the sweep before it starts."""
    problems = []
    for tool in ("uv",):
        if not shutil.which(tool):
            problems.append(f"{tool} is not on the path; generate.py runs under it")
    benches = [config["paths"]["bench"]] + [variant["bench"] for variant in config["warm"]["variants"]]
    for bench in map(Path, benches):
        if not bench.is_file() or not os.access(bench, os.X_OK):
            problems.append(f"{bench} is not an executable carta-zarr-bench")
    if config["source"]["path"] and not (Path(config["source"]["path"]) / "zarr.json").is_file():
        problems.append(f"{config['source']['path']} is not a zarr dataset")
    if config["checks"]["ram"] and not problems:
        ram = ram_bytes()
        if cold_method(config) == "off" and uncompressed_bytes(config, False) <= ram:
            problems.append("caches cannot be emptied here (cold method off), so every dataset must be larger "
                            f"than RAM ({ram >> 30} GiB), and the stage1 copy is "
                            f"{uncompressed_bytes(config, False) >> 30} GiB")
        if has_validation(config) and cold_method(config) != "off" and uncompressed_bytes(config, True) <= ram:
            problems.append(f"the validation copy ({uncompressed_bytes(config, True) >> 30} GiB) is not larger than "
                            f"RAM ({ram >> 30} GiB)")
    return problems


def dry_run(config: dict[str, Any], output: Path) -> int:
    layouts, invalid = stage1_layouts(config)
    sweep_like = Sweep(config, output)  # reads state when there is one, writes nothing
    stage1 = sweep_like.stage1_runs()
    grid = reader_grid(config, sweep_like.cores)
    candidates = config["stage2"]["top"] + (1 if config["source"]["path"] else 0)
    print(f"cold reads by: {cold_method(config)}")
    print(f"stage1: {len(layouts)} layouts x {len(stage1)} bench runs = {len(layouts) * len(stage1)} runs")
    for layout in layouts:
        print(f"  {layout.name}: {layout.describe()}")
    for layout, reason in invalid:
        print(f"  skipped {layout.name} ({layout.describe()}): {reason}")
    print(f"stage2: up to {candidates} layouts x {len(grid)} settings = {candidates * len(grid)} runs")
    size = uncompressed_bytes(config, False)
    print(f"each dataset: up to {bytes_text(size)} uncompressed; one exists at a time")
    if has_validation(config):
        print(f"validation dataset: up to {bytes_text(uncompressed_bytes(config, True))} uncompressed")
    if config["warm"]["variants"]:
        warm = sweep_like.warm_runs(sweep_like.baseline)
        print(f"warm: {len(sweep_like.warm_variants())} builds x {len(warm) // len(sweep_like.warm_variants())} "
              f"bench runs = {len(warm)} runs, on the recommended layout")
    durations = [entry["seconds"] for key, entry in sweep_like.state["runs"].items()
                 if key.startswith("stage1|") and f"users={config['users']['target']}|exact" in key]
    if durations:
        stage1_trials = config["measure"]["trials"]
        stage2_trials = config["stage2"]["trials"] or stage1_trials
        hours = statistics.mean(durations) * stage2_trials / stage1_trials * candidates * len(grid) / 3600
        print(f"stage2 estimate from stage1's runs: about {hours:.1f} hours")
    problems = preflight(config)
    for stripe in sorted({layout.stripe for layout in layouts if layout.stripe}):
        checked = subprocess.run(["uv", "run", "--quiet", "--script", config["paths"]["generator"],
                                  "--check-stripe", stripe, "--output-root", config["paths"]["work"]],
                                 capture_output=True, text=True)
        if checked.returncode != 0:
            problems.append(f"striping {stripe} cannot be set: {checked.stderr.strip().splitlines()[-1:]}")
    for problem in problems:
        print(f"problem: {problem}")
    return 1 if problems else 0


# -- The report -----------------------------------------------------------------------------------


def seconds_text(value: float) -> str:
    if math.isinf(value):
        return "≥ timeout"
    if value < 1e-3:
        return f"{value * 1e6:.0f} µs"
    if value < 1:
        return f"{value * 1e3:.1f} ms"
    return f"{value:.2f} s"


def rate_text(stats: Stats) -> str:
    """MiB/s, or a dash for operations that read no pixels."""
    if stats.logical_bytes == 0:
        return "–"
    rate = stats.mib_per_s
    return f"{rate:.0f}" if rate >= 10 else f"{rate:.2g}"


def bytes_text(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    for unit, shift in (("TiB", 40), ("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if number >= 1 << shift:
            return f"{number / (1 << shift):.1f} {unit}"
    return f"{number} B"


def ratio_text(value: float) -> str:
    return "–" if math.isinf(value) else f"{value:.2f}×"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def axis_lengths(text: str) -> dict[str, int]:
    """A shape as the bench writes one, "time=1;frequency=16;l=512", as lengths by axis."""
    lengths = {}
    for item in filter(None, text.split(";")):
        name, _, value = item.partition("=")
        if value.isdigit():
            lengths[name] = int(value)
    return lengths


def chunk_text(chunk: dict[str, int]) -> str:
    """A chunk shape as l × m × channels."""
    return f"{chunk.get('l', 1)}×{chunk.get('m', 1)}×{chunk.get('frequency', 1)}"


def pareto(rows: dict[str, dict[str, float]]) -> set[str]:
    """The names no other row is at least as fast as in every mode and faster than in one."""
    front = set()
    for name, medians in rows.items():
        dominated = any(all(other[mode] <= medians[mode] for mode in medians) and
                        any(other[mode] < medians[mode] for mode in medians)
                        for other_name, other in rows.items() if other_name != name)
        if not dominated and all(math.isfinite(value) for value in medians.values()):
            front.add(name)
    return front


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def first_touch_advice(mode: str, shape: dict[str, int], chunk: dict[str, int], users: int, ops: int) -> str:
    """What would give a mode first touches enough: the chunks it reads are too few for its operations."""
    total = users * ops
    depth = chunk.get("frequency", 1)
    if mode == "plane":
        # Planes of different polarizations share a chunk only when it holds more than one of them.
        slabs = math.ceil(shape.get("frequency", 1) / depth) * \
            math.ceil(shape.get("polarization", 1) / chunk.get("polarization", 1))
        fewer = f", or set measure.ops plane = {slabs // users}" if slabs >= 2 * users else ""
        return (f"{total} planes a trial fall in {plural(slabs, 'run')} of chunks {plural(depth, 'channel')} deep: "
                f"crop at least {depth * total} channels{fewer}.")
    tiles = math.ceil(shape.get("l", 1) / chunk.get("l", 1)) * math.ceil(shape.get("m", 1) / chunk.get("m", 1))
    where = (f"{total} {mode} operations a trial fall in {plural(tiles, 'spatial chunk')} of "
             f"{chunk.get('l', 1)}×{chunk.get('m', 1)}")
    if tiles < 2 * users:
        # Users who share a chunk are each other's cache, and no count of operations parts them.
        return (f"{where}, too few for {users} users to read chunks of their own: only a cube with a larger plane, "
                "or fewer users, can measure this.")
    return f"{where}: set measure.ops {mode} = {tiles // users}, or crop a larger plane."


def write_report(sweep: Sweep, layouts: list[Layout], invalid: list[tuple[Layout, str]]) -> list[str]:
    """Writes summary.md and returns the warnings, which also head it."""
    config = sweep.config
    analysis = Analysis(sweep, layouts)
    results = analysis.results
    target, modes = analysis.target, analysis.modes
    machine = json.loads((sweep.output / "machine.json").read_text()) if (sweep.output / "machine.json").is_file() else {}
    warnings: list[str] = []
    body: list[str] = []

    for what, reason in sorted(sweep.state["failures"].items()):
        warnings.append(f"**Failed:** `{what}`: {reason}")
    for layout, reason in invalid:
        warnings.append(f"**Skipped layout {layout.name}** ({layout.describe()}): {reason}")
    for mismatch in results.mismatches:
        warnings.append(f"**Checksums disagree** between layouts or settings: {mismatch}. One of them read wrongly.")
    if any(build != "Release" for build in results.build_types):
        warnings.append(f"**carta-zarr-bench was not a Release build** ({', '.join(sorted(results.build_types))}); "
                        "its timings are not what a deployed backend would see.")
    if results.cold_methods - {"command"}:
        warnings.append("**Cold reads were cold on this host only** (" + ", ".join(sorted(results.cold_methods)) +
                        "): a parallel filesystem's servers kept their caches. Give measure.drop_cache_cmd to "
                        "empty them too.")
    if results.cold_failures:
        warnings.append(f"**{results.cold_failures} operations ran after a failed cache drop** and may be warm.")
    errors = sum(stats.errors for stats in results.groups.values())
    if errors:
        warnings.append(f"**{errors} operations failed** with an error; their rows in results.csv say why.")
    # One warning per layout and mode, about the most users it was short with.
    short: dict[tuple[str, str], tuple[int, Stats, str]] = {}
    for (stage, dataset, _, users, mode, _), stats in results.groups.items():
        entry = sweep.state["datasets"].get(dataset)
        if stats.short and entry and users >= short.get((entry["layout"], mode), (0,))[0]:
            short[(entry["layout"], mode)] = (users, stats, dataset)
    for (name, mode), (users, stats, dataset) in sorted(short.items()):
        chunk = axis_lengths(results.chunk_shapes.get(dataset, ""))
        ops = int(config["measure"]["ops"].get(mode, BENCH_DEFAULT_OPS[mode]))
        advice = first_touch_advice(mode, axis_lengths(results.shapes.get(dataset, "")), chunk, users, ops)
        warnings.append(f"**Too few first touches for {mode} on {name}**: with {users} user{'s' if users > 1 else ''}, "
                        f"{stats.first_touches} of {len(stats.ops)} operations read no chunk another one read, so "
                        f"{mode} is ranked on every operation there, cache hits included, and so is every layout "
                        f"compared with it. The crop is small for chunks of {chunk_text(chunk)}. {advice}")

    # -- Conclusion
    now, after = analysis.recommendations()
    conclusion: list[str] = []
    if after:
        if now and now.layout == after.layout:
            labelled = [("For the data as it is, which is already in the recommended layout", now)]
        elif now and now.setting == after.setting:
            labelled = [("For the data as it is, and in the recommended layout too", now)]
        else:
            labelled = [("For the data as it is (current layout)", now),
                        ("Once the data is in the recommended layout", after)]
        for label, choice in labelled:
            if choice is None:
                continue
            conclusion += [f"**{label}:** {choice.setting.describe()}", "", "```", f"{choice.setting.backend_flags()}",
                           "```", "", f"or in settings.json: `{choice.setting.settings_json()}`", ""]
            ranked = analysis.ranked_settings(choice.layout)
            own = next((entry for entry in ranked if entry.setting == choice.setting), None)
            if own and own.worst > SACRIFICE:
                mode = worst_mode(own.medians, bests([entry.medians for entry in ranked], modes), analysis.weights)
                warnings.append(f"**A mode is given up** at the recommended settings on {choice.layout.name}: {mode} "
                                f"runs {own.worst:.2f}× slower than at its own best setting.")
        conclusion.append(f"**Recommended layout:** {after.layout.name} ({after.layout.describe()}).")
        if now and now.layout != after.layout:
            conclusion.append(f"The current layout is {now.score / after.score:.2f}× slower overall "
                              f"(weighted geometric mean over modes, {target} users).")
        conclusion.append("")
        # The read budget: does any mode gain from one the backend cannot set yet?
        for layout in {choice.layout.name: choice.layout for choice in (now, after) if choice}.values():
            dataset = dataset_key(layout, False)
            for mode in modes:
                library = [results.get("stage2", dataset, s, target, mode) for s in analysis.grid if s.read_budget == 0]
                budgeted = [(results.get("stage2", dataset, s, target, mode), s) for s in analysis.grid
                            if s.read_budget != 0]
                plain = min((stats.median for stats in library if stats), default=math.inf)
                best = min(((stats.median, s) for stats, s in budgeted if stats), default=(math.inf, None),
                           key=lambda entry: entry[0])
                if math.isfinite(plain) and best[1] and best[0] <= plain * (1 - READ_BUDGET_GAIN):
                    warnings.append(f"**A read budget helps {mode}** on {layout.name}: {best[1].read_budget >> 20} MiB "
                                    f"is {(1 - best[0] / plain) * 100:.0f}% faster than the library's own. The backend "
                                    "has no setting for it yet; this is the case for adding one.")
    else:
        conclusion.append("No recommendation yet: stage2 has no complete results.\n")

    # -- Per-mode best layout from stage1
    ranking = analysis.stage1_ranking(target)
    if ranking:
        conclusion.append(f"Best layout per mode, at the baseline settings with {target} users:\n")
        rows = []
        for mode in modes:
            entries = [(results.get("stage1", dataset_key(layout, False), sweep.baseline, target, mode), layout)
                       for layout, _, _ in ranking]
            entries = sorted(((stats.median, layout) for stats, layout in entries if stats), key=lambda e: e[0])
            if not entries:
                continue
            best_median, best_layout = entries[0]
            current = next((median for median, layout in entries if layout.current), None)
            rows.append([mode, f"{best_layout.name}", seconds_text(best_median),
                         ratio_text(current / best_median) if current is not None and best_median > 0 else "–"])
        conclusion += table(["mode", "best layout", "median", "current is"], rows)

    # -- Sensitivity: the candidates with other numbers of users
    confirm_rows = []
    recommended = {choice.setting for choice in (now, after) if choice}
    for layout, settings in analysis.confirm_plan():
        for users in sweep.user_counts("confirm"):
            measured = {setting: results.medians("confirm", dataset_key(layout, False), setting, users)
                        for setting in settings}
            best = bests(list(measured.values()), modes)
            for setting, medians in measured.items():
                value, worst = score(medians, best, analysis.weights)
                if setting in recommended and math.isfinite(worst) and worst > SACRIFICE:
                    mode = worst_mode(medians, best, analysis.weights)
                    warnings.append(f"**With {users} user{'s' if users > 1 else ''}** the recommended settings run "
                                    f"{mode} {worst:.2f}× slower than the best candidate on {layout.name}.")
                confirm_rows.append([layout.name, str(users), setting.describe(), ratio_text(value), ratio_text(worst)])

    # -- Validation verdict
    validation = []
    possible, why_not = validation_possible(config)
    if after and possible:
        dataset = dataset_key(after.layout, True)
        rows = []
        drifted = False
        compared = {after.setting: "recommended"}
        compared.setdefault(sweep.baseline, "baseline")
        for setting, label in compared.items():
            small_stage = "stage2" if setting in analysis.grid else "stage1"
            for mode in modes:
                large = results.get("validate", dataset, setting, target, mode)
                small = results.get(small_stage, dataset_key(after.layout, False), setting, target, mode)
                if not large or not small:
                    continue
                if small.rates and large.rates:
                    # Faster is a higher rate, so the change is said as time would say it.
                    drift = small.median_rate / large.median_rate - 1 if large.median_rate > 0 else math.inf
                    before, after_text = f"{small.median_rate:.3g} MiB/s", f"{large.median_rate:.3g} MiB/s"
                else:
                    drift = large.median / small.median - 1 if small.median > 0 else math.inf
                    before, after_text = seconds_text(small.median), seconds_text(large.median)
                drifted = drifted or abs(drift) > VALIDATE_DRIFT
                rows.append([label, mode, before, after_text, f"{drift * 100:+.0f}%"])
        if rows:
            validation += ["Each mode's median operation, as MiB/s where it reads pixels -- the copies differ in "
                           "size, and a spectrum of more channels takes longer without being slower -- and as "
                           "time for open. The change is in time per byte: positive is slower.", ""]
            validation += table(["settings", "mode", "stage", "larger than RAM", "change"], rows)
            if drifted:
                warnings.append(f"**Validation moved a median by more than {VALIDATE_DRIFT * 100:.0f}%**: the smaller "
                                "copy did not predict reads larger than RAM well.")
            recommended = results.medians("validate", dataset, after.setting, target)
            base = results.medians("validate", dataset, sweep.baseline, target)
            best = bests([recommended, base], modes)
            if score(recommended, best, analysis.weights)[0] > score(base, best, analysis.weights)[0]:
                warnings.insert(0, "**The recommended settings failed validation**: larger than RAM, they were slower "
                                "than the baseline.")
        else:
            validation.append("Not run yet.\n")
    else:
        validation.append(f"Skipped: {why_not or 'no recommendation to validate'}.\n")

    # Before the conclusion is assembled, since it adds to the warnings that head it.
    warm = warm_section(sweep, analysis, after, warnings)

    # -- Assemble
    lines = [f"# Storage tuning: {machine.get('host', platform.node())}", "",
             f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by tools/zarr-bench/sweep.py.", ""]
    lines += ["## Conclusion", ""]
    lines += [f"- {warning}" for warning in warnings] + ([""] if warnings else [])
    lines += conclusion

    lines += ["## Machine and environment", ""]
    sizes = {"ram_bytes", "l2_cache", "l3_cache"}
    rows = [[key, bytes_text(value) if key in sizes and str(value).isdigit() else str(value)]
            for key, value in machine.items()]
    rows += [["cold reads", ", ".join(sorted(results.cold_methods)) or "–"],
             ["carta-zarr-bench commit", ", ".join(sorted(results.commits)) or "–"],
             ["build type", ", ".join(sorted(results.build_types)) or "–"],
             ["baseline settings", sweep.baseline.describe()],
             ["users", f"target {target}, typical {config['users']['typical'] or '–'}"]]
    lines += table(["", ""], rows)

    lines += ["## Stage 1: layouts", "",
              f"At the baseline settings ({sweep.baseline.describe()}). Times are per operation, and an animation's "
              "per frame; makespan is the slowest user's whole trial. plane, spectrum and region are ranked on their "
              "first touches -- operations that read no chunk another operation of the trial read before or "
              "alongside them -- which the first touches column counts; \"all ranked\" marks a mode ranked on "
              "every operation, because some layout had too few first touches.", ""]
    for layout in layouts:
        entry = sweep.state["datasets"].get(dataset_key(layout, False))
        if entry:
            stripe = entry["stripe_effective"] or "–"
            lines.append(f"- **{layout.name}**: {layout.describe()}; compression {entry['compression_ratio']:.2f}, "
                         f"{entry['files']} files, stripe {str(stripe).splitlines()[0][:60]}")
    lines.append("")
    for mode in modes:
        rows = []
        for layout, _, _ in ranking:
            cells = [layout.name]
            for users in sweep.user_counts("stage1"):
                stats = results.get("stage1", dataset_key(layout, False), sweep.baseline, users, mode)
                if not stats:
                    cells += ["–"] * 3
                    continue
                cells += [seconds_text(stats.median), seconds_text(stats.p90), rate_text(stats)]
            stats = results.get("stage1", dataset_key(layout, False), sweep.baseline, target, mode)
            cells += [seconds_text(stats.minimum) if stats else "–",
                      seconds_text(stats.makespan) if stats and target > 1 else "–"]
            if mode in FIRST_TOUCH_MODES:
                ranked_on_all = stats and (stats.short or stats.every_operation)
                cells.append(f"{stats.first_touches}/{len(stats.ops)}{' (all ranked)' if ranked_on_all else ''}"
                             if stats else "–")
            rows.append(cells)
        header = ["layout"]
        for users in sweep.user_counts("stage1"):
            header += [f"median ({users})", f"p90 ({users})", f"MiB/s ({users})"]
        header += [f"min ({target})", f"makespan ({target})"]
        if mode in FIRST_TOUCH_MODES:
            header.append(f"first touches ({target})")
        lines += [f"### {mode}", ""] + table(header, rows)
    if ranking:
        lines += ["Weighted geometric mean of each mode's slowdown against its best layout, with "
                  f"{target} users (lower is better):", ""]
        lines += table(["layout", "score", "worst mode"],
                       [[layout.name, ratio_text(value), ratio_text(worst)] for layout, value, worst in ranking])
        lines += tradeoff_section(analysis, ranking)

    lines += ["## Stage 2: reader settings", "", f"With {target} users. Score and worst are against the best of "
              "every layout and setting here; settings with a read budget cannot be given to the backend yet.", ""]
    choices = analysis.stage2_choices()
    for layout in analysis.stage2_layouts():
        rows = [[choice.setting.describe(), ratio_text(choice.score), ratio_text(choice.worst)] +
                [seconds_text(choice.medians[mode]) for mode in modes]
                for choice in choices if choice.layout == layout]
        if rows:
            lines += [f"### {layout.name}", ""] + table(["settings", "score", "worst"] + modes, rows)

    lines += ["## Sensitivity to the number of users", "",
              "The candidates of stage 2 with other numbers of users: each mode against the fastest of them at "
              "that number.", ""]
    lines += table(["layout", "users", "settings", "score", "worst"], confirm_rows) if confirm_rows else \
        ["Not run: there are no other numbers of users to try, or no recommendation yet.", ""]

    lines += ["## Validation", "",
              "The recommended layout again, larger than RAM, against the stage it was chosen in.", ""] + validation

    lines += warm

    lines += ["## Reference: one-pass cube histograms", "",
              "ComputeCubeHistogram's single pass against the backend's exact two passes, at the baseline. Not "
              "recommended either way: it moves where the bin edges land.", ""]
    reference = []
    for method in config["measure"]["histogram_reference"]:
        for layout, _, _ in ranking:
            for users in sweep.user_counts("stage1"):
                exact = results.get("stage1", dataset_key(layout, False), sweep.baseline, users, "cube-histogram")
                other = results.get("stage1", dataset_key(layout, False), sweep.baseline, users, "cube-histogram", method)
                if exact and other:
                    reference.append([method, layout.name, str(users), seconds_text(exact.median),
                                      seconds_text(other.median), ratio_text(exact.median / other.median)])
    lines += table(["method", "layout", "users", "exact", "one pass", "speedup"], reference) if reference else \
        ["Not measured: measure.histogram_reference is empty.", ""]

    lines += ["## Configuration", "", "```toml", dump_config(config).rstrip(), "```", ""]
    (sweep.output / "summary.md").write_text("\n".join(lines))
    return warnings


def warm_section(sweep: Sweep, analysis: Analysis, after: Choice | None, warnings: list[str]) -> list[str]:
    """The tuning variants against the default build, and a warning for each that gains enough."""
    config, results = sweep.config, analysis.results
    if not config["warm"]["variants"]:
        return []
    lines = ["## Warm stage: tuning constants", "",
             "Builds of carta-zarr-bench whose library replaces constants of src/reduce/tuning.h, against the "
             "default build, on the recommended layout and settings with the data in the page cache. This is "
             "evidence for ADR 0014, not a setting: the backend is built without overrides, and the report "
             "never recommends one.", ""]
    if not after:
        return lines + ["Not run: there is no recommendation to run it on.", ""]
    dataset = dataset_key(after.layout, False)
    default_tuning = results.tunings.get("warm-default", set())
    if default_tuning and default_tuning != {"default"}:
        warnings.append(f"**The default build of the warm stage has tuning overrides** "
                        f"({', '.join(sorted(default_tuning))}): paths.bench should be built without them.")
    columns = [(mode, "exact") for mode in config["warm"]["modes"]] + [
        ("cube-histogram", method) for method in sweep.warm_methods()[1:]]
    rows = []
    for name, _ in sweep.warm_variants():
        label = f"warm-{name}"
        tuning = ", ".join(sorted(results.tunings.get(label, set()))) or "–"
        if name != "default" and results.tunings.get(label) == {"default"}:
            warnings.append(f"**Tuning variant {name} has no overrides**: its bench was built like the default.")
        for users in sorted({1, analysis.target}):
            cells = [name, tuning.replace(",", ", "), str(users)]
            for mode, method in columns:
                stats = results.get(label, dataset, after.setting, users, mode, method)
                base = results.get("warm-default", dataset, after.setting, users, mode, method)
                if not stats:
                    cells.append("–")
                    continue
                text = seconds_text(stats.median)
                if name != "default" and base and stats.median > 0 and math.isfinite(base.median):
                    gain = base.median / stats.median - 1
                    text += f" ({gain * 100:+.0f}%)"
                    if gain >= TUNING_GAIN:
                        what = mode if method == "exact" else f"{mode} ({method})"
                        warnings.append(f"**Tuning variant {name} makes {what} {gain * 100:.0f}% faster** with {users} "
                                        f"user{'s' if users > 1 else ''} ({tuning}): evidence for ADR 0014, not a "
                                        "setting to deploy.")
                cells.append(text)
            rows.append(cells)
    lines += ["Medians, warm; in brackets, how much faster than the default build (positive is faster).", ""]
    header = [mode if method == "exact" else f"{mode} ({method})" for mode, method in columns]
    return lines + table(["variant", "overrides", "users"] + header, rows)


def tradeoff_section(analysis: Analysis, ranking: list[tuple[Layout, float, float]]) -> list[str]:
    """Plane against spectrum: what each layout's chunk depth buys one and costs the other."""
    sweep, results, target = analysis.sweep, analysis.results, analysis.target
    modes = [mode for mode in TRADEOFF_MODES if mode in analysis.modes]
    if not modes:
        return []
    medians: dict[str, dict[str, float]] = {}
    rows = []
    for layout, _, _ in sorted(ranking, key=lambda entry: entry[0].name):
        dataset = dataset_key(layout, False)
        chunk = axis_lengths(results.chunk_shapes.get(dataset, ""))
        medians[layout.name] = {mode: stats.median if (stats := results.get("stage1", dataset, sweep.baseline,
                                                                             target, mode)) else math.inf
                                for mode in modes}
        rows.append((layout, chunk))
    rows.sort(key=lambda entry: (entry[1].get("frequency", 1), entry[0].name))
    front = pareto(medians)
    lines = ["### Plane against spectrum", "",
             "A deeper chunk holds more channels of each pixel, so a spectrum reads fewer chunks, and a plane reads "
             "more pixels than it shows: the channels of its chunks that are not its own. A layout is on the front "
             "when no other is at least as fast in every mode here and faster in one; which of those to take is "
             f"what [weights] says. With {target} user{'s' if target > 1 else ''}, at the baseline settings.", ""]
    header = ["layout", "chunk (l×m×channels)", "depth"] + [
        "animation per frame" if mode == "animation" else mode for mode in modes] + ["front"]
    lines += table(header, [[layout.name, chunk_text(chunk), str(chunk.get("frequency", 1))] +
                            [seconds_text(medians[layout.name][mode]) for mode in modes] +
                            ["yes" if layout.name in front else ""] for layout, chunk in rows])

    best = ranking[0][0].name if math.isfinite(ranking[0][1]) else None
    steady = []
    for mode, other in (("spectrum", "plane"), ("plane", "spectrum")):
        if mode not in analysis.modes or not best:
            continue
        weights = dict(analysis.weights)
        weights[mode] = weights.get(mode, 0.0) * 2
        if weights[mode] <= 0:
            continue
        shifted = analysis.stage1_ranking(target, weights)
        if shifted and math.isfinite(shifted[0][1]) and shifted[0][0].name != best:
            lines.append(f"Doubling weights.{mode} makes {shifted[0][0].name} the best layout instead of {best}: "
                         f"the choice turns on how much {mode} matters against {other}.")
            lines.append("")
        else:
            steady.append(f"weights.{mode}")
    if steady:
        lines.append(f"Doubling {' or '.join(steady)} leaves {best} the best layout.")
    if lines[-1] != "":
        lines.append("")
    return lines


# -- Across shapes ---------------------------------------------------------------------------------

# The modes whose operations cover every channel, and so take longer the deeper the cube.
DEPTH_MODES = ("spectrum", "region", "cube-histogram")


def weighted_geomean(values: list[tuple[float, float]]) -> float:
    """The geometric mean of (value, weight) pairs; infinite when any weighted value is."""
    total = weights = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        if not math.isfinite(value) or value <= 0:
            return math.inf
        total += weight * math.log(value)
        weights += weight
    return math.exp(total / weights) if weights else math.inf


def write_site_report(config: dict[str, Any], output: Path) -> None:
    """The top-level summary.md of a sweep over several shapes: each shape's own answer, and the one
    layout and the one setting that would serve them all, since a server reads every cube it holds
    with the same flags and an archive is seldom rewritten per cube size."""
    shapes = config["source"]["shape"]
    entries = []
    for shape in shapes:
        sub_config = shape_config(config, shape)
        directory = output / shape["name"]
        if not (directory / "sweep-state.json").is_file():
            continue
        layouts, _ = stage1_layouts(sub_config)
        sweep = Sweep(sub_config, directory)
        analysis = Analysis(sweep, layouts)
        entries.append((shape, sub_config, sweep, analysis, analysis.recommendations()[1]))
    target = config["users"]["target"]
    lines = [f"# Storage tuning across cube shapes: {platform.node()}", "",
             f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by tools/zarr-bench/sweep.py. "
             "Each shape is a sweep of its own, reported in full in its directory; this compares them.", ""]

    # -- Each shape's own answer
    rows = []
    for shape, sub_config, sweep, analysis, after in entries:
        lengths = cube_lengths(sub_config, False)
        size = f"{lengths.get('l', '?')}×{lengths.get('m', '?')}×{lengths.get('frequency', '?')}"
        if shape.get("channels"):
            size += f" (of {shape['channels']})"
        warnings = (sweep.output / "summary.md").read_text().split("## Machine")[0].count("\n- **") \
            if (sweep.output / "summary.md").is_file() else 0
        rows.append([f"[{shape['name']}]({shape['name']}/summary.md)", size,
                     f"{after.layout.name} ({after.layout.describe()})" if after else "–",
                     after.setting.describe() if after else "–", str(warnings)])
    lines += ["## Each shape", "",
              "The layout and settings each shape's own sweep recommends, with the number of warnings its report "
              "opens with.", ""]
    lines += table(["shape", "l×m×channels", "layout", "settings", "warnings"], rows)

    # -- One layout for every shape
    names = None
    for _, _, _, analysis, _ in entries:
        measured = {layout.name for layout in analysis.layouts if analysis.done(layout) and not layout.current}
        names = measured if names is None else names & measured
    site_layouts = []
    for name in sorted(names or []):
        scores = []
        for shape, _, _, analysis, _ in entries:
            ranked = {layout.name: value for layout, value, _ in analysis.stage1_ranking(target)}
            scores.append((ranked.get(name, math.inf), float(shape.get("weight", 1.0))))
        layout = next(layout for layout in entries[0][3].layouts if layout.name == name)
        site_layouts.append((weighted_geomean(scores), layout, scores))
    site_layouts.sort(key=lambda entry: (entry[0], entry[1].name))
    lines += ["## One layout for every shape", "",
              "Each layout's stage 1 score in every shape -- its weighted slowdown against that shape's best layout, "
              f"with {plural(target, 'user')} -- and their geometric mean, weighted by each shape's weight. Only layouts "
              "every "
              "shape measured are compared.", ""]
    if site_layouts and math.isfinite(site_layouts[0][0]):
        best = site_layouts[0][1]
        lines += [f"**One layout for every shape:** {best.name} ({best.describe()}), "
                  f"{site_layouts[0][0]:.2f}× its shapes' own best on average.", ""]
    lines += table(["layout", "describe"] + [shape["name"] for shape, *_ in entries] + ["all shapes"],
                   [[layout.name, layout.describe()] + [ratio_text(value) for value, _ in scores] + [ratio_text(value)]
                    for value, layout, scores in site_layouts])

    # -- One setting for every shape
    grid = [setting for setting in reader_grid(config, os.cpu_count() or 1) if setting.read_budget == 0]
    site_settings = []
    for setting in grid:
        scores = []
        for shape, _, _, analysis, after in entries:
            if not after:
                scores.append((math.inf, float(shape.get("weight", 1.0))))
                continue
            ranked = {choice.setting: choice.score for choice in analysis.ranked_settings(after.layout)}
            scores.append((ranked.get(setting, math.inf), float(shape.get("weight", 1.0))))
        site_settings.append((weighted_geomean(scores), setting, scores))
    site_settings.sort(key=lambda entry: entry[0])
    lines += ["## One setting for every shape", "",
              "carta-backend reads every cube with the same flags. Each setting's stage 2 score on each shape's "
              "recommended layout, against the best setting there, and their weighted geometric mean.", ""]
    if site_settings and math.isfinite(site_settings[0][0]):
        best = site_settings[0][1]
        lines += [f"**One setting for every shape:** {best.describe()}", "", "```", best.backend_flags(), "```", "",
                  f"or in settings.json: `{best.settings_json()}`", ""]
    lines += table(["settings"] + [shape["name"] for shape, *_ in entries] + ["all shapes"],
                   [[setting.describe()] + [ratio_text(value) for value, _ in scores] + [ratio_text(value)]
                    for value, setting, scores in site_settings])

    # -- Full depth, for shapes measured on fewer channels than they stand for
    deep = []
    for shape, sub_config, sweep, analysis, _ in entries:
        measured = cube_lengths(sub_config, False).get("frequency", 0)
        full = shape.get("channels", 0)
        if not measured or full <= measured:
            continue
        factor = full / measured
        for layout, _, _ in analysis.stage1_ranking(target):
            cells = [shape["name"], layout.name]
            for mode in DEPTH_MODES:
                stats = analysis.results.get("stage1", dataset_key(layout, False), sweep.baseline, target, mode)
                cells.append(f"≈ {seconds_text(stats.median * factor)}" if stats else "–")
            deep.append(cells)
    if deep:
        lines += ["## At full depth", "",
                  "Shapes measured on fewer channels than the cubes they stand for, to keep each layout's copy small "
                  "enough to write: the modes that cover every channel, scaled by the channels they stand for over the "
                  "channels measured. An estimate, which assumes those modes take time in proportion to the channels "
                  "they cover; planes and animations are measured at their full size and need none.", ""]
        lines += table(["shape", "layout"] + list(DEPTH_MODES), deep)

    lines += ["## Configuration", "", "```toml", dump_config(config).rstrip(), "```", ""]
    (output / "summary.md").write_text("\n".join(lines))


# -- Main -----------------------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, help="where results go (zarr-bench-results next to the config)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one setting of the config, e.g. users.target=32")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="say what would run, and check it can, writing nothing")
    action.add_argument("--report-only", action="store_true", help="rewrite summary.md from the results there are")
    return parser.parse_args(argv)


def sweep_one(config: dict[str, Any], output: Path, args: argparse.Namespace) -> int | None:
    """One sweep, as far as the arguments say: a dry run, a report from what there is, or the whole
    of it. Returns the exit status, or None when it did not get as far as a report."""
    layouts, invalid = stage1_layouts(config)
    if args.dry_run:
        return dry_run(config, output)
    if not args.report_only:
        problems = preflight(config)
        if problems:
            for problem in problems:
                log(f"error: {problem}")
            return 2
    output.mkdir(parents=True, exist_ok=True)
    sweep = Sweep(config, output)
    if args.report_only:
        if not sweep.csv.is_file():
            log(f"{sweep.csv} does not exist: there is nothing to report")
            return None
    else:
        sweep.logs.mkdir(exist_ok=True)
        # What failed last time is tried again: the cause may have been fixed since.
        sweep.state["failures"] = {}
        (output / "config.toml").write_text(dump_config(config))
        (output / "machine.json").write_text(json.dumps(machine_info(sweep.work), indent=2) + "\n")
        for layout, reason in invalid:
            log(f"skipping {layout.name} ({layout.describe()}): {reason}")
        run_sweep(sweep, layouts)

    warnings = write_report(sweep, layouts, invalid)
    log(f"wrote {output / 'summary.md'}")
    failed = bool(sweep.state["failures"]) or any(warning.startswith("**Checksums") for warning in warnings)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    config = load_config(args.config, args.set)
    output = (args.output or args.config.parent / "zarr-bench-results").resolve()
    shapes = config["source"]["shape"]
    if not shapes:
        status = sweep_one(config, output, args)
        if status is None:
            raise SystemExit(f"{output / 'results.csv'} does not exist: there is nothing to report")
        return status

    statuses = []
    for shape in shapes:
        log(f"== shape {shape['name']}")
        if args.dry_run:
            print(f"== shape {shape['name']}")
        statuses.append(sweep_one(shape_config(config, shape), output / shape["name"], args))
    if args.dry_run:
        return max(statuses)
    if all(status is None for status in statuses):
        raise SystemExit(f"no shape under {output} has results to report")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.toml").write_text(dump_config(config))
    write_site_report(config, output)
    log(f"wrote {output / 'summary.md'}")
    return max(status or 0 for status in statuses)


if __name__ == "__main__":
    sys.exit(main())
