from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config import (
    _parse_value,
    get_config_value,
    list_config_keys,
    load_config,
    load_raw_config,
    serialize_config,
    set_config_value,
    validate_raw_config,
    write_raw_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and edit the shared project config.")
    parser.add_argument("--config", default=None, help="Path to the project YAML config.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show", help="Print the resolved config.")

    get_parser = subparsers.add_parser("get", help="Read one dotted config key.")
    get_parser.add_argument("key", help="Dotted config key, e.g. train.epochs")

    set_parser = subparsers.add_parser("set", help="Update one dotted config key in place.")
    set_parser.add_argument("key", help="Dotted config key, e.g. train.epochs")
    set_parser.add_argument("value", help="New scalar value.")

    subparsers.add_parser("list-keys", help="List available dotted config keys.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "show":
        config = load_config(args.config)
        print(yaml.safe_dump(serialize_config(config), sort_keys=False), end="")
        return

    raw, config_path = load_raw_config(args.config)

    if args.command == "get":
        try:
            value = get_config_value(raw, args.key)
        except KeyError:
            parser.error(f"Unknown config key: {args.key}")
        if isinstance(value, (dict, list)):
            print(yaml.safe_dump(value, sort_keys=False), end="")
        else:
            print(value)
        return

    if args.command == "set":
        try:
            set_config_value(raw, args.key, _parse_value(args.value))
            validate_raw_config(raw, config_path)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        write_raw_config(raw, config_path)
        print(f"updated {args.key} in {Path(config_path)}")
        return

    if args.command == "list-keys":
        config = load_config(args.config)
        for key in list_config_keys(serialize_config(config)):
            print(key)
        return

    parser.error(f"Unsupported command: {args.command}")
