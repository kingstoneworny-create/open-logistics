"""Run logistics operations from UTF-8 JSON or start the embedded HTTP server."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from open_logistics import LogisticsEngine, __version__
from open_logistics.api.server import serve
from open_logistics.core.exceptions import LogisticsError


def _port(value: str) -> int:
    port = int(value)
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON result on success or exit with argparse's error status."""
    parser = argparse.ArgumentParser(prog="open-logistics", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    operations = {
        "dispatch": ("Assign delivery orders to vehicles", LogisticsEngine.dispatch_from_payload),
        "inventory": ("Evaluate stock for replenishment", LogisticsEngine.inventory_from_payload),
        "wave": ("Batch orders and generate pick tasks", LogisticsEngine.wave_from_payload),
    }
    for name, (description, _) in operations.items():
        command = commands.add_parser(name, help=description, description=description)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--stdin", action="store_true", help="read JSON from standard input")
        source.add_argument("--input", type=Path, metavar="FILE", help="read a UTF-8 JSON file")

    server = commands.add_parser("serve", help="start the HTTP API")
    server.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    server.add_argument("--port", type=_port, default=8080, help="listen port (default: %(default)s)")
    server.add_argument("--verbose", action="store_true", help="log HTTP requests")
    args = parser.parse_args(argv)

    if args.command == "serve":
        try:
            serve(host=args.host, port=args.port, verbose=args.verbose)
        except OSError as exc:
            server.error(str(exc))
        return 0

    try:
        if args.stdin:
            payload = json.load(getattr(sys.stdin, "buffer", sys.stdin), parse_constant=_reject_json_constant)
        else:
            with args.input.open(encoding="utf-8-sig") as source_file:
                payload = json.load(source_file, parse_constant=_reject_json_constant)
        _, operation = operations[args.command]
        result = operation(LogisticsEngine(), payload)
        output = json.dumps(result, allow_nan=False)
    except (LogisticsError, OSError, ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    print(output)
    return 0
