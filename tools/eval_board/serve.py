#!/usr/bin/env python3
"""Serve an exported eval board read-only, with HTTP range requests so browsers can seek videos.

Standard library only and Python 3.6 compatible, so it runs on a plain host that mirrors the board
(see sync.py). Files under media/ are content-addressed and never change, so browsers may cache
them for good; everything else is revalidated on every load.

  python3 serve.py --root /data/eval-board --host 0.0.0.0 --port 8766
"""

import argparse
import contextlib
import mimetypes
import os
import re
import socketserver
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

CHUNK = 1 << 20
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
TYPES = {".mp4": "video/mp4", ".json": "application/json", ".html": "text/html; charset=utf-8"}


class Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def resolve(root, url_path):
    """The file a URL path names inside root, or None for anything outside it or hidden."""
    path = unquote(urlparse(url_path).path)
    if path.endswith("/"):
        path += "index.html"
    parts = [part for part in path.split("/") if part]
    if not parts or any(part.startswith(".") for part in parts):
        return None
    full = os.path.realpath(os.path.join(root, *parts))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        return None
    return full


def make_handler(root):
    root = os.path.realpath(root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "eval-board/1"

        def log_message(self, format, *args):
            pass

        def do_HEAD(self):
            self._respond(body=False)

        def do_GET(self):
            self._respond(body=True)

        def _respond(self, body):
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send(body)

        def _send(self, body):
            path = resolve(root, self.path)
            if path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = os.path.getsize(path)
            start, end = 0, size - 1
            match = RANGE.match(self.headers.get("Range") or "")
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                else:
                    start = max(size - int(match.group(2)), 0)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            kind = TYPES.get(os.path.splitext(path)[1]) or mimetypes.guess_type(path)[0]
            immutable = os.path.relpath(path, root).startswith("media" + os.sep)
            self.send_header("Content-Type", kind or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header(
                "Cache-Control", "public, max-age=31536000, immutable" if immutable else "no-cache"
            )
            self.end_headers()
            if not body:
                return
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = Server((args.host, args.port), make_handler(args.root))
    print(f"serving {args.root} on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
