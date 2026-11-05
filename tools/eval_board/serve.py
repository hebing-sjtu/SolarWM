#!/usr/bin/env python3
"""Serve an exported eval board read-only, with HTTP range requests so browsers can seek videos.

Standard library only and Python 3.6 compatible, so it runs on a plain host that mirrors the board
(see sync.py). Files under media/ are content-addressed and never change, so browsers may cache
them for good; everything else is revalidated on every load.

With --answers DIR, labeling pages may also save their state: a POST of a JSON body to
/answers/<name>.json replaces DIR/<name>.json, and a GET of that path returns it. DIR sits
outside --root, so sync.py never overwrites what raters saved.

  python3 serve.py --root /data/eval-board --host 0.0.0.0 --port 8766
  python3 serve.py --root /data/eval-board --answers /data/eval-board-answers --port 8768
"""

import argparse
import contextlib
import json
import mimetypes
import os
import re
import socketserver
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

CHUNK = 1 << 20
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
ANSWER = re.compile(r"^/answers/([A-Za-z0-9][A-Za-z0-9._-]{0,99})\.json$")
MAX_ANSWER_BYTES = 4 << 20
TYPES = {
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
}


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


def answer_path(answers, url_path):
    """The answers file a URL path names, or None if answers are off or the name is not allowed."""
    if answers is None:
        return None
    match = ANSWER.match(unquote(urlparse(url_path).path))
    return os.path.join(answers, match.group(1) + ".json") if match else None


def make_handler(root, answers=None):
    root = os.path.realpath(root)
    answers = os.path.realpath(answers) if answers else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "eval-board/1"

        def log_message(self, format, *args):
            pass

        def do_HEAD(self):
            self._respond(body=False)

        def do_GET(self):
            self._respond(body=True)

        def do_POST(self):
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._save()

        def _respond(self, body):
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send(body)

        def _reply(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _save(self):
            target = answer_path(answers, self.path)
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_ANSWER_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            body = self.rfile.read(length)
            try:
                json.loads(body.decode("utf-8"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "body is not JSON")
                return
            handle, scratch = tempfile.mkstemp(dir=answers, prefix=".", suffix=".json")
            with os.fdopen(handle, "wb") as out:
                out.write(body)
            os.chmod(scratch, 0o644)
            os.replace(scratch, target)
            self._reply(HTTPStatus.OK, {"saved": os.path.basename(target), "bytes": length})

        def _send(self, body):
            saved = answer_path(answers, self.path)
            if saved is not None and not os.path.isfile(saved):
                self._reply(HTTPStatus.OK, {})
                return
            path = saved if saved is not None else resolve(root, self.path)
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
    parser.add_argument("--answers", help="directory for labeling-page state; off when omitted")
    args = parser.parse_args()
    if args.answers:
        os.makedirs(args.answers, exist_ok=True)
    server = Server((args.host, args.port), make_handler(args.root, args.answers))
    print(f"serving {args.root} on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
