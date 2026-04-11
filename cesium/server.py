import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import cgi

ROOT_DIR = Path(__file__).resolve().parent.parent
CESIUM_DIR = ROOT_DIR / "cesium"
MODELS_DIR = CESIUM_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if not safe:
        safe = "model.glb"
    if not safe.lower().endswith((".glb", ".gltf")):
        safe = safe + ".glb"
    return safe


class CesiumHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/upload-model":
            self.send_error(404, "Not Found")
            return

        try:
            ctype, _ = cgi.parse_header(self.headers.get("content-type", ""))
            if ctype != "multipart/form-data":
                self._send_json(400, {"ok": False, "error": "content-type must be multipart/form-data"})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("content-type", ""),
                },
            )

            if "file" not in form:
                self._send_json(400, {"ok": False, "error": "missing file field"})
                return

            file_item = form["file"]
            if isinstance(file_item, list):
                file_item = file_item[0] if file_item else None

            if file_item is None or getattr(file_item, "file", None) is None:
                self._send_json(400, {"ok": False, "error": "missing file content"})
                return

            original_name = file_item.filename or "model.glb"
            safe_name = sanitize_filename(original_name)
            target = MODELS_DIR / safe_name

            stem = target.stem
            suffix = target.suffix
            n = 1
            while target.exists():
                target = MODELS_DIR / f"{stem}_{n}{suffix}"
                n += 1

            with open(target, "wb") as f:
                data = file_item.file.read()
                f.write(data)

            relative_web_path = "/cesium/models/" + target.name
            self._send_json(200, {
                "ok": True,
                "path": relative_web_path,
                "size": len(data),
                "filename": target.name,
            })
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"upload failed: {exc}"})

    def _send_json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run(host: str = "127.0.0.1", port: int = 8000):
    os.chdir(ROOT_DIR)
    httpd = ThreadingHTTPServer((host, port), CesiumHandler)
    print(f"Serving {ROOT_DIR} at http://{host}:{port}")
    print("Open: http://127.0.0.1:8000/cesium/index.html")
    print("Upload API: POST /api/upload-model")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
