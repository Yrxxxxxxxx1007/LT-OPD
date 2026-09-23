#!/usr/bin/env python3
"""Download the pinned upstream scoring files listed in sources.json."""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

MANIFEST = json.loads((Path(__file__).with_name("sources.json")).read_text())


class RemoteZip(io.RawIOBase):
    """Read ZIP members by HTTP range without downloading the benchmark archive."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError(f"Server does not support HTTP range requests: {url}")
            self.length = int(response.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length)
        if not 0 <= self.position <= self.length:
            raise ValueError("ZIP seek is outside the archive")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size == 0:
            return b""
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.position}-{self.position + size - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Archive server stopped honoring HTTP range requests")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Incomplete archive range")
        self.position += len(data)
        return data


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, item in MANIFEST.items():
        path = directory / name
        if path.exists():
            data = path.read_bytes()
        else:
            if "zip_member" in item:
                with zipfile.ZipFile(RemoteZip(item["url"])) as archive:
                    data = archive.read(item["zip_member"])
            else:
                with urllib.request.urlopen(item["url"], timeout=60) as response:
                    data = response.read()
        if not path.exists():
            path.write_bytes(data)
        print(name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    fetch(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
