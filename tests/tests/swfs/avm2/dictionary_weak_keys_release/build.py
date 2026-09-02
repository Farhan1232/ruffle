#!/usr/bin/env python3
"""Builds test.swf from Test.as, with the asc.jar Ruffle uses for playerglobal.

Usage: build.py <path to playerglobal_import.abc>
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASC_JAR = HERE.parents[4] / "tools/asc/asc.jar"

SWF_VERSION = 15
STAGE_W, STAGE_H, FPS = 550, 400, 30


def rect(xmin, xmax, ymin, ymax):
    nbits = max(abs(v).bit_length() for v in (xmin, xmax, ymin, ymax)) + 1
    bits = []
    for i in range(4, -1, -1):
        bits.append((nbits >> i) & 1)
    for v in (xmin, xmax, ymin, ymax):
        v &= (1 << nbits) - 1
        for i in range(nbits - 1, -1, -1):
            bits.append((v >> i) & 1)
    bits += [0] * (-len(bits) % 8)
    return bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))


def tag(code, body):
    if len(body) < 0x3F:
        return struct.pack("<H", (code << 6) | len(body)) + body
    return struct.pack("<HI", (code << 6) | 0x3F, len(body)) + body


def swf(tags):
    body = rect(0, STAGE_W * 20, 0, STAGE_H * 20)
    body += struct.pack("<HH", FPS << 8, 1)
    body += tag(69, struct.pack("<I", 0x08))  # FileAttributes: ActionScript 3
    body += b"".join(tags)
    body += tag(1, b"") + tag(0, b"")
    header = b"FWS" + bytes([SWF_VERSION])
    return header + struct.pack("<I", len(header) + 4 + len(body)) + body


def compile_abc(source, playerglobal):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "java",
                "-classpath",
                str(ASC_JAR),
                "macromedia.asc.embedding.ScriptCompiler",
                "-optimize",
                "-import",
                str(playerglobal),
                "-outdir",
                tmp,
                "-out",
                source.stem,
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return (Path(tmp) / f"{source.stem}.abc").read_bytes()


def main():
    playerglobal = Path(sys.argv[1])
    abc = compile_abc(HERE / "Test.as", playerglobal)
    do_abc = tag(82, struct.pack("<I", 1) + b"\x00" + abc)
    symbol_class = tag(76, struct.pack("<HH", 1, 0) + b"Test\x00")
    (HERE / "test.swf").write_bytes(swf([do_abc, symbol_class]))


if __name__ == "__main__":
    main()
