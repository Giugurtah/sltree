"""
build_c.py — compile the three C backends of SL-TREE
====================================================
Replaces the per-directory Makefiles.  A single Python script, so the build
works identically on macOS, Linux and Windows without requiring `make`,
`uname` or any POSIX shell.

    python build_c.py            # build and install into the package
    python build_c.py --check    # only report what would be done
    python build_c.py --clean    # remove the built libraries

It is also imported by setup.py, which calls build_all() during
`pip install .`.

Choosing the compiler
---------------------
The environment variable ``CC`` wins when set, so a packager can pin a
specific toolchain.  Otherwise the first of ``cc``, ``gcc``, ``clang`` found
on PATH is used.  On Windows that means a MinGW-w64 ``gcc``: MSVC is not
supported and cannot be, because the C sources use variable-length arrays,
a C99 feature MSVC has never implemented.

A note on -ffp-contract=off
---------------------------
Without it, compilers are free to fuse ``a*b + c`` into a single
fused-multiply-add with one rounding step instead of two.  That is faster
and usually more accurate, but the result differs in the last bit from a
compiler that does not fuse.  In a tree algorithm a last-bit difference can
flip a `pi > best_pi` comparison and change every split below it, so the
same data would grow different trees on different platforms.  Turning
contraction off costs a little speed and buys cross-platform agreement,
which is what a replication script needs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ------------------------------------------------------------------ #
#  What to build
# ------------------------------------------------------------------ #

LIBRARIES = [
    {
        "name":    "pitree",
        "sources": ["pitree/pitree_core.c"],
        "opt":     "-O2",
        "install": "sltree/_backend",
    },
    {
        "name":    "slbt",
        "sources": ["slbt/slbt_core.c", "slbt/slbt_als.c", "slbt/linalg.c"],
        "opt":     "-O3",
        "install": "sltree/_backend",
    },
    {
        "name":    "categorizer",
        "sources": ["categorizer/categorizer_core.c", "categorizer/kmeans.c",
                    "categorizer/kmeans_init.c", "categorizer/kmeans_metrics.c"],
        "opt":     "-O2",
        "install": "sltree/_preprocessing/_backend",
    },
]

CSRC = HERE / "csrc"
COMMON = CSRC / "common"


# ------------------------------------------------------------------ #
#  Platform
# ------------------------------------------------------------------ #

# Diagnostic build flag, set by --init-pattern.  See build_one().
_INIT_PATTERN = False

# Cross-compilation target, set by --target.  None means "this machine".
_TARGET = None


def target() -> str:
    """The platform being built for: 'linux', 'macos' or 'windows'."""
    if _TARGET:
        return _TARGET
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def lib_suffix() -> str:
    return {"macos": ".dylib", "windows": ".dll", "linux": ".so"}[target()]


def find_compiler() -> str:
    env = os.environ.get("CC")
    if env:
        return env
    for candidate in ("cc", "gcc", "clang"):
        found = shutil.which(candidate)
        if found:
            return candidate
    raise RuntimeError(
        "No C compiler found on PATH.\n"
        "  macOS   : xcode-select --install\n"
        "  Linux   : sudo apt install build-essential\n"
        "  Windows : install MSYS2 from https://www.msys2.org and then\n"
        "            pacman -S mingw-w64-x86_64-gcc\n"
        "            (MSVC will not work: the C sources use C99 "
        "variable-length arrays, which MSVC does not support)"
    )


def link_flags() -> list[str]:
    """Platform-specific flags applied at link time."""
    if target() == "windows":
        # Windows reserves 1 MB of stack by default against 8 MB on Linux and
        # macOS.  kplusplus_init() puts an array the size of the dataset on the
        # stack, so a large fit would overflow it.  Ask for 8 MB to match.
        #
        # -static-libgcc keeps the DLL free of MinGW runtime dependencies, so
        # only KERNEL32.dll and msvcrt.dll are needed and nothing extra has to
        # be shipped alongside.
        return ["-static", "-static-libgcc", "-Wl,--stack,8388608"]
    return []


# ------------------------------------------------------------------ #
#  Build
# ------------------------------------------------------------------ #

def build_one(spec: dict, cc: str, dry: bool = False) -> Path:
    out_name = f"lib{spec['name']}{lib_suffix()}"
    out_path = CSRC / spec["name"] / out_name
    sources = [str(CSRC / s) for s in spec["sources"]]

    missing = [s for s in sources if not Path(s).exists()]
    if missing:
        raise FileNotFoundError(f"missing C sources: {missing}")

    cmd = [
        cc, spec["opt"], "-Wall",
        "-fPIC", "-shared",
        "-ffp-contract=off",          # see the note in the module docstring
        f"-I{COMMON}",
        "-o", str(out_path),
        *sources,
        *link_flags(),
        "-lm",
    ]
    if _INIT_PATTERN:
        # Diagnostic build.  Fills every uninitialised automatic variable with
        # a non-zero byte pattern instead of whatever the stack happened to
        # hold.  Code that never reads uninitialised memory gives identical
        # results with and without it; code that does gives different ones.
        #
        # This is the detector for the DX bug fixed in slbt_als.c, and for
        # anything of its kind.  A plain build cannot find them: Linux hands
        # out zeroed stack pages, so the garbage read is usually zero and the
        # answer comes out right by luck; the failure then surfaces on another
        # platform, or once other code has run first in the same process.
        #
        #     python build_c.py --init-pattern && pytest
        #
        # Any difference from a normal build is a bug, not a tolerance issue.
        # Needs gcc >= 12 or clang >= 8.
        cmd.insert(3, "-ftrivial-auto-var-init=pattern")

    print(f"  {spec['name']}")
    print(f"    {' '.join(cmd)}")
    if dry:
        return out_path

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr.strip():
        for line in result.stderr.strip().split("\n"):
            print(f"    | {line}")
    if result.returncode != 0:
        raise RuntimeError(f"compilation of {spec['name']} failed "
                           f"(exit code {result.returncode})")
    if not out_path.exists():
        raise RuntimeError(f"{out_path} was not produced")
    print(f"    -> {out_path.name} ({out_path.stat().st_size:,} bytes)")
    return out_path


def install_one(spec: dict, built: Path, dry: bool = False) -> None:
    dest_dir = HERE / spec["install"]
    if not dry:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, dest_dir / built.name)
    print(f"    -> {spec['install']}/{built.name}")


def build_all(dry: bool = False) -> None:
    cc = find_compiler()
    print(f"[sltree] compiler: {cc}    target: {target()} ({lib_suffix()})")
    for spec in LIBRARIES:
        built = build_one(spec, cc, dry)
        install_one(spec, built, dry)
    print("[sltree] all three C backends built.")


def clean() -> None:
    n = 0
    for spec in LIBRARIES:
        for suffix in (".so", ".dylib", ".dll"):
            for p in (CSRC / spec["name"]).glob(f"lib{spec['name']}{suffix}"):
                p.unlink(); n += 1; print(f"  removed {p.relative_to(HERE)}")
            for p in (HERE / spec["install"]).glob(f"lib{spec['name']}{suffix}"):
                p.unlink(); n += 1; print(f"  removed {p.relative_to(HERE)}")
    print(f"[sltree] {n} files removed.")


def main() -> int:
    global _TARGET, _INIT_PATTERN
    ap = argparse.ArgumentParser(
        description="Build the SL-TREE C backends.",
        formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--clean", action="store_true",
                    help="remove the built libraries")
    ap.add_argument("--init-pattern", action="store_true",
                    help="diagnostic build: fill uninitialised stack variables\n"
                         "with a byte pattern.  Results must be identical to a\n"
                         "normal build; any difference means the C code reads\n"
                         "memory it never wrote.  Run it, then pytest.")
    ap.add_argument("--target", choices=("linux", "macos", "windows"),
                    help="cross-compile for another platform.  Needs a matching "
                         "cross-compiler in CC, e.g.\n"
                         "  CC=x86_64-w64-mingw32-gcc python build_c.py --target windows\n"
                         "Meant for producing test builds; the CI compiles each "
                         "platform natively.")
    args = ap.parse_args()
    _TARGET = args.target
    _INIT_PATTERN = args.init_pattern

    if args.clean:
        clean()
        return 0
    try:
        build_all(dry=args.check)
    except Exception as exc:
        print(f"[sltree] BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
