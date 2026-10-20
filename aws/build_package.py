"""Build the Lambda zip for arm64 (Graviton) without Docker.

Downloads Linux aarch64 wheels for Python 3.12 (OpenCV 5 publishes them), unpacks them next to
the StageSight code and zips everything. boto3 is left out because the Lambda runtime has it.

Usage:  python aws/build_package.py            ->  build/stagesight_lambda.zip
"""
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "build", "lambda")
OUT = os.path.join(ROOT, "build", "stagesight_lambda.zip")
LIMIT_MB = 250   # Lambda's unzipped size limit for zip packages


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    reqs = [r.strip() for r in open(os.path.join(ROOT, "requirements.txt"))
            if r.strip() and not r.startswith("boto3")]
    subprocess.check_call(["uv", "pip", "install", "--quiet", "--target", BUILD,
                           "--python-platform", "aarch64-manylinux_2_28", "--python-version", "3.12",
                           "--only-binary", ":all:", *reqs])
    for d in ("stagesight", "sites"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(BUILD, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(ROOT, "aws", "handler.py"), BUILD)
    os.makedirs(os.path.join(BUILD, "web"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "web", "index.html"), os.path.join(BUILD, "web"))
    shutil.copytree(os.path.join(ROOT, "web", "replay"), os.path.join(BUILD, "web", "replay"))  # replay mode
    for junk in ("bin",):
        shutil.rmtree(os.path.join(BUILD, junk), ignore_errors=True)
    size = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(BUILD) for f in fs) / 1e6
    if size > LIMIT_MB:
        sys.exit(f"unzipped size {size:.0f} MB is over Lambda's {LIMIT_MB} MB limit")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p, _, fs in os.walk(BUILD):
            for f in fs:
                full = os.path.join(p, f)
                z.write(full, os.path.relpath(full, BUILD))
    print(f"{OUT}: {os.path.getsize(OUT) / 1e6:.0f} MB zipped, {size:.0f} MB unzipped")


if __name__ == "__main__":
    main()
