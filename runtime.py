import hashlib
import importlib.metadata as packageMetadata
import json
import os
from pathlib import Path
import platform
import sys
import time


repoPath = Path(__file__).resolve().parent


def configureRuntime(device="gpu"):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if device == "gpu":
        if sys.platform != "linux":
            raise RuntimeError("CUDA JAX requires Linux; use WSL2 on Windows")
    os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"


def readJson(filePath):
    return json.loads(Path(filePath).read_text(encoding="utf-8"))


def writeJson(filePath, payload):
    targetPath = Path(filePath)
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = targetPath.with_suffix(targetPath.suffix + ".tmp")
    temporaryPath.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporaryPath, targetPath)


def appendRecord(filePath, payload):
    with Path(filePath).open("a", encoding="utf-8") as recordFile:
        recordFile.write(json.dumps(payload, allow_nan=False) + "\n")
        recordFile.flush()
        os.fsync(recordFile.fileno())


def externalDirectory(directoryPath):
    resolvedPath = Path(directoryPath).expanduser().resolve()
    if resolvedPath == repoPath or repoPath in resolvedPath.parents:
        raise ValueError("store environments, checkpoints and results outside the repository")
    return resolvedPath


def sourceDigest():
    sourceHash = hashlib.sha256()
    sourcePaths = sorted(repoPath.glob("*.py")) + [repoPath / "requirements.txt"]
    for sourcePath in sourcePaths:
        sourceHash.update(sourcePath.name.encode("utf-8"))
        sourceHash.update(sourcePath.read_bytes().replace(b"\r\n", b"\n"))
    return sourceHash.hexdigest()


def runtimeManifest():
    packageVersions = {}
    for packageName in ("jax", "jaxlib", "flax", "optax", "craftax", "numpy", "jax-cuda12-plugin"):
        try:
            packageVersions[packageName] = packageMetadata.version(packageName)
        except packageMetadata.PackageNotFoundError:
            packageVersions[packageName] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packageVersions,
        "sourceDigest": sourceDigest(),
        "recordedAt": time.time(),
    }
