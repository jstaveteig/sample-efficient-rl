from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import errno
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
from types import BuiltinFunctionType, FunctionType, MethodType

import numpy as np

from runtime import externalDirectory


checkpointMagic = b"craftax-training-state\x01\n"
digestSize = hashlib.sha256().digest_size
hostChunkBytes = 64 * 1024 * 1024


@dataclass(frozen=True)
class TypedKeyData:
    bits: np.ndarray
    implementation: str


def hostArray(value):
    import jax

    if value.nbytes < hostChunkBytes or value.ndim == 0:
        return np.asarray(jax.device_get(value))
    hostValue = np.empty(value.shape, dtype=value.dtype)
    transferValue = value
    targetValue = hostValue
    rowBytes = value.nbytes // value.shape[0]
    if rowBytes > hostChunkBytes:
        transferValue = value.reshape(-1)
        targetValue = hostValue.reshape(-1)
        rowBytes = value.dtype.itemsize
    chunkRows = max(1, hostChunkBytes // rowBytes)
    for rowStart in range(0, transferValue.shape[0], chunkRows):
        rowEnd = min(rowStart + chunkRows, transferValue.shape[0])
        np.copyto(
            targetValue[rowStart:rowEnd],
            np.asarray(jax.device_get(transferValue[rowStart:rowEnd])),
            casting="no",
        )
    return hostValue


def validatePayload(payload):
    import jax

    pendingValues = [payload]
    visitedIds = set()
    while pendingValues:
        value = pendingValues.pop()
        if id(value) in visitedIds:
            continue
        visitedIds.add(id(value))
        if isinstance(value, (FunctionType, BuiltinFunctionType, MethodType)):
            raise TypeError("checkpoint payloads cannot contain functions")
        if type(value).__module__.startswith("flax.nnx.graph"):
            raise TypeError("checkpoint payloads cannot contain graph definitions")
        if isinstance(value, (np.ndarray, np.generic)):
            if value.dtype.hasobject:
                raise TypeError("checkpoint arrays must not contain Python objects")
            continue
        if isinstance(value, (jax.Array, type, str, bytes, int, float, complex, bool, type(None))):
            continue
        if isinstance(value, Mapping):
            pendingValues.extend(value.keys())
            pendingValues.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pendingValues.extend(value)
        elif is_dataclass(value):
            pendingValues.extend(getattr(value, field.name) for field in fields(value))
        else:
            pendingValues.extend(getattr(value, "__dict__", {}).values())
            for parentClass in type(value).__mro__:
                slotNames = getattr(parentClass, "__slots__", ())
                if isinstance(slotNames, str):
                    slotNames = (slotNames,)
                pendingValues.extend(
                    getattr(value, slotName)
                    for slotName in slotNames
                    if slotName not in {"__dict__", "__weakref__"} and hasattr(value, slotName)
                )


def hostTrainingTree(payload):
    import jax

    validatePayload(payload)
    implementationNames = {
        "threefry2x32": "threefry2x32",
        "fry": "threefry2x32",
        "rbg": "rbg",
        "unsafe_rbg": "unsafe_rbg",
        "urbg": "unsafe_rbg",
    }

    def hostLeaf(value):
        if isinstance(value, jax.Array):
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                implementationTag = str(jax.random.key_impl(value))
                if implementationTag not in implementationNames:
                    raise ValueError(f"unsupported PRNG implementation: {implementationTag}")
                return TypedKeyData(
                    hostArray(jax.random.key_data(value)),
                    implementationNames[implementationTag],
                )
            return hostArray(value)
        return value

    return jax.tree.map(hostLeaf, payload)


def restoreTrainingTree(payload):
    import jax

    def restoreLeaf(value):
        if isinstance(value, TypedKeyData):
            return jax.random.wrap_key_data(jax.device_put(value.bits), impl=value.implementation)
        return value

    return jax.tree.map(restoreLeaf, payload, is_leaf=lambda value: isinstance(value, TypedKeyData))


class DigestWriter:
    def __init__(self, outputFile):
        self.outputFile = outputFile
        self.digest = hashlib.sha256()

    def write(self, contents):
        self.digest.update(contents)
        return self.outputFile.write(contents)


def verifyCheckpoint(checkpointPath):
    with checkpointPath.open("rb") as checkpointFile:
        payloadEnd = os.fstat(checkpointFile.fileno()).st_size - digestSize
        if payloadEnd <= len(checkpointMagic):
            raise ValueError("checkpoint is truncated")
        if checkpointFile.read(len(checkpointMagic)) != checkpointMagic:
            raise ValueError("checkpoint format differs")
        checkpointFile.seek(0)
        digest = hashlib.sha256()
        remainingBytes = payloadEnd
        while remainingBytes:
            chunk = checkpointFile.read(min(8 * 1024 * 1024, remainingBytes))
            if not chunk:
                raise ValueError("checkpoint is truncated")
            digest.update(chunk)
            remainingBytes -= len(chunk)
        if checkpointFile.read(digestSize) != digest.digest():
            raise ValueError("checkpoint checksum differs")
    return payloadEnd


def syncDirectory(outputPath):
    if os.name != "posix":
        return
    descriptor = os.open(outputPath, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as failure:
            if failure.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
    finally:
        os.close(descriptor)


def saveTrainingState(outputPath, payload):
    outputPath = externalDirectory(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    primaryPath = outputPath / "checkpoint.pkl"
    previousPath = outputPath / "previous.pkl"
    hostPayload = hostTrainingTree(payload)
    descriptor, temporaryName = tempfile.mkstemp(
        prefix="checkpoint-", suffix=".tmp", dir=outputPath
    )
    temporaryPath = Path(temporaryName)
    try:
        with os.fdopen(descriptor, "wb") as checkpointFile:
            writer = DigestWriter(checkpointFile)
            writer.write(checkpointMagic)
            pickle.dump(hostPayload, writer, protocol=5)
            checkpointFile.write(writer.digest.digest())
            checkpointFile.flush()
            os.fsync(checkpointFile.fileno())
        if primaryPath.is_file():
            try:
                verifyCheckpoint(primaryPath)
            except (OSError, ValueError):
                pass
            else:
                # retain a durable fallback before replacing the primary
                os.replace(primaryPath, previousPath)
                syncDirectory(outputPath)
        os.replace(temporaryPath, primaryPath)
        syncDirectory(outputPath)
    finally:
        if temporaryPath.exists():
            temporaryPath.unlink()
    return primaryPath


def loadTrainingState(outputPath):
    outputPath = externalDirectory(outputPath)
    failures = []
    for fileName in ("checkpoint.pkl", "previous.pkl"):
        checkpointPath = outputPath / fileName
        try:
            payloadEnd = verifyCheckpoint(checkpointPath)
            with checkpointPath.open("rb") as checkpointFile:
                checkpointFile.seek(len(checkpointMagic))
                payload = pickle.load(checkpointFile)
                if checkpointFile.tell() != payloadEnd:
                    raise ValueError("checkpoint contains trailing payload bytes")
            return payload
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as failure:
            failures.append(f"{fileName}: {failure}")
    raise RuntimeError("no readable training checkpoint; " + "; ".join(failures))
