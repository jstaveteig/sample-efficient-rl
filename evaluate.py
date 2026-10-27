from configuration import configurationFor
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
from train import loadEngine
from runtime import (
    appendRecord,
    configureRuntime,
    externalDirectory,
    readJson,
    runtimeManifest,
    writeJson,
)


def parseArguments():
    parser = argparse.ArgumentParser(
        description="evaluate checkpoints on 96 complete native episodes"
    )
    parser.add_argument("--runDir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--wallSeconds", type=float)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--outputFile", type=Path)
    arguments = parser.parse_args()
    if arguments.step is not None and arguments.step < 1:
        parser.error("step must be positive")
    if arguments.batchSize < 1:
        parser.error("batchSize must be positive")
    if not 0 <= arguments.seed < 2**32:
        parser.error("seed must fit an unsigned 32-bit integer")
    if arguments.wallSeconds is not None and (
        not math.isfinite(arguments.wallSeconds) or arguments.wallSeconds <= 0.0
    ):
        parser.error("wallSeconds must be finite and positive")
    return arguments


def attemptAccounting(ledgerPath):
    if not ledgerPath.is_file():
        return {
            "wallSeconds": 0.0,
            "closedAttemptCount": 0,
            "openAttemptCount": 0,
            "latestStatus": None,
            "ledgerAvailable": False,
        }
    records = [
        json.loads(line)
        for line in ledgerPath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = {record["attemptId"] for record in records if record.get("event") == "start"}
    endings = {
        record["attemptId"]: record
        for record in records
        if record.get("event") in {"end", "failure"}
    }
    latestEnding = next(
        (record for record in reversed(records) if record.get("event") in {"end", "failure"}), {}
    )
    return {
        "wallSeconds": sum((float(record.get("wallSeconds", 0.0)) for record in endings.values())),
        "closedAttemptCount": len(endings),
        "openAttemptCount": len(starts - endings.keys()),
        "latestStatus": latestEnding.get("status", latestEnding.get("event")),
        "ledgerAvailable": True,
    }


def restoreAgent(
    engineModule, config, environment, environmentParams, checkpointRoot, requestedStep
):
    import jax
    from flax import nnx
    from checkpointState import loadTrainingState, restoreTrainingTree
    from train import trainingIdentity

    jax.device_get(jax.jit(lambda value: value + 1)(jax.numpy.asarray(0.0)))
    trainer = object.__new__(engineModule.Trainer)
    trainer.config = config
    trainer.numActions = environment.action_space(environmentParams).n
    unusedAgent, optimizer = trainer.buildAgent()
    del unusedAgent
    graphDefinition, unusedState = nnx.split(optimizer)
    del unusedState
    runPath = checkpointRoot.parent.parent
    collectionSteps = config.batchSize * config.rolloutHorizon
    selectedStep = requestedStep or config.totalEnvInteractions // collectionSteps * collectionSteps
    snapshotPath = runPath / "models" / str(selectedStep)
    if not (snapshotPath / "checkpoint.pkl").is_file():
        snapshotPath = runPath
    payload = loadTrainingState(snapshotPath)
    savedStep = int(payload["metadata"]["counts"]["realSteps"])
    if requestedStep is not None and requestedStep != savedStep:
        raise FileNotFoundError("no snapshot matches the requested real-interaction step")
    runConfig = readJson(runPath / "config.json")
    if payload["identity"] != trainingIdentity(config, runConfig["method"]):
        raise ValueError("training checkpoint source, configuration or method differs")
    policyState = restoreTrainingTree(payload["state"]["policyState"])
    del payload
    optimizer = nnx.merge(graphDefinition, jax.device_put(policyState))
    jax.block_until_ready(nnx.state(optimizer))
    return (optimizer.model, savedStep, snapshotPath / "checkpoint.pkl")


def evaluatePolicy(runDir, runConfig, arguments, startedAt, computeState):
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name as makeEnvironment
    from flax import nnx

    engineModule = loadEngine()
    trainingConfig = runConfig["configuration"]
    config = configurationFor(runConfig["method"], trainingConfig["seed"])
    config.ckptPath = trainingConfig["ckptPath"]
    config.wandbConfig.enable = trainingConfig["wandbConfig"]["enable"]
    if asdict(config) != trainingConfig:
        raise ValueError("saved configuration differs from the local profile and planned seed")
    if (
        config.envConfig.envType != "craftax"
        or config.envConfig.envName != "Craftax-Classic-Pixels-v1"
    ):
        raise ValueError("these comparisons require craftax classic pixels")
    environmentName = config.envConfig.envName
    environment = makeEnvironment(environmentName, auto_reset=False)
    environmentParams = environment.default_params
    maximumAchievementReward = 22.0
    checkpointRoot = runDir / "checkpoints" / f"fixed_step_seed{config.seed}"
    agent, checkpointStep, policyPath = restoreAgent(
        engineModule, config, environment, environmentParams, checkpointRoot, arguments.step
    )
    setupSeconds = time.perf_counter() - startedAt

    @jax.jit
    def resetEnvironments(resetKeys):
        return jax.vmap(environment.reset, in_axes=(0, None))(resetKeys, environmentParams)

    @nnx.jit
    def policyStep(policyAgent, observations, memoryState):
        batchSize = observations.shape[0]
        distribution, unusedValues, nextMemory = policyAgent(
            observations[:, None], jnp.zeros((batchSize, 1), dtype=jnp.bool_), memoryState
        )
        del unusedValues
        return (distribution.logits[:, 0], nextMemory)

    def stepGroup(
        actionLogits,
        nextMemory,
        observations,
        environmentState,
        memoryState,
        activeFlags,
        episodeKeys,
        stepIndex,
    ):
        batchSize = observations.shape[0]
        actionKeys = jax.vmap(
            lambda episodeKey: jax.random.fold_in(jax.random.fold_in(episodeKey, 1), stepIndex)
        )(episodeKeys)
        stepKeys = jax.vmap(
            lambda episodeKey: jax.random.fold_in(jax.random.fold_in(episodeKey, 2), stepIndex)
        )(episodeKeys)
        actions = jax.vmap(jax.random.categorical)(actionKeys, actionLogits).astype(jnp.int32)
        nextObservations, nextState, rewards, dones, unusedInfo = jax.vmap(
            environment.step, in_axes=(0, 0, 0, None)
        )(stepKeys, environmentState, actions, environmentParams)
        del unusedInfo

        def retainInactive(nextLeaf, previousLeaf):
            maskShape = (batchSize,) + (1,) * (nextLeaf.ndim - 1)
            return jnp.where(activeFlags.reshape(maskShape), nextLeaf, previousLeaf)

        nextState = jax.tree.map(retainInactive, nextState, environmentState)
        nextObservations = jnp.where(
            activeFlags[:, None, None, None], nextObservations, observations
        )
        nextMemory = jnp.where(activeFlags[:, None], nextMemory, memoryState)
        return (
            nextObservations,
            nextState,
            nextMemory,
            activeFlags & ~dones,
            jnp.where(activeFlags, rewards, 0.0),
            activeFlags & dones,
            nextState.achievements,
        )

    def budgetExpired():
        return (
            arguments.wallSeconds is not None
            and time.perf_counter() - startedAt >= arguments.wallSeconds
        )

    episodeCount = 96
    episodeRows = []
    baseKey = jax.random.PRNGKey(arguments.seed)
    resetSeconds = 0.0
    firstStepSeconds = None
    startedEpisodeCount = 0
    for groupStart in range(0, episodeCount, arguments.batchSize):
        if budgetExpired():
            break
        groupSize = min(arguments.batchSize, episodeCount - groupStart)
        episodeIds = np.arange(groupStart, groupStart + groupSize, dtype=np.uint32)
        episodeKeys = jax.vmap(lambda episodeId: jax.random.fold_in(baseKey, episodeId))(
            jnp.asarray(episodeIds)
        )
        jax.device_get(episodeKeys)
        resetStartedAt = time.perf_counter()
        computeState["pendingResetSlots"] = groupSize
        observations, environmentState = resetEnvironments(episodeKeys)
        jax.block_until_ready((observations, environmentState))
        resetSeconds += time.perf_counter() - resetStartedAt
        computeState["resetWorldSlots"] += groupSize
        computeState["pendingResetSlots"] = 0
        startedEpisodeCount += groupSize
        memoryState = agent.rnn.initializeCarry(groupSize)
        activeFlags = np.ones(groupSize, dtype=np.bool_)
        episodeReturns = np.zeros(groupSize, dtype=np.float64)
        episodeLengths = np.zeros(groupSize, dtype=np.int64)
        stepIndex = 0
        while activeFlags.any() and (not budgetExpired()):
            activeBefore = activeFlags.copy()
            stepStartedAt = time.perf_counter()
            computeState["pendingEnvironmentSteps"] = int(activeBefore.sum())
            computeState["pendingStepSlots"] = groupSize
            actionLogits, policyMemory = policyStep(agent, observations, memoryState)
            jax.block_until_ready((actionLogits, policyMemory))
            stepOutput = stepGroup(
                actionLogits,
                policyMemory,
                observations,
                environmentState,
                memoryState,
                jnp.asarray(activeFlags),
                episodeKeys,
                jnp.asarray(stepIndex, dtype=jnp.uint32),
            )
            jax.block_until_ready(stepOutput)
            (
                observations,
                environmentState,
                memoryState,
                nextActive,
                rewards,
                justFinished,
                achievements,
            ) = stepOutput
            hostRewards, hostFinished, hostAchievements, activeFlags = jax.device_get(
                (rewards, justFinished, achievements, nextActive)
            )
            if firstStepSeconds is None:
                firstStepSeconds = time.perf_counter() - stepStartedAt
            episodeReturns += np.asarray(hostRewards, dtype=np.float64)
            episodeLengths += activeBefore.astype(np.int64)
            computeState["environmentSteps"] += int(activeBefore.sum())
            computeState["environmentStepSlots"] += groupSize
            computeState["pendingEnvironmentSteps"] = 0
            computeState["pendingStepSlots"] = 0
            stepIndex += 1
            for groupIndex in np.flatnonzero(hostFinished):
                episodeRows.append(
                    {
                        "episodeIndex": int(episodeIds[groupIndex]),
                        "return": float(episodeReturns[groupIndex]),
                        "length": int(episodeLengths[groupIndex]),
                        "achievements": np.asarray(
                            hostAchievements[groupIndex], dtype=np.bool_
                        ).tolist(),
                    }
                )
        if activeFlags.any():
            break
    episodeRows.sort(key=lambda episodeRow: episodeRow["episodeIndex"])
    completed = len(episodeRows) == episodeCount
    completionMean = float(np.mean([row["return"] for row in episodeRows])) if episodeRows else None
    achievementRates = (
        np.asarray([row["achievements"] for row in episodeRows], dtype=np.float64).mean(axis=0)
        if completed
        else None
    )
    completedIds = {row["episodeIndex"] for row in episodeRows}
    return {
        "completed": completed,
        "status": "complete" if completed else "wallLimit",
        "checkpointStep": checkpointStep,
        "checkpointDirectory": str(policyPath.parent),
        "trainingRealSteps": checkpointStep,
        "trainingConfig": trainingConfig,
        "expectedFinalCheckpointStep": config.totalEnvInteractions
        // (config.batchSize * config.rolloutHorizon)
        * (config.batchSize * config.rolloutHorizon),
        "trainingSeed": config.seed,
        "seed": arguments.seed,
        "policyMode": "stochastic",
        "environmentName": environmentName,
        "requestedEpisodeCount": episodeCount,
        "episodeCount": len(episodeRows),
        "startedEpisodeCount": startedEpisodeCount,
        "batchSize": arguments.batchSize,
        "nativeHorizon": int(environmentParams.max_timesteps),
        "meanReturn": completionMean if completed else None,
        "rewardPercent": 100.0 * completionMean / maximumAchievementReward if completed else None,
        "maximumAchievementReward": maximumAchievementReward,
        "achievementRewardPercent": float(100.0 * achievementRates.mean()) if completed else None,
        "geometricScore": (
            float(np.expm1(np.log1p(100.0 * achievementRates).mean())) if completed else None
        ),
        "meanEpisodeLength": (
            float(np.mean([row["length"] for row in episodeRows])) if completed else None
        ),
        "achievementRates": achievementRates.tolist() if completed else None,
        "completionConditionalMean": completionMean,
        "selectionBiasPossible": not completed,
        "episodes": episodeRows,
        "unfinishedEpisodeIds": [
            index for index in range(episodeCount) if index not in completedIds
        ],
        "environmentSteps": computeState["environmentSteps"],
        "environmentStepSlots": computeState["environmentStepSlots"],
        "paddingStepSlots": computeState["environmentStepSlots"] - computeState["environmentSteps"],
        "policyObservationSlots": computeState["environmentStepSlots"],
        "resetWorldSlots": computeState["resetWorldSlots"],
        "setupSeconds": setupSeconds,
        "resetSeconds": resetSeconds,
        "firstStepSeconds": firstStepSeconds,
        "wallSeconds": time.perf_counter() - startedAt,
        "wallLimitSeconds": arguments.wallSeconds,
        "device": arguments.device,
        "visibleDeviceCount": len(jax.devices()),
    }


def main():
    startedAt = time.perf_counter()
    arguments = parseArguments()
    runDir = externalDirectory(arguments.runDir)
    runConfig = readJson(runDir / "config.json")
    outputFile = arguments.outputFile or runDir / "evaluation.json"
    if not outputFile.is_absolute():
        outputFile = runDir / outputFile
    outputFile = outputFile.resolve()
    externalDirectory(outputFile.parent)
    if outputFile.exists():
        raise FileExistsError(f"evaluation output already exists: {outputFile}")
    configureRuntime(arguments.device)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("MPLCONFIGDIR", str(runDir / "matplotlib"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(runDir / "jaxCache"))
    manifest = runtimeManifest()
    if manifest["packages"]["craftax"] != "1.4.5":
        raise ValueError("checkpoint evaluation requires craftax 1.4.5")
    attemptId = uuid.uuid4().hex
    ledgerPath = runDir / "evaluationAttempts.jsonl"
    previousEvaluation = attemptAccounting(ledgerPath)
    trainingAccounting = attemptAccounting(runDir / "trainingAttempts.jsonl")
    appendRecord(
        ledgerPath,
        {
            "event": "start",
            "attemptId": attemptId,
            "recordedAt": time.time(),
            "method": runConfig["method"],
            "seed": arguments.seed,
            "outputFile": str(outputFile),
            "runtime": manifest,
        },
    )
    computeState = {
        "environmentSteps": 0,
        "environmentStepSlots": 0,
        "resetWorldSlots": 0,
        "pendingEnvironmentSteps": 0,
        "pendingStepSlots": 0,
        "pendingResetSlots": 0,
    }
    try:
        result = evaluatePolicy(runDir, runConfig, arguments, startedAt, computeState)
    except BaseException as failure:
        appendRecord(
            ledgerPath,
            {
                "event": "failure",
                "attemptId": attemptId,
                "recordedAt": time.time(),
                "wallSeconds": time.perf_counter() - startedAt,
                "failure": repr(failure),
                "computeLowerBounds": {
                    name: computeState[name]
                    for name in ("environmentSteps", "environmentStepSlots", "resetWorldSlots")
                },
                "computeUpperBounds": {
                    "environmentSteps": computeState["environmentSteps"]
                    + computeState["pendingEnvironmentSteps"],
                    "environmentStepSlots": computeState["environmentStepSlots"]
                    + computeState["pendingStepSlots"],
                    "resetWorldSlots": computeState["resetWorldSlots"]
                    + computeState["pendingResetSlots"],
                },
            },
        )
        raise
    result.update(
        {
            "attemptId": attemptId,
            "method": runConfig["method"],
            "runtime": manifest,
            "trainingRunCompleted": trainingAccounting["latestStatus"] == "complete"
            and (not trainingAccounting["openAttemptCount"]),
            "trainingWallSeconds": trainingAccounting["wallSeconds"],
            "previousEvaluationWallSeconds": previousEvaluation["wallSeconds"],
            "totalAccountedWallSeconds": trainingAccounting["wallSeconds"]
            + previousEvaluation["wallSeconds"]
            + result["wallSeconds"],
            "trainingAttemptAccounting": trainingAccounting,
            "previousEvaluationAttemptAccounting": previousEvaluation,
            "computeAccountingIncomplete": bool(
                not trainingAccounting["ledgerAvailable"]
                or trainingAccounting["openAttemptCount"]
                or previousEvaluation["openAttemptCount"]
            ),
        }
    )
    result["trainingCompleted"] = (
        result["trainingRunCompleted"]
        and result["checkpointStep"] == result["expectedFinalCheckpointStep"]
    )
    writeJson(outputFile, result)
    appendRecord(
        ledgerPath,
        {
            "event": "end",
            "attemptId": attemptId,
            "recordedAt": time.time(),
            "completed": result["completed"],
            "wallSeconds": time.perf_counter() - startedAt,
            "environmentSteps": result["environmentSteps"],
            "environmentStepSlots": result["environmentStepSlots"],
        },
    )
    print(f"evaluation {result['status']}: {outputFile}")
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
