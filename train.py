from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from checkpointState import hostArray, loadTrainingState, restoreTrainingTree, saveTrainingState
from runtime import appendRecord, externalDirectory, readJson, sourceDigest, writeJson
import argparse
import os
from configuration import configurationFor
from runtime import configureRuntime


def buildPolicyTrainer(engineModule, config):
    jax = engineModule.jax
    jnp = engineModule.jnp
    if config.batchSize <= 0 or config.rolloutHorizon <= 0:
        raise ValueError("batch size and rollout horizon must be positive")
    if config.acConfig.numRealEpochs <= 0 or config.acConfig.numRealMinibatches <= 0:
        raise ValueError("real policy update counts must be positive")
    if config.batchSize % config.acConfig.numRealMinibatches:
        raise ValueError("real policy minibatches must divide the environment batch")
    trainer = object.__new__(engineModule.Trainer)
    trainer.config = config
    trainer.multiGpu = len(jax.devices()) > 1
    if config.ckptPath is not None:
        config.ckptPath = config.ckptPath.format(config=config)
        config.ckptPath = f"{config.ckptPath}/fixed_step_seed{config.seed}"
    trainer.rng = jax.random.PRNGKey(config.seed)
    trainer.env, trainer.envParams, trainer.numActions, trainer.Achievement, trainer.Action = (
        trainer.buildEnvironment(config.batchSize)
    )
    trainer.rollout = trainer.buildRollout(trainer.env, trainer.envParams)
    trainer.agent, trainer.policyTrainState = trainer.buildAgent()
    # match policy initialization across methods
    trainer.rng = jax.random.split(trainer.rng)[0]
    trainer.bufferState = None
    trainer.bufferAdd = lambda previousState, transitions: previousState
    trainer.rng, environmentKey = jax.random.split(trainer.rng)
    trainer.currObs, trainer.envState = trainer.env.reset(environmentKey, trainer.envParams)
    trainer.currDone = jnp.ones((config.batchSize,), dtype=jnp.bool_)
    trainer.agentState = trainer.agent.rnn.initializeCarry(config.batchSize)
    trainer.successRates = jnp.zeros((len(trainer.Achievement),))
    trainer.numEpisodes = jnp.array(0)
    trainer.tgtMean = 0
    trainer.tgtStd = 0
    trainer.debiasing = 0
    return trainer


def normalizedConfiguration(config):
    configuration = asdict(config)
    if configuration["ckptPath"] is not None:
        checkpointPath = Path(configuration["ckptPath"]).expanduser().resolve()
        checkpointName = f"fixed_step_seed{config.seed}"
        if checkpointPath.name != checkpointName:
            checkpointPath /= checkpointName
        configuration["ckptPath"] = str(checkpointPath)
    return configuration


def trainingIdentity(config, method):
    configuration = normalizedConfiguration(config)
    configurationDigest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    return {
        "version": 4,
        "method": method,
        "sourceDigest": sourceDigest(),
        "configurationDigest": configurationDigest,
        "configuration": configuration,
    }


def episodeMetrics(records):
    if not records:
        return {
            "completedEpisodes": 0,
            "meanReturn": None,
            "meanLength": None,
            "rewardPercent": None,
            "achievementRewardPercent": None,
            "geometricScore": None,
        }
    achievementRates = np.asarray(
        [record["achievements"] for record in records], dtype=np.float64
    ).mean(axis=0)
    meanReturn = float(np.mean([record["episodeReturn"] for record in records]))
    return {
        "completedEpisodes": len(records),
        "meanReturn": meanReturn,
        "meanLength": float(np.mean([record["episodeLength"] for record in records])),
        "rewardPercent": 100.0 * meanReturn / 22.0,
        "achievementRewardPercent": float(100.0 * achievementRates.mean()),
        "geometricScore": float(np.expm1(np.log1p(100.0 * achievementRates).mean())),
        "achievementRates": achievementRates.tolist(),
    }


def trainAgent(engineModule, config, outputPath, method, resume=False, collectionLimit=None):
    startedAt = time.monotonic()
    outputPath = externalDirectory(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    if method not in {"ppo", "dedieu", "baseline", "itc"}:
        raise ValueError("unknown native method")
    if collectionLimit is not None and collectionLimit < 1:
        raise ValueError("collectionLimit must be positive")
    if config.batchSize < 1 or config.rolloutHorizon < 1 or config.totalEnvInteractions < 1:
        raise ValueError("interaction and collection sizes must be positive")
    collectionSteps = config.batchSize * config.rolloutHorizon
    targetCollections = config.totalEnvInteractions // collectionSteps
    expectedRealSteps = targetCollections * collectionSteps
    if targetCollections < 1:
        raise ValueError("interaction budget must contain a complete collection")
    identity = trainingIdentity(config, method)
    savedPayload = loadTrainingState(outputPath) if resume else None
    if resume and savedPayload is None:
        raise FileNotFoundError("resume requires a completed training checkpoint")
    if savedPayload is not None and savedPayload["identity"] != identity:
        raise ValueError("checkpoint source, configuration or method differs")
    if not resume and (outputPath / "trainingSummary.json").exists():
        raise FileExistsError("existing training requires resume or a fresh output directory")
    jax, jnp, nnx = (engineModule.jax, engineModule.jnp, engineModule.nnx)

    def scalarMetrics(logs):
        values = (
            {
                name: float(np.mean([np.asarray(jax.device_get(log[name])).mean() for log in logs]))
                for name in logs[0]
            }
            if logs
            else {}
        )
        if not all((np.isfinite(value) for value in values.values())):
            raise FloatingPointError("a policy training metric is not finite")
        return values

    @jax.jit
    def finiteParameters(parameters):
        return jnp.stack([jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(parameters)]).all()

    jax.device_get(jax.jit(lambda value: value + 1)(jnp.asarray(0.0)))
    if config.ckptPath is not None:
        checkpointPath = Path(config.ckptPath).expanduser().resolve()
        if checkpointPath.name == f"fixed_step_seed{config.seed}":
            config.ckptPath = str(checkpointPath.parent)
    if method == "ppo":
        trainer = buildPolicyTrainer(engineModule, config)
        rng = trainer.rng
    else:
        trainer = engineModule.Trainer(config)
        rng, environmentKey = jax.random.split(trainer.rng)
        trainer.currObs, trainer.envState = trainer.env.reset(environmentKey, trainer.envParams)
        trainer.currDone = jnp.ones((config.batchSize,), dtype=jnp.bool_)
        trainer.agentState = trainer.agent.rnn.initializeCarry(config.batchSize)
        trainer.codebook = trainer.tokenizer.initCodebook()
        trainer.codebookSize = jnp.array(0)
        trainer.successRates = jnp.zeros((len(trainer.Achievement),))
        trainer.numEpisodes = jnp.array(0)
        trainer.tgtMean, trainer.tgtStd, trainer.debiasing = (0, 0, 0)
    jax.block_until_ready((trainer.currObs, trainer.envState, nnx.state(trainer.policyTrainState)))
    writeJson(outputPath / "config.json", {"method": method, "configuration": asdict(config)})
    initializationSeconds = time.monotonic() - startedAt
    completedCollections = 0
    episodeRecords = []
    runningReturns = np.zeros(config.batchSize, dtype=np.float64)
    runningLengths = np.zeros(config.batchSize, dtype=np.int64)
    policyCheckpointSteps = []
    counts = {
        "realSteps": 0,
        "realPolicyOptimizerSteps": 0,
        "worldModelOptimizerSteps": 0,
        "tokenizerFullCalls": 0,
        "imaginedPolicyOptimizerSteps": 0,
        "imaginedTransitionSlots": 0,
    }
    timingTotals = {
        "initializationSeconds": initializationSeconds,
        "realCollectionSeconds": 0.0,
        "realPolicySeconds": 0.0,
        "tokenizerAndWorldModelSeconds": 0.0,
        "imaginedCollectionSeconds": 0.0,
        "imaginedPolicySeconds": 0.0,
        "metricSeconds": 0.0,
        "checkpointSeconds": 0.0,
    }
    restoredCollections = 0
    previousSessionWallSeconds = 0.0
    if savedPayload is None:
        actorDigest = hashlib.sha256()
        actorParameters = nnx.state(trainer.agent, nnx.Param)
        for parameter in jax.tree.leaves(actorParameters):
            array = hostArray(parameter)
            actorDigest.update(str((array.shape, array.dtype.str)).encode())
            actorDigest.update(np.ascontiguousarray(array).tobytes())
        normLayers = [
            norm
            for block in trainer.agent.encoder.blocks
            for norm in (block.norm, block.resnet1.norm, block.resnet2.norm)
        ]
        if len(normLayers) != 9 or any((type(norm).__name__ != "GroupNorm" for norm in normLayers)):
            raise ValueError("the released recipe requires nine GroupNorm layers")
        initialState = {
            "freshInitialization": True,
            "seed": config.seed,
            "sourceIdentity": identity,
            "actorParameterSha256": actorDigest.hexdigest(),
            "actorParameterCount": sum(
                (parameter.size for parameter in jax.tree.leaves(actorParameters))
            ),
            "policyOptimizerStep": int(
                np.asarray(jax.device_get(trainer.policyTrainState.step.value))
            ),
            "worldModelOptimizerStep": (
                None
                if method == "ppo"
                else int(np.asarray(jax.device_get(trainer.worldModelTrainState.step)))
            ),
            "codebookSize": (
                None if method == "ppo" else int(np.asarray(jax.device_get(trainer.codebookSize)))
            ),
            "replayCurrentIndex": (
                None
                if method == "ppo"
                else int(np.asarray(jax.device_get(trainer.bufferState.current_index)))
            ),
            "batchStatisticLeaves": len(jax.tree.leaves(nnx.state(trainer.agent, nnx.BatchStat))),
            "counts": dict(counts),
            "criticCoefficient": config.acConfig.params.tdLossCoef,
        }
        if initialState["policyOptimizerStep"] != 0 or initialState[
            "worldModelOptimizerStep"
        ] not in {0, None}:
            raise ValueError("a fresh run must start with zero optimizer steps")
        appendRecord(outputPath / "progress.jsonl", {"event": "initialization", **initialState})
    if savedPayload is not None:
        # restore stochastic state alongside every optimizer
        state = jax.device_put(restoreTrainingTree(savedPayload["state"]))
        metadata = savedPayload["metadata"]
        graphDefinition, unusedState = nnx.split(trainer.policyTrainState)
        del unusedState
        trainer.policyTrainState = nnx.merge(graphDefinition, state["policyState"])
        trainer.agent = trainer.policyTrainState.model
        for attributeName, stateName in (
            ("envState", "environmentState"),
            ("currObs", "observation"),
            ("currDone", "done"),
            ("agentState", "memory"),
            ("tgtMean", "targetMean"),
            ("tgtStd", "targetStd"),
            ("debiasing", "debiasing"),
            ("successRates", "successRates"),
            ("numEpisodes", "episodeCount"),
        ):
            setattr(trainer, attributeName, state[stateName])
        rng = state["rng"]
        if method != "ppo":
            trainer.worldModelTrainState = trainer.worldModelTrainState.replace(
                **state["worldModelState"]
            )
            trainer.bufferState = state["replayState"]
            trainer.codebook, trainer.codebookSize = (state["codebook"], state["codebookSize"])
        jax.block_until_ready(
            (nnx.state(trainer.policyTrainState), trainer.currObs, trainer.envState)
        )
        completedCollections = int(metadata["completedCollections"])
        if not 0 <= completedCollections <= targetCollections:
            raise ValueError("checkpoint collection count is invalid")
        restoredCollections = completedCollections
        episodeRecords = metadata["episodeRecords"]
        runningReturns = np.asarray(metadata["runningReturns"], dtype=np.float64)
        runningLengths = np.asarray(metadata["runningLengths"], dtype=np.int64)
        counts = metadata["counts"]
        if counts["realSteps"] != completedCollections * collectionSteps:
            raise ValueError("checkpoint interaction count disagrees with its boundary")
        timingTotals = metadata["timingTotals"]
        timingTotals["initializationSeconds"] += initializationSeconds
        previousSessionWallSeconds = float(metadata["accountedSessionWallSeconds"])
        policyCheckpointSteps = metadata["policyCheckpointSteps"]
        appendRecord(
            outputPath / "progress.jsonl",
            {
                "event": "resume",
                "restoredCollections": completedCollections,
                "counts": dict(counts),
                "policyOptimizerStep": int(
                    np.asarray(jax.device_get(trainer.policyTrainState.step.value))
                ),
                "worldModelOptimizerStep": (
                    None
                    if method == "ppo"
                    else int(np.asarray(jax.device_get(trainer.worldModelTrainState.step)))
                ),
                "rng": np.asarray(jax.device_get(rng)).tolist(),
            },
        )
        del savedPayload, state
    capturedRollout = {}
    nativeRollout = trainer.rollout

    def captureRollout(*arguments, **options):
        result = nativeRollout(*arguments, **options)
        capturedRollout["rewards"] = result[1][4]
        capturedRollout["dones"] = result[1][5]
        capturedRollout["achievements"] = result[1][6]["returned_episode_achievements"]
        return result

    trainer.rollout = captureRollout

    def measured(name, function, stateFunction=None):
        operationStartedAt = time.monotonic()
        result = function()
        jax.block_until_ready(result)
        if stateFunction is not None:
            jax.block_until_ready(stateFunction())
        timingTotals[name] += time.monotonic() - operationStartedAt
        return result

    def optimizerStep():
        return int(np.asarray(jax.device_get(trainer.policyTrainState.step.value)))

    # checkpoint only after completing all collection updates
    def saveBoundary():
        state = {
            "policyState": nnx.split(trainer.policyTrainState)[1],
            "environmentState": trainer.envState,
            "observation": trainer.currObs,
            "done": trainer.currDone,
            "memory": trainer.agentState,
            "targetMean": trainer.tgtMean,
            "targetStd": trainer.tgtStd,
            "debiasing": trainer.debiasing,
            "successRates": trainer.successRates,
            "episodeCount": trainer.numEpisodes,
            "rng": rng,
        }
        if method != "ppo":
            modelState = trainer.worldModelTrainState
            state.update(
                {
                    "worldModelState": {
                        "params": modelState.params,
                        "opt_state": modelState.opt_state,
                        "step": modelState.step,
                    },
                    "replayState": trainer.bufferState,
                    "codebook": trainer.codebook,
                    "codebookSize": trainer.codebookSize,
                }
            )
        metadata = {
            "completedCollections": completedCollections,
            "counts": dict(counts),
            "episodeRecords": episodeRecords,
            "runningReturns": runningReturns.tolist(),
            "runningLengths": runningLengths.tolist(),
            "timingTotals": dict(timingTotals),
            "policyCheckpointSteps": list(policyCheckpointSteps),
            "accountedSessionWallSeconds": previousSessionWallSeconds
            + time.monotonic()
            - startedAt,
        }
        operationStartedAt = time.monotonic()
        saveTrainingState(outputPath, {"identity": identity, "state": state, "metadata": metadata})
        timingTotals["checkpointSeconds"] += time.monotonic() - operationStartedAt
        appendRecord(
            outputPath / "progress.jsonl",
            {
                "event": "checkpoint",
                "completedCollections": completedCollections,
                "realSteps": counts["realSteps"],
                "checkpointSeconds": timingTotals["checkpointSeconds"],
                "sessionWallSeconds": time.monotonic() - startedAt,
            },
        )

    def saveModels(step):
        modelState = {"policyState": nnx.split(trainer.policyTrainState)[1]}
        if method != "ppo":
            modelState["worldModelState"] = {
                "params": trainer.worldModelTrainState.params,
                "opt_state": trainer.worldModelTrainState.opt_state,
                "step": trainer.worldModelTrainState.step,
            }
        saveTrainingState(
            outputPath / "models" / str(step),
            {
                "identity": identity,
                "state": modelState,
                "metadata": {
                    "counts": dict(counts),
                    "completedCollections": completedCollections,
                    "snapshotKind": "models",
                },
            },
        )

    localCollections = 0
    stopReason = "completed"
    lastSavedCollection = completedCollections if resume else None
    while completedCollections < targetCollections:
        if (outputPath / "stop").exists():
            stopReason = "stopMarker"
            break
        if collectionLimit is not None and localCollections >= collectionLimit:
            stopReason = "collectionLimit"
            break
        collectionIndex = completedCollections
        step = (collectionIndex + 1) * collectionSteps
        if method == "ppo":
            learningRate = config.acConfig.learningRate * (
                1.0 - collectionIndex / targetCollections
            )
            # preserve adam moments while annealing the learning rate
            trainer.policyTrainState.tx = engineModule.optax.chain(
                engineModule.optax.clip_by_global_norm(config.maxGradNorm),
                engineModule.optax.adam(learning_rate=learningRate, eps=1e-05),
            )
        else:
            learningRate = config.acConfig.learningRate
        rng, rolloutKey = jax.random.split(rng)
        data, nextMemory = measured(
            "realCollectionSeconds",
            lambda: trainer.collectFromEnv(rolloutKey, step),
            lambda: trainer.bufferState,
        )
        beforeStep = optimizerStep()
        policyLogs = measured(
            "realPolicySeconds",
            lambda: trainer.learnPolicy(
                data,
                trainer.agentState,
                config.acConfig.numRealEpochs,
                config.acConfig.numRealMinibatches,
                enableTqdm=False,
            ),
            lambda: nnx.state(trainer.policyTrainState),
        )
        scalarLogs = scalarMetrics(policyLogs)
        counts["realPolicyOptimizerSteps"] += optimizerStep() - beforeStep
        trainer.agentState = nextMemory
        if method != "ppo":
            rng, sampleKey = jax.random.split(rng)
            modelBefore = int(np.asarray(jax.device_get(trainer.worldModelTrainState.step)))
            measured(
                "tokenizerAndWorldModelSeconds",
                lambda: trainer.learnWorldModel(sampleKey, step),
                lambda: (trainer.codebook, trainer.codebookSize, trainer.worldModelTrainState),
            )
            if not bool(jax.device_get(finiteParameters(trainer.worldModelTrainState.params))):
                raise FloatingPointError("world-model parameters are not finite")
            counts["worldModelOptimizerSteps"] += (
                int(np.asarray(jax.device_get(trainer.worldModelTrainState.step))) - modelBefore
            )
            counts["tokenizerFullCalls"] += config.tokenConfig.numUpdates
            if step >= config.warmupInteractions:
                for imaginationIndex in range(config.acConfig.numImaginationUpdates):
                    rng, imaginationKey = jax.random.split(rng)
                    imaginedData, imaginedMemory, unusedLogs = measured(
                        "imaginedCollectionSeconds", lambda: trainer.collectFromWm(imaginationKey)
                    )
                    del unusedLogs
                    beforeStep = optimizerStep()
                    imaginedLogs = measured(
                        "imaginedPolicySeconds",
                        lambda: trainer.learnPolicy(
                            imaginedData,
                            imaginedMemory,
                            config.acConfig.numImaginationEpochs,
                            1,
                            enableTqdm=False,
                        ),
                        lambda: nnx.state(trainer.policyTrainState),
                    )
                    scalarMetrics(imaginedLogs)
                    counts["imaginedPolicyOptimizerSteps"] += optimizerStep() - beforeStep
                    counts["imaginedTransitionSlots"] += config.batchSize * config.wmRolloutHorizon
        operationStartedAt = time.monotonic()
        rewards, dones, achievements = jax.device_get(
            (
                capturedRollout.pop("rewards"),
                capturedRollout.pop("dones"),
                capturedRollout.pop("achievements"),
            )
        )
        for timeIndex in range(config.rolloutHorizon):
            runningReturns += rewards[:, timeIndex]
            runningLengths += 1
            for environmentIndex in np.flatnonzero(dones[:, timeIndex]):
                episodeRecords.append(
                    {
                        "realStep": counts["realSteps"]
                        + timeIndex * config.batchSize
                        + int(environmentIndex)
                        + 1,
                        "environmentIndex": int(environmentIndex),
                        "episodeReturn": float(runningReturns[environmentIndex]),
                        "episodeLength": int(runningLengths[environmentIndex]),
                        "achievements": np.asarray(
                            achievements[environmentIndex, timeIndex], dtype=np.bool_
                        ).tolist(),
                    }
                )
                runningReturns[environmentIndex] = 0.0
                runningLengths[environmentIndex] = 0
        completedCollections += 1
        localCollections += 1
        counts["realSteps"] = step
        recentEpisodes = [record for record in episodeRecords if record["realStep"] > step - 50000]
        collectionRecord = {
            "event": "collection",
            "method": method,
            "completedCollections": completedCollections,
            "counts": dict(counts),
            "learningRate": learningRate,
            "policy": scalarLogs,
            "trainingEpisodes": episodeMetrics(episodeRecords),
            "last50kTrainingEpisodes": episodeMetrics(recentEpisodes),
            "sessionWallSeconds": time.monotonic() - startedAt,
        }
        appendRecord(outputPath / "progress.jsonl", collectionRecord)
        print(
            json.dumps(
                {
                    "method": method,
                    "realSteps": step,
                    "trainingRewardPercentLast50k": collectionRecord["last50kTrainingEpisodes"][
                        "rewardPercent"
                    ],
                    "sessionWallSeconds": collectionRecord["sessionWallSeconds"],
                }
            ),
            flush=True,
        )
        timingTotals["metricSeconds"] += time.monotonic() - operationStartedAt
        milestone = any(
            (step <= requested < step + collectionSteps for requested in config.ckptSteps)
        )
        cleanStop = (outputPath / "stop").exists() or (
            collectionLimit is not None and localCollections >= collectionLimit
        )
        if config.ckptPath is not None and milestone and (step not in policyCheckpointSteps):
            measured("checkpointSeconds", lambda: saveModels(step))
            policyCheckpointSteps.append(step)
        if (
            completedCollections == 1
            or completedCollections % 10 == 0
            or completedCollections == targetCollections
            or cleanStop
        ):
            saveBoundary()
            lastSavedCollection = completedCollections
    trainingCompleted = completedCollections == targetCollections
    if completedCollections and lastSavedCollection != completedCollections:
        saveBoundary()
    recentEpisodes = [
        record for record in episodeRecords if record["realStep"] > counts["realSteps"] - 50000
    ]
    summary = {
        "method": method,
        "trainingCompleted": trainingCompleted,
        "stopReason": stopReason,
        "resumed": resume,
        "restoredCollections": restoredCollections,
        "completedCollections": completedCollections,
        "sessionCollections": localCollections,
        "actualRealSteps": counts["realSteps"],
        "expectedRealSteps": expectedRealSteps,
        "counts": counts,
        "timingTotals": timingTotals,
        "sessionWallSeconds": time.monotonic() - startedAt,
        "checkpointAccountedSessionWallSeconds": previousSessionWallSeconds
        + time.monotonic()
        - startedAt,
        "trainingEpisodes": episodeMetrics(episodeRecords),
        "last50kTrainingEpisodes": episodeMetrics(recentEpisodes),
        "episodeRecords": episodeRecords,
        "unfinishedEpisodeReturns": runningReturns.tolist(),
        "unfinishedEpisodeLengths": runningLengths.tolist(),
        "policyCheckpointSteps": policyCheckpointSteps,
        "identity": identity,
        "configuration": asdict(config),
    }
    writeJson(outputPath / "trainingSummary.json", summary)
    return summary


def loadEngine():
    import trainer

    return trainer


def trainingPlan(method, outputDir, seed=0, resume=False):
    config = configurationFor(method, seed)
    return {
        "method": method,
        "outputDir": str(externalDirectory(outputDir)),
        "resumeRequested": resume,
        "configuration": asdict(config),
    }


def executeTraining(plan):
    startedAt = time.monotonic()
    method = plan["method"]
    seed = plan["configuration"]["seed"]
    outputPath = externalDirectory(plan["outputDir"])
    configureRuntime("gpu")
    resume = plan["resumeRequested"]
    if outputPath.exists() and any(outputPath.iterdir()) and (not resume):
        raise FileExistsError("training requires an empty output directory or resume")
    if resume:
        savedConfig = readJson(outputPath / "config.json")
        if savedConfig["method"] != method or savedConfig["configuration"]["seed"] != seed:
            raise ValueError("resume method or seed differs")
    outputPath.mkdir(parents=True, exist_ok=True)
    (outputPath / "linuxPid.txt").write_text(str(os.getpid()))
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(outputPath / "matplotlib"))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(outputPath / "jaxCache"))
    attemptId = str(time.time_ns())
    ledgerPath = outputPath / "trainingAttempts.jsonl"
    appendRecord(
        ledgerPath,
        {
            "event": "start",
            "attemptId": attemptId,
            "time": time.time(),
            "method": method,
        },
    )
    loggingRun = None
    try:
        engineModule = loadEngine()
        config = configurationFor(method, seed)
        config.ckptPath = str(outputPath / "checkpoints")
        if config.wandbConfig.enable:
            import wandb

            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = str(outputPath)
            loggingRun = wandb.init(
                project=config.wandbConfig.projectName,
                config=asdict(config),
                name=f"{method}_seed{seed}",
                mode="offline",
                dir=str(outputPath),
            )
        trainingSummary = trainAgent(
            engineModule, config, outputPath, method, resume=resume
        )
        if loggingRun is not None:
            loggingRun.finish()
            loggingRun = None
        appendRecord(
            ledgerPath,
            {
                "event": "end",
                "attemptId": attemptId,
                "time": time.time(),
                "wallSeconds": time.monotonic() - startedAt,
                "status": "complete" if trainingSummary["trainingCompleted"] else "partial",
            },
        )
    except BaseException as failure:
        appendRecord(
            ledgerPath,
            {
                "event": "end",
                "attemptId": attemptId,
                "time": time.time(),
                "wallSeconds": time.monotonic() - startedAt,
                "status": "failed",
                "failure": repr(failure),
            },
        )
        raise
    finally:
        if loggingRun is not None:
            loggingRun.finish()


def main():
    parser = argparse.ArgumentParser(description="Train local Craftax PPO, Dedieu or ITC models")
    parser.add_argument("method", choices=("ppo", "dedieu", "itc", "baseline"))
    parser.add_argument("--outputDir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.seed < 2**32:
        parser.error("seed must fit an unsigned 32-bit integer")
    plan = trainingPlan(arguments.method, arguments.outputDir, arguments.seed, arguments.resume)
    if arguments.execute:
        executeTraining(plan)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
