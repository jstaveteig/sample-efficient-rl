from dataclasses import asdict
import functools

from craftax import craftax_env as craftaxEnvironment
import flashbax as fbx
from flax import nnx
from flax.nnx import scan as nnxScan
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from ott.geometry.geometry import Geometry
from ott.solvers import linear
from tqdm import tqdm

from configuration import TrainConfig
from environment import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper
from policy import Agent
from tokenizer import NearestNeighborTokenizer
from worldModel import GPT2WorldModelConfig, FlaxGPT2WorldModelModule, getDefaultPositionIds


@jax.jit
def terminalAdvantages(rewards, dones, oldValues, gamma, lambdaValue):
    continuations = 1.0 - dones.astype(oldValues.dtype)
    deltas = rewards + continuations * gamma * oldValues[:, 1:] - oldValues[:, :-1]

    def accumulate(nextAdvantages, stepValues):
        stepDeltas, stepContinuations = stepValues
        advantages = stepDeltas + gamma * lambdaValue * stepContinuations * nextAdvantages
        return advantages, advantages

    # terminal masking belongs inside the backward recurrence
    unusedCarry, advantages = jax.lax.scan(
        accumulate,
        jnp.zeros_like(deltas[:, 0]),
        (deltas.swapaxes(0, 1), continuations.swapaxes(0, 1)),
        reverse=True,
    )
    del unusedCarry
    advantages = advantages.swapaxes(0, 1)
    return advantages, advantages + oldValues[:, :-1]


class Trainer:

    def __init__(self, config: TrainConfig):
        self.config = config
        self.multiGpu = len(jax.devices()) > 1
        if self.config.ckptPath is not None:
            self.config.ckptPath = self.config.ckptPath.format(config=self.config)
            self.config.ckptPath = f"{self.config.ckptPath}/fixed_step_seed{self.config.seed}"
        self.rng = jax.random.PRNGKey(config.seed)
        self.env, self.envParams, self.numActions, self.Achievement, self.Action = (
            self.buildEnvironment(config.batchSize)
        )
        self.rollout = self.buildRollout(self.env, self.envParams)
        self.agent, self.policyTrainState = self.buildAgent()
        self.rng, wmRng = jax.random.split(self.rng)
        self.worldModel, self.worldModelTrainState, self.worldModelConfig = (
            self.buildWorldModel(wmRng)
        )
        self.tokenizer = self.buildTokenizer()
        self.buffer, self.bufferState, self.bufferAdd = self.buildBuffer()
        self.wmLossFn = self.buildWmLossFn()
        self.wmRollout = self.buildWmRollout(self.config.wmConfig.decodeStrategy)
        self.batchSinkhorn = jax.vmap(Trainer.singleSinkhorn, in_axes=(0, None, None, None))

    def buildEnvironment(self, batchSize: int):
        config = self.config
        if config.envConfig.envType == "craftax":
            from craftax.craftax_classic.constants import Achievement, Action

            env = craftaxEnvironment.make_craftax_env_from_name(
                "Craftax-Classic-Pixels-v1", auto_reset=False
            )
            envParams = env.default_params
            numActions = env.action_space(envParams).n
            env = LogWrapper(env)
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, batchSize)
        elif config.envConfig.envType == "craftax_full":
            from craftax.craftax.constants import Achievement, Action

            env = craftaxEnvironment.make_craftax_env_from_name(
                "Craftax-Pixels-v1", auto_reset=False
            )
            envParams = env.default_params
            numActions = env.action_space(envParams).n
            env = LogWrapper(env)
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, batchSize)
        else:
            raise ValueError(f"Environment type {config.envConfig.envType} not supported.")
        return (env, envParams, numActions, Achievement, Action)

    def buildRollout(self, env, envParams):

        @functools.partial(nnx.jit, static_argnames=("horizon",))
        def rollout(agent, agentState, currObs, currDone, envState, horizon, rolloutRng):

            def oneStep(state, rng):
                observations, done, envState, agentState = state
                distribution, value, agentState = jax.lax.stop_gradient(
                    agent(observations[:, None, ...], done[:, None, ...], agentState)
                )
                rng, sampleRng = jax.random.split(rng)
                action, logProb = distribution.sample_and_log_prob(seed=sampleRng)
                action = action.squeeze(axis=1)
                logProb = logProb.squeeze(axis=1)
                value = value.squeeze(axis=1)
                rng, stepRng = jax.random.split(rng)
                nextObs, envState, reward, done, info = env.step(
                    stepRng, envState, action, envParams
                )
                return (
                    (nextObs, done, envState, agentState),
                    (observations, action, logProb, value, reward, done, info),
                )

            (currObs, currDone, envState, agentState), (
                observations,
                action,
                logProb,
                value,
                reward,
                done,
                info,
            ) = nnxScan(oneStep, out_axes=(nnx.transforms.iteration.Carry, 1), length=horizon)(
                (currObs, currDone, envState, agentState), jax.random.split(rolloutRng, horizon)
            )
            _, lastValue, _ = jax.lax.stop_gradient(
                agent(currObs[:, None, ...], currDone[:, None, ...], agentState)
            )
            return (
                (currObs, currDone, envState, agentState),
                (
                    observations,
                    action,
                    logProb,
                    jnp.concatenate((value, lastValue), axis=1),
                    reward,
                    done,
                    info,
                ),
            )

        return rollout

    def buildAgent(self):
        config = self.config
        agent = Agent(
            numActions=self.numActions,
            resize=config.envConfig.envType != "craftax",
            acParams=config.acConfig.params,
            rngs=nnx.Rngs(config.seed),
        )
        optimizerTransform = optax.chain(
            optax.clip_by_global_norm(config.maxGradNorm),
            optax.adam(learning_rate=config.acConfig.learningRate, eps=1e-05),
        )
        policyTrainState = nnx.Optimizer(agent, optimizerTransform)
        return (agent, policyTrainState)

    def buildTokenizer(self):
        config = self.config
        tokenizer = NearestNeighborTokenizer(
            config.tokenConfig.params.codebookSize,
            config.tokenConfig.params.patchSize,
            config.tokenConfig.params.gridRow,
            config.tokenConfig.params.gridCol,
            config.tokenConfig.params.threshold,
        )
        return tokenizer

    def buildWorldModel(self, rng):
        config = self.config
        modelConfig = GPT2WorldModelConfig(
            numActions=self.numActions, **asdict(config.wmConfig.params)
        )
        inputShape = (config.batchSize, modelConfig.maxTokens)
        worldModel = FlaxGPT2WorldModelModule(
            modelConfig,
            rewardLossCoef=config.wmConfig.rewardLossCoef,
            terminationLossCoef=config.wmConfig.terminationLossCoef,
        )
        rng, initWeightsRng = jax.random.split(rng)
        worldModelParams = worldModel.initWeights(initWeightsRng, inputShape)
        worldModelTx = optax.chain(
            optax.clip_by_global_norm(config.maxGradNorm),
            optax.adam(config.wmConfig.learningRate, eps=1e-05),
        )
        worldModelTrainState = TrainState.create(
            apply_fn=worldModel.apply, params=worldModelParams, tx=worldModelTx
        )
        return (worldModel, worldModelTrainState, modelConfig)

    def buildBuffer(self):
        config = self.config
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=config.batchSize,
            sample_batch_size=config.batchSize,
            sample_sequence_length=max(config.wmRolloutHorizon + 1, config.burnInHorizon),
            period=1,
            min_length_time_axis=max(config.wmRolloutHorizon + 1, config.burnInHorizon),
            max_size=config.replayBufferSize,
        )
        bufferState = buffer.init(
            {
                "obs": jnp.zeros(
                    (
                        config.tokenConfig.params.gridRow * config.tokenConfig.params.patchSize,
                        config.tokenConfig.params.gridCol * config.tokenConfig.params.patchSize,
                        3,
                    ),
                    dtype=jnp.float16,
                ),
                "action": jnp.zeros((), dtype=jnp.uint8),
                "reward": jnp.zeros((), dtype=jnp.float16),
                "done": jnp.zeros((), dtype=jnp.bool),
            }
        )
        if self.multiGpu:
            bufferState = jax.device_put(bufferState, device=jax.devices()[1], donate=True)
        bufferAdd = jax.jit(buffer.add, donate_argnums=(0,))
        return (buffer, bufferState, bufferAdd)

    def buildWmLossFn(self):

        @functools.partial(jax.jit, static_argnums=(1,))
        def wmLossFn(params, worldModel, dropoutKey, stateActionIds, rewards, terminations):
            return worldModel.loss(params, dropoutKey, stateActionIds, rewards, terminations)

        return wmLossFn

    @staticmethod
    def getDistanceCosts(gridRow, gridCol, inventoryHeight, useZeroNeighborDistance):
        gridHeight = gridRow
        gridWidth = gridCol
        horizontalCoordinates = jnp.arange(gridHeight * gridWidth) % gridWidth
        verticalCoordinates = jnp.arange(gridHeight * gridWidth) // gridWidth
        maskCenter = (
            (horizontalCoordinates > 0)
            * (horizontalCoordinates < gridWidth - 1)
            * (verticalCoordinates > 0)
            * (verticalCoordinates < gridHeight - 1 - inventoryHeight)
        )
        distanceCosts = (
            horizontalCoordinates[:, None] - horizontalCoordinates[None, :]
        ) ** 2 + (verticalCoordinates[:, None] - verticalCoordinates[None, :]) ** 2
        distanceLimit = 4.1
        distanceInf = 100
        distanceCosts = jnp.where(distanceCosts > distanceLimit, distanceInf, distanceCosts)
        if useZeroNeighborDistance:
            distanceCosts = jnp.where(distanceCosts <= 1.1, 0, distanceCosts)
        return (distanceCosts, maskCenter)

    @staticmethod
    def greedySingleStep(index, outputMatrix):
        target = outputMatrix.argmax(axis=-2, keepdims=True)
        plan = jnp.zeros_like(outputMatrix)
        plan = jnp.put_along_axis(plan, target, 1, axis=-2, inplace=False)
        outputConflict = jnp.where(plan, outputMatrix, -jnp.inf)
        source = outputConflict.argmax(axis=-1, keepdims=True)
        planWin = jnp.zeros_like(outputConflict)
        planWin = jnp.put_along_axis(planWin, source, 1, axis=-1, inplace=False)
        planWin = plan * planWin
        planLose = plan * (1 - planWin)
        outputMatrix = jnp.where(planLose, -jnp.inf, outputMatrix)
        return outputMatrix

    @staticmethod
    def greedyArgmax(outputMatrix, tokensPerBlock):
        # resolve competing copies before assigning each destination
        outputMatrix = jax.lax.fori_loop(0, tokensPerBlock, Trainer.greedySingleStep, outputMatrix)
        return outputMatrix.argmax(axis=0)

    @staticmethod
    def singleSinkhorn(costMatrix, observationCount, sinkhornEpsilon, tokensPerBlock):
        geom = Geometry(cost_matrix=costMatrix, epsilon=sinkhornEpsilon, relative_epsilon="std")
        output = linear.solve(geom)
        return (
            Trainer.greedyArgmax(output.matrix.at[:, observationCount:].set(0), tokensPerBlock),
            output.matrix,
        )

    def solveOptimalTransport(
        self, inputIds, tokensPerBlock, nextStateLogits, nextStateIds, decodeStrategy="sinkhorn"
    ):
        currStateIds = inputIds[:, -tokensPerBlock:-1]
        distanceCosts, maskCenter = Trainer.getDistanceCosts(
            self.config.tokenConfig.params.gridRow,
            self.config.tokenConfig.params.gridCol,
            2 if self.config.envConfig.envType == "craftax" else 4,
            self.config.wmConfig.useZeroNeighborDistance,
        )
        costs = (
            distanceCosts * self.config.wmConfig.distanceCoef
            + jnp.take_along_axis(
                -jax.nn.softmax(nextStateLogits[..., None, :, :]),
                currStateIds[..., :, None, None],
                axis=-1,
            ).squeeze(axis=-1)
            - self.config.wmConfig.trashCost
        )
        predCosts = jnp.take_along_axis(
            -jax.nn.softmax(nextStateLogits[..., :, :]), nextStateIds[..., :, None], axis=-1
        ).squeeze(axis=-1)[..., None, :]
        predCosts = jnp.repeat(predCosts, self.config.wmConfig.numDummy, axis=-2)
        shape = predCosts.shape
        eye = jnp.eye(shape[-1])
        # diagonal dummy sources represent newly sampled observation tokens
        predCosts = (1 - eye) * 100 + eye * predCosts
        trashCosts = jnp.zeros((*shape[:-2], shape[-1], shape[-2]))
        dummyCosts = jnp.zeros((*shape[:-2], shape[-2], shape[-2]))
        finalCosts = jnp.concatenate(
            (
                jnp.concatenate((costs, trashCosts), axis=-1),
                jnp.concatenate((predCosts, dummyCosts), axis=-1),
            ),
            axis=-2,
        )
        if decodeStrategy == "sinkhorn":
            nextStateFrom, partialTransport = self.batchSinkhorn(
                finalCosts, shape[-1], self.config.wmConfig.sinkhornEpsilon, tokensPerBlock
            )
        else:
            raise ValueError("correspondence decoding requires sinkhorn")
        nextStateFrom = nextStateFrom[..., : tokensPerBlock - 1]
        return (nextStateFrom, maskCenter)

    def buildWmRollout(self, decodeStrategy: str = "original"):

        @functools.partial(
            nnx.jit,
            static_argnames=("worldModel", "tokenizer", "horizon", "maxTokens", "tokensPerBlock"),
        )
        def wmRollout(
            agent,
            agentState,
            currObs,
            currDone,
            worldModel,
            worldModelParams,
            tokenizer,
            codebook,
            codebookSize,
            horizon,
            maxTokens,
            tokensPerBlock,
            rolloutRng,
        ):

            def oneStep(state, rng):
                observations, done, agentState, positionIds, pastKeyValues = state
                distribution, value, agentState = jax.lax.stop_gradient(
                    agent(observations[:, None, ...], done[:, None, ...], agentState)
                )
                rng, sampleRng = jax.random.split(rng)
                action, logProb = distribution.sample_and_log_prob(seed=sampleRng)
                action = action.squeeze(axis=1)
                logProb = logProb.squeeze(axis=1)
                value = value.squeeze(axis=1)

                def imagineState(rng, worldModel, params, inputIds, positionIds, pastKeyValues):
                    outputs = worldModel(
                        params, inputIds, position_ids=positionIds, past_key_values=pastKeyValues
                    )
                    stateRng, rewardRng, doneRng = jax.random.split(rng, 3)
                    rewardLogits = outputs.rewardLogits[:, -1]
                    reward = jax.random.categorical(rewardRng, rewardLogits)
                    doneLogits = outputs.terminationLogits[:, -1]
                    done = jax.random.categorical(doneRng, doneLogits)
                    tokensPerState = tokensPerBlock - 1
                    nextStateLogits = outputs.observationLogits[:, -tokensPerState:]
                    maskCodebook = jnp.where(
                        jnp.arange(self.config.tokenConfig.params.codebookSize) >= codebookSize,
                        -jnp.inf,
                        0.0,
                    )
                    nextStateLogits = nextStateLogits + maskCodebook
                    # gumbel argmax samples the categorical token distribution
                    nextStateLogits = nextStateLogits + jax.random.gumbel(
                        stateRng, nextStateLogits.shape
                    )
                    nextStateIds = jnp.argmax(nextStateLogits, axis=-1)
                    if decodeStrategy == "sinkhorn":
                        nextStateFrom, maskCenter = self.solveOptimalTransport(
                            inputIds, tokensPerBlock, nextStateLogits, nextStateIds, decodeStrategy
                        )
                        currStateIds = inputIds[:, -tokensPerBlock:-1]
                        # terminal predictions cannot copy previous world tokens
                        shouldSkip = done[:, None] == 0
                        nextStateIds = jnp.where(
                            (nextStateFrom < tokensPerBlock - 1) * maskCenter * shouldSkip,
                            jnp.take_along_axis(currStateIds, nextStateFrom, axis=-1),
                            nextStateIds,
                        )
                    if self.config.wmConfig.params.useSpatioTemporal:
                        nextPositionIds = positionIds + 2
                    else:
                        nextPositionIds = positionIds + tokensPerBlock
                    return (nextStateIds, reward, done, nextPositionIds, outputs.past_key_values)

                stateIds = tokenizer(observations, codebook)
                stateActionIds = jnp.concatenate((stateIds, action[:, None]), axis=-1)
                rng, stepRng = jax.random.split(rng)
                nextStateIds, reward, done, positionIds, pastKeyValues = imagineState(
                    stepRng,
                    worldModel,
                    worldModelParams,
                    stateActionIds,
                    positionIds,
                    pastKeyValues,
                )
                nextObs = tokenizer.decode(nextStateIds, codebook)
                return (
                    (nextObs, done, agentState, positionIds, pastKeyValues),
                    (observations, action, logProb, value, reward, done),
                )

            batchSize = currObs.shape[0]
            positionIds = getDefaultPositionIds(
                batchSize,
                tokensPerBlock,
                tokensPerBlock,
                self.config.wmConfig.params.useSpatioTemporal,
            )
            pastKeyValues = worldModel.initCache(batchSize, maxTokens)
            (currObs, currDone, agentState, _, _), (
                observations,
                action,
                logProb,
                value,
                reward,
                done,
            ) = nnx.scan(oneStep, out_axes=(nnx.transforms.iteration.Carry, 1))(
                (currObs, currDone, agentState, positionIds, pastKeyValues),
                jax.random.split(rolloutRng, horizon),
            )
            _, lastValue, _ = jax.lax.stop_gradient(
                agent(currObs[:, None, ...], currDone[:, None, ...], agentState)
            )
            return (
                observations,
                action,
                logProb,
                jnp.concatenate((value, lastValue), axis=1),
                reward,
                done,
            )

        return wmRollout

    def collectFromEnv(self, rolloutRng, step):
        config = self.config
        (self.currObs, nextDone, self.envState, nextAgentState), (
            observations,
            action,
            logProb,
            value,
            reward,
            done,
            info,
        ) = self.rollout(
            self.agent,
            self.agentState,
            self.currObs,
            self.currDone,
            self.envState,
            config.rolloutHorizon,
            rolloutRng,
        )
        self.bufferState = self.bufferAdd(
            self.bufferState,
            {
                "obs": observations.astype(jnp.float16),
                "action": action.astype(jnp.uint8),
                "reward": reward.astype(jnp.float16),
                "done": done,
            },
        )
        if config.wandbConfig.enable:
            import wandb

            wandb.log(
                {
                    "rollout/reward": reward.mean(),
                    "rollout/done": done.mean(),
                    "rollout/log_prob": logProb.mean(),
                    "rollout/value": value.mean(),
                    "target_mean": self.tgtMean,
                    "target_std": self.tgtStd,
                    "debiasing": self.debiasing,
                },
                step=step,
            )
        if info["returned_episode"].any():
            avgEpisodeReturns = jnp.average(
                info["returned_episode_returns"], weights=info["returned_episode"]
            )
            numEpisodeEnds = info["returned_episode"].sum()
            weights = jnp.broadcast_to(
                info["returned_episode"][..., None], info["returned_episode_achievements"].shape
            )
            avgEpisodeAchievements = jnp.average(
                info["returned_episode_achievements"],
                weights=weights,
                axis=range(len(info["returned_episode_achievements"].shape))[:-1],
            )
            score = jnp.exp(jnp.mean(jnp.log(1 + avgEpisodeAchievements * 100.0))) - 1.0
            self.successRates = (
                self.successRates * self.numEpisodes + avgEpisodeAchievements * numEpisodeEnds
            ) / (self.numEpisodes + numEpisodeEnds)
            self.numEpisodes = self.numEpisodes + numEpisodeEnds
            accScore = jnp.exp(jnp.mean(jnp.log(1 + self.successRates * 100.0))) - 1.0
            if config.wandbConfig.enable:
                import wandb

                wandb.log(
                    {
                        "rollout/return": avgEpisodeReturns,
                        "rollout/ends": numEpisodeEnds,
                        **{
                            f"rollout/achievements/{achievement.name}": avgEpisodeAchievements[
                                achievement.value
                            ]
                            for achievement in self.Achievement
                        },
                        "rollout/single_score": score,
                        "rollout/accumulated_score": accScore,
                    },
                    step=step,
                )
        reset = jnp.concatenate((self.currDone[:, None], done[:, :-1]), axis=1)
        # invert debiased target normalization before computing advantages
        value = value * jnp.maximum(
            self.tgtStd / jnp.maximum(self.debiasing, 0.1), 0.1
        ) + self.tgtMean / jnp.maximum(self.debiasing, 0.01)
        advantages, targets = terminalAdvantages(
            reward, done, value, config.acConfig.gamma, config.acConfig.ld
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        self.currDone = nextDone
        return ((observations, reset, action, logProb, advantages, targets), nextAgentState)

    def learnPolicy(self, data, agentState, nEpochs, nMinibatches, enableTqdm=True):
        config = self.config
        observations, reset, action, logProb, advantages, targets = data
        miniLogs = []
        for epochIndex in tqdm(range(nEpochs), desc="Policy", disable=not enableTqdm):
            for index in range(nMinibatches):
                startIdx = index * (config.batchSize // nMinibatches)
                endIdx = (index + 1) * (config.batchSize // nMinibatches)
                tgtMini = targets[startIdx:endIdx]
                self.tgtMean = (
                    config.acConfig.tgtDiscount * self.tgtMean
                    + (1 - config.acConfig.tgtDiscount) * tgtMini.mean()
                )
                self.tgtStd = (
                    config.acConfig.tgtDiscount * self.tgtStd
                    + (1 - config.acConfig.tgtDiscount) * tgtMini.std()
                )
                self.debiasing = (
                    config.acConfig.tgtDiscount * self.debiasing
                    + (1 - config.acConfig.tgtDiscount) * 1
                )
                tgtMini = (
                    tgtMini - self.tgtMean / jnp.maximum(self.debiasing, 0.01)
                ) / jnp.maximum(self.tgtStd / jnp.maximum(self.debiasing, 0.1), 0.1)
                lossFn = lambda model: model.loss(
                    observations[startIdx:endIdx],
                    reset[startIdx:endIdx],
                    agentState[startIdx:endIdx],
                    action[startIdx:endIdx],
                    logProb[startIdx:endIdx],
                    advantages[startIdx:endIdx],
                    tgtMini,
                )
                (loss, metrics), gradients = nnx.value_and_grad(lossFn, has_aux=True)(
                    self.policyTrainState.model
                )
                self.policyTrainState.update(grads=gradients)
                jax.block_until_ready(nnx.state(self.policyTrainState))
                miniLogs.append({"total_loss": loss, **metrics})
        return miniLogs

    def learnWorldModel(self, rng, step):
        config = self.config
        for _ in tqdm(range(config.tokenConfig.numUpdates), desc="Tokenizer"):
            rng, sampleRng = jax.random.split(rng)
            data = self.buffer.sample(self.bufferState, sampleRng)
            if self.multiGpu:
                data = jax.device_get(data)
            observations = data.experience["obs"].astype(jnp.float32)
            self.codebook, self.codebookSize = self.tokenizer.update(
                observations, self.codebook, self.codebookSize
            )
            jax.block_until_ready((self.codebook, self.codebookSize))
        miniLogs = []
        for _ in tqdm(range(config.wmConfig.numUpdates), desc="World Model"):
            rng, sampleRng = jax.random.split(rng)
            data = self.buffer.sample(self.bufferState, sampleRng)
            if self.multiGpu:
                data = jax.device_get(data)
            for index in range(config.wmConfig.numMinibatches):
                startIdx = index * (config.batchSize // config.wmConfig.numMinibatches)
                endIdx = (index + 1) * (config.batchSize // config.wmConfig.numMinibatches)
                observations = data.experience["obs"][startIdx:endIdx].astype(jnp.float32)
                action = data.experience["action"][startIdx:endIdx].astype(jnp.int32)
                reward = data.experience["reward"][startIdx:endIdx].astype(jnp.float32)
                done = data.experience["done"][startIdx:endIdx]
                batchSize = observations.shape[0]
                stateIds = self.tokenizer(observations, self.codebook)
                stateActionIds = jnp.concatenate((stateIds, action[:, :, None]), axis=-1)
                stateActionIds = stateActionIds.reshape(batchSize, -1)
                rng, dropoutRng = jax.random.split(rng)
                (loss, metrics), gradients = jax.value_and_grad(self.wmLossFn, has_aux=True)(
                    self.worldModelTrainState.params,
                    self.worldModel,
                    dropoutRng,
                    stateActionIds,
                    reward[:, :-1],
                    done[:, :-1].astype(jnp.int32),
                )
                self.worldModelTrainState = self.worldModelTrainState.apply_gradients(
                    grads=gradients
                )
                jax.block_until_ready(self.worldModelTrainState)
                miniLogs.append({"total_loss": loss, **metrics})
        if config.wandbConfig.enable and len(miniLogs) > 0:
            import wandb

            logs = {}
            for metricName in miniLogs[0].keys():
                logs[f"world_model/{metricName}"] = jnp.array(
                    [metricEntry[metricName] for metricEntry in miniLogs]
                ).mean()
            wandb.log(logs, step=step)

    def collectFromWm(self, rng, wmRollout=None):
        if wmRollout is None:
            wmRollout = self.wmRollout
        config = self.config
        rng, sampleRng = jax.random.split(rng)
        data = self.buffer.sample(self.bufferState, sampleRng)
        observations = data.experience["obs"].astype(jnp.float32)
        done = data.experience["done"]
        if self.multiGpu:
            observations = jax.device_get(observations[:, : config.burnInHorizon + 1])
            done = jax.device_get(done[:, : config.burnInHorizon + 1])
        # resets describe the transition preceding each observation
        _, _, imaginationAgentState = self.agent(
            observations[:, : config.burnInHorizon],
            jnp.concatenate(
                (jnp.ones_like(done[:, :1]), done[:, : config.burnInHorizon - 1]), axis=1
            ),
            self.agent.rnn.initializeCarry(config.batchSize),
        )
        currObs = observations[:, config.burnInHorizon]
        currDone = done[:, config.burnInHorizon - 1]
        rng, rolloutRng = jax.random.split(rng)
        observations, action, logProb, value, reward, done = wmRollout(
            self.agent,
            imaginationAgentState,
            currObs,
            currDone.astype(jnp.int32),
            self.worldModel,
            self.worldModelTrainState.params,
            self.tokenizer,
            self.codebook,
            self.codebookSize,
            config.wmRolloutHorizon,
            self.worldModelConfig.maxTokens,
            self.worldModelConfig.tokensPerBlock,
            rolloutRng,
        )
        reset = jnp.concatenate((currDone[:, None], done[:, :-1]), axis=1).astype(jnp.bool)
        value = value * jnp.maximum(
            self.tgtStd / jnp.maximum(self.debiasing, 0.1), 0.1
        ) + self.tgtMean / jnp.maximum(self.debiasing, 0.01)
        advantages, targets = terminalAdvantages(
            reward, done, value, config.acConfig.gamma, config.acConfig.ld
        )
        singleLog = {
            "reward": reward.mean(),
            "done": done.mean(),
            "log_prob": logProb.mean(),
            "value": value.mean(),
            "return": (reward.sum(axis=-1) + value[:, -1]).mean(),
        }
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        return (
            (observations, reset, action, logProb, advantages, targets),
            imaginationAgentState,
            singleLog,
        )
