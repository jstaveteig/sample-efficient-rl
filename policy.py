from dataclasses import asdict

import distrax
from flax import nnx
import jax
import jax.numpy as jnp

from configuration import ActorCriticParams


class ResNetBlock(nnx.Module):
    def __init__(self, features, rngs):
        self.norm = nnx.GroupNorm(
            num_features=features,
            num_groups=None,
            group_size=1,
            rngs=rngs,
        )
        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs,
        )

    def __call__(self, features):
        residual = features
        features = nnx.relu(features)
        features = self.norm(features)
        return residual + self.conv(features)


class ImpalaBlock(nnx.Module):
    def __init__(self, inputFeatures, outputFeatures, rngs):
        self.norm = nnx.GroupNorm(
            num_features=inputFeatures,
            num_groups=None,
            group_size=1,
            rngs=rngs,
        )
        self.conv = nnx.Conv(
            in_features=inputFeatures,
            out_features=outputFeatures,
            kernel_size=(3, 3),
            kernel_init=nnx.initializers.xavier_normal(),
            rngs=rngs,
        )
        self.resnet1 = ResNetBlock(outputFeatures, rngs)
        self.resnet2 = ResNetBlock(outputFeatures, rngs)

    def __call__(self, features):
        features = self.norm(features)
        features = self.conv(features)
        features = nnx.max_pool(features, (3, 3), (2, 2), "SAME")
        features = self.resnet1(features)
        return self.resnet2(features)


class ImpalaCNN(nnx.Module):
    def __init__(self, channels, rngs):
        self.blocks = []
        inputChannels = 3
        for outputChannels in channels:
            self.blocks.append(ImpalaBlock(inputChannels, outputChannels, rngs))
            inputChannels = outputChannels

    def __call__(self, features):
        for block in self.blocks:
            features = block(features)
        return features


class RNN(nnx.Module):
    def __init__(self, inputFeatures, hiddenFeatures, *, rngs):
        self.hiddenFeatures = hiddenFeatures
        self.rnnCell = nnx.GRUCell(
            in_features=inputFeatures,
            hidden_features=hiddenFeatures,
            rngs=rngs,
        )

    def __call__(self, previousState, features, reset):
        def singleStep(memory, observationFeatures, resetFlags):
            memory = jnp.where(resetFlags[:, None], jnp.zeros_like(memory), memory)
            return self.rnnCell(memory, observationFeatures)

        return nnx.scan(
            singleStep,
            in_axes=(nnx.transforms.iteration.Carry, 1, 1),
            out_axes=(nnx.transforms.iteration.Carry, 1),
        )(previousState, features, reset)

    def initializeCarry(self, batchSize):
        return self.rnnCell.initialize_carry((batchSize, self.hiddenFeatures))


class Actor(nnx.Module):
    def __init__(self, inputDim, intermediateDim, numActions, *, rngs):
        self.norm1 = nnx.LayerNorm(num_features=inputDim, rngs=rngs)
        self.linear1 = nnx.Linear(
            in_features=inputDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediateDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediateDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_features=intermediateDim, rngs=rngs)
        self.linear4 = nnx.Linear(
            in_features=intermediateDim,
            out_features=numActions,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )

    def __call__(self, features):
        features = self.norm1(features)
        features = nnx.relu(self.linear1(features))
        features = features + nnx.relu(self.linear2(features))
        features = features + nnx.relu(self.linear3(features))
        features = self.norm2(features)
        return distrax.Categorical(logits=self.linear4(features))


class Critic(nnx.Module):
    def __init__(self, inputDim, intermediateDim, *, rngs):
        self.norm1 = nnx.LayerNorm(num_features=inputDim, rngs=rngs)
        self.linear1 = nnx.Linear(
            in_features=inputDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=intermediateDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.linear3 = nnx.Linear(
            in_features=intermediateDim,
            out_features=intermediateDim,
            kernel_init=nnx.initializers.orthogonal(0.01),
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_features=intermediateDim, rngs=rngs)
        self.linear4 = nnx.Linear(
            in_features=intermediateDim,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(1.0),
            rngs=rngs,
        )

    def __call__(self, features):
        features = self.norm1(features)
        features = nnx.relu(self.linear1(features))
        features = features + nnx.relu(self.linear2(features))
        features = features + nnx.relu(self.linear3(features))
        features = self.norm2(features)
        return jnp.squeeze(self.linear4(features), axis=-1)


class ActorCritic(nnx.Module):
    def __init__(self, inputDim, numActions, *, eps=0.2, tdLossCoef=2.0, entLossCoef=0.01, rngs):
        self.eps = eps
        self.tdLossCoef = tdLossCoef
        self.entLossCoef = entLossCoef
        self.actor = Actor(inputDim, 2048, numActions, rngs=rngs)
        self.critic = Critic(inputDim, 2048, rngs=rngs)

    def __call__(self, state):
        return self.actor(state), self.critic(state)

    def loss(self, state, action, oldLogProbability, advantages, targets):
        distribution, values = self(state)
        probabilityRatio = jnp.exp(distribution.log_prob(action) - oldLogProbability)
        clippedRatio = jnp.clip(probabilityRatio, 1 - self.eps, 1 + self.eps)
        policyLoss = -jnp.minimum(probabilityRatio * advantages, clippedRatio * advantages).mean()
        valueLoss = jnp.square(values - targets).mean()
        entropyLoss = -distribution.entropy().mean()
        totalLoss = policyLoss + self.tdLossCoef * valueLoss + self.entLossCoef * entropyLoss
        return totalLoss, {
            "policy_loss": policyLoss,
            "value_loss": valueLoss,
            "ent_loss": entropyLoss,
            "advantage": advantages,
            "target": targets,
        }


class Agent(nnx.Module):
    def __init__(self, numActions, resize=False, acParams=None, *, rngs):
        self.resize = resize
        self.encoder = ImpalaCNN([64, 64, 128], rngs)
        self.norm = nnx.LayerNorm(num_features=8 * 8 * 128, rngs=rngs)
        self.linear = nnx.Linear(
            in_features=8 * 8 * 128,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
            rngs=rngs,
        )
        self.rnn = RNN(256, 256, rngs=rngs)
        self.actorCritic = ActorCritic(
            8 * 8 * 128 + 256,
            numActions,
            **asdict(ActorCriticParams() if acParams is None else acParams),
            rngs=rngs,
        )

    def encode(self, observations, reset, previousState):
        batchSize, timeSteps = observations.shape[:2]
        if self.resize:
            observations = jax.image.resize(
                observations, (batchSize, timeSteps, 63, 63, 3), "nearest"
            )
        features = self.encoder(observations.reshape(batchSize * timeSteps, 63, 63, 3))
        features = nnx.relu(features).reshape(batchSize, timeSteps, -1)
        memoryInputs = nnx.relu(self.linear(self.norm(features)))
        nextState, memoryFeatures = self.rnn(previousState, memoryInputs, reset)
        # retain visual features alongside the recurrent bottleneck
        state = jnp.concatenate((features, nnx.relu(memoryFeatures)), axis=-1)
        return state, nextState

    def __call__(self, observations, reset, previousState):
        state, nextState = self.encode(observations, reset, previousState)
        distribution, values = self.actorCritic(state)
        return distribution, values, nextState

    @nnx.jit
    def loss(
        self, observations, reset, previousState, action, oldLogProbability, advantages, targets
    ):
        state, unusedMemory = self.encode(observations, reset, previousState)
        del unusedMemory
        return self.actorCritic.loss(state, action, oldLogProbability, advantages, targets)
