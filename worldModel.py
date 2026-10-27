import math
from typing import Any, ClassVar, Optional

import flax
import flax.linen as nn
from flax.core.frozen_dict import unfreeze
from flax.linen.attention import dot_product_attention_weights
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_flax_gpt2 import (
    FlaxGPT2Attention,
    FlaxGPT2Block,
    FlaxGPT2BlockCollection,
    FlaxGPT2MLP,
    FlaxGPT2Module,
)


class GPT2WorldModelConfig(GPT2Config):
    def __init__(self, numActions=17, tokensPerBlock=81, maxBlocks=21, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numActions = numActions
        self.tokensPerBlock = tokensPerBlock
        self.maxBlocks = maxBlocks

    @property
    def maxTokens(self):
        return self.tokensPerBlock * self.maxBlocks

    def __hash__(self):
        return hash(self.to_json_string())


@flax.struct.dataclass
class FlaxGPT2WorldModelOutput(FlaxBaseModelOutputWithPastAndCrossAttentions):
    observationLogits: Optional[jnp.ndarray] = None
    rewardLogits: Optional[jnp.ndarray] = None
    terminationLogits: Optional[jnp.ndarray] = None


def blockCausalMask(sequenceLength, blockSize):
    blockCount = math.ceil(sequenceLength / blockSize)
    # tokens attend throughout their current observation block
    blockMask = jnp.tril(jnp.ones((blockCount, blockCount), dtype=jnp.bool_))
    tokenMask = jnp.repeat(jnp.repeat(blockMask, blockSize, axis=0), blockSize, axis=1)
    return tokenMask[None, None, :sequenceLength, :sequenceLength]


def sliceObservations(stateActionSequence, tokensPerBlock):
    blockCount = stateActionSequence.shape[1] // tokensPerBlock
    tokenIndices = jnp.arange(tokensPerBlock - 1)[None, :]
    blockOffsets = jnp.arange(blockCount)[:, None] * tokensPerBlock
    return stateActionSequence[:, (blockOffsets + tokenIndices).reshape(-1)]


class ObservationHead(nn.Module):
    preserve_adopted_names: ClassVar[bool] = True
    headModule: nn.Module
    tokensPerBlock: int

    def __call__(self, hiddenStates):
        return self.headModule(sliceObservations(hiddenStates, self.tokensPerBlock))


class ActionHead(nn.Module):
    preserve_adopted_names: ClassVar[bool] = True
    headModule: nn.Module
    tokensPerBlock: int

    def __call__(self, hiddenStates):
        sequenceLength = hiddenStates.shape[1]
        actionIndices = jnp.arange(self.tokensPerBlock - 1, sequenceLength, self.tokensPerBlock)
        return self.headModule(hiddenStates[:, actionIndices])


class Embedder(nn.Module):
    preserve_adopted_names: ClassVar[bool] = True
    tokensPerBlock: int
    maxBlocks: int
    observationEmbedding: nn.Embed
    actionEmbedding: nn.Embed
    absoluteEmbedding: Optional[nn.Embed]

    def setup(self):
        observationPattern = jnp.ones(self.tokensPerBlock, dtype=jnp.bool_).at[-1].set(False)
        self.observationMask = jnp.tile(observationPattern, self.maxBlocks)

    def __call__(self, tokens):
        observationMask = self.observationMask[: tokens.shape[1]]
        observationEmbeddings = self.observationEmbedding(tokens)
        actionEmbeddings = self.actionEmbedding(tokens)
        embeddings = jnp.where(
            observationMask[None, :, None],
            observationEmbeddings,
            actionEmbeddings,
        )
        if self.absoluteEmbedding is not None:
            tokenPositions = jnp.arange(tokens.shape[1], dtype=jnp.int32) % self.tokensPerBlock
            embeddings += self.absoluteEmbedding(tokenPositions[None, :])
        return embeddings


def rotateHalf(tensor):
    halfWidth = tensor.shape[-1] // 2
    return jnp.concatenate((-tensor[..., halfWidth:], tensor[..., :halfWidth]), axis=-1)


def applyRotaryPosition(tensor, sinePositions, cosinePositions):
    return tensor * cosinePositions + rotateHalf(tensor) * sinePositions


class FlaxLlamaRotaryEmbeddingReal(nn.Module):
    config: Any
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.headDimension = self.config.hidden_size // self.config.num_attention_heads
        self.inverseFrequencies = 1.0 / (
            10000 ** (np.arange(0, self.headDimension, 2) / self.headDimension)
        )

    def __call__(self, key, query, positionIds):
        positionShape = positionIds.shape
        frequencies = jnp.einsum(
            "i, j -> i j",
            positionIds.flatten(),
            self.inverseFrequencies,
        ).astype("float32")
        embeddings = jnp.concatenate((frequencies, frequencies), axis=-1)
        sinePositions = jnp.sin(embeddings)[:, None, :].reshape(
            *positionShape, 1, self.headDimension
        )
        cosinePositions = jnp.cos(embeddings)[:, None, :].reshape(
            *positionShape, 1, self.headDimension
        )
        key = applyRotaryPosition(key, sinePositions, cosinePositions)
        query = applyRotaryPosition(query, sinePositions, cosinePositions)
        return jnp.asarray(key, dtype=self.dtype), jnp.asarray(query, dtype=self.dtype)


class FlaxGPT2BlockAttention(FlaxGPT2Attention):
    def setup(self):
        super().setup()
        if self.causal:
            self.blockMask = blockCausalMask(
                self.config.max_position_embeddings,
                self.config.tokensPerBlock,
            )
        self.rotaryEmbedding = FlaxLlamaRotaryEmbeddingReal(
            self.config,
            dtype=self.dtype,
            name="rotary_emb",
        )
        headDimension = self.config.hidden_size // self.config.num_attention_heads
        temporalUnits, spatialUnits = 2, 3
        scale = headDimension // 2 // (temporalUnits + spatialUnits * 2)
        # partition rotary dimensions across temporal and spatial coordinates
        rotaryPatterns = [
            jnp.array([False, False] * spatialUnits * scale + [True] * temporalUnits * scale),
            jnp.array([True, False] * spatialUnits * scale + [False] * temporalUnits * scale),
            jnp.array([False, True] * spatialUnits * scale + [False] * temporalUnits * scale),
        ]
        self.rotaryPatterns = [jnp.concatenate((pattern, pattern)) for pattern in rotaryPatterns]

    def __call__(
        self,
        hidden_states,
        position_ids,
        key_value_states=None,
        attention_mask=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
    ):
        isCrossAttention = key_value_states is not None
        batchSize = hidden_states.shape[0]
        if not isCrossAttention:
            queryKeyValue = self.c_attn(hidden_states)
            query, key, value = jnp.split(queryKeyValue, 3, axis=2)
        else:
            query = self.q_attn(hidden_states)
            (query,) = jnp.split(query, 1, axis=2)
            key, value = jnp.split(self.c_attn(key_value_states), 2, axis=2)
        query, key, value = (self._split_heads(tensor) for tensor in (query, key, value))
        for coordinateIndex, pattern in enumerate(self.rotaryPatterns):
            rotatedQuery, rotatedKey = self.rotaryEmbedding(
                query, key, position_ids[..., coordinateIndex]
            )
            query = jnp.where(pattern, rotatedQuery, query)
            key = jnp.where(pattern, rotatedKey, key)
        queryLength, keyLength = query.shape[1], key.shape[1]
        if self.causal:
            if self.has_variable("cache", "cached_key"):
                # cache offsets select the current attention rows
                maskOffset = self.variables["cache"]["cache_index"]
                maxDecoderLength = self.variables["cache"]["cached_key"].shape[1]
                attentionMask = lax.dynamic_slice(
                    self.blockMask,
                    (0, 0, maskOffset, 0),
                    (1, 1, queryLength, maxDecoderLength),
                )
            else:
                attentionMask = self.blockMask[:, :, :queryLength, :keyLength]
            attention_mask = jnp.broadcast_to(attentionMask, (batchSize,) + attentionMask.shape[1:])
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        dropoutRng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropoutRng = self.make_rng("dropout")
        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key, value, attention_mask = self._concatenate_to_cache(
                key, value, query, attention_mask
            )
        if attention_mask is not None:
            attentionBias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attentionBias = None
        attentionWeights = dot_product_attention_weights(
            query,
            key,
            bias=attentionBias,
            dropout_rng=dropoutRng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )
        attentionOutput = jnp.einsum("...hqk,...khd->...qhd", attentionWeights, value)
        attentionOutput = self._merge_heads(attentionOutput)
        attentionOutput = self.c_proj(attentionOutput)
        attentionOutput = self.resid_dropout(attentionOutput, deterministic=deterministic)
        return (attentionOutput, attentionWeights) if output_attentions else (attentionOutput,)


class FlaxGPT2WorldModelBlock(FlaxGPT2Block):
    def setup(self):
        hiddenSize = self.config.hidden_size
        innerDimension = self.config.n_inner if self.config.n_inner is not None else 4 * hiddenSize
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxGPT2BlockAttention(
                config=self.config,
                dtype=self.dtype,
                causal=False,
                is_cross_attention=True,
            )
            self.ln_cross_attn = nn.LayerNorm(
                epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
            )
        self.mlp = FlaxGPT2MLP(self.config, innerDimension, dtype=self.dtype)
        self.attn = FlaxGPT2BlockAttention(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        position_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attentionOutputs = self.attn(
            hidden_states,
            position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        outputs = attentionOutputs[1:]
        hidden_states = attentionOutputs[0] + residual
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError("encoder states require cross-attention layers")
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            crossAttentionOutputs = self.crossattention(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            hidden_states = residual + crossAttentionOutputs[0]
            outputs += crossAttentionOutputs[1:]
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states, deterministic=deterministic)
        return (hidden_states,) + outputs


class FlaxGPT2WorldModelBlockCollection(FlaxGPT2BlockCollection):
    def setup(self):
        self.blocks = [
            FlaxGPT2WorldModelBlock(self.config, name=str(layerIndex), dtype=self.dtype)
            for layerIndex in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        position_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        allAttentions = () if output_attentions else None
        allHiddenStates = () if output_hidden_states else None
        allCrossAttentions = () if output_attentions and encoder_hidden_states is not None else None
        for block in self.blocks:
            if output_hidden_states:
                allHiddenStates += (hidden_states,)
            layerOutputs = block(
                hidden_states,
                position_ids,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layerOutputs[0]
            if output_attentions:
                allAttentions += (layerOutputs[1],)
                if encoder_hidden_states is not None:
                    allCrossAttentions += (layerOutputs[2],)
        return hidden_states, allHiddenStates, allAttentions, allCrossAttentions


class FlaxGPT2WorldModelModule(FlaxGPT2Module):
    config: GPT2WorldModelConfig
    rewardLossCoef: float = 1.0
    terminationLossCoef: float = 1.0

    def setup(self):
        embeddingSize = self.config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size,
            embeddingSize,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.h = FlaxGPT2WorldModelBlockCollection(self.config, dtype=self.dtype)
        actionEmbedding = nn.Embed(
            self.config.numActions,
            embeddingSize,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            name="action_embedding",
            parent=None,
        )
        if self.config.useAbsoluteEmbedding:
            absoluteEmbedding = nn.Embed(
                self.config.tokensPerBlock,
                embeddingSize,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
                dtype=self.dtype,
                name="absolute_embedding",
                parent=None,
            )
        else:
            absoluteEmbedding = None
        self.embedder = Embedder(
            tokensPerBlock=self.config.tokensPerBlock,
            maxBlocks=self.config.maxBlocks,
            observationEmbedding=self.wte,
            actionEmbedding=actionEmbedding,
            absoluteEmbedding=absoluteEmbedding,
        )
        self.observationHead = ObservationHead(
            tokensPerBlock=self.config.tokensPerBlock,
            headModule=nn.Sequential(
                [nn.Dense(embeddingSize), nn.relu, nn.Dense(self.config.vocab_size)],
                name="head_module",
                parent=None,
            ),
            name="observation_head",
        )
        self.rewardHead = ActionHead(
            tokensPerBlock=self.config.tokensPerBlock,
            headModule=nn.Sequential(
                [nn.Dense(embeddingSize), nn.relu, nn.Dense(2)],
                name="head_module",
                parent=None,
            ),
            name="reward_head",
        )
        self.terminationHead = ActionHead(
            tokensPerBlock=self.config.tokensPerBlock,
            headModule=nn.Sequential(
                [nn.Dense(embeddingSize), nn.relu, nn.Dense(2)],
                name="head_module",
                parent=None,
            ),
            name="termination_head",
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        batchSize, sequenceLength = input_ids.shape
        if position_ids is None:
            position_ids = getDefaultPositionIds(
                batchSize,
                sequenceLength,
                self.config.tokensPerBlock,
                self.config.useSpatioTemporal,
            )
        inputEmbeddings = self.embedder(input_ids.astype("i4"))
        hiddenStates = self.dropout(inputEmbeddings, deterministic=deterministic)
        outputs = self.h(
            hiddenStates,
            position_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hiddenStates = self.ln_f(outputs[0])
        observationLogits = self.observationHead(hiddenStates)
        rewardLogits = self.rewardHead(hiddenStates)
        terminationLogits = self.terminationHead(hiddenStates)
        if output_hidden_states:
            allHiddenStates = outputs[1] + (hiddenStates,)
            outputs = (hiddenStates, allHiddenStates) + outputs[2:]
        else:
            outputs = (hiddenStates,) + outputs[1:]
        outputs += (observationLogits, rewardLogits, terminationLogits)
        if not return_dict:
            return tuple(value for value in outputs if value is not None)
        return FlaxGPT2WorldModelOutput(
            last_hidden_state=hiddenStates,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
            observationLogits=observationLogits,
            rewardLogits=rewardLogits,
            terminationLogits=terminationLogits,
        )

    @staticmethod
    def quantizeRewards(rewards):
        return (rewards >= 0.5).astype(jnp.int32)

    def trainingLoss(self, stateActionIds, rewards, terminations):
        # each block predicts the following observation and transition
        inputIds = stateActionIds[:, : -self.config.tokensPerBlock]
        outputs = self.forward(inputIds, jnp.ones_like(inputIds), deterministic=False)
        observationLabels = stateActionIds[:, self.config.tokensPerBlock :]
        observationLabels = sliceObservations(observationLabels, self.config.tokensPerBlock)
        observationLabels = nn.one_hot(observationLabels, self.config.vocab_size)
        observationLoss = optax.softmax_cross_entropy(
            outputs.observationLogits, observationLabels
        ).mean()
        rewardLabels = nn.one_hot(self.quantizeRewards(rewards), 2)
        rewardLoss = optax.softmax_cross_entropy(outputs.rewardLogits, rewardLabels).mean()
        terminationLabels = nn.one_hot(terminations, 2)
        terminationLoss = optax.softmax_cross_entropy(
            outputs.terminationLogits, terminationLabels
        ).mean()
        loss = (
            observationLoss
            + rewardLoss * self.rewardLossCoef
            + terminationLoss * self.terminationLossCoef
        )
        return loss, {
            "observation_loss": observationLoss,
            "reward_loss": rewardLoss,
            "termination_loss": terminationLoss,
        }

    def loss(self, params, dropoutRng, *args, **kwargs):
        return self.apply(
            {"params": params},
            *args,
            **kwargs,
            method="trainingLoss",
            rngs={"dropout": dropoutRng},
        )

    def initWeights(self, rng, inputShape):
        inputIds = jnp.zeros(inputShape, dtype="i4")
        attentionMask = jnp.ones_like(inputIds)
        batchSize, sequenceLength = inputShape
        positionIds = getDefaultPositionIds(
            batchSize,
            sequenceLength,
            self.config.tokensPerBlock,
            self.config.useSpatioTemporal,
        )
        parameterRng, dropoutRng = jax.random.split(rng)
        variables = self.init(
            {"params": parameterRng, "dropout": dropoutRng},
            inputIds,
            attentionMask,
            positionIds,
            method="forward",
            return_dict=False,
        )
        return variables["params"]

    def initCache(self, batchSize, maxLength):
        inputIds = jnp.ones((batchSize, maxLength))
        positionIds = getDefaultPositionIds(
            batchSize,
            maxLength,
            self.config.tokensPerBlock,
            self.config.useSpatioTemporal,
        )
        variables = self.init(
            jax.random.PRNGKey(0),
            inputIds,
            jnp.ones_like(inputIds),
            positionIds,
            method="forward",
            return_dict=False,
            init_cache=True,
        )
        return unfreeze(variables["cache"])

    def __call__(
        self,
        params,
        input_ids,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        dropout_rng=None,
        train=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            encoder_attention_mask = jnp.ones(encoder_hidden_states.shape[:2])
        batchSize, sequenceLength = input_ids.shape
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("cached inference requires position_ids")
            position_ids = getDefaultPositionIds(
                batchSize,
                sequenceLength,
                self.config.tokensPerBlock,
                self.config.useSpatioTemporal,
            )
        if attention_mask is None:
            attention_mask = jnp.ones((batchSize, sequenceLength))
        rngs = {} if dropout_rng is None else {"dropout": dropout_rng}
        variables = {"params": params}
        if past_key_values:
            variables["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False
        outputs = self.apply(
            variables,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            encoder_hidden_states,
            encoder_attention_mask,
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            method="forward",
            rngs=rngs,
            mutable=mutable,
        )
        if past_key_values is not None and return_dict:
            outputs, updatedVariables = outputs
            outputs["past_key_values"] = unfreeze(updatedVariables["cache"])
        elif past_key_values is not None:
            outputs, updatedVariables = outputs
            outputs = outputs[:1] + (unfreeze(updatedVariables["cache"]),) + outputs[1:]
        return outputs


def getDefaultPositionIds(batchSize, sequenceLength, tokensPerBlock, useSpatioTemporal):
    if useSpatioTemporal:
        timeSteps = sequenceLength // tokensPerBlock
        gridWidth = int(math.sqrt(tokensPerBlock - 1))
        temporalPosition = (jnp.arange(sequenceLength) // tokensPerBlock) * 2
        actionOffset = jnp.concatenate((jnp.zeros((tokensPerBlock - 1,)), jnp.ones((1,))))
        temporalPosition += jnp.tile(actionOffset, timeSteps)
        horizontalPosition = jnp.concatenate(
            (
                jnp.arange(tokensPerBlock - 1) % gridWidth,
                jnp.array([gridWidth // 2]),
            )
        )
        verticalPosition = jnp.concatenate(
            (
                jnp.arange(tokensPerBlock - 1) // gridWidth,
                jnp.array([gridWidth // 2]),
            )
        )
        positionIds = jnp.stack(
            (
                temporalPosition,
                jnp.tile(horizontalPosition, timeSteps) + temporalPosition,
                jnp.tile(verticalPosition, timeSteps) + temporalPosition,
            ),
            axis=-1,
        )
    else:
        temporalPosition = jnp.arange(sequenceLength)
        positionIds = jnp.stack((temporalPosition, temporalPosition, temporalPosition), axis=-1)
    return jnp.broadcast_to(positionIds[None], (batchSize, sequenceLength, 3)).astype("i4")
