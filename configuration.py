from dataclasses import dataclass, field


@dataclass
class ActorCriticParams:
    eps: float = 0.2
    tdLossCoef: float = 2.0
    entLossCoef: float = 0.01


@dataclass
class ActorCriticConfig:
    params: ActorCriticParams = field(default_factory=ActorCriticParams)
    numRealEpochs: int = 4
    numRealMinibatches: int = 8
    numImaginationUpdates: int = 150
    numImaginationEpochs: int = 1
    gamma: float = 0.925
    ld: float = 0.625
    tgtDiscount: float = 0.95
    learningRate: float = 0.00045


@dataclass
class WorldModelParams:
    tokensPerBlock: int = 82
    maxBlocks: int = 20
    vocab_size: int = 4096
    n_positions: int = 1640
    n_embd: int = 128
    n_layer: int = 3
    n_head: int = 8
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    useAbsoluteEmbedding: bool = False
    useSpatioTemporal: bool = False


@dataclass
class WorldModelConfig:
    params: WorldModelParams = field(default_factory=WorldModelParams)
    numUpdates: int = 500
    numMinibatches: int = 3
    rewardLossCoef: float = 1.0
    terminationLossCoef: float = 1.0
    numDummy: int = 0
    distanceCoef: float = 1.0
    trashCost: float = 1.0
    sinkhornEpsilon: float = 1e-5
    useZeroNeighborDistance: bool = False
    learningRate: float = 0.001
    decodeStrategy: str = "original"


@dataclass
class TokenizerParams:
    codebookSize: int = 4096
    patchSize: int = 7
    gridRow: int = 9
    gridCol: int = 9
    threshold: float = 0.75


@dataclass
class TokenizerConfig:
    params: TokenizerParams = field(default_factory=TokenizerParams)
    numUpdates: int = 25


@dataclass
class WandBConfig:
    enable: bool = False
    projectName: str = "sample-efficient-rl"


@dataclass
class EnvConfig:
    envType: str = "craftax"
    envName: str = "Craftax-Classic-Pixels-v1"


@dataclass
class TrainConfig:
    seed: int = 0
    totalEnvInteractions: int = 1_000_000
    warmupInteractions: int = 200_000
    ckptPath: str | None = None
    ckptSteps: list[int] = field(default_factory=lambda: [50000, 500000, 1000000])
    replayBufferSize: int = 128_000
    batchSize: int = 48
    rolloutHorizon: int = 96
    wmRolloutHorizon: int = 20
    maxGradNorm: float = 0.5
    burnInHorizon: int = 5
    wandbConfig: WandBConfig = field(default_factory=WandBConfig)
    acConfig: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    wmConfig: WorldModelConfig = field(default_factory=WorldModelConfig)
    tokenConfig: TokenizerConfig = field(default_factory=TokenizerConfig)
    envConfig: EnvConfig = field(default_factory=EnvConfig)


def configurationFor(method, seed=0):
    if method not in {"ppo", "dedieu", "baseline", "itc"}:
        raise ValueError("unknown training method")
    config = TrainConfig(seed=seed)
    if method == "ppo":
        config.acConfig.numImaginationUpdates = 0
        config.wmConfig.numUpdates = 0
        config.tokenConfig.numUpdates = 0
    if method in {"baseline", "itc"}:
        config.warmupInteractions = 50_000
        config.acConfig.numImaginationUpdates = 300
    if method == "itc":
        config.wmConfig.params.useAbsoluteEmbedding = True
        config.wmConfig.params.useSpatioTemporal = True
        config.wmConfig.numDummy = 81
        config.wmConfig.distanceCoef = 0.6
        config.wmConfig.trashCost = 0.3
        config.wmConfig.decodeStrategy = "sinkhorn"
    return config
