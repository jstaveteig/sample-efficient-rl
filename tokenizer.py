import functools

from flax import nnx
import jax
import jax.numpy as jnp


class NearestNeighborTokenizer:
    def __init__(self, codebookSize=4096, patchSize=7, gridRow=9, gridCol=9, threshold=0.75):
        self.codebookSize = codebookSize
        self.patchSize = patchSize
        self.gridRow = gridRow
        self.gridCol = gridCol
        self.threshold = threshold

    def initCodebook(self):
        return jnp.zeros((self.codebookSize, self.patchSize, self.patchSize, 3)) - 1

    @staticmethod
    def updateCodebook(codebookState, patch, threshold, maxSize):
        codebook, currentSize = codebookState
        if threshold is None:
            matches = jnp.isclose(codebook, patch).all(axis=(-3, -2, -1))
            shouldUpdate = jnp.logical_and(~matches.any(), currentSize < maxSize)
        else:
            distances = jnp.square(codebook - patch).sum(axis=(-3, -2, -1))
            shouldUpdate = jnp.logical_and((distances > threshold).all(), currentSize < maxSize)
        codebook = jnp.where(shouldUpdate, codebook.at[currentSize].set(patch), codebook)
        currentSize += shouldUpdate
        return (codebook, currentSize), None

    @functools.partial(nnx.jit, static_argnums=(0,))
    def updateChunk(self, images, codebook, codebookSize):
        channels = images.shape[-1]
        patches = images.reshape(
            -1, self.gridRow, self.patchSize, self.gridCol, self.patchSize, channels
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            -1, self.patchSize, self.patchSize, channels
        )
        (codebook, codebookSize), unusedOutputs = nnx.scan(
            self.updateCodebook,
            in_axes=(nnx.transforms.iteration.Carry, 0, None, None),
        )((codebook, codebookSize), patches, self.threshold, self.codebookSize)
        del unusedOutputs
        return codebook, codebookSize

    def update(self, images, codebook, codebookSize):
        frames = images.reshape(-1, *images.shape[-3:])
        # sequential chunks preserve patch insertion and bound memory
        for frameStart in range(0, frames.shape[0], 16):
            codebook, codebookSize = self.updateChunk(
                frames[frameStart : frameStart + 16], codebook, codebookSize
            )
            jax.block_until_ready((codebook, codebookSize))
        return codebook, codebookSize

    @functools.partial(nnx.jit, static_argnums=(0,))
    def __call__(self, images, codebook):
        leadingShape = images.shape[:-3]
        channels = images.shape[-1]
        patches = images.reshape(
            -1, self.gridRow, self.patchSize, self.gridCol, self.patchSize, channels
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            -1, self.patchSize, self.patchSize, channels
        )
        distances = jnp.square(patches[:, None] - codebook[None]).sum(axis=(-3, -2, -1))
        tokenIds = jnp.argmin(distances, axis=-1)
        return tokenIds.reshape(*leadingShape, self.gridRow * self.gridCol)

    @functools.partial(nnx.jit, static_argnums=(0,))
    def decode(self, tokenIds, codebook):
        leadingShape = tokenIds.shape[:-1]
        tokenIds = tokenIds.reshape(-1, self.gridRow, self.gridCol)
        patches = jnp.clip(jnp.take(codebook, tokenIds, axis=0), 0.0, 1.0)
        images = patches.transpose(0, 1, 3, 2, 4, 5)
        return images.reshape(
            *leadingShape,
            self.gridRow * self.patchSize,
            self.gridCol * self.patchSize,
            images.shape[-1],
        )
