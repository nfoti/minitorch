from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        if (len(out_shape) == len(in_shape)
            and np.all(out_shape == in_shape)
            and np.all(out_strides == in_strides)):
            # stride-aligned just a 1-1 for-loop
            for li in prange(out.shape[0]):
                out[li] = fn(in_storage[li])
        else:
            # Broadcast and index to get strides right
            # inline version of shape_broadcast using numpy array operations.
            # Assumes in_shape has fewer dimensions than out_shape
            ndims = out_shape.shape[0]
            pad_len = ndims - in_shape.shape[0]
            in_shape_pad = np.ones(ndims, dtype=np.int32)
            in_shape_pad[pad_len:] = out_shape
            bcshape = np.zeros(ndims)
            for i in range(ndims-1, -1, -1):
                in_dim = in_shape_pad[i]
                out_dim = out_shape[i]
                if in_dim == out_dim or in_dim == 1:
                    bcshape[i] = out_dim
                elif out_dim == 1:
                    bcshape[i] = in_dim

            for li in prange(out.shape[0]):
                idx = np.zeros(bcshape.shape[0], dtype=np.int32)
                in_idx = np.zeros(in_shape.shape[0], dtype=np.int32)
                to_index(li, bcshape, idx)
                broadcast_index(idx, bcshape, in_shape, in_idx)
                in_li = index_to_position(in_idx, in_strides)
                out[li] = fn(in_storage[in_li])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        
        if (len(a_shape) == len(b_shape)
            and np.all(a_shape == b_shape)
            and np.all(a_strides == b_strides)
            and np.all(out_strides == a_strides)):
            # stride-aligned just a 1-1 for-loop
            for li in prange(out.shape[0]):
                out[li] = fn(a_storage[li], b_storage[li])
            pass
        else:
            # We can just assume that a and b broadcast to out_shape as per the documentation
            bcshape = out_shape
            for li_out in prange(out.shape[0]):
                idx = np.zeros(bcshape.shape[0], dtype=np.int32)
                a_idx = np.zeros(a_shape.shape[0], dtype=np.int32)
                b_idx = np.zeros(b_shape.shape[0], dtype=np.int32)
                to_index(li_out, bcshape, idx)
                broadcast_index(idx, bcshape, a_shape, a_idx)
                broadcast_index(idx, bcshape, b_shape, b_idx)
                li_a = index_to_position(a_idx, a_strides)
                li_b = index_to_position(b_idx, b_strides)
                out[li_out] = fn(a_storage[li_a], b_storage[li_b])


    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:

        # Remember: out_shape is a_shape with out_shape[reduce_dim] == 1.
        for li in prange(out.shape[0]):
            idx = np.zeros_like(out_shape, dtype=np.int32)
            a_idx = np.zeros_like(a_shape, dtype=np.int32)

            to_index(li, out_shape, idx)
            for j in range(a_shape[reduce_dim]):
                a_idx[:] = idx[:]
                a_idx[reduce_dim] = j
                li_a = index_to_position(a_idx, a_strides)
                out[li] = fn(out[li], a_storage[li_a])

        return

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0


    # Specify only need to work for 3d tensors
    #for n:
    #  for i:
    #    for j:
    #      for k:
    #        out[n, i, j] += a[n, i, k] * b[n, k, j]


    # make outer loop parallel
    for li in prange(out.shape[0]):

        idx = np.zeros_like(out_shape, dtype=np.int32)
        to_index(li, out_shape, idx)

        a_idx = np.zeros_like(a_shape, dtype=np.int32)
        broadcast_index(idx, out_shape, a_shape, a_idx)
        b_idx = np.zeros_like(b_shape, dtype=np.int32)
        broadcast_index(idx, out_shape, b_shape, b_idx)

        for k in range(a_shape[-1]):
            a_idx[-1] = k
            b_idx[-2] = k
            a_li = index_to_position(a_idx, a_strides)
            b_li = index_to_position(b_idx, b_strides)
            out[li] += a_storage[a_li] * b_storage[b_li]

    return



tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
