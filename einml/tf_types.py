from __future__ import annotations as _annotations

import typing as _typing

from tensorflow.python.data.util.structure import NoneTensorSpec, NoneTensor
if _typing.TYPE_CHECKING:
    from tensorflow.python.types.core import GenericFunction
    from tensorflow.python.data.ops.dataset_ops import DatasetSpec
    from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
    from tensorflow.python.framework.ops import Tensor
    from tensorflow.python.framework.dtypes import DType
    from tensorflow.python.framework.tensor_shape import TensorShape
    from tensorflow.python.framework.tensor_spec import TensorSpec
    from tensorflow.python.types.core import TensorLike, Value as ValueTensor
else:
    from tensorflow.python.types.core import Value as ValueTensor
    from tensorflow import Tensor, DType, TensorShape, TensorSpec
    from tensorflow.types.experimental import GenericFunction, TensorLike
    from tensorflow.data import DatasetSpec
    from tensorflow.experimental import DynamicRaggedShape

AnyTensorSpec = _typing.Union[TensorSpec, DatasetSpec, NoneTensorSpec]

NestedTensor = _typing.Union[
    Tensor,
    dict[str, 'NestedTensor'],
    list['NestedTensor'],
    tuple['NestedTensor'],
]

NestedTensorSpec = _typing.Union[
    AnyTensorSpec,
    dict[str, 'NestedTensorSpec'],
    list['NestedTensorSpec'],
    tuple['NestedTensorSpec'],
]
