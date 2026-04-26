from typing import List

import torch

from mergekit.common import ImmutableMap, ModelReference
from mergekit.merge_methods.base import TensorDictWrapper
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
from tests.test_graph import DummyTask


def _dummy_tensor_task(value: torch.Tensor) -> DummyTask:
    return DummyTask(
        result=value,
        dependencies=ImmutableMap(data={}),
        name="tensor",
    )


def test_easy_define_dynamic_task_recomputes_abstract_methods():
    @merge_method(name="test_easy_define_sum")
    def _sum_merge(tensors: List[torch.Tensor]) -> torch.Tensor:
        return sum(tensors)

    model_a = ModelReference(model="test/model-a")
    model_b = ModelReference(model="test/model-b")
    tensor_input = TensorDictWrapper(
        tensors=ImmutableMap(
            data={
                model_a: _dummy_tensor_task(torch.tensor([1.0])),
                model_b: _dummy_tensor_task(torch.tensor([2.0])),
            }
        )
    )

    task = REGISTERED_MERGE_METHODS["test_easy_define_sum"].make_task(
        output_weight=None,
        tensors=tensor_input,
        parameters=ImmutableMap(data={}),
        tensor_parameters=ImmutableMap(data={}),
        base_model=None,
    )

    assert task.__class__.__abstractmethods__ == frozenset()
    assert task.arguments() == {"tensors": tensor_input}
    result = task.execute(
        tensors={model_a: torch.tensor([1.0]), model_b: torch.tensor([2.0])}
    )
    torch.testing.assert_close(result, torch.tensor([3.0]))
