from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
    ModelArchitecture,
    ModuleArchitecture,
    ModuleDefinition,
    WeightInfo,
)


class _ToyModuleArchitecture(ModuleArchitecture):
    def pre_weights(self, config):
        return [WeightInfo(name="embed.weight")]

    def post_weights(self, config):
        return [WeightInfo(name="head.weight")]

    def layer_weights(self, index, config):
        return [WeightInfo(name=f"layers.{index}.weight")]


def test_configured_architectures_resolve_transformers_type_namespace():
    config = PretrainedConfig(num_hidden_layers=2)
    module = _ToyModuleArchitecture()
    configured_module = ConfiguredModuleArchitecture(info=module, config=config)
    model = ModelArchitecture(
        modules={"default": ModuleDefinition(architecture=module)},
        architectures=["ToyModel"],
        model_type="toy",
    )
    configured_model = ConfiguredModelArchitecture(info=model, config=config)

    assert ConfiguredModuleArchitecture.__pydantic_complete__ is True
    assert ConfiguredModelArchitecture.__pydantic_complete__ is True
    assert configured_module.num_layers() == 2
    assert configured_model.get_module("default").num_layers() == 2
