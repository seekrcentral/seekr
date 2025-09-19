"""

"""

import typing

import seekr.modules.scales.molecular_dynamics.structures as md_structures
import seekr.modules.scales.brownian_dynamics.structures as bd_structures

Scale = typing.Union[md_structures.Scale_md, bd_structures.Scale_bd]
Stage = typing.Union[md_structures.Stage_md, bd_structures.Stage_bd]

def register_structure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the structure hooks for all scales and their sampling methods.
    """
    md_structures.register_structure_hooks(converter)
    bd_structures.register_structure_hooks(converter)
    return

def register_unstructure_hooks(
        converter: typing.Any
        ) -> None:
    """
    Register the unstructure hooks for all scales and their sampling methods.
    """
    md_structures.register_unstructure_hooks(converter)
    bd_structures.register_unstructure_hooks(converter)
    return

def get_sampling_methods() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    return {
        "molecular_dynamics": md_structures.sampling_methods_md,
        "brownian_dynamics": bd_structures.sampling_methods_bd,
    }

def update_sampling_methods(
        new_sampling_methods: typing.Dict[str, typing.Dict[str, typing.Any]]
        ) -> None:
    """
    Update the sampling methods input dictionaries for all scales.
    """
    md_structures.sampling_methods_md.update(
        new_sampling_methods.get("molecular_dynamics", {}))
    bd_structures.sampling_methods_bd.update(
        new_sampling_methods.get("brownian_dynamics", {}))