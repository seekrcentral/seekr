"""
test_data_structures.py

Test the modules/data_structures.py module.
"""

import seekr.modules.data_structures as data_structures

def test_model_get_md_timestep(
        host_guest_mmvt_model: data_structures.Seekr_model
        ) -> None:
    """
    Assure that the model.get_type() function is working properly
    """
    assert host_guest_mmvt_model.get_md_timestep() == 0.002
    return

# TODO: implement toy model test
def test_model_get_bulk_index(
        host_guest_mmvt_model: data_structures.Seekr_model
        ) -> None: #, toy_multi_model):
    assert host_guest_mmvt_model.get_bulk_index() == 13
    #assert toy_multi_model.get_bulk_index() == None
    return