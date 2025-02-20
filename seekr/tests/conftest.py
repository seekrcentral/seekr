"""
conftest.py

This file is used to define fixtures that are used in the tests.
"""

import os
import pytest
import copy

import seekr.prepare as prepare
import seekr.tests.create_seekr_input as create_seekr_input

TEST_DIRECTORY = os.path.dirname(__file__)

@pytest.fixture(scope="session")
def host_guest_mmvt_model_input_persistent(tmpdir_factory):
    """
    Create a model object that is persistent across the tests in this file.
    """
    rootdir = str(tmpdir_factory.mktemp("hostguest_mmvt"))
    host_guest_mmvt_model_input_persisent_obj \
        = create_seekr_input.create_host_guest_example_input(
            rootdir, bd=True)
    return host_guest_mmvt_model_input_persisent_obj

@pytest.fixture()
def host_guest_mmvt_model_input(host_guest_mmvt_model_input_persistent):
    """
    Create a copy of the model input that is not persistent. But this 
    at least doesn't require us to generate an entirely new model 
    input.
    """
    host_guest_mmvt_model_input_obj = copy.deepcopy(
        host_guest_mmvt_model_input_persistent)
    return host_guest_mmvt_model_input_obj

@pytest.fixture(scope="session")
def host_guest_mmvt_model_persistent(host_guest_mmvt_model_input_persistent):
    """
    Create a model object that is persistent across the tests in this file.
    """
    os.chdir(TEST_DIRECTORY)
    host_guest_mmvt_model_obj, model_xml_path \
        = prepare.prepare(host_guest_mmvt_model_input_persistent, 
                          force_overwrite=False)
    model_dir = os.path.dirname(model_xml_path)
    host_guest_mmvt_model_obj.directory = os.path.abspath(model_dir)
    return host_guest_mmvt_model_obj

@pytest.fixture
def host_guest_mmvt_model(host_guest_mmvt_model_persistent):
    """
    Create a copy of the model that is not persistent. But this at least
    doesn't require us to generate an entirely new model
    """
    # TODO: attempt to change the directory path to a different temporary
    #  directory to prevent clashes?
    host_guest_mmvt_model = copy.deepcopy(host_guest_mmvt_model_persistent)
    return host_guest_mmvt_model