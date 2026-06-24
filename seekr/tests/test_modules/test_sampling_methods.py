from __future__ import annotations
"""
test_sampling_methods.py

Smoke tests for the Conventional sampling method's heat-ramp feature. The ramp
must run both for an integrator that exposes a "thermal_energy" global variable
(Langevin) and for one that does not (Verlet), where the temperature is only
applied via setVelocitiesToTemperature.
"""

import os

import pytest

import seekr.prepare as prepare
import seekr.run as run
import seekr.modules.base as base
import seekr.tests.create_seekr_input as create_seekr_input


@pytest.mark.parametrize("integrator_type, ensemble_cls", [
    ("langevin", base.NVT_ensemble),
    ("verlet", base.NVE_ensemble),
])
def test_heat_ramp_runs(tmp_path, integrator_type, ensemble_cls):
    """A Conventional heat ramp runs to completion for integrators with and
    without a 'thermal_energy' global variable."""
    seekr_input = create_seekr_input.create_toy_input(str(tmp_path))
    seekr_input.engine_settings_list[0].integrator_type = integrator_type
    stage1 = seekr_input.stages[0]
    stage1.ensemble = ensemble_cls()
    stage1.sampling_method.heat_ramp_temperatures = [100.0, 200.0, 300.0]
    stage1.sampling_method.heat_ramp_step_interval = 100
    stage1.completion_criteria.number_of_steps = 500
    old_dir = os.getcwd()
    try:
        model, model_xml_path = prepare.prepare(
            seekr_input, force_overwrite=False)
        model.directory = os.path.abspath(os.path.dirname(model_xml_path))
        run.run(model, "equilibration")
    finally:
        os.chdir(old_dir)
    assert os.path.isdir(model.directory)
