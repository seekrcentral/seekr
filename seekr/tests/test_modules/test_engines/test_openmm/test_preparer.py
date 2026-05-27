from __future__ import annotations
"""
test_preparer.py

Unit tests for seekr.modules.engines.openmm.preparer.Openmm_settings_input.
"""

import pytest

import seekr.modules.engines.openmm.preparer as openmm_preparer
import seekr.modules.engines.preparer as engines_preparer
import seekr.modules.engines.structures as engines_structures


_TEMPERATURE = 298.15


def _make_input(integrator_type="langevin", platform_type="reference"):
    """Openmm_settings_input with minimal unpartitioned_settings."""
    s = openmm_preparer.Openmm_settings_input(
        integrator_type=integrator_type,
        platform_type=platform_type,
    )
    s.unpartitioned_settings = engines_preparer.MD_unpartitioned_settings_input()
    return s


# ---------------------------------------------------------------------------
# make_model_engine -- integrator branches
# ---------------------------------------------------------------------------

def test_make_model_engine_langevinMiddle():
    engine = _make_input("langevinMiddle").make_model_engine(_TEMPERATURE)
    assert isinstance(
        engine.integrator_settings,
        engines_structures.Langevin_middle_integrator_settings)
    assert engine.integrator_settings.target_temperature == _TEMPERATURE
    assert engine.integrator_settings.timestep == pytest.approx(0.002)


def test_make_model_engine_brownian():
    engine = _make_input("brownian").make_model_engine(_TEMPERATURE)
    assert isinstance(
        engine.integrator_settings,
        engines_structures.Brownian_integrator_settings)
    assert engine.integrator_settings.target_temperature == _TEMPERATURE
    assert engine.integrator_settings.timestep == pytest.approx(0.002)


def test_make_model_engine_verlet():
    engine = _make_input("verlet").make_model_engine(_TEMPERATURE)
    assert isinstance(
        engine.integrator_settings,
        engines_structures.Verlet_integrator_settings)
    assert engine.integrator_settings.timestep == pytest.approx(0.002)
    assert not hasattr(engine.integrator_settings, "target_temperature")


def test_make_model_engine_reference_platform():
    engine = _make_input("langevin", "reference").make_model_engine(_TEMPERATURE)
    assert engine.platform_settings is None


def test_make_model_engine_cuda_platform():
    engine = _make_input("langevin", "cuda").make_model_engine(_TEMPERATURE)
    assert isinstance(engine.platform_settings, engines_structures.Cuda_platform_settings)


# ---------------------------------------------------------------------------
# start_from_unpartitioned
# ---------------------------------------------------------------------------

def test_start_from_unpartitioned_no_positions():
    """positions=None -> unpartitioned stage not used."""
    assert _make_input().start_from_unpartitioned(using_toy=False) is False


def test_start_from_unpartitioned_with_positions_toy():
    """toy mode with positions set -> True without requiring a topology."""
    s = _make_input()
    s.unpartitioned_settings.positions = [[0.0, 0.0, 0.0]]
    assert s.start_from_unpartitioned(using_toy=True) is True


def test_start_from_unpartitioned_no_unpartitioned_settings():
    """unpartitioned_settings=None -> always False."""
    s = openmm_preparer.Openmm_settings_input(
        integrator_type="langevin",
        platform_type="reference",
    )
    s.unpartitioned_settings = None
    assert s.start_from_unpartitioned(using_toy=False) is False
