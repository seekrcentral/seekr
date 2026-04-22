"""
test_structures.py

Unit tests for seekr.modules.engines.openmm.structures.Openmm_md_engine.
"""

import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import seekr.modules.engines.structures as engines_structures
import seekr.modules.engines.openmm.structures as openmm_structures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    """Engine with Langevin integrator and CUDA platform."""
    engine = openmm_structures.Openmm_md_engine()
    engine.integrator_settings = engines_structures.Langevin_integrator_settings()
    engine.platform_settings = engines_structures.Cuda_platform_settings()
    return engine


def _make_engine_with_unpartitioned(names):
    """Engine with unpartitioned settings containing named selections."""
    engine = _make_engine()
    unpart = engines_structures.MD_unpartitioned_settings()
    unpart.selections = [
        engines_structures.Selection(name=n, atom_indices=[]) for n in names
    ]
    engine.unpartitioned_settings = unpart
    return engine


def _write_single_frame_pdb(path):
    """Write a minimal 1-atom PDB using OpenMM."""
    topology = openmm_app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("ALA", chain)
    topology.addAtom("CA", openmm_app.element.carbon, res)
    positions = [openmm.Vec3(0, 0, 0)] * unit.angstroms
    with open(path, "w") as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)


# ---------------------------------------------------------------------------
# Tests: get_pdb_object / get_pdb_num_frames
# ---------------------------------------------------------------------------

def test_get_pdb_object_returns_pdbfile(tmp_path):
    pdb_path = str(tmp_path / "test.pdb")
    _write_single_frame_pdb(pdb_path)
    engine = _make_engine()
    pdb_obj = engine.get_pdb_object(pdb_path)
    assert isinstance(pdb_obj, openmm_app.PDBFile)


def test_get_pdb_num_frames_single_frame(tmp_path):
    pdb_path = str(tmp_path / "test.pdb")
    _write_single_frame_pdb(pdb_path)
    engine = _make_engine()
    assert engine.get_pdb_num_frames(pdb_path) == 1


# ---------------------------------------------------------------------------
# Tests: get_random_seed / set_random_seed
# ---------------------------------------------------------------------------

def test_get_random_seed_default():
    engine = _make_engine()  # Langevin integrator with random_seed=0
    assert engine.get_random_seed() == 0


def test_get_random_seed_no_integrator():
    engine = _make_engine()
    engine.integrator_settings = None
    assert engine.get_random_seed() == 0


def test_get_random_seed_verlet_no_attr():
    """Verlet has no random_seed field -> falls through to 0."""
    engine = _make_engine()
    engine.integrator_settings = engines_structures.Verlet_integrator_settings()
    assert engine.get_random_seed() == 0


def test_set_random_seed():
    engine = _make_engine()
    engine.set_random_seed(42)
    assert engine.get_random_seed() == 42


# ---------------------------------------------------------------------------
# Tests: get_device_index / set_device_index
# ---------------------------------------------------------------------------

def test_get_device_index_default():
    engine = _make_engine()
    assert engine.get_device_index() == "0"


def test_get_device_index_no_platform():
    engine = _make_engine()
    engine.platform_settings = None
    assert engine.get_device_index() is None


def test_set_device_index():
    engine = _make_engine()
    engine.set_device_index("2")
    assert engine.get_device_index() == "2"


def test_set_device_index_no_platform_is_noop():
    engine = _make_engine()
    engine.platform_settings = None
    engine.set_device_index("2")  # must not raise


# ---------------------------------------------------------------------------
# Tests: get_selection_names
# ---------------------------------------------------------------------------

def test_get_selection_names():
    engine = _make_engine_with_unpartitioned(["alpha", "beta", "gamma"])
    assert engine.get_selection_names() == ["alpha", "beta", "gamma"]


def test_get_selection_names_empty():
    engine = _make_engine_with_unpartitioned([])
    assert engine.get_selection_names() == []
