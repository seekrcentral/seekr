from __future__ import annotations
"""
Tests for modules/engines/structures.py
"""

import os

import openmm
import openmm.app as openmm_app
import openmm.unit as unit

import seekr.modules.engines.structures as engines_structures


def _write_minimal_pdb(path: str) -> None:
    topology = openmm_app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("ALA", chain)
    topology.addAtom("CA", openmm_app.element.carbon, res)
    positions = [openmm.Vec3(0, 0, 0)] * unit.angstroms
    with open(path, "w") as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)


def _write_minimal_system_xml(path: str) -> None:
    system = openmm.System()
    system.addParticle(12.0 * unit.amu)
    with open(path, "w") as f:
        f.write(openmm.XmlSerializer.serialize(system))


def test_openmm_system_copy_files_autofills_topology_from_positions(tmp_path):
    system_path = str(tmp_path / "system.xml")
    positions_path = str(tmp_path / "positions.pdb")
    _write_minimal_system_xml(system_path)
    _write_minimal_pdb(positions_path)

    src = engines_structures.Openmm_system(
        system_filename=system_path, topology_filename="")
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    new = engines_structures.Openmm_system()

    src.copy_files(str(dest_dir), new, positions_pdb=positions_path)

    assert new.system_filename == "system.xml"
    assert new.topology_filename == engines_structures.TOPOLOGY_PDB_BASENAME
    assert os.path.isfile(dest_dir / "system.xml")
    assert os.path.isfile(dest_dir / engines_structures.TOPOLOGY_PDB_BASENAME)


def test_openmm_system_copy_files_explicit_topology_wins_over_positions(tmp_path):
    system_path = str(tmp_path / "system.xml")
    topology_path = str(tmp_path / "explicit_topology.pdb")
    positions_path = str(tmp_path / "positions.pdb")
    _write_minimal_system_xml(system_path)
    _write_minimal_pdb(topology_path)
    _write_minimal_pdb(positions_path)

    src = engines_structures.Openmm_system(
        system_filename=system_path, topology_filename=topology_path)
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    new = engines_structures.Openmm_system()

    src.copy_files(str(dest_dir), new, positions_pdb=positions_path)

    assert new.topology_filename == "explicit_topology.pdb"
    assert os.path.isfile(dest_dir / "explicit_topology.pdb")
    assert not os.path.isfile(dest_dir / engines_structures.TOPOLOGY_PDB_BASENAME)


def test_forcefield_copy_files_autofills_topology_from_positions(tmp_path):
    positions_path = str(tmp_path / "positions.pdb")
    _write_minimal_pdb(positions_path)

    src = engines_structures.Forcefield_parameters(
        built_in_forcefield_filenames=[],
        custom_forcefield_filenames=[],
        topology_filename="")
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    new = engines_structures.Forcefield_parameters()

    src.copy_files(str(dest_dir), new, positions_pdb=positions_path)

    assert new.topology_filename == engines_structures.TOPOLOGY_PDB_BASENAME
    assert os.path.isfile(dest_dir / engines_structures.TOPOLOGY_PDB_BASENAME)


def test_forcefield_copy_files_explicit_topology_wins_over_positions(tmp_path):
    topology_path = str(tmp_path / "explicit_topology.pdb")
    positions_path = str(tmp_path / "positions.pdb")
    _write_minimal_pdb(topology_path)
    _write_minimal_pdb(positions_path)

    src = engines_structures.Forcefield_parameters(
        built_in_forcefield_filenames=[],
        custom_forcefield_filenames=[],
        topology_filename=topology_path)
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    new = engines_structures.Forcefield_parameters()

    src.copy_files(str(dest_dir), new, positions_pdb=positions_path)

    assert new.topology_filename == "explicit_topology.pdb"
    assert os.path.isfile(dest_dir / "explicit_topology.pdb")
    assert not os.path.isfile(dest_dir / engines_structures.TOPOLOGY_PDB_BASENAME)


def test_openmm_system_copy_files_requires_topology_or_positions(tmp_path):
    system_path = str(tmp_path / "system.xml")
    _write_minimal_system_xml(system_path)
    src = engines_structures.Openmm_system(
        system_filename=system_path, topology_filename="")
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    new = engines_structures.Openmm_system()

    try:
        src.copy_files(str(dest_dir), new, positions_pdb=None)
        raised = False
    except AssertionError:
        raised = True
    assert raised
