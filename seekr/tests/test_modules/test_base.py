from __future__ import annotations
"""
test_base.py

Test the modules/base.py module.
"""

import os
import json
import pytest
import random
import logging

import numpy as np
import mdtraj

import seekr.modules.base as base

TEST_DIRECTORY = os.path.dirname(__file__)

def test_noindent():
    obj = base.NoIndent([1, 2, 3])
    assert obj.value == [1, 2, 3]

def test_noindent_list_encoder():
    data = {"key": base.NoIndent([1, 2, 3])}
    encoded = json.dumps(data, cls=base.NoIndentListEncoder, indent=4)
    assert "[1,2,3]" in encoded

def test_configure_logging_file_only(tmp_path):
    log_file = tmp_path / "test.log"
    base.configure_logging(str(log_file), console=False)
    logging.getLogger("seekr.test").info("Test log message1")
    with open(log_file, "r") as f:
        assert "Test log message1" in f.read()

def test_configure_logging_file_and_stdout(tmp_path, capsys):
    log_file = tmp_path / "test.log"
    base.configure_logging(str(log_file))
    logging.getLogger("seekr.test").info("Test log message2")
    captured = capsys.readouterr()
    assert "Test log message2" in captured.out
    with open(log_file, "r") as f:
        assert "Test log message2" in f.read()

def test_configure_logging_stdout_only(capsys):
    base.configure_logging()
    logging.getLogger("seekr.test").info("Test log message3")
    captured = capsys.readouterr()
    assert "Test log message3" in captured.out

def test_order_files_numerically():
    string_list = ["/path/to/anchor0/output0_0", "/path/to/anchor0/output0_1",
                   "/path/to/anchor0/output0_2", "/path/to/anchor0/output1_0",
                   "/path/to/anchor0/output1_1", "/path/to/anchor0/output1_2",
                   "/path/to/anchor1/output0_0", "/path/to/anchor1/output0_1",
                   "/path/to/anchor1/output2_0", "/path/to/anchor1/output10_0"]
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    for item1, item2 in zip(ordered_list, desired_list):
        assert item1==item2
    
    scrambled_list = ["me2.345.pdb", "me1.234.pdb", "me-3.456.pdb"]
    correct_list = ["me-3.456.pdb", "me1.234.pdb", "me2.345.pdb"]
    ordered_list = base.order_files_numerically(scrambled_list, func=float)
    for item1, item2 in zip(ordered_list, correct_list):
        assert item1==item2
    
    string_list = ["mmvt.restart1.out", "mmvt.restart2.out"]
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    for item1, item2 in zip(ordered_list, desired_list):
        assert item1==item2
    
    string_list = ["/path/to/anchor_0.1/output0_0", 
                   "/path/to/anchor_0.2/output0_1"]
    
    desired_list = string_list[:]
    random.shuffle(string_list)
    ordered_list = base.order_files_numerically(string_list)
    
    return

def test_wrap_noindent_integers():
    data = {"key": [1, 2, 3], "nested": {"key2": [4.0, 5.0]}}
    wrapped = base.wrap_noindent_integers(data)
    assert isinstance(wrapped["key"], base.NoIndent)
    assert isinstance(wrapped["nested"]["key2"], base.NoIndent)

def test_make_tree(tmp_path):
    tree = {"dir1": {"subdir1": {}, "subdir2": {}}, "dir2": {}}
    base.make_tree(str(tmp_path), tree)
    assert (tmp_path / "dir1" / "subdir1").is_dir()
    assert (tmp_path / "dir1" / "subdir2").is_dir()
    assert (tmp_path / "dir2").is_dir()

def test_string_lower():
    assert base.string_lower("TEST") == "test"
    assert base.string_lower(None) is None
    with pytest.raises(TypeError):
        base.string_lower(123)

def test_get_pdb_num_frames(tmp_path):
    pdb_file = tmp_path / "test.pdb"
    with open(pdb_file, "w") as f:
        f.write("MODEL\n"+
                "HETATM    1 X0   UNK A   1       0.000  -7.000   0.000  1.00  0.00          X0\n"+
                "TER       2      UNK A   1\n"+
                "ENDMDL\n"+
                "MODEL\n"+
                "HETATM    1 X0   UNK A   1       0.000  -7.010   0.000  1.00  0.00          X0\n"+
                "TER       2      UNK A   1\n"+
                "ENDMDL\n")
    num_frames = base.get_pdb_num_frames(str(pdb_file))
    assert num_frames == 2

def test_glob_or(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.log").touch()
    (tmp_path / "file3.txt").touch()
    result = base.glob_or(["*.txt", "*.log"], str(tmp_path))
    assert len(result) == 3

def test_mdtraj_center_of_mass(tmp_path):
    pdb_file = tmp_path / "test.pdb"
    with open(pdb_file, "w") as f:
        f.write("MODEL\n"+
                "HETATM    1 X0   UNK A   1       0.000  -7.000   0.000  1.00  0.00          X0\n"+
                "TER       2      UNK A   1\n"+
                "ENDMDL\n"+
                "MODEL\n"+
                "HETATM    1 X0   UNK A   1       0.000  -7.010   0.000  1.00  0.00          X0\n"+
                "TER       2      UNK A   1\n"+
                "ENDMDL\n")
    traj = mdtraj.load(str(pdb_file))
    com = base.mdtraj_center_of_mass(traj)
    assert com.shape == (2, 3)
    assert np.allclose(com, np.array([[0.0, -0.7, 0.0], [0.0, -0.701, 0.0]]))
