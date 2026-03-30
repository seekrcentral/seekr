"""
modules/reporters/openmm_interface.py

Reporters that interface with OpenMM.
"""

import os
import sys
import json
import typing
import importlib

from attrs import define, field
import numpy as np
import openmm
import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.collective_variables.base as cv_base
import seekr.modules.structures as structures
import seekr.modules.reporters.structures as reporters_structures
import seekr.modules.engines.runner as engines_runner
import seekr.modules.sampling_methods.openmm_interface as sampl_meth_omm_int

class Base_reporter_openmm_interface(typing.Protocol):
    """
    A base class for a reporter using the OpenMM engine.
    """
    pass

@define
class Position_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A position reporter for SEEKR.
    """
    _reporter: reporters_structures.Position_reporter = field()
    _sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface | None \
        = field(default=None)
    def add_reporter_to_simulation(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add the reporter to the OpenMM simulation object.
        """
        simulation = kwargs["simulation"]
        filename = kwargs["filename"]
        toy_mode = kwargs["toy_mode"]
        if toy_mode:
            enforce_periodic_box = False
        else:
            enforce_periodic_box = self._reporter.enforce_periodic_box
        if self._reporter.extension == "dcd":
            openmm_reporter = openmm_app.DCDReporter(
                file=filename, reportInterval=self._reporter.interval, append=False,
                enforcePeriodicBox=enforce_periodic_box,)
                # TODO: implement this one day when available for the standard OpenMM version.
                #atomSubset=self._reporter.atom_subset)
        elif self._reporter.extension == "pdb":
            openmm_reporter = openmm_app.PDBReporter(
                file=filename, reportInterval=self._reporter.interval, append=False,
                enforcePeriodicBox=enforce_periodic_box,
                atomSubset=self._reporter.atom_subset)
        simulation.reporters.append(openmm_reporter)
        return
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self._reporter.get_reporter_extension()

    def add_sampling_method_interface(
            self,
            sampling_method_interface
            ) -> None:
        self._sampling_method_interface = sampling_method_interface
        return
    
@define
class State_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A state reporter for SEEKR.
    """
    _reporter: reporters_structures.State_reporter = field()
    _sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface | None \
        = field(default=None)
    def add_reporter_to_simulation(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add the reporter to the OpenMM simulation object.
        """
        simulation = kwargs["simulation"]
        if self._reporter.to_stdout:
            filename = sys.stdout
        else:
            filename = kwargs["filename"]
        openmm_reporter = openmm_app.StateDataReporter(
            file=filename, reportInterval=self._reporter.interval, append=False,
            step=self._reporter.step, time=self._reporter.time,
            potentialEnergy=self._reporter.potential_energy,
            kineticEnergy=self._reporter.kinetic_energy,
            totalEnergy=self._reporter.total_energy,
            temperature=self._reporter.temperature,
            volume=self._reporter.volume,
            density=self._reporter.density,
            speed=self._reporter.speed,
            elapsedTime=self._reporter.elapsed_time)
        simulation.reporters.append(openmm_reporter)
        return
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self._reporter.get_reporter_extension()
    
    def add_sampling_method_interface(
            self,
            sampling_method_interface
            ) -> None:
        self._sampling_method_interface = sampling_method_interface
        return

class CVValueReporter(object):
    """
    Custom reporter to write structures corresponding to certain CV values(ranges).
    """
    
    def __init__(
            self, 
            file: str, 
            checkInterval: int,
            cv_index: int, 
            cv_values: list[float],
            extension: str,
            sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface,
            enforcePeriodicBox: bool | None = None,
            swarm_id: int = 0,
            mode: str = "save_last_frame",
            json_filename: str | None = None,
            saveInterval: int | None = None
            ) -> None:
        if extension not in ("pdb", "dcd"):
            raise Exception(
                f"Unsupported extension {extension} for CV value reporter.")
        self._filepath = file
        self._checkInterval = checkInterval
        self._cv_index = cv_index
        assert len(cv_values) >= 2, \
            "At least two CV values must be provided for the CV value reporter."
        self._cv_values = cv_values
        n_bins = len(cv_values) - 1
        self._cv_values_indices = np.array([-1] * n_bins)
        self._extension = extension
        self._enforcePeriodicBox = enforcePeriodicBox
        self._swarm_id = swarm_id
        self._sampling_method_interface = sampling_method_interface
        assert mode in ["save_first_frame", "save_last_frame"] #, "save_closest_frame"]
        self._mode = mode
        if json_filename is None:
            # Use 'file' with .json extension in the same directory as the reporter file.
            base_filename = os.path.splitext(os.path.basename(file))[0]
            self._json_filename = os.path.join(os.path.dirname(os.path.abspath(file)), 
                                         base_filename + ".json")
        else:
            self._json_filename = json_filename
        if saveInterval is None:
            self._saveInterval = checkInterval
        else:
            assert saveInterval >= checkInterval, \
                "saveInterval must be >= checkInterval."
            assert saveInterval % checkInterval == 0, \
                "saveInterval must be a multiple of checkInterval."
            self._saveInterval = saveInterval
        # One buffered frame per bin; written in bin order at finalize().
        self._buffered_positions = [None] * n_bins
        self._buffered_box_vectors = [None] * n_bins
        self._topology = None
        self._dt = None

    def __del__(self):
        self.finalize()

    def describeNextReport(
            self, 
            simulation: openmm_app.Simulation):
        steps = self._checkInterval - simulation.currentStep % self._checkInterval
        return {'steps': steps, 'periodic': self._enforcePeriodicBox, 'include': ['positions']}

    def _write_json(self) -> None:
        full_dict = {}
        for i in range(len(self._cv_values) - 1):
            cv_value_dict = {}
            cv_value_dict["index"] = i
            cv_value_dict["range"] = (self._cv_values[i], self._cv_values[i+1])
            cv_value_dict["frame_index"] = int(self._cv_values_indices[i])
            full_dict[f"cv_value_{i}"] = cv_value_dict
        with open(self._json_filename, "w") as json_file:
            json.dump(full_dict, json_file, indent=4)
        return

    def report(
            self, 
            simulation: openmm_app.Simulation, 
            state: openmm.State
            ) -> None:
        # Capture topology and timestep on first call regardless of interval throttle.
        if self._topology is None:
            self._topology = simulation.topology
            self._dt = simulation.integrator.getStepSize()

        cv_values = self._sampling_method_interface.get_cv_values()
        if cv_values is None:
            #There is a problem - previous restart had no CV values provided
            cv_value = None
        else:
            cv_value = cv_values[self._cv_index]
        
        if cv_value is None:
            return

        positions = state.getPositions(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors()

        # Buffer the frame for the matching bin every checkInterval.
        for i in range(len(self._cv_values) - 1):
            if self._cv_values[i] <= cv_value < self._cv_values[i+1]:
                if self._mode == "save_last_frame" \
                        or (self._mode == "save_first_frame" \
                            and self._buffered_positions[i] is None):
                    self._buffered_positions[i] = positions
                    self._buffered_box_vectors[i] = box_vectors
                break
        
        # Flush buffers to disk every saveInterval.
        if simulation.currentStep % self._saveInterval == 0:
            self.finalize()

        return

    def finalize(self) -> None:
        """
        Write all buffered frames to the output file in ascending bin order,
        then emit the JSON index file. Re-entrant: safe to call at every
        saveInterval and from __del__. Frames are written in bin order so
        _cv_values_indices is always monotonically increasing over sampled bins.
        """
        if not hasattr(self, '_buffered_positions'):
            return

        sampled_indices = [i for i, p in enumerate(self._buffered_positions)
                           if p is not None]
        if not sampled_indices or self._topology is None:
            return

        dt = self._dt if self._dt is not None else 1.0
        frame_counter = 0
        file_mode = "wb" if self._extension == "dcd" else "w"
        with open(self._filepath, file_mode) as f:
            if self._extension == "dcd":
                dcd = openmm_app.DCDFile(
                    f, self._topology, dt=dt,
                    firstStep=0, interval=1, append=False)
                for i in sampled_indices:
                    dcd.writeModel(
                        self._buffered_positions[i],
                        periodicBoxVectors=self._buffered_box_vectors[i])
                    self._cv_values_indices[i] = frame_counter
                    frame_counter += 1
            elif self._extension == "pdb":
                openmm_app.PDBFile.writeHeader(self._topology, f)
                for i in sampled_indices:
                    # PDB MODEL records are 1-indexed; frame_index in JSON is 0-indexed.
                    openmm_app.PDBFile.writeModel(
                        self._topology, self._buffered_positions[i],
                        f, frame_counter + 1)
                    self._cv_values_indices[i] = frame_counter
                    frame_counter += 1
                openmm_app.PDBFile.writeFooter(self._topology, f)

        self._write_json()
        return

@define
class CV_value_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A CV value reporter for SEEKR.
    """
    _reporter: reporters_structures.CV_value_reporter = field()
    _cv_index: int = field()
    _swarm_id: int = field()
    _sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface | None \
        = field(default=None)

    def add_reporter_to_simulation(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add the reporter to the OpenMM simulation object.
        """
        simulation = kwargs["simulation"]
        filename = kwargs["filename"]
        toy_mode = kwargs["toy_mode"]
        if toy_mode:
            enforce_periodic_box = False
        else:
            enforce_periodic_box = self._reporter.enforce_periodic_box
        
        openmm_reporter = CVValueReporter(
            filename, self._reporter.interval, self._cv_index, 
            self._reporter.cv_values, self._reporter.extension,
            self._sampling_method_interface,
            self._reporter.enforce_periodic_box, 
            swarm_id=self._swarm_id, mode=self._reporter.mode,
            saveInterval=self._reporter.save_interval)
        simulation.reporters.append(openmm_reporter)
        return
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self._reporter.get_reporter_extension()
    
    def add_sampling_method_interface(
            self,
            sampling_method_interface
            ) -> None:
        self._sampling_method_interface = sampling_method_interface
        return

class AnchorReporter(object):
    """
    Custom reporter to write structures corresponding to anchors in a model.
    """
    
    def __init__(
            self, 
            file: str, 
            checkInterval: int,
            extension: str,
            sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface,
            enforcePeriodicBox: bool | None = None,
            swarm_id: int = 0,
            mode: str = "save_last_frame",
            json_filename: str | None = None,
            saveInterval: int | None = None
            ) -> None:
        if extension not in ("pdb", "dcd"):
            raise Exception(
                f"Unsupported extension {extension} for CV value reporter.")
        self._filepath = file
        self._checkInterval = checkInterval
        self._extension = extension
        self._enforcePeriodicBox = enforcePeriodicBox
        self._swarm_id = swarm_id
        self._sampling_method_interface = sampling_method_interface
        assert mode in ["save_first_frame", "save_last_frame"] #, "save_closest_frame"]
        self._mode = mode
        if json_filename is None:
            # Use 'file' with .json extension in the same directory as the reporter file.
            base_filename = os.path.splitext(os.path.basename(file))[0]
            self._json_filename \
                = os.path.join(os.path.dirname(os.path.abspath(file)), 
                                base_filename + ".json")
        else:
            self._json_filename = json_filename
        if saveInterval is None:
            self._saveInterval = checkInterval
        else:
            assert saveInterval >= checkInterval, \
                "saveInterval must be >= checkInterval."
            assert saveInterval % checkInterval == 0, \
                "saveInterval must be a multiple of checkInterval."
            self._saveInterval = saveInterval
        # One buffered frame per bin; written in bin order at finalize().
        n_anchors = len(self._sampling_method_interface._model.anchors)
        self._buffered_positions = [None] * n_anchors
        self._buffered_box_vectors = [None] * n_anchors
        self._topology = None
        self._dt = None

    def __del__(self):
        self.finalize()

    def describeNextReport(
            self, 
            simulation: openmm_app.Simulation):
        steps = self._checkInterval - simulation.currentStep % self._checkInterval
        return {'steps': steps, 'periodic': self._enforcePeriodicBox, 'include': ['positions']}

    def _write_json(self) -> None:
        full_dict = {}
        for i in range(len(self._cv_values) - 1):
            cv_value_dict = {}
            cv_value_dict["index"] = i
            cv_value_dict["range"] = (self._cv_values[i], self._cv_values[i+1])
            cv_value_dict["frame_index"] = int(self._cv_values_indices[i])
            full_dict[f"cv_value_{i}"] = cv_value_dict
        with open(self._json_filename, "w") as json_file:
            json.dump(full_dict, json_file, indent=4)
        return

    def report(
            self, 
            simulation: openmm_app.Simulation, 
            state: openmm.State
            ) -> None:
        # Capture topology and timestep on first call regardless of interval throttle.
        if self._topology is None:
            self._topology = simulation.topology
            self._dt = simulation.integrator.getStepSize()

        cv_values = self._sampling_method_interface.get_cv_values()
        anchor_index = self._sampling_method_interface\
            .find_current_anchor_index(cv_values)

        positions = state.getPositions(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors()

        # Buffer the frame for the matching bin every checkInterval.
        if self._mode == "save_last_frame" \
                or (self._mode == "save_first_frame" \
                    and self._buffered_positions[anchor_index] is None):
            self._buffered_positions[anchor_index] = positions
            self._buffered_box_vectors[anchor_index] = box_vectors
        
        # Flush buffers to disk every saveInterval.
        if simulation.currentStep % self._saveInterval == 0:
            self.finalize()

        return

    def finalize(self) -> None:
        """
        Write all buffered frames to the output file in ascending bin order,
        then emit the JSON index file. Re-entrant: safe to call at every
        saveInterval and from __del__. Frames are written in bin order so
        _cv_values_indices is always monotonically increasing over sampled bins.
        """
        if not hasattr(self, '_buffered_positions'):
            return

        sampled_indices = [i for i, p in enumerate(self._buffered_positions)
                           if p is not None]
        if not sampled_indices or self._topology is None:
            return

        dt = self._dt if self._dt is not None else 1.0
        frame_counter = 0
        file_mode = "wb" if self._extension == "dcd" else "w"
        with open(self._filepath, file_mode) as f:
            if self._extension == "dcd":
                dcd = openmm_app.DCDFile(
                    f, self._topology, dt=dt,
                    firstStep=0, interval=1, append=False)
                for i in sampled_indices:
                    dcd.writeModel(
                        self._buffered_positions[i],
                        periodicBoxVectors=self._buffered_box_vectors[i])
                    self._cv_values_indices[i] = frame_counter
                    frame_counter += 1
            elif self._extension == "pdb":
                openmm_app.PDBFile.writeHeader(self._topology, f)
                for i in sampled_indices:
                    # PDB MODEL records are 1-indexed; frame_index in JSON is 0-indexed.
                    openmm_app.PDBFile.writeModel(
                        self._topology, self._buffered_positions[i],
                        f, frame_counter + 1)
                    self._cv_values_indices[i] = frame_counter
                    frame_counter += 1
                openmm_app.PDBFile.writeFooter(self._topology, f)

        self._write_json()
        return

@define
class Anchor_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    An anchor reporter for SEEKR.
    """
    _reporter: reporters_structures.Anchor_reporter = field()
    _swarm_id: int = field()
    _sampling_method_interface: sampl_meth_omm_int.Base_md_openmm_interface | None \
        = field(default=None)

    def add_reporter_to_simulation(
            self, 
            **kwargs: typing.Any
            ) -> None:
        """
        Add the reporter to the OpenMM simulation object.
        """
        simulation = kwargs["simulation"]
        filename = kwargs["filename"]
        toy_mode = kwargs["toy_mode"]
        if toy_mode:
            enforce_periodic_box = False
        else:
            enforce_periodic_box = self._reporter.enforce_periodic_box
        
        openmm_reporter = AnchorReporter(
            """
            filename, self._reporter.interval, self._cv_index, 
            self._reporter.cv_values, self._reporter.extension,
            self._sampling_method_interface,
            self._reporter.enforce_periodic_box, 
            swarm_id=self._swarm_id, mode=self._reporter.mode,
            saveInterval=self._reporter.save_interval)
            """
        simulation.reporters.append(openmm_reporter)
        return
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self._reporter.get_reporter_extension()
    
    def add_sampling_method_interface(
            self,
            sampling_method_interface
            ) -> None:
        self._sampling_method_interface = sampling_method_interface
        return

#Reporter_openmm_interface = typing.Union[
#    DCD_reporter_openmm_interface,
#    PDB_reporter_openmm_interface,
#    State_reporter_openmm_interface
#]

def cv_index_from_name(
        cv_name: str,
        collective_variables: list[cv_base.Base_CV]
        ) -> int:
    """
    Get the index of a CV from its name.
    """
    cv_index = None
    for cv in collective_variables:
        if cv.name == cv_name:
            cv_index = cv.index
    assert cv_index is not None, \
        f"No CV found by the name of {cv_name}"
    return cv_index

def assign_reporter_interface(
        reporter: reporters_structures.Base_reporter, 
        md_engine: structures.MD_engine,
        plugin_list: list[str],
        collective_variables: list[cv_base.Base_CV],
        anchors: list[structures.Anchor],
        swarm_id: int = 0,
        ) -> Base_reporter_openmm_interface | None:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if reporter.type == "position":
        return Position_reporter_openmm_interface(reporter)
    elif reporter.type == "state":
        return State_reporter_openmm_interface(reporter)
    elif reporter.type == "cv_value":
        cv_index = cv_index_from_name(reporter.cv_name, collective_variables)
        interface = CV_value_reporter_openmm_interface(
            reporter, cv_index, swarm_id)
        return interface
    elif reporter.type == "anchor":
         return Anchor_reporter_openmm_interface(reporter)
    else:
        # Run through plugins
        for plugin_name in plugin_list:
            reporters_module_str = base.get_plugin_str(plugin_name, submodule_name="reporters")
            if reporters_module_str is not None:
                plugin_module_reporter_interface = importlib.import_module(
                    reporters_module_str+".openmm_interface")
                found, reporter_interface = plugin_module_reporter_interface.assign_reporter_interface(
                reporter)
            if found:
                return reporter_interface

    raise Exception("Reporter not found: " + reporter.type)