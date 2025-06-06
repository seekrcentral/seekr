"""
modules/reporters/openmm_interface.py

Reporters that interface with OpenMM.
"""

import sys
import typing
import importlib

from attrs import define, field
import openmm.app as openmm_app

import seekr.modules.base as base
import seekr.modules.structures as structures
import seekr.modules.reporters.structures as reporters_structures

class Base_reporter_openmm_interface(typing.Protocol):
    """
    A base class for a reporter using the OpenMM engine.
    """
    pass

@define
class DCD_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A DCD reporter for SEEKR.
    """
    _reporter: reporters_structures.DCD_reporter = field()
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
        openmm_reporter = openmm_app.DCDReporter(
            file=filename, reportInterval=self._reporter.interval, append=False,
            enforcePeriodicBox=enforce_periodic_box,)
            # TODO: implement this one day when available for the standard OpenMM version.
            #atomSubset=self._reporter.atom_subset)
        simulation.reporters.append(openmm_reporter)
        return
    
    def get_reporter_extension(self) -> str:
        """
        Get the file extension for the reporter.
        """
        return self._reporter.get_reporter_extension()
    
@define
class PDB_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A PDB reporter for SEEKR.
    """
    _reporter: reporters_structures.PDB_reporter = field()
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
    
@define
class State_reporter_openmm_interface(Base_reporter_openmm_interface):
    """
    A state reporter for SEEKR.
    """
    _reporter: reporters_structures.State_reporter = field()
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
    
#Reporter_openmm_interface = typing.Union[
#    DCD_reporter_openmm_interface,
#    PDB_reporter_openmm_interface,
#    State_reporter_openmm_interface
#]

def assign_reporter_interface(
        reporter: reporters_structures.Base_reporter, 
        md_engine: structures.MD_engine,
        plugin_list: list[str]
        ) -> Base_reporter_openmm_interface | None:
    """
    Given a sampling_method and an md_engine, confirm that the OpenMM
    MD engine is being used, and return the corresponding interface
    """
    assert md_engine.type == "openmm", \
        "Only OpenMM engines can be used in the openmm_interface module."
    if reporter.type == "dcd":
        return DCD_reporter_openmm_interface(reporter)
    elif reporter.type == "pdb":
        return PDB_reporter_openmm_interface(reporter)
    elif reporter.type == "state":
        return State_reporter_openmm_interface(reporter)
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