"""
modules/engines/browndye/xml_structures.py

Any objects that are serialized to a Browndye XML file are
structured here.
"""

import typing
import xml.etree.ElementTree as ET
from xml.dom import minidom

from attrs import define, field, validators, Factory
from attrs.validators import and_, optional, instance_of, gt, ge, deep_iterable

import seekr.modules.engines.browndye.base as browndye_base

@define
class Ion:
    """
    Represent an Ion object in Browndye2 for generating APBS grids
    and Debye Length.
    
    Attributes:
    -----------
    radius : float
        The radius of the ion in units of Angstroms.
    charge : float
        The charge of the ion in units of 
    conc : float
        The concentration of the ion in solution in units of moles per
        liter.
    """
    
    radius: float = field(
        default=1.0, validator=[instance_of(float), gt(0)])
    charge: float = field(
        default=0.0, validator=[instance_of(float)])
    conc: float = field(
        default=0.0, validator=[instance_of(float), ge(0)])
    
    def serialize(
            self, 
            xmlIon: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlIon : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        assert self.radius >= 0.0, "Ion radius must be set"
        xmlIonRadius = ET.SubElement(xmlIon, 'radius')
        xmlIonRadius.text = str(self.radius)
        xmlIonCharge = ET.SubElement(xmlIon, 'charge')
        xmlIonCharge.text = str(self.charge)
        assert self.conc >= 0.0, "Ion concentration must be set"
        xmlIonConc = ET.SubElement(xmlIon, 'conc')
        xmlIonConc.text = str(self.conc)
        return

@define
class Solvent:
    """
    Parameters to represent the solvent within the BD simulation.
    
    Attributes:
    -----------
    debye_length : float
        The Debye length is a distance inversely related to the 
        strength and concentration of ions in solution.
    dielectric : float, Default 78.0
        The dielectric of solvent, relative to vacuum permittivity.
    relative_viscosity : float, Default 1.0
        Relative to water viscosity.
    kT : float
        Thermal energy relative to Boltzmann's constant times 298 K.
    desolvation_parameter : float, Default 1.0
        Factor that multiplies desolvation energy.
    ions : list
        A list of Ion() objects for APBS input
    """
    
    debye_length: float = field(
        default=9e9, validator=[instance_of(float), gt(0)])
    dielectric: float = field(
        default=78.0, validator=[instance_of(float), gt(0)])
    relative_viscosity: float = field(
        default=1.0, validator=[instance_of(float), gt(0)])
    kT: float = field(
        default=1.0, validator=[instance_of(float), gt(0)])
    desolvation_parameter: float = field(
        default=1.0, validator=[instance_of(float), ge(0)])
    ions: typing.List[Ion] = field(default=Factory(list))
    
    def serialize(
            self, 
            xmlSolvent: ET.Element, 
            make_apbs_mode: bool = True
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlSolvent : ElementTree.SubElement
            All sub elements get added to this root.
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        if not make_apbs_mode:
            assert self.debye_length > 0.0, "Solvent Debye length must be assigned."
            xmlSolventDebye = ET.SubElement(xmlSolvent, 'debye_length')
            xmlSolventDebye.text = str(self.debye_length)
        assert self.dielectric > 0.0
        xmlSolventDielectric = ET.SubElement(xmlSolvent, 'dielectric')
        xmlSolventDielectric.text = str(self.dielectric)
        assert self.relative_viscosity > 0.0
        xmlSolventRelVisc = ET.SubElement(xmlSolvent, 'relative_viscosity')
        xmlSolventRelVisc.text = str(self.relative_viscosity)
        assert self.kT > 0.0
        xmlSolventKT = ET.SubElement(xmlSolvent, 'kT')
        xmlSolventKT.text = str(self.kT)
        assert self.desolvation_parameter >= 0.0
        xmlSolventDesolv = ET.SubElement(xmlSolvent, 'desolvation_parameter')
        xmlSolventDesolv.text = str(self.desolvation_parameter)
        if make_apbs_mode:
            xmlSolventIons = ET.SubElement(xmlSolvent, 'ions')
            for ion in self.ions:
                xmlIon = ET.SubElement(xmlSolventIons, 'ion')
                ion.serialize(xmlIon)
        
        return

@define
class Time_step_tolerances:
    """
    Parameters to represent the limitations on the time step sizes
    within the BD simulation.
    
    Attributes:
    -----------
    force : float or None, Default None
        Dimensionless parameter that governs the splitting of the 
        adaptive time step in response to changes in force and torque.
    reaction : float or None, Default None
        Dimensionless parameter that governs the splitting of the 
        adaptive time step in response to changes in the reaction
        coordinate near a reaction.
    minimum_core_dt : float, default 0.0
        For a system with only rigid cores and frozen chains, this is
        the smallest possible time step size in picoseconds.
    minimum_chain_dt : float or None, default None
        For a system with unfrozen chains, this is the smallest 
        possible time step size in picoseconds.
    minimum_core_reaction_dt : float, default 0.0
        If a reaction coordinate is small, then the minimum step size
        given by minimum_core_dt can be overridden by the value given
        here.
    minimum_chain_reaction_dt : float or None, default None
        This has the same relation to minimum_chain_dt.
    """
    
    force: float | None = field(
        default=None, validator=optional(and_(instance_of(float), gt(0))))
    reaction: float | None = field(
        default=None, validator=optional(and_(instance_of(float), gt(0))))
    minimum_core_dt: float = field(
        default=0.0, validator=[instance_of(float), ge(0)])
    minimum_chain_dt: float | None = field(
        default=None, validator=optional(and_(instance_of(float), ge(0))))
    minimum_core_reaction_dt: float = field(
        default=0.0, validator=[instance_of(float), ge(0)])
    minimum_chain_reaction_dt: float | None = field(
        default=None, validator=optional(and_(instance_of(float), ge(0))))

    def serialize(
            self, 
            xmlTime: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlTime : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        if self.force is not None:
            assert self.force > 0.0
            xmlTimeForce = ET.SubElement(xmlTime, "force")
            xmlTimeForce.text = str(self.force)
        if self.reaction is not None:
            assert self.reaction > 0.0
            xmlTimeReaction = ET.SubElement(xmlTime, "reaction")
            xmlTimeReaction.text = str(self.reaction)
        assert self.minimum_core_dt >= 0.0
        xmlTimeMinCoreDt = ET.SubElement(xmlTime, "minimum_core_dt")
        xmlTimeMinCoreDt.text = str(self.minimum_core_dt)
        if self.minimum_chain_dt is not None:
            assert self.minimum_chain_dt >= 0.0
            xmlTimeMinChainDt = ET.SubElement(xmlTime, "minimum_chain_dt")
            xmlTimeMinChainDt.text = str(self.minimum_chain_dt)
        assert self.minimum_core_reaction_dt >= 0.0
        xmlTimeMinCoreRxnDt = ET.SubElement(xmlTime, "minimum_core_reaction_dt")
        xmlTimeMinCoreRxnDt.text = str(self.minimum_core_reaction_dt)
        if self.minimum_chain_reaction_dt is not None:
            assert self.minimum_chain_reaction_dt >= 0.0
            xmlTimeMinChainRxnDt = ET.SubElement(
                xmlTime, "minimum_chain_reaction_dt")
            xmlTimeMinChainRxnDt.text = str(self.minimum_chain_reaction_dt)
        return

@define
class Electric_field:
    """
    In Browndye2, Electric_field tags define the various electric
    electric forces applied to a molecule
    
    Attributes:
    -----------
    grid_list : list, Default []
        A list of DX file names output from APBS.
    multipole_field : str or None, Default None
        A file of multipole extension outside grids.
    """
    
    grid_list: typing.List[str] = field(
        default=Factory(list), validator=[deep_iterable(
            member_validator=instance_of(str),
            iterable_validator=instance_of(list),)])
    multipole_field: str | None = field(
        default=None, validator=optional(instance_of(str)))

    def serialize(
            self, 
            xmlE: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlE : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        for grid in self.grid_list:
            xmlGrid_i = ET.SubElement(xmlE, 'grid')
            xmlGrid_i.text = grid
        if self.multipole_field is not None:
            assert self.multipole_field
            xmlEMultipole = ET.SubElement(xmlE, "multipole_field")
            xmlEMultipole.text = self.multipole_field
        return

@define
class Link:
    """
    In Browndye2, chains are connected to cores - defining the Link.
    
    Attributes:
    -----------
    core_name : str
        The name of the Core to link.
    core_residue : int
        The residue index within the Core to link to.
    chain_residue : int
        The residue index within the chain to link to.
    """
    
    core_name: str = field(
        default="", validator=instance_of(str))
    core_residue: int = field(
        default=0, validator=[instance_of(int), ge(0)])
    chain_residue: int = field(
        default=0, validator=[instance_of(int), ge(0)])
    def serialize(
            self, 
            xmlLink: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlLink : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        assert self.core_name, "A name for the core must be provided"
        xmlLinkCoreName = ET.SubElement(xmlLink, "core_name")
        xmlLinkCoreName.text = self.core_name
        assert self.core_residue >= 0
        xmlLinkCoreResid = ET.SubElement(xmlLink, "core_residue")
        xmlLinkCoreResid.text = str(self.core_residue)
        assert self.chain_residue >= 0
        xmlLinkChainResid = ET.SubElement(xmlLink, "chain_residue")
        xmlLinkChainResid.text = str(self.chain_residue)
        return

@define
class Chain:
    """
    In Browndye2, chains are flexible assemblies of spheres that can
    interact with each other and the cores with user-specified force
    fields.
    
    Attributes:
    -----------
    name : str
        The name of this Chain.
    atoms : str
        The filename of the atoms in this chain. When using the
        COFFDROP model, the atoms files for cores and chains
        must contain the coarse-grained "beads".
    link_list : list
        A list of Links, or connections between the chain and core. The
        name of the core, the number of the linking residue, and the
        number of the chain's linking residue are specified.
    """
    
    name: str = field(default="", validator=instance_of(str))
    atoms: str = field(default="", validator=instance_of(str))
    link_list: typing.List[Link] = field(default=Factory(list))

    def serialize(
            self, 
            xmlChain: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlChain : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        assert self.name, "A name for this chain must be provided"
        xmlChainName = ET.SubElement(xmlChain, "name")
        xmlChainName.text = self.name
        assert self.atoms, "An atoms file must be provided for this Chain"
        xmlChainAtoms = ET.SubElement(xmlChain, "atoms")
        xmlChainAtoms.text = self.atoms
        for link in self.link_list:
            xmlLink_i = ET.SubElement(xmlChain, 'link')
            link.serialize(xmlLink_i)
        return

@define
class Core:
    """
    In Browndye2, cores are large rigid bodies composed of many smaller
    atoms.
    
    Attributes:
    -----------
    name : str
        The name to identify this core.
    atoms : str
        The PQRXML file of the atoms of this core, created by the 
        program pqr2xml.
    all_in_surface : str or None, Default "false"
        If this is true, then all of the atoms of the core are
        included, not just a shell made of the surface atoms.
    electric_field : Electric_field() or None
        Object containing APBS grids, desolvation grids, and other
        information pertaining to electrical forces.
    desolvation_field : str or None
        This block describes the desolvation field; it is just like the
        electric-field block but does not use the multipole field.
    eff_charges : str or None
        A file with effective charges and lumping information; output
        of lumped_charges.
    eff_charges_squared : str or None
        File like eff_charges above, but with the charges squared.
    copy : CoreCopy() or None
        One can specify a copy of a previously defined core.
    is_protein : str or None, Default "false"
        If this is true, then it affects how the effective charges are
        generated.
    dielectric : float, default 4.0
        The interior dielectric of the core.
    """
    
    name: str = field(default="", validator=instance_of(str))
    atoms: str = field(default="", validator=instance_of(str))
    all_in_surface: str | None = field(default="false")
    electric_field: Electric_field | None = field(
        default=None, validator=optional(instance_of(Electric_field)))
    desolvation_field: str | None = field(
        default=None,validator=optional(instance_of(str)))
    eff_charges: str | None = field(
        default=None, validator=optional(instance_of(str)))
    eff_charges_squared: str | None = field(
        default=None, validator=optional(instance_of(str)))
    copy: typing.Union["CoreCopy", None] = field(default=None)
    is_protein: str | None = field(default="false")
    dielectric: float = field(
        default=4.0, validator=[instance_of(float), gt(0)])
    grid_spacing: float = field(
        default=0.5, validator=[instance_of(float), gt(0)])
    
    def serialize(
            self, 
            xmlCore: ET.Element, 
            make_apbs_mode: bool = True
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlCore : ElementTree.SubElement
            All sub elements get added to this root.
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        assert self.name, "A name must be assigned to a Core"
        xmlCoreName = ET.SubElement(xmlCore, "name")
        xmlCoreName.text = self.name
        assert self.atoms, "An atoms XML file must be provided for this Core"
        xmlCoreAtoms = ET.SubElement(xmlCore, "atoms")
        xmlCoreAtoms.text = self.atoms
        if self.all_in_surface is not None:
            assert self.all_in_surface.lower() in ["true","false"]
            xmlCoreAllInSurf = ET.SubElement(xmlCore, "all_in_surface")
            xmlCoreAllInSurf.text = self.all_in_surface.lower()
        if not make_apbs_mode:
            if self.electric_field is not None:
                xmlCoreElecField = ET.SubElement(xmlCore, 'electric_field')
                self.electric_field.serialize(xmlCoreElecField)
        if self.desolvation_field is not None:
            assert self.desolvation_field
            xmlEDesolv = ET.SubElement(xmlCore, "desolvation_field")
            xmlEDesolv.text = self.desolvation_field
        if self.eff_charges is not None:
            assert self.eff_charges
            xmlEEff = ET.SubElement(xmlCore, "eff_charges")
            xmlEEff.text = self.eff_charges
        if self.eff_charges_squared is not None:
            assert self.eff_charges_squared
            xmlEEff2 = ET.SubElement(xmlCore, "eff_charges_squared")
            xmlEEff2.text = self.eff_charges_squared
        if self.copy is not None:
            xmlECopy = ET.SubElement(xmlCore, 'copy')
            self.copy.serialize(xmlECopy)
        if self.is_protein is not None:
            assert self.is_protein.lower() in ["true","false"]
            xmlCoreIsProt = ET.SubElement(xmlCore, "is_protein")
            xmlCoreIsProt.text = self.is_protein.lower()
        assert self.dielectric > 0.0
        xmlCoreDielec = ET.SubElement(xmlCore, "dielectric")
        xmlCoreDielec.text = str(self.dielectric)
        if make_apbs_mode:
            assert self.grid_spacing > 0.0
            xmlCoreGridSpacing = ET.SubElement(xmlCore, "grid_spacing")
            xmlCoreGridSpacing.text = str(self.grid_spacing)
        return

@define
class CoreCopy:
    """
    In Browndye2, cores may have all information copied to another
    core with a different translational/rotational position.
    
    Attributes:
    -----------
    parent : str
        The name of the core to copy from
    translation : list
        A list of 3 floats that describes the new translation.
    rotation : list
        A list of 9 floats that describes the new rotation.
    """
    
    parent: str = field(default="", validator=[instance_of(str)])
    translation: typing.List[float] = field(
        default=Factory(list), validator=[validators.deep_iterable(
            member_validator=instance_of(float),
            iterable_validator=instance_of(list),)])
    rotation: typing.List[float] = field(
        default=Factory(list), validator=[validators.deep_iterable(
            member_validator=instance_of(float),
            iterable_validator=instance_of(list),)])

    def serialize(
            self, 
            xmlCoreCopy: ET.Element
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlCoreCopy : ElementTree.SubElement
            All sub elements get added to this root.
        """
        assert self.parent, "A name must be assigned to a CoreCopy"
        xmlCoreCpName = ET.SubElement(xmlCoreCopy, "parent")
        xmlCoreCpName.text = self.parent
        assert len(self.translation) == 3
        xmlCoreCpTrans = ET.SubElement(xmlCoreCopy, "translation")
        xmlCoreCpTrans.text = " ".join([str(i) for i in self.translation])
        assert len(self.translation) == 9
        xmlCoreCpRot = ET.SubElement(xmlCoreCopy, "rotation")
        xmlCoreCpRot.text = " ".join([str(i) for i in self.rotation])
        return

@define
class Group:
    """
    In Browndye2, Cores and Chains can be assembled into groups. With
    two groups present, the second-order rate constant of the encounter
    of the groups may be computed.
    
    Cores may be connected by chains within a group. At this time, 
    however, SEEKR only supports a single Core in a group.
    
    Attributes:
    -----------
    name : str
        The name of this group
    core_list : list, Default []
        A list of Core() objects for this group
    chain_list : list, Default []
        A list of Chain() objects for this group
    """
    
    name: str = field(default="", validator=[instance_of(str)])
    core_list: typing.List[Core] = field(default=Factory(list))
    chain_list: typing.List[Chain] = field(default=Factory(list))
    
    def serialize(
            self, 
            xmlGroup: ET.Element, 
            make_apbs_mode: bool = True
            ) -> None:
        """
        Serialize this object to XML
        
        Parameters:
        -----------
        xmlCoreCopy : ElementTree.SubElement
            All sub elements get added to this root.
        """
        
        assert self.name, "A name must be assigned to a Group"
        xmlGroupName = ET.SubElement(xmlGroup, "name")
        xmlGroupName.text = self.name
        for core in self.core_list:
            xmlCore_i = ET.SubElement(xmlGroup, 'core')
            core.serialize(xmlCore_i, make_apbs_mode)
        for chain in self.chain_list:
            xmlChain_i = ET.SubElement(xmlGroup, 'chain')
            chain.serialize(xmlChain_i)
        return

@define
class System:
    """
    A Browndye2 system description.
    
    Attributes:
    -----------
    force_field : str or None
        Accepts "molecular_mechanics" or "spline" as arguments to
        select the type of forcefield.
    parameters : str or None
        Parameters file for the force field. Optional for 
        "molecular_mechanics" with no chains.
    start_at_site : str or None
        if true, then all the molecular components are started 
        according to their atom input files; otherwise, the 
        second group is started on the b-sphere and the second-
        order rate constant is computed.
    b_radius : float or None
        This can be automatically computed, but if a value is given
        it overrides the automatic value.
    reaction_file : str
        Describes the reaction pathways and criteria.
    hydrodynamic_interactions : str or None
        Include hydrodynamic interactions.
    n_steps_between_hi_updates : int or None
        Number of steps between updates to HI.
    density_field : str or None
        DX file containing an external density from microscopy or
        tomography.
    atom_weights : str or None
        File of atomic weights for each atom name in the atoms file.
    solvent : Solvent() or None
        Solvent object with parameters.
    time_step_tolerances : Time_step_tolerances()
        Time_step_tolerances object with parameters.
    group_list : list
        A list of Group() objects for the system.
    """
    
    force_field: str | None = field(
        default=None, validator=optional(instance_of(str)))
    parameters: str | None = field(
        default=None, validator=optional(instance_of(str)))
    start_at_site: str | None = field(
        default="false") #, validator=optional(instance_of(str)))
    b_radius: float | None = field(
        default=None, validator=optional(instance_of(float)))
    reaction_file: str = field(
        default="", validator=instance_of(str))
    hydrodynamic_interactions: str | None = field(
        default="true") #, validator=optional(instance_of(str)))
    n_steps_between_hi_updates: int | None = field(
        default=None, validator=optional(and_(instance_of(int), gt(0))))
    density_field: str | None = field(
        default=None, validator=optional(instance_of(str)))
    atom_weights: str | None = field(
        default=None, validator=optional(instance_of(str)))
    solvent: Solvent = field(default=Factory(Solvent))
    time_step_tolerances: Time_step_tolerances = field(
        default=Factory(Time_step_tolerances))
    group_list: typing.List[Group] = field(default=Factory(list))
    
    def serialize(
            self, 
            xmlSys: ET.Element, 
            make_apbs_mode: bool = True
            ) -> None:
        """
        This function needs to be defined explicitly because the 
        serializer will not format this XML block the way that 
        Browndye2 will accept as input.
        
        Parameters:
        -----------
        xmlChain : ElementTree.SubElement
            All sub elements get added to this root.
        
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        if self.force_field is not None:
            xmlSysFF = ET.SubElement(xmlSys, "force_field")
            xmlSysFF.text = self.force_field
        if self.parameters is not None:
            xmlSysParam = ET.SubElement(xmlSys, "parameters")
            xmlSysParam.text = self.parameters
        if self.start_at_site is not None:
            assert self.start_at_site.lower() in ["true","false"]
            xmlSysStart = ET.SubElement(xmlSys, "start_at_site")
            xmlSysStart.text = self.start_at_site.lower()
        if self.b_radius is not None:
            xmlSysBRad = ET.SubElement(xmlSys, "b_radius")
            xmlSysBRad.text = str(self.b_radius)
        assert self.reaction_file
        xmlSysRxnFile = ET.SubElement(xmlSys, "reaction_file")
        xmlSysRxnFile.text = self.reaction_file
        if self.hydrodynamic_interactions is not None:
            assert self.hydrodynamic_interactions.lower() in ["true","false"]
            xmlSysHydro = ET.SubElement(xmlSys, "hydrodynamic_interactions")
            xmlSysHydro.text = self.hydrodynamic_interactions.lower()
        if self.n_steps_between_hi_updates is not None:
            assert self.n_steps_between_hi_updates > 0
            xmlSysNStepsHI = ET.SubElement(xmlSys, "n_steps_between_hi_updates")
            xmlSysNStepsHI.text = str(self.n_steps_between_hi_updates)
        if self.density_field is not None:
            xmlSysDens = ET.SubElement(xmlSys, "density_field")
            xmlSysDens.text = self.density_field
        if self.atom_weights is not None:
            xmlSysWeights = ET.SubElement(xmlSys, "atom_weights")
            xmlSysWeights.text = self.atom_weights
        xmlSysSolv = ET.SubElement(xmlSys, 'solvent')
        self.solvent.serialize(xmlSysSolv, make_apbs_mode)
        xmlSysTime = ET.SubElement(xmlSys, 'time_step_tolerances')
        self.time_step_tolerances.serialize(xmlSysTime)
        for group in self.group_list:
            xmlGroup_i = ET.SubElement(xmlSys, 'group')
            group.serialize(xmlGroup_i, make_apbs_mode)
        return

@define
class Root:
    """
    An entire Browndye2 Input XML object.
    
    Attributes:
    -----------
    n_threads : int or None
        The number of threads used.
    seed : int
        Used to seed the random number generator
    output : str
        Name of file where output is sent (not trajectories).
    n_trajectories : int
        Total number of trajectories to be run.
    n_trajectories_per_output : int or None
        Number of time steps between updates to the output file.
    max_n_steps : int
        Number of time steps taken in a trajectory before giving up
        and declaring that it is "stuck".
    trajectory_file : str or None
        Prefix of files used for outputting of trajectories.
    n_steps_per_output : int or None
        Number of time steps between output to trajectory file.
    min_rxn_dist_file : str or None
        File to output minimum reaction coordinate for each trajectory.
    system : System()
        Information about the physical system and its motions are
        described inside this section.
    """
    
    n_threads: int = field(
        default=1, validator=[instance_of(int), gt(0)])
    seed: int = field(
        default=11111113, validator=[instance_of(int)])
    output: str = field(
        default="results1.xml", validator=[instance_of(str)])
    n_trajectories: int = field(
        default=100, validator=[instance_of(int), gt(0)])
    n_trajectories_per_output: int | None = field(
        default=None, validator=optional(and_( instance_of(int), gt(0))))
    max_n_steps: int = field(
        default=1000000, validator=[instance_of(int), gt(0)])
    trajectory_file: str | None = field(
        default=browndye_base.BROWNDYE_TRAJ_PREFIX) #, validator=optional(instance_of(str)))
    n_steps_per_output: int | None = field(
        default=100000) #, validator=optional(gt(0)))
    min_rxn_dist_file: str | None = field(
        default=None, validator=optional(instance_of(str)))
    system: System = field(default=Factory(System))
    
    def serialize(
            self, 
            make_apbs_mode: bool = True
            ) -> ET.Element:
        """
        This function needs to be defined explicitly because the 
        serializer will not format this XML block the way that 
        Browndye2 will accept as input.
        
        Parameters:
        -----------
        xmlChain : ElementTree.SubElement
            All sub elements get added to this root.
        
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        xmlRoot = ET.Element('root')
        if self.n_threads is not None:
            assert self.n_threads > 0
            xmlNThreads = ET.SubElement(xmlRoot, "n_threads")
            xmlNThreads.text = str(self.n_threads)
        xmlSeed = ET.SubElement(xmlRoot, "seed")
        xmlSeed.text = str(self.seed)
        xmlOutput = ET.SubElement(xmlRoot, "output")
        xmlOutput.text = self.output
        assert self.n_trajectories > 0, "n_trajectories must be set."
        xmlNTraj = ET.SubElement(xmlRoot, "n_trajectories")
        xmlNTraj.text = str(self.n_trajectories)
        if self.n_trajectories_per_output is not None:
            assert self.n_trajectories_per_output > 0
            xmlNTrajPerOut = ET.SubElement(xmlRoot, "n_trajectories_per_output")
            xmlNTrajPerOut.text = str(self.n_trajectories_per_output)
        assert self.max_n_steps > 0
        xmlMaxN = ET.SubElement(xmlRoot, "max_n_steps")
        xmlMaxN.text = str(self.max_n_steps)
        if self.trajectory_file is not None:
            assert self.trajectory_file
            xmlTrajFile = ET.SubElement(xmlRoot, "trajectory_file")
            xmlTrajFile.text = str(self.trajectory_file)
        if self.n_steps_per_output is not None:
            assert self.n_steps_per_output > 0
            xmlNStepsPerOut = ET.SubElement(xmlRoot, "n_steps_per_output")
            xmlNStepsPerOut.text = str(self.n_steps_per_output)
        if self.min_rxn_dist_file is not None:
            assert self.min_rxn_dist_file
            xmlMinRxnFile = ET.SubElement(xmlRoot, "min_rxn_dist_file")
            xmlMinRxnFile.text = str(self.min_rxn_dist_file)
        xmlSys = ET.SubElement(xmlRoot, "system")
        self.system.serialize(xmlSys, make_apbs_mode)
        return xmlRoot
    
    def write(
            self, 
            filename: str, 
            make_apbs_mode: bool = True
            ) -> None:
        """
        Write the Browndye2 calculation to XML.
        
        Parameters:
        -----------
        filename : str
            The name of the file to write XML to.
        
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        root = self.serialize(make_apbs_mode)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(
            indent="   ")
        with open(filename, 'w') as f:
            f.write(xmlstr)
        return

@define
class Reaction:
    """
    A Browndye2 reaction block in an XML file.
    """
    
    name: str = field(
        default="", validator=instance_of(str))
    state_before: str = field(
        default="", validator=instance_of(str))
    state_after: str = field(
        default="", validator=instance_of(str))
    molecule0_group: str = field(
        default="", validator=instance_of(str))
    molecule0_core: str = field(
        default="", validator=instance_of(str))
    molecule1_group: str = field(
        default="", validator=instance_of(str))
    molecule1_core: str = field(
        default="", validator=instance_of(str))
    n_needed: int = field(
        default=1, validator=[instance_of(int), ge(0)])
    pair_list: typing.List["Pair"] = field(default=Factory(list))
        
    def serialize(
            self, 
            xmlRoot: ET.Element
            ) -> None:
        """
        Convert this object to XML form.
        """
        
        assert self.name, "reaction name must be provided"
        xmlName = ET.SubElement(xmlRoot, "name")
        xmlName.text = self.name
        assert self.state_before, "reaction state_before must be provided"
        xmlStateBefore = ET.SubElement(xmlRoot, "state_before")
        xmlStateBefore.text = self.state_before
        assert self.state_after, "reaction state_after must be provided"
        xmlStateAfter = ET.SubElement(xmlRoot, "state_after")
        xmlStateAfter.text = self.state_after
        xmlCriterion = ET.SubElement(xmlRoot, "criterion")
        xmlMolecules = ET.SubElement(xmlCriterion, "molecules")
        xmlMolecule0 = ET.SubElement(xmlMolecules, "molecule0")
        assert self.molecule0_group != "" and self.molecule0_core != ""
        xmlMolecule0.text = self.molecule0_group + " " + self.molecule0_core
        xmlMolecule1 = ET.SubElement(xmlMolecules, "molecule1")
        assert self.molecule1_group != "" and self.molecule1_core != ""
        xmlMolecule1.text = self.molecule1_group + " " + self.molecule1_core
        assert self.n_needed >= 0
        xmlN_needed = ET.SubElement(xmlCriterion, "n_needed")
        xmlN_needed.text = str(self.n_needed)
        assert len(self.pair_list) > 0
        for pair in self.pair_list:
            xmlPair_i = ET.SubElement(xmlCriterion, "pair")
            pair.serialize(xmlPair_i)
        
        return

@define
class Pair:
    """
    A Browndye2 reaction pair.
    """
    
    atom1_index: int = field(
        default=-1, validator=[instance_of(int), ge(-1)])
    atom2_index: int = field(
        default=-1, validator=[instance_of(int), ge(-1)])
    distance: float = field(
        default=0.0, validator=[instance_of(float), ge(0.0)])
    
    def serialize(
            self, 
            xmlPair: ET.Element
            ) -> None:
        """
        Convert this object to XML output.
        """
        
        assert self.atom1_index >= 0, "Pair atom1_index must be assigned."
        assert self.atom2_index >= 0, "Pair atom2_index must be assigned."
        xmlAtoms = ET.SubElement(xmlPair, "atoms")
        xmlAtoms.text = str(self.atom1_index) + " " + str(self.atom2_index)
        assert self.distance >= 0.0, "Pair distance must be assigned."
        xmlDistance = ET.SubElement(xmlPair, "distance")
        xmlDistance.text = str(self.distance)
        return

@define
class Reaction_root:
    """
    The root XML of the simulation reaction file.
    """
    
    first_state: str = field(
        default="", validator=instance_of(str))
    reaction_list: typing.List[Reaction] = field(default=Factory(list))
    
    def serialize(self) -> ET.Element:
        """
        Convert this object to an XML file.
        """
        
        xmlRoot = ET.Element("roottag")
        assert self.first_state
        xmlFirstState = ET.SubElement(xmlRoot, "first_state")
        xmlFirstState.text = self.first_state
        assert len(self.reaction_list) > 0
        xmlReactions = ET.SubElement(xmlRoot, "reactions")
        for reaction in self.reaction_list:
            xmlReaction_i = ET.SubElement(xmlReactions, "reaction")
            reaction.serialize(xmlReaction_i)
        return xmlRoot
    
    def write(
            self, 
            filename: str
            ) -> None:
        """
        Write the Browndye2 reaction file to XML.
        
        Parameters:
        -----------
        filename : str
            The name of the file to write XML to.
        """
        
        root = self.serialize()
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(
            indent="   ")
        with open(filename, 'w') as f:
            f.write(xmlstr)
        return