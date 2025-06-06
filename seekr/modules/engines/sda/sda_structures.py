"""
modules/engines/sda/sda_structures.py

Any objects that are output as plaintext SDA input files.
Equivalent to the xml_structures.py in the Browndye engine.
"""

import os
import typing

import attrs
from attrs import define, field, validators, Factory
from attrs.validators import and_, in_, optional, instance_of, gt, ge, deep_iterable

import seekr.modules.engines.sda.base as sda_base

@define
class Ion():
    """
    Represent an Ion object in SDA for generating APBS grids and setting 
    ionic strength.
    
    Attributes:
    -----------
    radius : float
        The radius of the ion in units of Angstroms.
    charge : float
        The charge of the ion in units of e.
    conc : float
        The concentration of the ion in solution in units of moles per
        liter.
    """
    
    radius: float = field(default=0.9, 
                          validator=and_(instance_of(float), gt(0)))
    charge: float = field(default=1.0, 
                          validator=and_(instance_of(float)))
    conc: float = field(default=0.0, 
                        validator=and_(instance_of(float), ge(0)))

@define
class Solvent():
    """
    Parameters to represent the solvent within the BD simulation.
    
    Attributes:
    -----------
    dielectric : float, Default 78.0
        The dielectric of solvent, relative to vacuum permittivity.
    temperature : float, Default 298.15
        The temperature of the solvent in Kelvin.
    ions : list
        A list of Ion() objects for APBS input
    """
    dielectric: float = field(default=78.0,
                                validator=and_(instance_of(float), ge(1)))
    temperature: float = field(default=298.15,
                                validator=and_(instance_of(float), gt(0)))
    ions: typing.List[Ion] = field(default=Factory(list),)

# NOTE: setting slots=False allows the __dict__ attribute to be used
@define(slots=False)
class MainSDA():
    """
    Main SDA parameters (they don't belong to any group)
    
    Attributes:
    -----------
    dseed: integer, Default 0
        Seed number to initialize random forces. If 0, time clock is used.
    nrun: integer, Default 1
        Number of trajectories.
    timemax: float, Default 0
        Maximum time of trajectories in ps.
    probep: float, Default 1.4
        Radius of the probe used to compute the exclusion volume grid
    probew: float, Default 1.2
        Radius of solvent probe used to calculate the solvent accessibilities 
        of solute atoms
    hexcl: float, default 0.1
        Exclusion grid spacings
    threshold: float, default 0.0
        Solvent accessible surface area (SASA) threshold to be used.
    epfct: float, default 0.5
        Factor to multiply to electrostatic potentials.
    edfct: float, default None
        Factor to multiply to electrostatic desolvation potentials.
    hdfct: float, default None
        Factor to multiply to hydrophobic desolvation potentials.
    restart: string
        If this string is not empty, BD simulations will be restarted.
        It is necessary to provide a complexes file.
    save_access: integer, Default 1
        Saves the SASA values calculated. It saves computational time if solutes are big.
    rboost: float, Default 1.0
        Boost distance (in Å) to move the solutes apart when more than novers moves 
        were not accepted due to overlaps.
    novers: integer, Default 150
        Number of overlaps allowed before making a boost. 
        Recommended value: 100-200, to avoid boosts.
    """
    dseed: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    nrun: int = field(default=1, validator=and_(instance_of(int), gt(0)))
    timemax: float = field(default=0.0, validator=and_(instance_of(float), ge(0)))
    probep: float = field(default=1.4, validator=and_(instance_of(float), gt(0)))
    probew: float = field(default=1.2, validator=and_(instance_of(float), gt(0)))
    hexcl: float = field(default=0.1, validator=and_(instance_of(float), gt(0)))
    threshold: float = field(default=0.0, validator=and_(instance_of(float), ge(0)))
    epfct: float = field(default=0.5, validator=instance_of(float))
    # TODO: make sure this can be input by user, or automatically assign based on whether
    # this is a protein-protein vs. protein-ligand simulation. See SDA manual for details.
    # https://mcm.h-its.org/sda7/doc/doc_sda7/input.html
    edfct: typing.Optional[float] = field(default=0.36, 
                                          validator=optional(instance_of(float)))
    # Same as for edfct, see above.
    hdfct: typing.Optional[float] = field(default=-0.013,
                                          validator=optional(instance_of(float)))
    restart: typing.Optional[str] = field(default=None,
                                          validator=optional(instance_of(str)))
    save_access: int = field(default=1, validator=and_(instance_of(int), in_([0, 1])))
    rboost: float = field(default=1.0, validator=and_(instance_of(float), gt(0)))
    novers: int = field(default=150, validator=and_(instance_of(int), gt(0)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = attribute_name + " = " + str(value) + "\n"
                self.group[attribute_name] = string

# TODO: this seems to exist in Input() below. Is this redundant?
@define(slots=False)
class Type_Calculation():
    """
    Calculation type to perform in BD simulations
    
    Attributes:
    -----------
    type : string (sda_2proteins, sdamm, sda_koff, sda_energy, sda_pmf), 
        Default sda_2proteins. Indicates the type of simulation to run
    total solutes : integer, Default 2
        Total solutes to use.
    total grids : integer, Default 2
        Total grid types to use. Conformational ensamble counts as 1 type of grid.
    """
    type: typing.Optional[str] = field(default="sda_2proteins", 
                                       validator=and_(instance_of(str),
                                        in_(["sda_2proteins", "sdamm", "sda_koff", "sda_energy", 
                                             "sda_pmf"])))
    total_solutes: int = field(default=2, validator=and_(instance_of(int), ge(2)))
    total_grids: int = field(default=2, validator=and_(instance_of(int), ge(1)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        assert self.total_solutes >= self.total_grids, \
            "total_solutes must be greater than or equal to total_grids"
        if self.type != "sdamm":
            assert self.total_solutes == 2, \
                "total_solutes must be equal to 2 for type " + self.type
            assert self.total_grids == 2, \
                "total_grids must be equal to 2 for type " + self.type
        self.group = {}
        self.group["GROUP"] = "GROUP = Type_Calculation\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return

@define(slots=False)
class ReactionCriteria():
    """
    Parameters for reaction criteria.
    
    Attributes:
    -----------
    computation: string Default "association"
        Defines the type of calculations to perform during
    BD simulations. Options are:
            - association: calculates kon from trajectories that satisfy reaction criteria
            - docking: saves the encounter complexes that satisfy reaction criteria.
            - electron_transfer: computes electron transfer
            - all: save all the encounter complexes, reaction criteria is omited.
            - off_rc: deactivates reaction criteria.
    rxna12f: string
        Pathway and name of the reaction criteria file.
    et_sol1, et_sol2: string or None
        Pathways and names of the reaction criteria file 
        for the electron transfer calculations.
    dind: float, Default 0.
        Distance between contacts to be considered independent.
    nnnons: integer, Default 1
        Maximum number of independent contacts required to record encounter complexes.
    nwrec: integer, Default 1
        Windows distance number where start recording encounter complexes.
        The distance is defined as (win0+dwin*(nwrec-1))
    sdamd: integer, Default 0
        Saves the encounter complexes that are found in the nwrec windows distance.
    """    
    computation: str = field(default="association",
                            validator=optional(and_(instance_of(str), in_([
                            "association",])))) #"docking", "electron_transfer", "all", 
                            #"off_rc"]))))
    rxna12f: str = field(default=sda_base.REACTION_BASENAME,
                            validator=instance_of(str))
    et_sol1: typing.Optional[str] = field(default=None, 
                                          validator=optional(instance_of(str)))
    et_sol2: typing.Optional[str] = field(default=None,
                                          validator=optional(instance_of(str)))
    dind: float = field(default=0.0,
                       validator=and_(instance_of(float), ge(0)))
    nnnons: int = field(default=1, validator=and_(instance_of(int), ge(1)))
    nwrec: int = field(default=1, validator=and_(instance_of(int), ge(1)))
    sdamd: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = ReactionCriteria\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return
    
@define(slots=False)
class RateCalculation():
    """
    Parameters for rate calculations.
    
    Attributes:
    -----------
    win0 : float, Default None
        Closest windows distance reaction
    nwin: integer, Default None
        Number of windows distances to monitore association rates
    dwin: float, Default 0.5
        Separation between windows distances
    nb_contact: integer, Default 1
        Number of independent contacts to be considered for rate calculations
    bootstrap: integer, Default 1
        If 1, association rates and first passage time are printed for every
        trajectory to perform later bootstrap calculations.
    fpt: integer, Default 0
        If 1, first passage time is printed.
    stop_traj: integer, Default 0
        If 1, the trajectory will stop when the most strict reaction criteria
        definition is met.
    analytical_correction: integer, Default 0
        If 1, the analytical correction for centrosymmetric forces is computed.
    """
    win0: float = field(default=4.0, validator=and_(instance_of(float), ge(0)))
    nwin: int = field(default=35, validator=and_(instance_of(int), ge(1)))
    dwin: float = field(default=0.5, validator=and_(instance_of(float), ge(0)))
    nb_contact: int = field(default=1, validator=and_(instance_of(int), ge(1)))
    bootstrap: int = field(default=1, validator=and_(instance_of(int), ge(0)))
    fpt: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    stop_traj: int = field(default=1, validator=and_(instance_of(int), ge(0)))
    analytical_correction: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = RateCalculation\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return
    
@define(slots=False)
class Analytical():
    """
    Parameters for rate calculations.
    
    Attributes:
    -----------
    h_analytic : float, Default 0.01
        defines the bin size (in Å) of the arrays of the precomputed 
        debye-hueckel values.
    debyeh: integer, Default 0
        if 1, activates Debye-Hueckel interactions for all solutes.
    dh_imgchg: integer, Default 0
        if 1 and a surface is present, the image-charge model used 
        for modelling electrostatic interactions with conducting surfaces 
        is extended to use debye-huckel interactions for all solutes whose 
        images lie outside of their electrostatic grids.
    ionic_strength: float or None, Default None
        Ionic strength (in M) for Debye-Hueckel interactions.
    hydrodynamic: integer, Default 0
        If 1, activates the mean-field hydrodynamics interactions.
    lvol_rcut: float or None, Default None
        Defines the local volume radius for hydrodynamics. lvol_rcut must
        be greater than vol_radius of any solute.
    surface_heigth: float or None, Default None
        Defines the heigth (in A) above a surface at which the image method 
        is applied for surface hydrodynamic interactions.
    hom_charged_surf: integer, Default 0
        If 1, activates the long-range Debye-Hueckel electrostatic treatment
        for a surface
    surface_fact: float or None, Default None
        Prefactor for the surface charge potential. If the layer is thin 
        (i.e.: one atom layer), surface_fact = 1. If the layer is thin,
        surface_fact = 0.5.
    gouy_chapman: integer, Default 0
        If 1, uses Gouy-Chapman treatment instead of Debye-Huckel for the long range 
        electrostatics of the surface.
    """
    h_analytic: float = field(default=0.01, validator=and_(instance_of(float), gt(0)))
    debyeh: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    dh_imgchg: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    ionic_strength: typing.Optional[float] = field(default=None,
                                                   validator=optional(and_(instance_of(float), ge(0))))
    hydrodynamic: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    lvol_rcut: typing.Optional[float] = field(default=None,
                                              validator=optional(and_(instance_of(float), gt(0))))
    surface_heigth: typing.Optional[float] = field(default=None,
                                                   validator=optional(and_(instance_of(float), gt(0))))
    hom_charged_surf: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    surface_fact: typing.Optional[float] = field(default=None,
                                                 validator=optional(and_(instance_of(float), ge(0))))
    gouy_chapman: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = Analytical\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return

@define(slots=False)
class Solute_Grid():
    """
    Parameters for rate calculations.
    
    Attributes:
    -----------
    nb_solute : integer, Default 1
        Number of solutes of this type.
    pdb_filename : string
        Pathway where the pdb file is located.
    trans_diffusion : float or None. Default None.
        Translational diffusion coefficient value in A^2/ps
    rotat_diffusion : float or None. Default None.
        Rotational diffusion coefficient value in rad/ps
    real_net_charge: float or None. Default None.
        Total net formal charge of the solute. 
        Used if the Debye field is activated.
    surface_charge_dens: float or None. Default None.
        If the solute is a surface, this option can be used to define
        an average charge density for surface. The interaction with other solutes 
        is then modelled using a Debye-Hueckel sphere approximation where 
        spherical charge solutes interact with a homogeneously charged surface.
    rotate : integer, default 1.
        Defines whether the solute can rotate or not. If 0,
        rotational diffusion coefficient is not used.
    surface: str or None, Default None.
        Defines if the solute is a surface.
    flex: integer or None, Default None
        Defines if a conformational ensamble is read for the solute.
    image_charge: integer or None, Default None
        Defines if the electrostatic image charge of this solute(s) must be computed. 
        It is relevant in the case that one solute is a metal surface.
    dh_radius: float or None, Default None
        Indicates the radius used to represent the solute cavity in the Debye-Hueckel 
        sphere charge model.
    vol_radius: float or None, Default None
        Indicates the radius used for calculating the local occupied volume fractions 
        of the mean-field hydrodynamic interaction model.
    list_conformation: string or None, Default None
        Plain text file that indicates the pathways and names of the different grids
        and population of each conformer of the solute.
    total_conf: integer or None, Default None
        total number of conformations of the solute
    method: string (random, minimum, metropolis) or None, Default None.
        method to be used for changing the conformations on the fly. Options:
            - random: choose a random conformation if nearest > 1. Energies are
            not evaluated.
            - minimum: the minimum energy between actual and the closest conformations
            is selected if nearest = 1
            - metropolis: Metropolis-Hastings algorithm is applied to choose a new
            conformation based on the population and the intermolecular energy with other
            solutes.
    initial_conf: integer, Default None.
        selects the conformation at the beginning of each trajectory. If value is -1, the
        conformation will be selected randomly.
    frequency: float, Default None
        Delay time between changes in conformation (in ps).
        If SDAMM, the delay is computed as a Gaussian random with average frequency
        and standard deviation std_frequency to avoid syncronisation between solutes.
    std_frequency: float, Default None
        Standard deviation frequency (in ps).
    epf : string or None, Default None
        Pathway and name of the electrostatic potential grid file
    qef : string or None, Default None
        Pathway and name of the effective charges file
    edf : string or None, Default None
        Pathway and name of the electrostatic desolvation grid file
    hdf : string or None, Default None
        Pathway and name of the hydrophobic desolvation grid file
    lj_repf : string or None, Default None
        Pathway and name of the lennard-jones repulsion grid file
    """

    nb_solute: int = field(default=1, validator=and_(instance_of(int), ge(1)))
    pdb_filename: str = field(default="", validator=instance_of(str))
    trans_diffusion: typing.Optional[float] = field(default=None,
                                          validator=optional(and_(instance_of(float), gt(0))))
    rot_diffusion: typing.Optional[float] = field(default=None,
                                          validator=optional(and_(instance_of(float), gt(0))))
    real_net_charge: typing.Optional[float] = field(default=None,
                                          validator=optional(instance_of(float)))
    surface_charge_dens: typing.Optional[float] = field(default=None,
                                          validator=optional(instance_of(float)))
    rotate: int = field(default=1, validator=and_(instance_of(int), in_([0, 1])))
    surface: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    flex: typing.Optional[int] = field(default=None,
                                          validator=optional(and_(instance_of(int), in_([0, 1]))))
    image_charge: typing.Optional[int] = field(default=None,
                                          validator=optional(and_(instance_of(int), in_([0, 1]))))
    dh_radius: typing.Optional[float] = field(default=None,
                                          validator=optional(and_(instance_of(float), gt(0))))
    vol_radius: typing.Optional[float] = field(default=None,
                                          validator=optional(and_(instance_of(float), gt(0))))
    list_conformation: typing.Optional[str] = field(default=None,
                                          validator=optional(instance_of(str)))
    total_conf: typing.Optional[int] = field(default=None,
                                          validator=optional(and_(instance_of(int), ge(1))))
    method: typing.Optional[str] = field(default=None,
                                          validator=optional(and_(instance_of(str), in_([
                                              "random", "minimum", "metropolis"]))))
    initial_conf: int = field(default=None, validator=optional(and_(instance_of(int), ge(-1))))
    frequency: float = field(default=None, validator=optional(and_(instance_of(float), gt(0))))
    std_frequency: typing.Optional[float] = field(default=None,
                                          validator=optional(and_(instance_of(float), gt(0))))
    epf: str = field(default="", validator=instance_of(str))
    qef: str = field(default="", validator=instance_of(str))
    edf: str = field(default="", validator=instance_of(str))
    hdf: str = field(default="", validator=instance_of(str))
    lj_repf: typing.Optional[str] = field(default=None, validator=optional(instance_of(str)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = Solute_Grid\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return

@define
class Solute():

    """
    A solute for input to SDA
    
    Attributes:
    ----------

    type : string
        Indicates the type of molecule to be simulated (protein or \
        small organic compound).
    apbs_grid_dime : integer
        Size of the APBS grid dimensions.
    apbs_grid_spacing : float
        Space between the points conforming the grid.
    dielectric : float
        Dielectric constant of the solute.
    solute_grid : Solute_Grid()
        Solute grid object whose attributes will be used to build \
        an SDA input file.
    """
    type: str = field(default="", validator=instance_of(str))
    apbs_grid_dime: int = field(default=-1, validator=instance_of(int))
    apbs_grid_spacing: float = field(default=-1.0, validator=instance_of(float))
    dielectric: float = field(default=-1.0, validator=instance_of(float))
    solute_grid: Solute_Grid = field(default=Factory(Solute_Grid),
                                    validator=instance_of(Solute_Grid))
    pqr_filename: str = field(default="", validator=instance_of(str))

@define(slots=False)
class Geometry():
    """
    Parameters to represent the geometry properties in where simulations are going to run.
    
    Attributes:
    -----------
    type : string (sphere, box, nambox), Default sphere
        Indicates whether a simulation runs in a box or in a sphere. If nambox, association rates
        are calculated using the NAM algorithm in a box.
    pbc : integer, Default 0
        If 1, periodic boundary conditions are applied when using a box.
    escape : integer, Default 1
        Indicates if a solute can escape. If the solute escapes, the trajectory will stop.
    surface : integer, Default 0
        Indicates if a surface is present.
    start_pos : float, default None
        Center-to-center distance (in amstrongs) of solutes at the beginning of a trajectory. 
        If the simulation runs in a box, this distance will be considered on the z-ax.
        If the simulation runs in a sphere or nambox, this distance is the b-surface.
    c : float, default None
        c-surface (in amstrongs) in case that simulations run in a sphere or nambox.
    <AX>min, <AX>max: float or None, default None
        <AX> = x, y, z. Min and max indicate the size of the ax.
        If escape is activated, the trajectory ends when the solute reaches zmax.
    """

    type: str = field(default="sphere",
                    validator=and_(instance_of(str), in_(["sphere", "box", "nambox"])))
    pbc: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    surface: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    escape: int = field(default=1, validator=and_(instance_of(int), in_([0, 1])))
    start_pos: float = field(default=100.0, validator=and_(instance_of(float), ge(0)))
    c: float = field(default=-1.0, validator=instance_of(float))
    xmin: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    xmax: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    ymin: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    ymax: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    zmin: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    zmax: typing.Optional[float] = field(default=None, validator=optional(instance_of(float)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = Geometry\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return

@define(slots=False)
class Timestep():
    """
    Parameters to represent the timestep configuration.
    
    Attributes:
    -----------
    variable : integer, Default 1
        If 1, timestep is variable, describing a linear change between 
        dt1 and dt2 when solutes are within swd1 and swd2 distance.
        If 0, just dt1 is considered.
    dt1 : float or None. Default None.
        Smallest timestep. Employed when solutes are closer than swd1.
    swd1 : float or None. Default None.
        Center-to-center distance at which timestep = dt1.
    dt2 : float or None. Default None.
        Longest timestep. Employed when solutes are beyond swd2
    swd2 : float or None. Default None.
        Center-to-center distance at which timestep = dt2.
    """
    variable: int = field(default=1, validator=and_(instance_of(int), in_([0, 1])))
    dt1: typing.Optional[float] = field(default=1.0, 
                                        validator=optional(and_(instance_of(float), gt(0))))
    swd1: typing.Optional[float] = field(default=60.0, 
                                         validator=optional(and_(instance_of(float), gt(0))))
    dt2: typing.Optional[float] = field(default=20.0,
                                        validator=optional(and_(instance_of(float), gt(0))))
    swd2: typing.Optional[float] = field(default=100.0,
                                         validator=optional(and_(instance_of(float), gt(0))))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = Timestep\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return
    
@define(slots=False)
class Complexes():
    """
    Parameters to represent the timestep configuration.
    
    Attributes:
    -----------
    fcomplexes : string
        Pathway and name of the file where encounter complexes will be written.
    restart_complex : integer, Default 0.
        If 1, simulations will be restarted from the complexes file indicated
        in "restart" flag.
    binary_complex : integer, Default 0.
        If 1, complexes file will be in binary format.
    nb_complexes : integer, Default 500
        Number of encounter complexes to record. If this number is reached,
        the new encounter complexes with lower energy will replace the old
        encounter complexes with higher energy.
    rmsd_min : float, Default 1.0
        RMSD threshold (in A) between encounter complexes to consider them as
        the same encounter complex.
    ftrajectories : string
        Pathway and name of the trajectories file
    binary_trajectory : integer, Default 0.
        If 1, trajectories file will be in binary format.
    ntraj_rec : integer, Default -1
        Number of the trajectory to record. If -1, all trajectories
        are saved.
    freq_print : integer, Default 1000
        Frequency (in steps) with which snapshots of the trajectory are saved.
    """
    fcomplexes: str = field(default="", validator=instance_of(str))
    restart_complex: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    binary_complex: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    nb_complexes: int = field(default=500, validator=and_(instance_of(int), ge(1)))
    rmsd_min: float = field(default=1.0, validator=and_(instance_of(float), ge(0)))
    ftrajectories: str = field(default="", validator=instance_of(str))
    binary_trajectory: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    ntraj_rec: int = field(default=-1, validator=and_(instance_of(int), ge(-1)))
    freq_print: int = field(default=1000, validator=and_(instance_of(int), ge(1)))
    group: dict = field(default=Factory(dict), 
                        validator=instance_of(dict))
    
    def make_group(self):
        self.group = {}
        self.group["GROUP"] = "GROUP = Complexes\n"
        for attribute_name, value in self.__dict__.items():
            if attribute_name != "group" and value is not None:
                string = "    " + attribute_name + " = " \
                    + str(value) + "\n"
                self.group[attribute_name] = string
        self.group["END_GROUP"] = "END GROUP\n"
        return

# TODO: review some of these attributes within other objects in this module.
#  There seems to be some redundancy.
@define
class Input():
    """
    An SDA Input file object.
    
    Attributes:
    -----------
    nrun : int or None
        Number of trajectories.
    dseed : int
        Used to seed the random number generator.
    output : str
        Name of file used for outputting SDA results.
    timemax : float or None
        Maximum time per trajectory. If 0, trajectory ends when solute escapes.
    prot_data_grid_path : string
        Path of the protein's grid files.
    lig_data_grid_path : string
        Path of the ligand's grid files.
    trajectory_file : str or None
        Name of file used for outputting of trajectories.
    complexes_file: str or None
        Name of file used for outputting of encounter complexes.
    nb_complexes: int or None
        Number of encounter complexes to record
    ntraj: int or None
        Number of the trajectory to record. If -1, all are recorded.
    geometry: str or None
        Geometry used in the simulation (sphere, box, nambox)
    start_pos: float or None
        Position of the ligand at the beginning of a trajectory
    c: float or None
        c-surface
    <AX>min, <AX>max: float or None
        Initial and final position of box axes, being AX = x, y, z
    surface: int or None
        1 if a surface is present
    win0: float
        Closest windows distance
    nwin: integer
        Number of windows distances to monitor
    hydrodynamics : int or None, Default None
        If 1, mean-field hydrodynamics approximation will be used.
    """
    nrun: int = field(default=-1, validator=and_(instance_of(int), ge(-1)))
    timemax: float = field(default=0.0, validator=and_(instance_of(float), ge(0)))
    dseed: int = field(default=0, validator=and_(instance_of(int), ge(0)))
    type: str = field(default="sda_2proteins", 
                      validator=and_(instance_of(str),
                        in_(["sda_2proteins", "sdamm", "sda_koff", "sda_energy"])))
    solutes: typing.List[Solute] = field(factory=list, )
                                         #validator=and_(instance_of(list), 
                                         #               len_gt(0)))
    solutes_path: str = field(default="", validator=instance_of(str))
    solvent_ions: typing.List[Ion] = field(factory=list,)
    total_solutes: int = field(default=2, validator=and_(instance_of(int), ge(1)))
    total_grids: int = field(default=2, validator=and_(instance_of(int), ge(0)))
    rxna12f: str = field(default=sda_base.REACTION_BASENAME,
                         validator=instance_of(str))
    ftrajectories: str = field(default=sda_base.SDA_TRAJ_NAME,
                               validator=instance_of(str))
    fcomplexes: str = field(default=sda_base.SDA_COMPLEXES_NAME,
                            validator=instance_of(str))
    nb_complexes: int = field(default=10000, validator=and_(instance_of(int), ge(1)))
    ntraj: int = field(default=1, validator=and_(instance_of(int), ge(-1)))
    geom_type: str = field(default="sphere",
                            validator=and_(instance_of(str), in_(["sphere", "box", "nambox"])))
    start_pos: float = field(default=100.0, validator=and_(instance_of(float), ge(0)))
    c: float = field(default=200.0, validator=and_(instance_of(float), ge(0)))
    pbc: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    xmin: typing.Optional[float] = field(default=None, 
                                         validator=optional(instance_of(float)))
    xmax: typing.Optional[float] = field(default=None,
                                         validator=optional(instance_of(float)))
    ymin: typing.Optional[float] = field(default=None,
                                         validator=optional(instance_of(float)))
    ymax: typing.Optional[float] = field(default=None,  
                                         validator=optional(instance_of(float)))
    zmin: typing.Optional[float] = field(default=None,
                                         validator=optional(instance_of(float)))
    zmax: typing.Optional[float] = field(default=None,
                                         validator=optional(instance_of(float)))
    surface: int = field(default=0, validator=and_(instance_of(int), in_([0, 1])))
    rotate: int = field(default=1, validator=and_(instance_of(int), in_([0, 1])))
    win0: float = field(default=20.0, validator=and_(instance_of(float), ge(0)))
    nwin: int = field(default=2, validator=and_(instance_of(int), ge(1)))
    dwin: float = field(default=0.5, validator=and_(instance_of(float), ge(0)))
    dt_variable: typing.Optional[int] = field(default=1,
                                              validator=optional(and_(instance_of(int), in_([0, 1]))))
    dt1: typing.Optional[float] = field(default=1.0,
                                        validator=optional(and_(instance_of(float), gt(0))))
    hydrodynamics: typing.Optional[int] = field(default=None,
                                                validator=optional(and_(instance_of(int), in_([0, 1]))))
    main: MainSDA = field(factory=MainSDA,
                         validator=instance_of(MainSDA))
    type_calculation: Type_Calculation = field(factory=Type_Calculation,
                                               validator=instance_of(Type_Calculation))
    reaction_criteria: ReactionCriteria = field(factory=ReactionCriteria,
                                               validator=instance_of(ReactionCriteria))
    rate_calculation: RateCalculation = field(factory=RateCalculation,
                                               validator=instance_of(RateCalculation))
    solute_grids: typing.List[Solute_Grid] = field(factory=list,
                                                   validator=instance_of(list))
    analytical: typing.Optional[Analytical] = field(default=None,
                                                   validator=optional(instance_of(Analytical)))
    geometry: Geometry = field(factory=Geometry,
                              validator=instance_of(Geometry))
    timestep: Timestep = field(factory=Timestep,
                              validator=instance_of(Timestep))
    complexes: Complexes = field(factory=Complexes,
                              validator=instance_of(Complexes))
    
    def make_input(self, make_analytical=False):
        """
        Creates the SDA input file.
        
        Parameters:
        -----------
        make_analytical : bool, Default False
            If True, the analytical group will be written into the 
            SDA input file.
        """
        
        self.main = MainSDA()
        self.main.nrun = self.nrun
        self.main.timemax = self.timemax
        self.main.make_group()

        self.type_calculation = Type_Calculation()
        self.type_calculation.type = self.type
        self.type_calculation.total_solutes = self.total_solutes
        self.type_calculation.total_grids = self.total_grids
        self.type_calculation.make_group()

        self.reaction_criteria = ReactionCriteria()
        self.reaction_criteria.make_group()

        self.rate_calculation = RateCalculation()
        self.rate_calculation.win0 = self.win0
        self.rate_calculation.nwin = self.nwin
        self.rate_calculation.dwin = self.dwin
        self.rate_calculation.make_group()

        self.solute_grids = []
        for solute in self.solutes:
            pqr_filename = os.path.basename(solute.pqr_filename)
            solute_name = '.'.join(pqr_filename.split(".")[:-1])
            pdb_filename = solute_name + "_noh.pdb"
            solute_grid = Solute_Grid()
            solute_grid.nb_solute = solute.solute_grid.nb_solute
            solute_grid.trans_diffusion = solute.solute_grid.trans_diffusion
            solute_grid.rot_diffusion = solute.solute_grid.rot_diffusion
            solute_grid.pdb_filename = os.path.join(self.solutes_path, \
                                                pdb_filename)
            if hasattr(solute, 'rotate'):
                self.solute_grid.rotate = solute.solute_grid.rotate
            if hasattr(solute, 'surface') and solute.solute_grid.surface == "yes":
                self.solute_grid.surface = 1
            solute_grid.epf = os.path.join(self.solutes_path, \
                                                solute_name + "_ep.grd")
            solute_grid.qef = os.path.join(self.solutes_path, \
                                                solute_name + ".echa")
            solute_grid.edf = os.path.join(self.solutes_path, \
                                                solute_name + "_ed.grd")
            solute_grid.hdf = os.path.join(self.solutes_path, \
                                                solute_name + "_hd.grd")
            if hasattr(solute, 'lj_repf'):
                solute_grid.lj_repf = os.path.join(self.solutes_path, \
                                                solute_name + "_ljrep.grd")
            solute_grid.make_group()
            self.solute_grids.append(solute_grid)

        if make_analytical:
            ionic_strength = 0
            for ion in self.solvent_ions:
                ionic_strength += ion.conc*ion.charge**2 / 2
            self.analytical = Analytical()
            self.analytical.make_group()
        else:
            self.analytical = None

        self.geometry = Geometry()
        self.geometry.type = self.geom_type
        self.geometry.start_pos = self.start_pos
        self.geometry.pbc = self.pbc
        if self.geometry.type == "sphere" or self.geometry.type == "nambox":
            self.geometry.c = self.c
        if self.geometry.type == "box" or self.geometry.type == "nambox":
            self.geometry.xmin = self.xmin
            self.geometry.xmax = self.xmax
            self.geometry.ymin = self.ymin
            self.geometry.ymax = self.ymax
            self.geometry.zmin = self.zmin
            self.geometry.zmax = self.zmax
        self.geometry.surface = self.surface   
        self.geometry.make_group()
        
        self.timestep = Timestep()
        self.timestep.variable = self.dt_variable
        self.timestep.dt1 = self.dt1
        if self.dt_variable is None:
            self.timestep.swd1 = None
            self.timestep.dt2 = None
            self.timestep.swd2 = None
        self.timestep.make_group()

        self.complexes = Complexes()
        self.complexes.fcomplexes = self.fcomplexes
        self.complexes.ftrajectories = self.ftrajectories
        self.complexes.make_group()

        return
    
    # TODO: remove make_apbs_mode parameter, see if it is not used.
    def write(self, filename, make_apbs_mode=True):
        """
        Write the SDA input to plain text.
        
        Parameters:
        -----------
        filename : str
            The name of the file to write XML to.
        
        make_apbs_mode : bool
            Whether this object should be serialized for the
            make_apbs_inputs program.
        """
        
        self.make_input()
        with open(filename, 'w') as f:

            for parameter in self.type_calculation.group:            
                line = self.type_calculation.group[parameter]
                f.write(line)
            f.write("\n")

            for parameter in self.main.group:
                if parameter != "novers" and parameter != "rboost":            
                    line = self.main.group[parameter]
                    f.write(line)
            f.write("\n")

            for parameter in self.reaction_criteria.group:
                line = self.reaction_criteria.group[parameter]
                f.write(line)
            f.write("\n")

            for parameter in self.rate_calculation.group:
                line = self.rate_calculation.group[parameter]
                f.write(line)
            f.write("\n")

            if self.analytical is not None:
                for parameter in self.analytical.group:
                    line = self.analytical.group[parameter]
                    f.write(line)
                f.write("\n")

            for solute_grid in self.solute_grids:
                for parameter in solute_grid.group:
                    line = solute_grid.group[parameter]
                    f.write(line)
                f.write("\n")
            
            for parameter in self.geometry.group:
                line = self.geometry.group[parameter]
                f.write(line)
            f.write("\n")

            for parameter in self.timestep.group:
                line = self.timestep.group[parameter]
                f.write(line)
            f.write("\n")

            for parameter in self.complexes.group:
                line = self.complexes.group[parameter]
                f.write(line)
            f.write("\n")

            line = self.main.group["rboost"]
            f.write(line+"\n")
            line = self.main.group["novers"]
            f.write(line+"\n")

        return

@define
class Reaction():
    """
    Reaction file class.
    """
    atoms_1: typing.List[int] = field(factory=list, validator=instance_of(list))
    atoms_2: typing.List[int] = field(factory=list, validator=instance_of(list))
    
    def make_input(self):
        """
        Creates an SDA reaction file.
        """
        
        rxna_lines = []
        for ghost_atom_rec, ghost_atom_lig in zip(self.atoms_1, self.atoms_2):
            reaction = f"CNONS {ghost_atom_rec} |{8:>9.2f}| {ghost_atom_lig}"
            rxna_lines.append(reaction)

        return rxna_lines
    
    def write(self, filename):
        """
        Write the SDA reaction file to a plain text.
        
        Parameters:
        -----------
        filename : str
            The name of the rxna output file.
        """
        
        root = self.make_input()
        with open(filename, 'w') as f:
            for line in root:
                f.write(line)
        return

@define
class APBS_Input():
    temperature: float = field(default=298.15,
                              validator=and_(instance_of(float), ge(0)))
    solute_pqr: str = field(default="", validator=instance_of(str))
    solute_name: str = field(default="", validator=instance_of(str))
    solute_dielectric: float = field(default=2.0,
                                    validator=and_(instance_of(float), ge(1)))
    solvent_dielectric: float = field(default=78.5,
                                    validator=and_(instance_of(float), ge(1)))
    apbs_grid_spacing: float = field(default=0.5,
                                    validator=and_(instance_of(float), gt(0)))
    apbs_grid_dime: int = field(default=110, validator=and_(instance_of(int), ge(1)))
    ions: typing.List[Ion] = field(factory=list,)
    
    def make_input(self):
        """
        Prepares APBS input files.
        """
        pqr_filename = os.path.basename(\
            self.solute_pqr)
        solute_name = self.solute_name
        apbs_lines = []
        dime = self.apbs_grid_dime
        grid = self.apbs_grid_spacing
        apbs_lines.append("read\n" \
                          +"  mol pqr "+pqr_filename+" \n" \
                          +"end \n" \
                          +"\n" \
                          +"elec name viz \n" \
                          +"  mg-manual \n" \
                          +f"  dime {dime} {dime} {dime}\n" \
                          +f"  grid {grid} {grid} {grid}\n" \
                          +"  gcent  mol 1 \n" \
                          +"  mol 1 \n" \
                          +"  lpbe \n" \
                          +"  bcfl sdh \n")
        for ion in self.ions:
            apbs_lines.append("  ion charge " + str(ion.charge) + \
                              " conc " + str(ion.conc) + \
                              " radius " + str(ion.radius) +"\n")
        apbs_lines.append("  pdie " + str(self.solute_dielectric) + "\n" \
                          +"  sdie " + str(self.solvent_dielectric) + "\n" \
                          +"  chgm spl0 \n" \
                          +"  srfm smol \n" \
                          +"  srad 0.0 \n" \
                          +"  swin 0.3 \n" \
                          +"  temp " + str(self.temperature) + "\n" \
                          +"  write pot uhbd " + solute_name + "_apbs\n" \
                          +"end \n" \
                          +"quit \n")
        apbs_grid_filename = f"{solute_name}_apbs.grd"
        return apbs_lines, apbs_grid_filename

    def write(self):

        """
        Write the APBS input file to a plain text.
        """

        apbs_lines, apbs_grid_filename = self.make_input()
        apbs_input_filename = self.solute_name+"_apbs.in"
        with open(apbs_input_filename, "w") as apbs_file:
            for line in apbs_lines:
                apbs_file.write(line)
            
        return apbs_input_filename, apbs_grid_filename
    
@define
class SDA_grid_inputs():

    """
    Prepares SDA grid inputs.
    """
    temperature: float = field(default=298.15,
                              validator=and_(instance_of(float), ge(0)))
    ionic_strength: float = field(default=0.0,
                                  validator=and_(instance_of(float), ge(0)))
    solute_name: str = field(default="", validator=instance_of(str))
    solute_pdb: str = field(default="", validator=instance_of(str))
    #solvent: Solvent = field(default=Factory(Solvent),
    #                         validator=instance_of(Solvent))
    solvent_dielectric: float = field(default=78.5,
                                    validator=and_(instance_of(float), ge(1)))
    ecm_input: str = field(default="", validator=instance_of(str))
    ecm_lines: typing.List[str] = field(factory=list,
                                       validator=instance_of(list))
    tcha: str = field(default="", validator=instance_of(str))
    echa: str = field(default="", validator=instance_of(str))
    epf: str = field(default="", validator=instance_of(str))
    edf_input: str = field(default="", validator=instance_of(str))
    edf_lines: typing.List[str] = field(factory=list,
                                       validator=instance_of(list))
    hdf_input: str = field(default="", validator=instance_of(str))
    hdf_lines: typing.List[str] = field(factory=list,
                                       validator=instance_of(list))
    lj_repf_input: str = field(default="", validator=instance_of(str))
    lj_repf_lines: typing.List[str] = field(factory=list,
                                       validator=instance_of(list))
    edf: str = field(default="", validator=instance_of(str))
    hdf: str = field(default="", validator=instance_of(str))
    lj_repf: str = field(default="", validator=instance_of(str))
    dime: int = field(default=110, validator=and_(instance_of(int), ge(1)))
    grid_spacing: float = field(default=0.5, 
                                validator=and_(instance_of(float), gt(0)))
    def make_inputs(self):
        
        """
        Create the grid and effective charge input files
        """
        assert self.solute_pdb != "", "Solute PDB file name is not set."
        self.ecm_lines.append("#------------------ pdb file name\n" \
                              +self.solute_pdb+"\n" \
                              +"#------------------ file with test charges for a molecule\n" \
                              +self.tcha+"\n" \
                              +"#------------------ grid file name\n" \
                              +self.epf+"\n" \
                              +"#------------------ probe, skin: expansion happens in [probe; probe+skin] interval\n" \
                              +"4.0, 3.0\n" \
                              +"#------------------ ionic strength, solvent dielectric\n" \
                              +str(self.ionic_strength)+", "+str(self.solvent_dielectric)+"\n" \
                              +"#------------------ file to write effective charges\n" \
                              +self.echa+"\n" \
                              +"#------------------ NEW: ADDED in  version flex, reg_charge, if 0 will not print echa, and then will need ecm_mkreglev for finalising effective charges\n" \
                              +"1.0")


        self.edf_lines.append("#------------------------------ h,ndimx,ndimy,ndimz\n" \
                              +str(self.grid_spacing)+", "+str(self.dime)+", "+str(self.dime)+", "+str(self.dime)+"\n" \
                              +"#------------------------------ iostr,epssol,rion\n" \
                              +"0.  "+str(self.solvent_dielectric)+"  1.5\n" \
                              +"#------------------------------ pfile\n" \
                              +self.solute_pdb+"\n" \
                              +"#------------------------------ efile, iform\n" \
                              +self.edf+"\n" \
                              +"0")
        
        self.hdf_lines.append("#------------------------------ h,ndimx,ndimy,ndimz\n" \
                              +str(self.grid_spacing)+", "+str(self.dime)+", "+str(self.dime)+", "+str(self.dime)+"\n" \
                              +"#------------------------------ a,b, grid-value\n" \
                              +"3.10, 4.35, 0.5\n" \
                              +"#------------------------------ pfile\n" \
                              +self.solute_pdb+"\n" \
                              +"#------------------------------ efile, iform\n" \
                              +self.hdf+"\n" \
                              +"0")

        if len(self.lj_repf_input) > 0:
            self.lj_repf_lines.append("#------------------------------ h,ndimx,ndimy,ndimz\n" \
                                      +str(self.grid_spacing)+", "+str(self.dime)+", "+str(self.dime)+", "+str(self.dime)+"\n" \
                                      +"#------------------------------ factor, nexp, fraction\n" \
                                      +"4096.d0, 6, 1.5d0\n" \
                                      +"#------------------------------ pfile\n" \
                                      +self.solute_pdb+"\n" \
                                      +"#------------------------------ efile, iform\n" \
                                      +self.lj_repf+"\n" \
                                      +"0")
        return


    def write(self):
        
        """
        Write the grid and effective charge input files
        """

        self.make_inputs()

        with open(self.ecm_input, "w") as ecm:
            for line in self.ecm_lines:
                ecm.write(line)

        with open(self.edf_input, "w") as ed:
            for line in self.edf_lines:
                ed.write(line)
        
        with open(self.hdf_input, "w") as hd:
            for line in self.hdf_lines:
                hd.write(line)

        if len(self.lj_repf_input) > 0:
            with open(self.lj_repf_input, "w") as lj_rep:
                for line in self.lj_repf_lines:
                    lj_rep.write(line)

@define
class Hydropro():

    """
    Class to set the hydropro values

    Attributes:
    --------------------
    temperature: float
        Temperature of the simulation.
    pdbfile: str
        PDB file name containing the heavy atoms of the solute.
    """
    temperature: float = field(default=298.15,
                              validator=and_(instance_of(float), ge(0)))
    pdbfile: str = field(default="", validator=instance_of(str))
    
    # TODO: combine this with molecular weights provided somewhere else, or using
    # parmed?
    def get_MW(self, pdbfilename):

        """
        Calculate the molecular weight of the solute.
        
        Parameters:
        -----------
        pdbfile: str
            PDB file name containing the heavy atoms of the solute.
        """

        AMU_dict = {
            'H': 1.00794,
            'C': 12.0107,
            'CL': 35.453,
            'N': 14.0067,
            'O': 15.9994,
            'P': 30.973762,
            'S': 32.065,
            'BR': 79.904,
            'I': 126.90447,
            'F': 18.9984032,
        }
        MW = 0
        filelines = open(pdbfilename, "r").readlines()
        for line in filelines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atomname = line[12:16].strip()
                element_one_letter = atomname[0]
                element_two_letters = atomname[:2]
                if element_two_letters in AMU_dict:
                    MW += AMU_dict[element_two_letters]
                elif element_one_letter in AMU_dict:
                    MW += AMU_dict[element_one_letter]
                else:
                    print("WARNING:", atomname, "not recognized." )
        return MW
    
    def write_input(self, input_filename, small=False):

        """
        Write the HYDROpro input file to a plain text.
        
        Parameters:
        -----------
        input_filename : str
            The name of the HYDROpro input file to write.
        small : bool
            Boolean to indicate if the solute is a small
            organic compound. Necessary to set the AER value.
        """
        
        weight = self.get_MW(self.pdbfile)

        print("The Molecular Weight is %.2f \n" % weight)

        file = open(input_filename, 'w')
        file.write(self.pdbfile.split('.')[0].split('\\')[-1]+ "                        !Name of molecule\n")
        file.write(self.pdbfile.split('.')[0].split('\\')[-1]+ "                        !Name for output file\n")
        file.write(self.pdbfile.split('\\')[-1]+ "        !Strucutural (PBD) file\n")
        file.write("1               !Type of calculation\n")
        if small:
            file.write("1.2,            !AER, radius of primary elements\n")
        else:
            file.write("2.9,            !AER, radius of primary elements\n")
        file.write("-1,              !NSIG\n")

        self.temperature = self.temperature - 273.15
        file.write(str(self.temperature)+",            !T (temperature, centigrade)\n")

        if self.temperature >= 19.0 and self.temperature <= 21.0:
            file.write("0.01,           !ETA (Viscosity of the solvent in poises)\n")

        elif self.temperature >= 24.0 and self.temperature <= 26.0:
            file.write("0.0091,         !ETA (Viscosity of the solvent in poises)\n")

        else:
            file.write("0.0091,         !ETA (Viscosity of the solvent in poises)\n")

        file.write(str(weight)+",         !RM (Molecular weight)\n")
        psv = 0.730
        #float(raw_input("Enter the partial specific volume of molecule (cm3/g) : for example 0.702 for lysozyme protein\n"))
        file.write(str(psv)+ ",         !Partial specific volume, cm3/g\n")
        file.write("1.0,            !Solvent density, g/cm3\n")
        file.write("-1              !Number of values of Q\n")
        file.write("-1          !Number of intervals\n")
        file.write("0,              !Number of trials for MC calculation of covolume\n")
        file.write("1               !IDIF=1 (yes) for full diffusion tensors\n")
        file.write("*")
        file.close()
        print("The input file of HYDROpro is stored in the same directory with the name hydropro.dat")

    def read_output(self, output_filename):
        
        """
        Read the HYDROpro output file.
        
        output_filename:
        -----------
        filename : str
            Output file of the HYDROpro calculations, usually
            finishing in *-res.txt
        """

        filelines = open(output_filename,'r').readlines()
        trans = "Translational diffusion coefficient"
        rot = "Rotational diffusion coefficient"

        for line in filelines:
            if trans in line:
                #print("HYDROpro Result for Translational diffusion coefficient:\n")
                #print(line)
                trans_value= float(line[41:50])
                trans_value= trans_value* 10**4
                print("The value of Translational diffusion coefficient to be entered in SDA input file(unit: Ang^2/ps):   %f\n" %(trans_value))

            if rot in line:
                #print("HYDROpro Result for Rotational diffusion coefficient:\n")
                #print(line)
                rot_value= float(line[41:50])
                rot_value= rot_value/10**12
                print("The value of Rotational diffusion coefficient to be entered in SDA input file(unit: radian^2/ps):   %f\n" %(rot_value))
        
        return [trans_value, rot_value]

# TODO: combine this with the equivalent function in browndye modules?
# TODO: move to preparer
def create_ghost_atom_from_atoms_center_of_mass(
        pqr_filename, atom_index_list, new_pqr_filename=None,
        center_molecule=True):
    """
    Add a ghost atom to a PQR file at the location of the center
    of mass of the atoms listed.
    
    Parameters:
    -----------
    pqr_filename : str
        The name of the PQR file where the ghost atom will be added
        into the file itself.
        
    atom_index_list : list
        A list of integers which are atom indices. The center of mass
        will be found for these atoms, and the ghost atom will be 
        located at that center of mass.
        
    new_pqr_filename : str or None
        If None, the pqr_filename will be overwritten. Otherwise, the
        new PQR with the ghost atom will be written to this file.
    """
    
    if new_pqr_filename is None:
        new_pqr_filename = pqr_filename
    pqr_struct = parmed.load_file(pqr_filename, skip_bonds=True)


    # Compute the center of mass for the selected group of atoms
    center_of_mass = np.array([[0., 0., 0.]])
    total_mass = 0.0
    for atom_index in atom_index_list:
        atom_pos = pqr_struct.coordinates[atom_index,:]
        atom_mass = pqr_struct.atoms[atom_index].mass
        if atom_mass == 0.0:
            atom_mass = 0.0001
        center_of_mass += atom_mass * atom_pos
        total_mass += atom_mass
    center_of_mass = center_of_mass / total_mass
    
    if center_molecule:
        print("calculating center of mass")
        # Compute the center of mass of the entire molecule to be transposed
        mol_center_of_mass = np.array([[0., 0., 0.]])
        mol_total_mass = 0.0
        step_range = max(len(pqr_struct.atoms) // 100, 1)
        
        pqr_coordinates = pqr_struct.coordinates
        for atom_index, atom in enumerate(pqr_struct.atoms):
            atom_pos = pqr_coordinates[atom_index,:]
            atom_mass = atom.mass
            if atom_mass == 0.0:
                atom_mass = 0.0001
            mol_center_of_mass += atom_mass * atom_pos
            mol_total_mass += atom_mass
            if atom_index % step_range == 0:
                print(f"{((atom_index + 1) / len(pqr_struct.atoms)) * 100:.0f}%")

        mol_center_of_mass = mol_center_of_mass / mol_total_mass
    ghost_atom = parmed.Atom(name="GHO", mass=0.0, charge=0.0, solvent_radius=0.0)
    ghost_structure = parmed.Structure()
    ghost_structure.add_atom(ghost_atom, "GHO", 1)
    ghost_structure.coordinates = np.array(center_of_mass)

    #pqr_complex = pqr_struct + ghost_structure
    pqr_complex = pqr_struct
    for residue in pqr_complex.residues:
        residue.chain = ""


    if center_molecule:
        print("centering molecule")
        step_range = max(len(pqr_complex.atoms) // 100, 1)
        new_coordinates = np.zeros(pqr_complex.coordinates.shape)
        pqr_coordinates = pqr_complex.coordinates
        for atom_index, atom in enumerate(pqr_struct.atoms):
            new_coordinates[atom_index,:] = pqr_coordinates[atom_index,:] \
                - mol_center_of_mass[0,:]
            
            atom.occupancy = atom.charge
            atom.bfactor = atom.solvent_radius

            if atom_index % step_range == 0:
                print(f"{((atom_index + 1) / len(pqr_complex.atoms)) * 100:.0f}%")
                
        pqr_complex.coordinates = new_coordinates
                

    
    ghost_index = len(pqr_complex.atoms)
    x, y, z = ghost_structure.coordinates[0][:]
    ghost_atom = "ATOM{:>7s} {:4s} {:3s}  {:>4s}{:>12.3f}{:>8.3f}{:>8.3f}".format(
    str(ghost_index), ghost_atom.name, ghost_atom.residue.name, str(ghost_atom.residue.idx), 
    x, y, z)


    pqr_complex.save(new_pqr_filename, overwrite=True, format='pdb')

    return ghost_atom