seekr
=====

| **Status** | [![Github Actions Status](https://github.com/seekrcentral/seekr/workflows/CI/badge.svg)](https://github.com/seekrcentral/seekr/actions?query=workflow%3ACI)  [![Codecov coverage](https://codecov.io/gh/seekrcentral/seekr/branch/main/graph/badge.svg)](https://codecov.io/gh/seekrcentral/seekr/branch/main) |
| **Latest Release** | [![Last release tag](https://img.shields.io/github/v/release/seekrcentral/seekr.svg)](https://github.com/seekrcentral/seekr/releases)
| **Communication** | [![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://seekr.readthedocs.io/) [![user & dev discussions](https://img.shields.io/badge/user%20%26%20dev%20discussions-GitHub-red?logo=github)](https://github.com/seekrcentral/discussions/discussions) |
| **Platforms** | [![license](https://img.shields.io/github/license/seekrcentral/seekr.svg)](https://opensource.org/licenses/MIT) [![platforms](https://img.shields.io/badge/Platforms-Linux-orange.svg)](https://seekr.readthedocs.io/en/stable/installation.html) [![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://seekr.readthedocs.io/en/stable/installation.html)|
| **Installation** | [![Releases](https://img.shields.io/badge/obtain-latest-green.svg)](https://github.com/seekrcentral/seekr/releases) [![Conda](https://img.shields.io/conda/v/conda-forge/seekr.svg)](https://anaconda.org/conda-forge/seekr) [![Last updated](https://anaconda.org/conda-forge/seekr/badges/latest_release_relative_date.svg)](https://anaconda.org/conda-forge/seekr) [![Anaconda Cloud downloads](https://anaconda.org/conda-forge/seekr/badges/downloads.svg)](https://anaconda.org/conda-forge/seekr)          |


# Simulation-Enabled Estimation of Kinetic Rates

A multiscale scientific software package to compute molecular thermodynamics 
and kinetics.

Prepare, run, and analyze enhanced sampling methods (like milestoning or 
metadynamics) using molecular dynamics engines (like OpenMM), Brownian dynamics engines (like Browndye or SDA), and more to obtain the kinetics and thermodynamics of molecular processes
such as: ligand-receptor binding and unbinding, domain motion, toy systems, and more.

## Documentation

This README is only a quickstart to get seekr up and running as quickly as possible. Please
see the documentation in order to see more detailed 
[installation](https://seekr.readthedocs.io/en/stable/installation.html) and 
[tutorials](https://seekr.readthedocs.io/en/stable/installation.html) tutorials.

## Quick Install

It is recommended to install seekr using the Mamba package manager (conda may also
be used). If you don't already have Mamba installed, complete the following commands 
before proceeding.

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

One may then create and activate a Mamba environment for running seekr.

```
mamba create --name SEEKR python=3.12
mamba activate SEEKR
```

### Dependencies

Many of the dependencies for seekr will be installed alongside seekr, but some must
be installed separately, before seekr.

#### OpenMM (recommended)

OpenMM is strongly recommended when using the molecular dynamics scale of seekr.
OpenMM is also required for the use of "toy systems". While not strictly
required, any normal use of seekr will require OpenMM.

```
mamba install openmm
```

You may wish to specify the cuda version for your pre-installed version.

```
mamba install openmm cuda-nvrtc=##.# cuda-version=##.#
```

Where, of course, you replace the '##.#' with whatever Cuda version you have
installed, found using `nvidia-smi` or other such program.

#### Browndye (optional; may be used for k-on calculations)

Seekr can use Browndye if Brownian dynamics simulations will be run 
(necessary for k-on calculations). Please see (https://browndye.ucsd.edu/) 
for Browndye2 installation instructions. Cannot be used, at present,
for membrane-containing systems.

#### SDA (optional; may be used for k-on calculations, including on membrane systems)

Seekr may alternatively use SDA if Brownian dynamics (BD) simulations will 
be run (necessary for k-on calculations). Please see 
(https://www.h-its.org/downloads/sda7/) for SDA installation instructions.

### Install seekr

Once the dependencies are installed, one may install seekr. First, clone
the Github repository and install the package:

```
git clone https://github.com/seekrcentral/seekr.git
cd seekr
python -m pip install .
```

### Testing seekr (optional)

To test the seekr installation, run the following command in the seekr/
directory:

```
pytest
```

Additional tests may be run that, for instance, incorporate Browndye and
SDA. Please see the README.md in seekr/tests/.

## Run

A seekr calculation may be run in a number of ways. Most likely, one would
use [seekrflow](https://github.com/seekrcentral/seekrflow), a package for
preparing, launching, and monitoring seekr calculations more easily. Yet, 
one may use seekr alone for more customizable calculations using either 
a Model  JSON file, or by using the seekr API. The following commands 
demonstrate the use of the Model input JSON file and only requires OpenMM. 
From within the seekr/seekr directory, execute:

```
python prepare.py data/hostguest_sample_input.json
```

This will create a folder "test_seekr" in your home directory. You
may then execute the following commands to complete the four stages of
the calculation:

```
python run.py ~/test_seekr/model.json 1
python run.py ~/test_seekr/model.json 2
python run.py ~/test_seekr/model.json 3
python run.py ~/test_seekr/model.json 4
```

Before, during, or after any of the above stages are running, you may see 
details about the status of the seekr calculation using status.py:

```
python status.py ~/test_seekr/model.json
```

When the stages are complete, one may perform the final checks and
analysis:

```
python analyze.py ~/test_seekr/model.json
```

You will see the analysis printed to the screen. If you're curious, the 
experimentally-measured k-off for this compound is 3.8e8 1/s.
This calculation is artificially truncated for demonstration purposes -
a true seekr calculation should simulate much longer. The 
generated images can be seen in ~/test_seekr/images_and_plots.

### Important Options and Hints

* In general, seekr programs can be run with the '-h' argument to see all
available options. Please see https://seekr.readthedocs.io/en/latest for a
detailed description of programs and options.

* The **run.py** and **prepare.py** programs also take a '-f' option for
force overwrite. Without this argument, a previously run stage will not
be overwritten.

* When a setting to the model must be modified, it is usually necessary to 
make the change in the **input file** and not the **model file**, and then re-run 
prepare.py to make a new model file. In some exceptions, the model file may be
directly (and carefully) modified.

## Authors and Contributors

The following people have contributed directly to the coding and validation
efforts of seekr (listed an alphabetical order of last name). 
Thanks also to everyone who has helped or will help improve this project by 
providing feedback, bug reports, or other comments.

* Rommie Amaro (principal investigator)
* Ilker Deveci (developer)
* Hilliary Frank (contributor)
* Sasha Heyneman (contributor)
* Ben Jagger (developer)
* Abraham Muniz-Chicharro (developer)
* Anson Noland (developer)
* Anand Ojha (developer)
* Andy Stokely (developer)
* Natalie Timms (contributor)
* Lane Votapka (lead developer)
* Jeff Wagner (contributor)

### Citing seekr and Dependencies

The seekr v.3 manuscript is in preparation, so there is no manuscript yet to cite,
but feel free to cite the seekr v.2 paper for the time being:

* Votapka, L. W.; Stokely, A. M.; Ojha, A. A.; Amaro, R. E. SEEKR2: Versatile Multiscale Milestoning Utilizing the OpenMM Molecular Dynamics Engine. J. Chem. Inf. Mod. 2022 62 (13), 3253-3262. DOI: 10.1021/acs.jcim.2c00501

One should also cite seekr's dependencies:

* Van Der Walt, S., Colbert, S.C. & Varoquaux, G., 2011. The NumPy array: a structure for efficient numerical computation. Computing in Science & Engineering, 13(2), pp.22–30.

* Jones, E. et al., 2001. SciPy: Open source scientific tools for Python.

* Hunter, J.D., 2007. Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), pp.90–95.

* T.D. Swinburne and D.J. Wales, Defining, Calculating, and Converging Observables of a Kinetic Transition Network, J. Chemical Theory and Computation (2020), https://doi.org/10.1021/acs.jctc.9b01211

You may also optionally cite the following papers related to seekr:

* Ojha, A. A., Votapka L. W., Amaro, R. E. QMrebind: incorporating quantum mechanical force field reparameterization at the ligand binding site for improved drug-target kinetics through milestoning simulations. Chemical Science 14 (45), 13159-13175

* Ojha A. A., Srivastava A., Votapka L. W., and Amaro R. E. Selectivity and Ranking of Tight-Binding JAK-STAT Inhibitors Using Markovian Milestoning with Voronoi Tessellations. Journal of Chemical Information and Modeling 2023 63 (8), 2469-2482. DOI: 10.1021/acs.jcim.2c01589

* Votapka, L. W.; Jagger, B. R.; Heyneman, A. L.; Amaro, R. E. SEEKR: Simulation Enabled Estimation of Kinetic Rates, A Computational Tool to Estimate Molecular Kinetics and Its Application to Trypsin–Benzamidine Binding. J. Phys. Chem. B 2017, 121 (15), 3597–3606. https://doi.org/10.1021/acs.jpcb.6b09388. 

* Jagger, B. R.; Ojha, A. A.; Amaro, R. E. Predicting Ligand Binding Kinetics Using a Markovian Milestoning with Voronoi Tessellations Multiscale Approach. J. Chem. Theory Comput. 2020. https://doi.org/10.1021/acs.jctc.0c00495. 

* Jagger, B. R.; Lee, C. T.; Amaro, R. E. Quantitative Ranking of Ligand Binding Kinetics with a Multiscale Milestoning Simulation Approach. J. Phys. Chem. Lett. 2018, 9 (17), 4941–4948. https://doi.org/10.1021/acs.jpclett.8b02047. 

* Votapka LW, Amaro RE (2015) Multiscale Estimation of Binding Kinetics Using Brownian Dynamics, Molecular Dynamics and Milestoning. PLOS Computational Biology 11(10): e1004381. https://doi.org/10.1371/journal.pcbi.1004381

### Copyright

Copyright (c) 2026, Lane Votapka

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.