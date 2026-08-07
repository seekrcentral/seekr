Getting Started
===============

This guide will walk you through the process of installing Seekr. At this time,
seekr has only been tested on Linux systems. Therefore, all installation instructions
are for Linux only.

What is seekr?
--------------

Seekr is a multiscale scientific software framework for estimating molecular kinetics
and thermodynamics using enhanced sampling methods, such as milestoning or
metadynamics.

Install Mamba
-------------

It is recommended, though not mandatory, that you install Mamba (or Conda) to use with
seekr. Without Mamba/Conda, all dependencies will need to be installed by hand, or
from source (see sections below).

If you do not already have Mamba, it can be easily installed by completing the
following steps:

Download Mamba, run the script, and fill out the prompts:

.. code-block:: bash

   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
   bash Miniforge3-$(uname)-$(uname -m).sh

- When asked for an install destination, choose a directory owned by your user account.

- When asked if you'd like to initialize MambaForce, answer "yes".

- Close and reopen the window

Make sure Mamba is installed by running:

.. code-block:: bash

   which mamba

If you want you can create a Mamba environment. Python 3.12 is recommended,
but you can use any version of Python that is supported by the dependencies.

.. code-block:: bash

   mamba create --name SEEKR python=3.12

but you can also just install all packages straight to the base environment
if you wish to. If using an environment, whenever you're installing or running
anything involving OpenMM or seekr, make sure that you have activated your
environment by running ``mamba activate SEEKR``.

Install OpenMM and seekr with Mamba
-----------------------------------
This section describes the fasted and easiest way to get seekr working.

With your environment activated, install OpenMM and seekr with the following command:

.. code-block:: bash

   mamba install openmm

You may wish to specify the cuda version for your pre-installed version.

.. code-block:: bash

   mamba install openmm cuda-nvrtc=##.# cuda-version=##.#

Where, of course, you replace the '##.#' with whatever Cuda version you have
installed, found using ``nvidia-smi`` or other such program.

With OpenMM installed, install seekr with the following command:

.. code-block:: bash

   git clone https://github.com/seekrcentral/seekr.git
   cd seekr
   python -m pip install .


Install Browndye (optional)
---------------------------

If one wishes to use Brownian dynamics to obtain, for instance, k-on estimates,
one can install Browndye. A full installation guide is available at:
https://browndye.ucsd.edu/.

Install SDA (optional)
----------------------

If one wishes to use SDA for Brownian dynamics to compute k-on estimates,
one may use SDA with seekr. SDA is also the only BD program, at present,
that is able to run membrane systems. Please visit the SDA website for
downloads and installation instructions: https://www.h-its.org/downloads/sda7/

Testing (optional)
------------------

You may now test your installation by running the following command within
the seekr/ github repository:

.. code-block:: bash

   pytest

Many options are available for testing that validate, for instance, Browndye or
SDA. There is also a more comprehensive test suite that performs more in-depth
tests, but are normally skipped due to the time it takes to run.

.. code-block:: bash

   pytest seekr/tests --full
   pytest seekr/tests --full -m browndye     # If one installed Browndye
   pytest seekr/tests --full -m sda          # If one installed SDA

OpenMM Installation from Source (optional)
------------------------------------------

Compiling OpenMM from source is tricky, but may be desirable if the Mamba/Conda
installation doesn't work, or if you wish to optimize OpenMM's performance.

Please see the official
`OpenMM guide to installing from source <http://docs.openmm.org/latest/userguide/library.html#compiling-openmm-from-source-code>`_
for complete OpenMM installation instructions.

If you want to use a GPU to accelerate your OpenMM simulations (highly
recommended) you must ensure that a recent version of CUDA is installed and
loaded.

If you need to install CUDA, it is highly recommended that you contact your
system administrator about this, although if you have to install CUDA by
yourself, you should carefully read and follow all instructions from
`NVIDIA's CUDA toolkit installation instructions
<https://developer.nvidia.com/cuda-toolkit>`_ or
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
.

Many times, cuda is located in /usr/local/cuda:

.. code-block:: bash

   ls /usr/local/cuda

If CUDA is located here, then the OpenMM plugin should be able to automatically
detect it there. If CUDA is not in /usr/local/cuda, then you can also sometimes
find the CUDA compile 'nvcc' using 'which'. You can also see whether the
CUDA_HOME environmental variable is defined:

.. code-block:: bash

   which nvcc
   echo $CUDA_HOME

If the commands didn't return a path to nvcc, or a value or CUDA_HOME, seekr
is likely to have difficulty finding CUDA on it's own. You may have to take
more trouble to explicitly assign the necessary variables to the cmake or
ccmake commands.

In order to use CUDA, you may also need to define the following environmental
variable by placing it in your .bashrc file:

.. code-block:: bash

   export OPENMM_CUDA_COMPILER=/path/to/nvcc

Obviously, you'll need to modify "/path/to/nvcc" with the actual path. The
program "nvcc" will exist in your CUDA installation, and might be discoverable
by typing ``which nvcc``.

Next, install the necessary programs and packages into Mamba/Conda. (Replace
"mamba" with "conda" wherever necessary)

.. code-block:: bash

   mamba install numpy scipy netcdf4 mpi4py swig

Make sure 'git' is installed, if not already.

.. code-block:: bash

   mamba install git

Make sure 'ccmake' is installed

.. code-block:: bash

   which ccmake

If nothing happens, you may need to ask your system administrator to install
ccmake or you can install it yourself if you have sudo privileges:

.. code-block:: bash

   sudo apt-get install cmake-curses-gui

Make sure 'doxygen' is installed.

.. code-block:: bash

   mamba install -c conda-forge doxygen

Upgrade Cython:

.. code-block:: bash

   pip install --upgrade cython

Clone OpenMM and cd into OpenMM directory, then perform necessary build steps.

.. code-block:: bash

   git clone https://github.com/openmm/openmm.git
   cd openmm
   mkdir build
   cd build
   ccmake ..

The ccmake gui should come up. Press 'c' and then 't'

You should modify the following variables:

CMAKE_INSTALL_PREFIX: change to a local directory that exists (for example:
/home/USERNAME/bin/openmm). If such a directory doesn't exist, then make one.
You can also leave this variable at the default if you have sudo privileges
and don't mind installing OpenMM globally.

Check all the variables, and then type 'c' to configure. If there are any
problems, it will let you know.

When the configuration is successful, type 'g' to generate. Then ccmake
should close on its own.

If you are having trouble with assigning a variable, like
CUDA_CUDA_LIBRARY-NOTFOUND, then run 'cmake' (instead of 'ccmake') and
assign the missing variable using the -D argument:

For example:

.. code-block:: bash

   cmake -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs ..

Next, build, install, and test OpenMM:

.. code-block:: bash

   make
   make install
   make PythonInstall
   python -m openmm.testInstallation

If the PythonInstall step fails, then make sure you have upgraded cython

.. code-block:: bash

   pip install --upgrade cython

Next one may run a couple of quick tests to see if OpenMM's CUDA platform is
working:

.. code-block:: bash

   ./TestCudaLangevinMiddleIntegrator

And within the python interpreter:

.. code-block:: python

   import openmm
   openmm.Platform.getPlatformByName('CUDA')

One may also run comprehensive tests within the build/ directory to test
all aspects of OpenMM's installation:

.. code-block:: bash

   make test

Hopefully, if everything went according to plan, all the tests pass.
If a few fail, then determine if those failures will be necessary for
your calculations. If several or all fail, then you'll need to be sure that
you fix whatever problem caused those failures. If the CUDA tests failed,
then you either do not have a working CUDA installation, or the proper
environmental variables such as OPENMM_CUDA_COMPILER have not been set.

Finally, one must install seekr:

.. code-block:: bash

   cd
   git clone https://github.com/seekrcentral/seekr.git
   cd seekr
   python -m pip install .

OpenMM Installation from Source on Cluster or Shared Resource
-------------------------------------------------------------

A simple Mamba/Conda installation on a Cluster or Supercomputer would probably
work just fine, but if you wish to install from source, this section provides
some helpful information to that end.

Installation of OpenMM on a shared resource is almost identical to the
local installation of OpenMM as detailed in the previous section. However, the
shared resource is likely to have a number of specific features that will have
to be taken into account when installing OpenMM from source.

Some tips and advice:

* You should run all installation commands in an interactive node to avoid
  clogging up the login nodes. Consider using the debug or development queue,
  if available.

* Use "wget" to obtain mamba:

  .. code-block:: bash

     wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh

* If the cluster or shared resource has GPU computing capabilities, the
  administrators have likely made CUDA available. You should consult the
  resource's manual or reach out to the system administrators for how to
  load or utilize CUDA.

* If 'ccmake' is not available, you can still use 'cmake' to install OpenMM,
  you just must provide any arguments using '-D'. For instance:

  .. code-block:: bash

     cmake -DCMAKE_INSTALL_PREFIX=/path/to/openmm -DCMAKE_LIBRARY_PATH=/path/to/cuda/lib64/stubs ..
