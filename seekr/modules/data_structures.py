"""
data_structures.py

Contain data structure classes used for SEEKR inputs and models.
"""

#from seekr import __version__

# Idea list
# - Within blocks, where the substructure is unclear, add an
#    attribute keyword that allows the substructure to be correctly
#    identified and parsed.
# - Add a logger that writes a SYNOPSIS file that keeps track of
#    all procedures that occur, as well as pointers towards more
#    detailed log files
# - Username, hostname, datetime, version number, all included in 
#    model and maybe SYNOPSIS.

from attrs import define

@define
class SeekrModel:
    seekr_version: str

if __name__ == "__main__":
    print(__version__)