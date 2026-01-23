"""
modules/collective_variables/base.py

This module contains the base classes for collective variables and
any other common structures or functions related to CVs.
"""

import os
import json
import typing

import cattrs

import seekr.modules.base as base

class Base_CV:
    """
    Base class for all collective variables.
    """
    def save(
            self, 
            directory: str
            ) -> None:
        """
        Save the collective variable to a file.
        """
        converter: cattrs.Converter = cattrs.Converter()
        cvs_dict: dict = converter.unstructure(self)
        cvs_dict_noindent: dict = base.wrap_noindent_integers(cvs_dict)
        json_dump: str = json.dumps(cvs_dict_noindent, indent=4, 
                                    cls=base.NoIndentListEncoder)
        
        filename = os.path.join(directory, self.name + ".json")
        with open(filename, "w") as file:
            file.write(json_dump)
        return

class Base_CV_input(typing.Protocol):
    """
    Base class for all collective variable inputs.
    """
    pass