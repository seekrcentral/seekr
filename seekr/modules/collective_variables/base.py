"""
collective_variables/base.py

Base classes for collective variables, states, and anchors.
"""

import typing

from attrs import define, field, validators

# TODO: make a base class for CVs? - it could contain potentially
#  specialized functions for each CV.

if typing.TYPE_CHECKING:
    import seekr.modules.data_structures as data_structures

@define
class CV_input_2groups:
    """
    Distance CV input involving two groups of atoms.
    
    Attributes:
    -----------
    _index : int
        The index of this CV input object in the Model_input object.
        computed automatically - not entered by user.
        
    group1_name : str
        The string name of the first group of atoms, which will
        correspond to a Selection within the MD and BD settings,
        either systemwide or within an anchor
        
    group2_name : str
        The string name of the first group of atoms, which will
        correspond to a Selection within the MD and BD settings,
        either systemwide or within an anchor
        
    """
    _index: int = field(default=0,
                          validator=validators.instance_of(int))
    group1_name: str = field(default="",
                             validator=validators.instance_of(str))
    group2_name: str = field(default="",
                             validator=validators.instance_of(str))
    
    def check(
            self,
            input_anchors: list["data_structures.Anchor_input"]) -> None:
        # TODO: fill out generic checks, and implement in derived classes
        #  if necessary.
        pass
    
    """
    # TODO: move check to model CV object - and include in a base class?
    def check(self):
        ""
        Check user inputs to ensure they have been entered properly.
        ""
        
        last_radius = -1e9
        found_bulk_anchor = False
        
        if self.input_anchors is None:
            return
        
        assert len(self.group1) > 0, "Any input CV groups must contain atoms."
        assert len(self.group2) > 0, "Any input CV groups must contain atoms."
        
        for i, input_anchor in enumerate(self.input_anchors):
            assert input_anchor.__class__.__name__ == "Spherical_cv_anchor"
            radius = input_anchor.radius
            assert radius >= 0.0, "A radius must be greater than "\
                "or equal to zero."
            assert radius > last_radius, "Each subsequent anchor "\
                "radius must be greater than the last (sorted)."
            
            if input_anchor.bound_state is None:
                input_anchor.bound_state = False
            
            assert input_anchor.bound_state in [True, False], \
                "bound_state must be a boolean"
                
            if input_anchor.bulk_anchor is None:
                input_anchor.bulk_anchor = False
                
            assert input_anchor.bulk_anchor in [True, False], \
                "bulk_anchor must be a boolean"
            
            if input_anchor.bulk_anchor:
                assert not found_bulk_anchor, "Only one bulk anchor allowed "\
                    "per set of anchors in a CV."
                found_bulk_anchor = False
            else:
                assert not found_bulk_anchor, "Only the outermost anchor "\
                    "should be the bulk anchor."
            
            if i > 0:
                assert not input_anchor.bound_state, "Only the lowest"\
                    "anchor can be the bound state."
                    
            assert len(self.input_anchors) > 1, "A CV must contain "\
                "more than one anchor."
                
            last_radius = radius
        
        return
    
    def make_mmvt_milestone_between_two_anchors(
            self, anchor1, anchor2, input_anchor1, input_anchor2, 
            milestone_index):
        neighbor_index1 = anchor2.index
        neighbor_index2 = anchor1.index        
        milestone1 = base.Milestone()
        milestone1.index = milestone_index
        milestone1.neighbor_anchor_index = neighbor_index1
        milestone1.alias_index = len(anchor1.milestones)+1
        milestone1.cv_index = self.index
        if input_anchor1.upper_milestone_radius is None:
            value = 0.5 * (input_anchor1.radius + input_anchor2.radius)
        else:
            value = input_anchor2.upper_milestone_radius
        milestone2 = base.Milestone()
        milestone2.index = milestone_index
        milestone2.neighbor_anchor_index = neighbor_index2
        milestone2.alias_index = len(anchor2.milestones)+1
        milestone2.cv_index = self.index
        if input_anchor2.lower_milestone_radius is None:
            value = 0.5 * (input_anchor1.radius + input_anchor2.radius)
        else:
            value = input_anchor2.lower_milestone_radius
        # assign k
        milestone1.variables = {"k": 1.0, "radius": value}
        milestone2.variables = {"k": -1.0, "radius": value}
        for milestone in anchor1.milestones:
            if milestone.neighbor_anchor_index == neighbor_index1:
                # Then anchor1 already has this milestone added to it
                return milestone_index
        anchor1.milestones.append(milestone1)
        anchor2.milestones.append(milestone2)
        milestone_index += 1
        return milestone_index
    
    """