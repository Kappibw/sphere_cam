import numpy as np
import torch

class TangentialWarp:
    """
    A class for applying tangential warping and unwarping transformations on coordinates.

    Warping has the effect of barrel distoring the coordinates, which is useful for mapping cube 
    faces to spheres. Unwarping is the inverse operation, which restores the original coordinates.

    To see the effect of the warping transformation, use the vizualize_warp function in the visualizations.py module.

    Attributes
    ----------
    Theta : float
        Constants defining classic and optimized theta values for different warping transformations.

    Methods
    -------
    warp(cube_face_coordinates)
        Applies a tangential warp to map cube face coordinates to UV coordinates.
    unwarp(quad_sphere_coordinates)
        Applies an inverse tangential warp to retrieve cube face coordinates from UV coordinates.
    """
    
    class Theta:
        """Holds theta values for classic and optimized transformations."""
        classic = np.pi / 4
        optimized = 0.8687

    def __init__(self, theta=Theta.classic):
        """
        Initializes the TangentialWarp class with a specified theta value.

        Parameters
        ----------
        theta : float
            A free parameter > 0, representing the warp scaling factor. It can be set to one of
            the predefined constants in TangentialWarp.Theta.
        """
        self.theta = theta
        self.tan_theta = np.tan(self.theta)  # Precompute tan(theta) for optimized calculations.

    def warp(self, cube_face_coordinates):
        """
        Applies a tangential warp transformation to cube face coordinates.

        This function transforms the coordinates from the cube face to quad sphere coordinates using a
        tan(theta * x) / tan(theta) formula. The warp operation serves to normalize and
        project the coordinates while preserving angular information.

        Parameters
        ----------
        cube_face_coordinates : torch.Tensor
            A tensor of coordinates in the cube face domain, where values are expected to range from -1 to 1.

        Returns
        -------
        quad_sphere_coordinates : torch.Tensor
            The warped coordinates mapped to the quad sphere domain.
        """
        
        quad_sphere_coordinates = torch.tan(cube_face_coordinates * self.theta) / self.tan_theta
        return quad_sphere_coordinates

    def unwarp(self, quad_sphere_coordinates):
        """
        Applies the inverse tangential warp to retrieve cube face coordinates from UV coordinates.

        The function reverts the quad sphere coordinates back to the original cube face coordinates by
        performing an arctan transformation on the UV coordinates.

        Parameters
        ----------
        quad_sphere_coordinates : torch.Tensor
            A tensor of quad sphere domain coordinates.

        Returns
        -------
        cube_face_coordinates : torch.Tensor
            The unwrapped coordinates mapped back to the cube face domain.
        """
        cube_face_coordinates = torch.arctan(self.tan_theta * quad_sphere_coordinates) / self.theta
        return cube_face_coordinates
