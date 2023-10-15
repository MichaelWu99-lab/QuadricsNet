
import numpy as np

from src.primitive_forward import initialize_sphere_model,initialize_plane_model, initialize_cylinder_model, initialize_cone_model,forward_sphere,forward_plane,forward_cylinder,forward_cone

import torch
EPS = np.finfo(np.float32).eps

class FittingModule:
    def __init__(self, config):
        # get routine for the primitive prediction
        # self.ellipsoid_decoder = initialize_ellipsoid_model(config)
        self.sphere_decoder = initialize_sphere_model(config)
        self.plane_decoder = initialize_plane_model(config)
        self.cylinder_decoder = initialize_cylinder_model(config)
        self.cone_decoder = initialize_cone_model(config)
        
        self.parameters = {}

    def forward_pass_sphere(self, points,normals, ids, weights = None,if_fitting_normals=0):
        points = torch.unsqueeze(points, 0)
        normals = torch.unsqueeze(normals, 0)
        quadrics,trans_inv,C = forward_sphere(
            points,normals, self.sphere_decoder, weights=weights,if_fitting_normals=if_fitting_normals)
        self.parameters[ids] = ["sphere", quadrics,"trans_inv",trans_inv,"C",C]

        return quadrics,trans_inv,C
        
    def forward_pass_plane(self, points,normals, ids, weights = None, if_fitting_normals=0):
        points = torch.unsqueeze(points, 0)
        normals = torch.unsqueeze(normals, 0)
        quadrics,trans_inv,C = forward_plane(
            points,normals, self.plane_decoder, weights=weights,if_fitting_normals=if_fitting_normals)
        self.parameters[ids] = ["plane", quadrics,"trans_inv",trans_inv,"C",C]

        return quadrics,trans_inv,C
            
    def forward_pass_cylinder(self, points,normals, ids, weights = None, if_fitting_normals=0):
        points = torch.unsqueeze(points, 0)
        normals = torch.unsqueeze(normals, 0)
        quadrics,trans_inv,C = forward_cylinder(
            points,normals, self.cylinder_decoder, weights=weights,if_fitting_normals=if_fitting_normals)
        self.parameters[ids] = ["cylinder", quadrics,"trans_inv",trans_inv,"C",C]

        return quadrics,trans_inv,C
            
    def forward_pass_cone(self, points,normals, ids, weights = None, if_fitting_normals=0):
        points = torch.unsqueeze(points, 0)
        normals = torch.unsqueeze(normals, 0)
        quadrics,trans_inv,C = forward_cone(
            points,normals, self.cone_decoder, weights=weights,if_fitting_normals=if_fitting_normals)
        self.parameters[ids] = ["cone", quadrics,"trans_inv",trans_inv,"C",C]

        return quadrics,trans_inv,C
    
