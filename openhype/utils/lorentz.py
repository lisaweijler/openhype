# The functions in this files are taken and adapted from the MERU codebase
# https://github.com/facebookresearch/meru

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

from einops import rearrange
import torch
from torch import Tensor


def get_interpolated_hyperbolic_features(
    hyper_feats,  # features on the hyperboloid, space components
    max_dist: float = 11.1,  # asinh(2**15) ≈ 11.09, capping of our lorentz latent space
    steps: int = 20,
    curv: float = 1.0,
    return_mask: bool = False,  # return mask for extrapolated features (e.g. for visualization)
):
    """Expects hyper feats alread flattented ((h w) d) not (h w d)"""

    orignial_dtype = hyper_feats.dtype
    hyper_feats = hyper_feats.to(torch.float64)
    step_size = max_dist / float(steps)
    # get weights - each feature depending on distance to origin has different weight
    dist_to_origin_hyper = dist_to_origin(hyper_feats).squeeze(
        -1
    )  # same as torch.norm(feats)
    # int_weights
    # value that is used to multiply the mask feature with,
    # coming form mask-dist-to-origin * ? = step_size
    inter_weight = step_size / dist_to_origin_hyper

    inter_weight_levels = torch.stack(
        [i * inter_weight for i in range(1, steps + 1)]
    ).to(
        hyper_feats.device
    )  # each row are interpolation weights for each mask per level
    # row 0 is first level, row 1 is second level etc (starting from root/origin)

    # map to tangent space and do interpolation there
    feats = log_map0(hyper_feats, curv)
    root_feat = torch.zeros(
        *(feats.shape), device=hyper_feats.device, dtype=hyper_feats.dtype
    )
    # equidistant steps per feature
    interp_feats = [
        torch.lerp(root_feat, feats, inter_weight_levels[i, :].unsqueeze(1))
        for i in range(steps)
    ]

    # root first, "boundary" of space last
    interp_feats = torch.stack(interp_feats, dim=1)  # n_masks x steps x dim

    # set first one >=1 to 1 rest to nan
    # features beyond (= further away from origin than) the original feature are called extrapolated features
    mask_extrapolated = inter_weight_levels >= 1.0  # n_steps x n_masks
    first_idx = torch.argmax(mask_extrapolated.int(), dim=0)

    interp_feats[(torch.arange(interp_feats.shape[0]), first_idx)] = (
        feats  # set first extrapolated to original feat
    )

    # interp_feats shape n_masks x n_steps x dim

    # flip to have boundary first and root last.. so index 0 is the step at the edge of hyperb. space corresponding to small masks
    interp_feats = interp_feats.flip(1)
    interp_feats = rearrange(interp_feats, "n s d -> (n s) d")
    interp_feats = exp_map0(interp_feats, curv)  # Lift on the Hyperboloid

    if return_mask:
        return interp_feats.to(orignial_dtype), mask_extrapolated.T.flip(1)
    return interp_feats.to(orignial_dtype)


def dist_to_origin_given_xtime(
    x_time: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between a batch of points and the origin on
    the hyperboloid when time component is already given.

    Args:
        x_time: Tensor of shape `(B1)` giving time components of a batch
            of point on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1)` giving pairwise distance along the geodesics
        connecting the input points.
    """
    orig_time = torch.sqrt(1 / curv)
    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * -(orig_time * x_time)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def dist_to_origin(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Compute the pairwise geodesic distance between a batch of points and the origin on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1)` giving pairwise distance along the geodesics
        connecting the input points.
    """
    orig_time = (1 / curv) ** 0.5
    x_time = get_time_comp(x, curv)
    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * -(orig_time * x_time)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def get_time_comp(x: Tensor, curv: float | Tensor = 1.0):
    """
    Compute time component of a batch of space components of vectors
    on the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B1)` giving time component to given space components.
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    return x_time


def lorentz_norm_time_like(x: Tensor, curv: float | Tensor = 1.0):
    sq_norm_x = torch.sum(x**2, dim=-1, keepdim=True)

    # Compute time coordinates
    x_time = torch.sqrt(1 / curv + sq_norm_x)

    # Compute the Lorentzian inner product
    xyl = sq_norm_x - x_time**2

    return torch.sqrt(xyl)


def lorentz_norm(
    x: Tensor, curv: float | Tensor = 1.0, including_time_comp: bool = False
):
    if including_time_comp:
        sq_norm_x = torch.sum(x[:, 1:] ** 2, dim=-1, keepdim=True)
        x_time = x[:, 0].unsqueeze(-1)
    else:
        sq_norm_x = torch.sum(x**2, dim=-1, keepdim=True)

        # Compute time coordinates
        x_time = torch.sqrt(1 / curv + sq_norm_x)

    # Compute the Lorentzian inner product
    xyl = sq_norm_x - x_time**2

    if torch.all(xyl >= 0):  # space-like
        return torch.sqrt(xyl)
    elif torch.all(xyl < 0):  # time-like
        return torch.sqrt(-xyl)
    else:
        raise ValueError(
            f"some vectors are space-like and some are time-like in the given batch."
        )


def rowwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute rowwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B)` giving pairwise Lorentzian inner product
        between input vectors. input tensors must have same number of vextors
    """

    # x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    # y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    # xyl = x @ y.T - x_time @ y_time.T
    # # xyl =  torch.einsum('bnd,bkd->bnk', x, y) - torch.einsum('bnd,bkd->bnk', x_time, y_time)
    # Compute squared norm efficiently
    sq_norm_x = torch.sum(x**2, dim=-1, keepdim=True)
    sq_norm_y = torch.sum(y**2, dim=-1, keepdim=True)

    # Compute time coordinates
    x_time = torch.sqrt(1 / curv + sq_norm_x)
    y_time = torch.sqrt(1 / curv + sq_norm_y)

    # Compute the Lorentzian inner product
    xyl = torch.sum(
        x * y, dim=-1, keepdim=True
    )  # element wise mutliplication an summing
    xyl = xyl - x_time * y_time

    return xyl


def pairwise_inner_meru(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    # x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    # y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    # xyl = x @ y.T - x_time @ y_time.T
    # # xyl =  torch.einsum('bnd,bkd->bnk', x, y) - torch.einsum('bnd,bkd->bnk', x_time, y_time)
    # make sure shape as length 2 - eg if only one vector is given, other cases not covered here
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    # Compute squared norm efficiently
    sq_norm_x = torch.sum(x**2, dim=-1, keepdim=True)
    sq_norm_y = torch.sum(y**2, dim=-1, keepdim=True)

    # Compute time coordinates
    x_time = torch.sqrt(1 / curv + sq_norm_x)
    y_time = torch.sqrt(1 / curv + sq_norm_y)

    # Compute the Lorentzian inner product
    xyl = torch.mm(x, y.transpose(0, 1))
    xyl = xyl - x_time * y_time.transpose(0, 1)

    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def rowwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the rowwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B)` giving rowwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * rowwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(
        rc_xnorm, min=eps, max=math.asinh(2**15)
    )  # this max max=math.asinh(2**15) gives the max distance of points to origin of around 11.09...
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def pairwise_oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B1, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of shape `(B2,D)` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B1, B2)` giving the pairwise required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1)).unsqueeze(
        -1
    )  # column vector
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1)).unsqueeze(0)  # row vector

    # Calculate lorentzian inner product multiplied with curvature.
    c_xyl = curv * pairwise_inner(x, y, curv=curv)  # B1 x B2

    # Make the numerator and denominator for input to arc-cosh, shape: (B1, B2)
    # acos_numer = y_time + c_xyl * x_time
    acos_numer = y_time + torch.mul(c_xyl, x_time)  # c_xyl: B1 x B2, x_time: B1
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (
        torch.mul(torch.norm(x, dim=-1).unsqueeze(-1), acos_denom) + eps
    )
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def rowwise_oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle
