"""The displacements of the modes at points (hb.Model.mode_displacements): the rigid modes
about R in body axes and the generalized modes, for a placed and yawed body."""

import numpy as np
import pytest

from bodies import surface
from meshes import box_panels
from problems import model


def test_mode_displacements_rigid_and_generalized():
    mesh = surface(box_panels(2.0, 1.0, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz")
    bend = {"name": "bend", "pieces": [{"w": [[0.25, 2, 0, 0], [-1.0, 0, 0, 0]]}], "parity": "heave"}
    body = {"mesh": mesh, "position": [10.0, 5.0, 0.0], "yaw": 90.0, "rotation_center": [1.0, 0.0, -0.2],
            "generalized_modes": [bend]}  # fmt: skip
    m = model(body, [1.0], (), options={"diffraction": False})
    pn = m.panels()
    u = m.mode_displacements(pn.centroids, pn.body, pn.element)
    assert u.shape == (len(pn.element), 7, 3)
    # body-local points and the rotation R(90 deg): x_global = (10 - y, 5 + x, z)
    c = pn.centroids
    xl = np.c_[c[:, 1] - 5.0, 10.0 - c[:, 0], c[:, 2]]
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for j in range(3):
        e = np.eye(3)[j]
        assert np.allclose(u[:, j], rot @ e)
        assert np.allclose(u[:, 3 + j], np.cross(e, xl - [1.0, 0.0, -0.2]) @ rot.T)
    w = 0.25 * xl[:, 0] ** 2 - 1.0
    assert np.allclose(u[:, 6], np.c_[0 * w, 0 * w, w])
    with pytest.raises(ValueError, match="does not exist"):
        m.mode_displacements(c[:1], [0], [999])
