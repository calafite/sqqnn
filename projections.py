from typing import Tuple, NamedTuple
import numpy as np

class TrigState(NamedTuple):
    sa: np.ndarray
    ca: np.ndarray
    sb: np.ndarray
    cb: np.ndarray
    sg: np.ndarray
    cg: np.ndarray
    st: float
    ct: float
    sw: float
    cw: float

class ProjectionOperations:
    @staticmethod
    def make_state(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, theta: float, omega: float) -> TrigState:
        # caches trigonometric evaluations
        return TrigState(
            np.sin(alpha), np.cos(alpha),
            np.sin(beta), np.cos(beta),
            np.sin(gamma), np.cos(gamma),
            float(np.sin(theta)), float(np.cos(theta)),
            float(np.sin(omega)), float(np.cos(omega))
        )

    @staticmethod
    def _get_vw(s: TrigState) -> Tuple[np.ndarray, np.ndarray]:
        # groups data dependent features
        v1: np.ndarray = s.cb
        v2: np.ndarray = -s.ca * s.sb
        v3: np.ndarray = -s.sb * s.cg
        v4: np.ndarray = s.sa * s.sg
        v5: np.ndarray = -s.ca * s.cb * s.cg
        v_mat: np.ndarray = np.array([v1, v2, v3, v4, v5])

        # groups parameter dependent weights
        w1: float = s.ct * s.cw
        w2: float = s.st * s.cw
        w3: float = s.ct * s.sw
        w4: float = s.st * s.sw
        w5: float = s.st * s.sw
        w_vec: np.ndarray = np.array([w1, w2, w3, w4, w5])

        return v_mat, w_vec

    @staticmethod
    def compute_output(s: TrigState) -> np.ndarray:
        # returns sum of grouped terms via dot product
        v_mat, w_vec = ProjectionOperations._get_vw(s)
        return w_vec @ v_mat

    @staticmethod
    def compute_derivatives(s: TrigState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v_mat, w_vec = ProjectionOperations._get_vw(s)

        n_samples: int = s.sa.shape[0]
        z_arr: np.ndarray = np.zeros(n_samples)

        # computes alpha partials
        da1: np.ndarray = z_arr
        da2: np.ndarray = s.sa * s.sb
        da3: np.ndarray = z_arr
        da4: np.ndarray = s.ca * s.sg
        da5: np.ndarray = s.sa * s.cb * s.cg
        da_mat: np.ndarray = np.array([da1, da2, da3, da4, da5])

        # computes beta partials
        db1: np.ndarray = -s.sb
        db2: np.ndarray = -s.ca * s.cb
        db3: np.ndarray = -s.cb * s.cg
        db4: np.ndarray = z_arr
        db5: np.ndarray = s.ca * s.sb * s.cg
        db_mat: np.ndarray = np.array([db1, db2, db3, db4, db5])

        # computes gamma partials
        dg1: np.ndarray = z_arr
        dg2: np.ndarray = z_arr
        dg3: np.ndarray = s.sb * s.sg
        dg4: np.ndarray = s.sa * s.cg
        dg5: np.ndarray = s.ca * s.cb * s.sg
        dg_mat: np.ndarray = np.array([dg1, dg2, dg3, dg4, dg5])

        # computes theta partials
        dt1: float = -s.st * s.cw
        dt2: float = s.ct * s.cw
        dt3: float = -s.st * s.sw
        dt4: float = s.ct * s.sw
        dt5: float = s.ct * s.sw
        dt_vec: np.ndarray = np.array([dt1, dt2, dt3, dt4, dt5])

        # computes omega partials
        dw1: float = -s.ct * s.sw
        dw2: float = -s.st * s.sw
        dw3: float = s.ct * s.cw
        dw4: float = s.st * s.cw
        dw5: float = s.st * s.cw
        dw_vec: np.ndarray = np.array([dw1, dw2, dw3, dw4, dw5])

        # evaluates chain rules via vector products
        da: np.ndarray = w_vec @ da_mat
        db: np.ndarray = w_vec @ db_mat
        dg: np.ndarray = w_vec @ dg_mat

        dt: np.ndarray = dt_vec @ v_mat
        dw: np.ndarray = dw_vec @ v_mat

        return da, db, dg, dt, dw
