from lpm_model_functions import *
import numpy as np


def test_improved_euler_step():

    yk1 = improved_euler_step(pressure_ode_model, 0, 1, 1, 2, [0, 10, 1, 1, 1, 1])

    by_hand_soln = -3.5

    assert abs(yk1 - by_hand_soln) < 1.e-10
