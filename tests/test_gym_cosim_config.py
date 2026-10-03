"""Co-simulation timestep, sensors and coordinate validation."""

import math

import pytest

from ml_stack.gym.cosim import make_cosim
from ml_stack.gym.cosim_map import physics_heading, sumo_heading


@pytest.mark.parametrize("config", [{"physics_dt":0.03}, {"physics_dt":0},
                                   {"physics_dt":float("nan")}, {"lidar_num_lasers":3},
                                   {"lidar_distance":0}, {"lidar_distance":float("inf")}])
def test_invalid_bridge_settings_are_rejected_before_starting_simulators(config):
    with pytest.raises(ValueError):
        make_cosim(config)


@pytest.mark.parametrize("theta,degrees", [(0,90), (math.pi/2,0), (-math.pi/2,180), (math.pi,270)])
def test_world_heading_conversion_preserves_cardinal_directions(theta, degrees):
    assert sumo_heading(theta) == pytest.approx(degrees)
    assert math.cos(physics_heading(degrees)) == pytest.approx(math.cos(theta))
    assert math.sin(physics_heading(degrees)) == pytest.approx(math.sin(theta))
