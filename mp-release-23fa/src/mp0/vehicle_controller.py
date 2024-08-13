from enum import Enum, auto
import copy
from typing import List

class PedestrianMode(Enum):
    Normal=auto()

class VehicleMode(Enum):
    Normal = auto()
    Brake = auto()
    Accel = auto()
    HardBrake = auto()

class State:
    x: float 
    y: float 
    theta: float 
    v: float 
    agent_mode: VehicleMode 

    def __init__(self, x, y, theta, v, agent_mode: VehicleMode):
        pass 

def decisionLogic(ego: State, other: State):
    output = copy.deepcopy(ego)
    
    # TODO: Edit this part of decision logic

    if ego.agent_mode == VehicleMode.Normal:
        output.agent_mode = VehicleMode.Accel
    
    if ego.agent_mode == VehicleMode.Accel and other.dist < 20:
        output.agent_mode = VehicleMode.Brake
    elif ego.agent_mode == VehicleMode.Brake and other.dist > 20:
        output.agent_mode = VehicleMode.Accel

#    if ego.agent_mode == VehicleMode.Accel and other.dist < 15:
#        output.agent_mode = VehicleMode.HardBrake
#    elif ego.agent_mode == VehicleMode.HardBrake and other.dist > 15:
#        output.agent_mode = VehicleMode.Accel

    ###########################################

    assert other.dist > 2.0

    return output 
