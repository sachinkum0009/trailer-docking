from dataclasses import dataclass
import numpy as np
from geometry_msgs.msg import Quaternion


@dataclass
class Pos:
    x: float
    y: float
    theta: float
    phi: float = 0.0
    is_reverse: bool = False

    @staticmethod
    def default() -> "Pos":
        return Pos(x=0.0, y=0.0, theta=0.0)


def quaternion_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))
