from dataclasses import dataclass


@dataclass
class Pos:
    x: float
    y: float
    theta: float
    phi: float = 0.0

    @staticmethod
    def default() -> "Pos":
        return Pos(x=0.0, y=0.0, theta=0.0)
