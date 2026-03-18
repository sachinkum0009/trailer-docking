from dataclasses import dataclass


@dataclass
class Pos:
    x: float
    y: float
    theta: float

    @staticmethod
    def default() -> "Pos":
        return Pos(x=0.0, y=0.0, theta=0.0)
