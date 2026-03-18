import math
from hybrid_a_star_planner.mpc import MPC
from hybrid_a_star_planner.utils import Pos

def main():
    mpc = MPC(dt=0.1, horizon=10)
    mpc.set_physical_constraints(v_max=1.0, v_min=-0.5, omega_max=1.0)
    mpc.set_weights(position_weight=1.0, heading_weight=0.5, control_weight=0.1)

    pos_list = mpc.solve(Pos(x=0.0, y=0.0, theta=0.0), Pos(x=1.0, y=1.0, theta=math.pi / 4))
    for pos in pos_list:
        print(f"x={pos.x:.2f}, y={pos.y:.2f}, theta={pos.theta:.2f}")

    # initial_state = [0.0, 0.0, 0.0]
    # controls = [[0.5, 0.1] for _ in range(mpc.horizon)]
    # trajectory = mpc.predict_trajectory(initial_state, controls)

    # print("Predicted trajectory:")
    # for i in range(mpc.horizon):
    #     print(f"Step {i}: x={trajectory[i][0]:.2f}, y={trajectory[i][1]:.2f}, theta={trajectory[i][2]:.2f}")

if __name__ == "__main__":
    main()