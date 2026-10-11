import time

import pybullet as p
import pybullet_data


def main():

    physics_client = p.connect(
        p.GUI
    )

    if physics_client < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet GUI."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81,
    )

    p.loadURDF(
        "plane.urdf"
    )

    print(
        "PyBullet GUI started successfully."
    )

    print(
        "Close the PyBullet window to finish."
    )

    try:

        while p.isConnected():

            p.stepSimulation()

            time.sleep(
                1.0 / 240.0
            )

    finally:

        if p.isConnected():

            p.disconnect()


if __name__ == "__main__":
    main()
    