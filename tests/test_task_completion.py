from environment.multi_agv_env import MultiAGVEnv


def test_single_agv_can_complete_task():

    env = MultiAGVEnv(
        width=20,
        height=20,
        num_agvs=1,
        max_steps=50,
    )

    state, _ = env.reset(
        seed=42
    )

    # Start:
    # (0, 0)
    #
    # Pickup:
    # (3, 3)
    #
    # Delivery:
    # (19, 19)

    actions = (
        [1, 1, 1]
        +
        [3, 3, 3]
        +
        [1] * 16
        +
        [3] * 16
    )

    assert len(actions) == 38

    last_info = None

    for action in actions:

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            [action]
        )

        last_info = info

        if terminated or truncated:
            break

    assert last_info is not None

    assert (
        last_info["completed_tasks"]
        == 1
    )

    assert (
        env.tasks[0].status
        == "COMPLETED"
    )
    