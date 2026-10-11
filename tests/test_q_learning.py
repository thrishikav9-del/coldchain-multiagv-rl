import numpy as np

from algorithms.q_learning import (
    QLearningAgent,
)


def test_q_learning_initialization():

    agent = QLearningAgent(
        action_count=5,
        seed=42,
    )

    assert agent.action_count == 5

    assert agent.learning_rate == 0.1

    assert agent.discount_factor == 0.95

    assert agent.epsilon == 1.0

    assert agent.state_count == 0


def test_q_table_creation():

    agent = QLearningAgent(
        action_count=5,
        seed=42,
    )

    state = (
        0,
        0,
        1,
        1,
    )

    action = agent.select_action(
        state,
        training=False,
    )

    assert 0 <= action < 5

    assert agent.state_count == 1


def test_q_update():

    agent = QLearningAgent(
        action_count=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.0,
        seed=42,
    )

    state = (
        0,
        0,
    )

    next_state = (
        1,
        0,
    )

    old_value = agent.q_table[
        state
    ][0]

    agent.update(
        state=state,
        action=0,
        reward=10.0,
        next_state=next_state,
        terminated=True,
    )

    new_value = agent.q_table[
        state
    ][0]

    assert old_value == 0.0

    assert new_value == 1.0


def test_q_update_non_terminal():

    agent = QLearningAgent(
        action_count=2,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    state = (
        0,
    )

    next_state = (
        1,
    )

    agent.q_table[
        next_state
    ][1] = 10.0

    agent.update(
        state=state,
        action=0,
        reward=5.0,
        next_state=next_state,
        terminated=False,
    )

    # target = 5 + 0.9 * 10
    #       = 14
    #
    # Q = 0 + 0.1 * 14
    #   = 1.4

    assert np.isclose(
        agent.q_table[
            state
        ][0],
        1.4,
    )


def test_epsilon_decay():

    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.5

    agent.decay_epsilon()

    assert agent.epsilon == 0.25

    for _ in range(20):

        agent.decay_epsilon()

    assert agent.epsilon >= 0.05

def test_q_learning_save_load(
    tmp_path,
):

    agent = QLearningAgent(
        action_count=5,
        epsilon=0.2,
        seed=42,
    )

    state = (
        1,
        2,
        3,
    )

    agent.q_table[
        state
    ][2] = 15.5

    path = (
        tmp_path
        /
        "q_table.pkl"
    )

    agent.save(
        path
    )

    loaded = QLearningAgent(
        action_count=5,
        seed=100,
    )

    loaded.load(
        path
    )

    assert loaded.action_count == 5

    assert loaded.epsilon == 0.2

    assert loaded.q_table[
        state
    ][2] == 15.5    