from algorithms.q_learning_trainer import (
    QLearningTrainer,
)


def test_trainer_initialization():

    trainer = QLearningTrainer(
        episodes=1,
        max_steps=5,
        num_agvs=2,
    )

    assert trainer.num_agvs == 2

    assert len(
        trainer.agents
    ) == 2


def test_single_training_episode():

    trainer = QLearningTrainer(
        episodes=1,
        max_steps=5,
        num_agvs=2,
    )

    result = trainer.train_episode(
        episode=1
    )

    assert "reward" in result

    assert "steps" in result

    assert "completed" in result

    assert result["steps"] <= 5


def test_short_training_run():

    trainer = QLearningTrainer(
        episodes=2,
        max_steps=5,
        num_agvs=2,
    )

    results = trainer.train()

    assert len(
        results["rewards"]
    ) == 2

    assert len(
        results["steps"]
    ) == 2

    assert len(
        results["completions"]
    ) == 2

def test_multi_agv_state_contains_other_agents():
    
    trainer = QLearningTrainer(
        episodes=1,
        max_steps=5,
        num_agvs=3,
    )

    states, _ = trainer.env.reset(
        seed=42
    )

    agv_state = trainer._get_agv_state(
        states,
        0,
    )

    # 8 values for the AGV itself
    # + 8 values for AGV 2
    # + 8 values for AGV 3
    assert len(agv_state) == 24
