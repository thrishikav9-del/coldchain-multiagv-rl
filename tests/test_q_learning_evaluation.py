from algorithms.q_learning_trainer import (
    QLearningTrainer,
)

from evaluation.q_learning_evaluation import (
    evaluate_agent,
)


def test_q_learning_evaluation():

    trainer = QLearningTrainer(
        episodes=1,
        max_steps=5,
        num_agvs=2,
    )

    trainer.train_episode(
        episode=1
    )

    results = evaluate_agent(
        trainer,
        episodes=2,
    )

    assert results["episodes"] == 2

    assert len(
        results["rewards"]
    ) == 2

    assert len(
        results["steps"]
    ) == 2

    assert len(
        results["completions"]
    ) == 2
    