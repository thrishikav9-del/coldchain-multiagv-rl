from algorithms.q_learning_trainer import (
    QLearningTrainer,
)

from evaluation.baseline_evaluation import (
    evaluate_random_policy,
)


def test_random_baseline():

    trainer = QLearningTrainer(
        episodes=1,
        max_steps=5,
        num_agvs=2,
    )

    results = evaluate_random_policy(
        trainer,
        episodes=2,
    )

    assert results["episodes"] == 2

    assert len(
        results["rewards"]
    ) == 2

    assert len(
        results["completions"]
    ) == 2