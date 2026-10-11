"""
Compare a trained Q-learning policy against a random policy.
"""

from __future__ import annotations

from algorithms.q_learning_trainer import (
    QLearningTrainer,
)

from evaluation.q_learning_evaluation import (
    evaluate_agent,
)

from evaluation.baseline_evaluation import (
    evaluate_random_policy,
)


def main():

    print("=" * 64)
    print("BASE MULTI-AGV POLICY COMPARISON")
    print("=" * 64)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer = QLearningTrainer(
        episodes=100,
        max_steps=500,
        num_agvs=3,
        seed=42,
    )

    print()
    print("Training Q-learning...")

    trainer.train()

    # --------------------------------------------------
    # Evaluate Q-learning
    # --------------------------------------------------

    print()
    print("Evaluating Q-learning...")

    q_results = evaluate_agent(
        trainer,
        episodes=20,
    )

    # --------------------------------------------------
    # Evaluate random baseline
    # --------------------------------------------------

    print()
    print("Evaluating random baseline...")

    random_results = evaluate_random_policy(
        trainer,
        episodes=20,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 64)
    print("RESULTS")
    print("=" * 64)

    print(
        f"Q-learning average reward : "
        f"{q_results['average_reward']:.3f}"
    )

    print(
        f"Random average reward     : "
        f"{random_results['average_reward']:.3f}"
    )

    print()

    print(
        f"Q-learning completed      : "
        f"{q_results['average_completed']:.3f}"
    )

    print(
        f"Random completed          : "
        f"{random_results['average_completed']:.3f}"
    )

    print()

    print(
        f"Q-learning steps          : "
        f"{q_results['average_steps']:.3f}"
    )

    print(
        f"Random steps              : "
        f"{random_results['average_steps']:.3f}"
    )

    # --------------------------------------------------
    # Improvement
    # --------------------------------------------------

    q_reward = (
        q_results["average_reward"]
    )

    random_reward = (
        random_results["average_reward"]
    )

    if random_reward != 0:

        improvement = (
            (
                q_reward
                -
                random_reward
            )
            /
            abs(random_reward)
        ) * 100.0

        print()

        print(
            f"Reward improvement       : "
            f"{improvement:.2f}%"
        )

    print("=" * 64)


if __name__ == "__main__":
    main()
    