"""
Run base-paper Q-learning training.
"""

from algorithms.q_learning_trainer import (
    QLearningTrainer,
)


def main():

    trainer = QLearningTrainer(
        episodes=100,
        max_steps=500,
        num_agvs=3,
        seed=42,
    )

    results = trainer.train()

    rewards = results["rewards"]

    completions = results[
        "completions"
    ]

    print()
    print("=" * 60)
    print("Q-LEARNING TRAINING COMPLETE")
    print("=" * 60)

    if rewards:

        print(
            "Average reward:",
            sum(rewards) / len(rewards),
        )

        print(
            "Best reward:",
            max(rewards),
        )

    if completions:

        print(
            "Average completed tasks:",
            sum(completions)
            /
            len(completions),
        )


if __name__ == "__main__":
    main()
    