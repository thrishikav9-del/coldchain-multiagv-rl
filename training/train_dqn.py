import os
import numpy as np

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 2000
MAX_STEPS = 200

NUM_AGVS = 3
ACTION_SIZE = 5

LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.001

REPLAY_CAPACITY = 200000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99

CHECKPOINT_INTERVAL = 100


# ==========================================================
# DIRECTORIES
# ==========================================================

MODEL_DIR = "models"

CHECKPOINT_DIR = os.path.join(
    MODEL_DIR,
    "checkpoints",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # ENVIRONMENT
    # ======================================================

    env = MultiAGVEnv(
        num_agvs=NUM_AGVS,
        max_steps=MAX_STEPS,
    )

    # ------------------------------------------------------
    # Initial reset
    # ------------------------------------------------------

    reset_result = env.reset()

    if isinstance(
        reset_result,
        tuple,
    ):

        state = reset_result[0]

    else:

        state = reset_result

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    # ======================================================
    # STATE SIZE
    # ======================================================
    #
    # Current environment:
    #
    # 3 AGVs × 8 features = 24
    #
    # Features per AGV:
    #
    # 1. x
    # 2. y
    # 3. goal_x
    # 4. goal_y
    # 5. risk
    # 6. urgency
    # 7. future_risk_60
    # 8. temperature
    #
    # ======================================================

    STATE_SIZE = int(
        env.observation_space.shape[0]
    )

    EXPECTED_STATE_SIZE = (
        NUM_AGVS * 8
    )

    if STATE_SIZE != EXPECTED_STATE_SIZE:

        raise RuntimeError(
            "Unexpected observation size. "
            f"Expected {EXPECTED_STATE_SIZE} "
            f"for {NUM_AGVS} AGVs, "
            f"but environment returned {STATE_SIZE}."
        )

    # ------------------------------------------------------
    # Also validate the actual reset state.
    # ------------------------------------------------------

    if state.size != STATE_SIZE:

        raise RuntimeError(
            "Invalid initial state size. "
            f"Expected {STATE_SIZE}, "
            f"got {state.size}."
        )

    # ======================================================
    # DQN AGENT
    # ======================================================

    agent = DQNAgent(
        state_size=STATE_SIZE,
        num_agvs=NUM_AGVS,
        action_size=ACTION_SIZE,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,

        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
    )

    # ======================================================
    # CONNECT DQN TO ACTIVE ENVIRONMENT
    # ======================================================
    #
    # Required by the centralized collision-free
    # joint-action safety system.
    #
    # The DQN agent uses:
    #
    #     env._get_target_position()
    #
    # to determine where every AGV would move.
    #
    # This allows the agent to reject:
    #
    # 1. Same-destination collisions
    # 2. Moving into stationary AGV
    # 3. Swap collisions
    #
    # ======================================================

    agent._evaluation_env = env

    # ======================================================
    # TRAINING HISTORY
    # ======================================================

    rewards_history = []
    steps_history = []
    completion_history = []
    collision_history = []
    loss_history = []

    # ======================================================
    # HEADER
    # ======================================================

    print()
    print("=" * 72)
    print("PHASE 4.2 CENTRALIZED JOINT-ACTION DQN TRAINING")
    print("=" * 72)

    print(f"Episodes       : {EPISODES}")
    print(f"Max steps      : {MAX_STEPS}")
    print(f"AGVs           : {NUM_AGVS}")
    print(f"State size     : {STATE_SIZE}")
    print(f"Action size    : {ACTION_SIZE}")
    print(
        f"Joint actions  : "
        f"{agent.joint_action_size}"
    )
    print(f"Device         : {agent.device}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Gamma          : {GAMMA}")
    print(f"Tau            : {TAU}")
    print(f"Replay capacity: {REPLAY_CAPACITY}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Epsilon start  : {EPSILON_START}")
    print(f"Epsilon min    : {EPSILON_MIN}")
    print(f"Epsilon decay  : {EPSILON_DECAY}")
    print("=" * 72)

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for episode in range(
        1,
        EPISODES + 1,
    ):

        # ==================================================
        # RESET
        # ==================================================

        reset_result = env.reset()

        if isinstance(
            reset_result,
            tuple,
        ):

            state = reset_result[0]

        else:

            state = reset_result

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.size != STATE_SIZE:

            raise RuntimeError(
                f"Invalid state size: "
                f"{state.size}. "
                f"Expected {STATE_SIZE}."
            )

        # ==================================================
        # EPISODE VARIABLES
        # ==================================================

        episode_reward = 0.0

        episode_losses = []

        collision_count = 0
        completed_count = 0
        pickup_count = 0

        steps = 0

        # ==================================================
        # EPISODE LOOP
        # ==================================================

        for step in range(
            MAX_STEPS
        ):

            steps = step + 1

            # ==============================================
            # JOINT ACTION
            # ==============================================
            #
            # DQN selects one centralized joint action:
            #
            #     [AGV1_action,
            #      AGV2_action,
            #      AGV3_action]
            #
            # Example:
            #
            #     [4, 0, 1]
            #
            # ==============================================

            joint_action = agent.select_joint_action(
                state
            )

            joint_action = np.asarray(
                joint_action,
                dtype=np.int64,
            )

            # ------------------------------------------------
            # Shape validation
            # ------------------------------------------------

            if joint_action.shape != (
                NUM_AGVS,
            ):

                raise RuntimeError(
                    "Invalid joint action shape: "
                    f"{joint_action.shape}. "
                    f"Expected ({NUM_AGVS},)."
                )

            # ------------------------------------------------
            # Action range validation
            # ------------------------------------------------

            if np.any(
                joint_action < 0
            ) or np.any(
                joint_action >= ACTION_SIZE
            ):

                raise RuntimeError(
                    "Joint action contains invalid "
                    f"action values: {joint_action}"
                )

            # ==============================================
            # ENVIRONMENT STEP
            # ==============================================

            result = env.step(
                joint_action
            )

            if len(result) != 5:

                raise RuntimeError(
                    "Environment step() must return "
                    "(observation, reward, terminated, "
                    "truncated, info)."
                )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = result

            done = bool(terminated)

            # ==============================================
            # NEXT STATE
            # ==============================================

            next_state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if next_state.size != STATE_SIZE:

                raise RuntimeError(
                    "Invalid next-state size: "
                    f"{next_state.size}. "
                    f"Expected {STATE_SIZE}."
                )

            # ==============================================
            # STORE ONE JOINT EXPERIENCE
            # ==============================================
            #
            # Important:
            # We store the complete centralized action,
            # not separate experiences for each AGV.
            #
            # state
            # joint_action
            # reward
            # next_state
            # done
            #
            # ==============================================

            agent.remember(
                state,
                joint_action,
                reward,
                next_state,
                done,
            )

            # ==============================================
            # DQN TRAINING STEP
            # ==============================================

            loss = agent.train_step()

            if loss is not None:

                loss = float(loss)

                if np.isfinite(loss):

                    episode_losses.append(
                        loss
                    )

            # ==============================================
            # REWARD
            # ==============================================

            episode_reward += float(
                reward
            )

            # ==============================================
            # INFO
            # ==============================================

            if isinstance(
                info,
                dict,
            ):

                collision_count += int(
                    info.get(
                        "collision_count",
                        info.get(
                            "collisions",
                            0,
                        ),
                    )
                )

                completed_count += int(
                    info.get(
                        "completed_tasks",
                        info.get(
                            "completed",
                            0,
                        ),
                    )
                )

                pickup_count += int(
                    info.get(
                        "pickup_tasks_this_step",
                        info.get(
                            "pickup_tasks",
                            info.get(
                            "pickups",
                            0,
                        ),
                    ),
                )
            )

            # ==============================================
            # NEXT STATE
            # ==============================================

            state = next_state

            # ==============================================
            # TERMINATION
            # ==============================================

            if done:

                break

        # ==================================================
        # EPSILON DECAY
        # ==================================================

        agent.decay_epsilon()

        # ==================================================
        # AVERAGE LOSS
        # ==================================================

        if episode_losses:

            average_loss = float(
                np.mean(
                    episode_losses
                )
            )

        else:

            average_loss = 0.0

        # ==================================================
        # HISTORY
        # ==================================================

        rewards_history.append(
            episode_reward
        )

        steps_history.append(
            steps
        )

        completion_history.append(
            completed_count
        )

        collision_history.append(
            collision_count
        )

        loss_history.append(
            average_loss
        )

        # ==================================================
        # PRINT
        # ==================================================

        if (
            episode == 1
            or episode % 10 == 0
        ):

            print(
                f"Episode "
                f"{episode:04d}/{EPISODES:04d} "
                f"| Reward={episode_reward:.2f} "
                f"| Steps={steps} "
                f"| Pickup={pickup_count} "
                f"| Completed={completed_count} "
                f"| Collision={collision_count} "
                f"| Loss={average_loss:.5f} "
                f"| Epsilon={agent.epsilon:.4f} "
                f"| Replay={len(agent.replay_buffer)}"
            )

        # ==================================================
        # CHECKPOINT
        # ==================================================

        if (
            episode % CHECKPOINT_INTERVAL == 0
        ):

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"dqn_checkpoint_ep_{episode:05d}.pt",
            )

            agent.save(
                checkpoint_path
            )

            print(
                f"CHECKPOINT SAVED: "
                f"{checkpoint_path}"
            )

    # ======================================================
    # FINAL MODEL
    # ======================================================

    final_model_path = os.path.join(
        MODEL_DIR,
        "dqn_model.pt",
    )

    agent.save(
        final_model_path
    )

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"MODEL SAVED: {final_model_path}"
    )

    print(
        f"FINAL REPLAY SIZE: "
        f"{len(agent.replay_buffer)}"
    )

    print(
        f"FINAL EPSILON: "
        f"{agent.epsilon:.6f}"
    )

    print(
        f"FINAL TRAINING STEPS: "
        f"{agent.training_steps}"
    )

    print("=" * 72)
    
        # ======================================================
    # TRAINED DQN EVALUATION
    # ======================================================

    print()
    print("=" * 72)
    print("TRAINED DQN EVALUATION")
    print("=" * 72)

    EVAL_EPISODES = 100

    # ------------------------------------------------------
    # Create a fresh evaluation environment
    # ------------------------------------------------------

    eval_env = MultiAGVEnv(
        num_agvs=NUM_AGVS,
        max_steps=MAX_STEPS,
    )

    # ------------------------------------------------------
    # Create evaluation agent with identical architecture
    # ------------------------------------------------------

    eval_agent = DQNAgent(
        state_size=STATE_SIZE,
        num_agvs=NUM_AGVS,
        action_size=ACTION_SIZE,

        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,

        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,

        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
    )

    # ------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------

    eval_agent.load(
        final_model_path
    )

    # ------------------------------------------------------
    # Force greedy evaluation
    # ------------------------------------------------------

    eval_agent.evaluation_mode()

    # IMPORTANT:
    # Collision-safe joint-action filtering needs the
    # active environment reference.
    eval_agent._evaluation_env = eval_env

    # ------------------------------------------------------
    # Evaluation statistics
    # ------------------------------------------------------

    eval_rewards = []
    eval_steps = []
    eval_pickups = []
    eval_completions = []
    eval_collisions = []
    successful_episodes = 0

    # ======================================================
    # EVALUATION EPISODES
    # ======================================================

    for eval_episode in range(
        1,
        EVAL_EPISODES + 1,
    ):

        reset_result = eval_env.reset()

        if isinstance(
            reset_result,
            tuple,
        ):
            state = reset_result[0]
        else:
            state = reset_result

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.size != STATE_SIZE:

            raise RuntimeError(
                "Evaluation state size mismatch: "
                f"{state.size}. "
                f"Expected {STATE_SIZE}."
            )

        episode_reward = 0.0
        episode_pickups = 0
        episode_completions = 0
        episode_collisions = 0

        actual_steps = 0

        # --------------------------------------------------
        # Run one complete episode
        # --------------------------------------------------

        for step in range(
            MAX_STEPS
        ):

            actual_steps += 1

            action = eval_agent.select_action(
                state,
                training=False,
            )

            action = np.asarray(
                action,
                dtype=np.int64,
            )

            if action.shape != (
                NUM_AGVS,
            ):

                raise RuntimeError(
                    "Evaluation action shape mismatch: "
                    f"{action.shape}. "
                    f"Expected ({NUM_AGVS},)."
                )

            if np.any(
                action < 0
            ) or np.any(
                action >= ACTION_SIZE
            ):

                raise RuntimeError(
                    "Evaluation produced invalid "
                    f"action: {action}"
                )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = eval_env.step(
                action
            )

            episode_reward += float(
                reward
            )

            if isinstance(
                info,
                dict,
            ):

                episode_collisions += int(
                    info.get(
                        "collision_count",
                        
                            0,
                        ),
                    )
                

                episode_completions += int(
                    info.get(
                        "completed_tasks_this_step",
                            0,
                        )
                    )

                episode_pickups += int(
                    info.get(
                        "pickup_tasks_this_step",
                            0,
                        )
                    )

            state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if state.size != STATE_SIZE:

                raise RuntimeError(
                    "Evaluation next-state size mismatch: "
                    f"{state.size}. "
                    f"Expected {STATE_SIZE}."
                )

            done = bool(terminated)
            
            if done:
                break

            if truncated:
                break
        # --------------------------------------------------
        # Store episode statistics
        # --------------------------------------------------

        eval_rewards.append(
            episode_reward
        )

        eval_steps.append(
            actual_steps
        )

        eval_pickups.append(
            episode_pickups
        )

        eval_completions.append(
            episode_completions
        )

        eval_collisions.append(
            episode_collisions
        )

        if bool(terminated):

            successful_episodes += 1

        # --------------------------------------------------
        # Progress output
        # --------------------------------------------------

        if (
            eval_episode == 1
            or
            eval_episode % 10 == 0
        ):

            print(
                f"Evaluation Episode "
                f"{eval_episode:03d}/{EVAL_EPISODES:03d} "
                f"| Reward={episode_reward:.2f} "
                f"| Steps={actual_steps} "
                f"| Pickup={episode_pickups} "
                f"| Completed={episode_completions} "
                f"| Collision={episode_collisions}"
            )

    # ======================================================
    # EVALUATION SUMMARY
    # ======================================================

    average_eval_reward = float(
        np.mean(
            eval_rewards
        )
    )

    best_eval_reward = float(
        np.max(
            eval_rewards
        )
    )

    average_eval_steps = float(
        np.mean(
            eval_steps
        )
    )

    average_eval_pickups = float(
        np.mean(
            eval_pickups
        )
    )

    average_eval_completions = float(
        np.mean(
            eval_completions
        )
    )

    total_eval_collisions = int(
        np.sum(
            eval_collisions
        )
    )

    average_eval_collisions = float(
        np.mean(
            eval_collisions
        )
    )

    completion_rate = (
        100.0
        *
        successful_episodes
        /
        EVAL_EPISODES
    )

    total_eval_completions = int(
        np.sum(
            eval_completions
        )
    )

    total_eval_pickups = int(
        np.sum(
            eval_pickups
        )
    )

    # ======================================================
    # FINAL EVALUATION REPORT
    # ======================================================

    print()
    print("=" * 72)
    print("FINAL DQN EVALUATION SUMMARY")
    print("=" * 72)

    print(
        f"Episodes evaluated : "
        f"{EVAL_EPISODES}"
    )

    print(
        f"Best reward        : "
        f"{best_eval_reward:.2f}"
    )

    print(
        f"Average reward     : "
        f"{average_eval_reward:.2f}"
    )

    print(
        f"Average steps      : "
        f"{average_eval_steps:.2f}"
    )

    print(
        f"Total pickups      : "
        f"{total_eval_pickups}"
    )

    print(
        f"Average pickups    : "
        f"{average_eval_pickups:.2f}"
    )

    print(
        f"Total completions  : "
        f"{total_eval_completions}"
    )

    print(
        f"Average completions: "
        f"{average_eval_completions:.2f}"
    )

    print(
        f"Total collisions   : "
        f"{total_eval_collisions}"
    )

    print(
        f"Average collisions : "
        f"{average_eval_collisions:.2f}"
    )

    print(
        f"Successful episodes: "
        f"{successful_episodes}"
    )

    print(
        f"Completion rate    : "
        f"{completion_rate:.2f}%"
    )

    print(
        f"Evaluation epsilon : "
        f"{eval_agent.epsilon:.4f}"
    )

    print("=" * 72)
    # ======================================================
    # CLOSE ENVIRONMENT
    # ======================================================

    env.close()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()