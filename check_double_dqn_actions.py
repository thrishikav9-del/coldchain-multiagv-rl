from algorithms.dqn_trainer import DQNTrainer
t = DQNTrainer(
    episodes=1,
    max_steps=300,
    num_agvs=3,
    seed=42,
    model_path="models/double_dqn_2000ep.pt",
    resume=False,
    verbose=False,
)
t.agent.evaluation_mode()
env = t.env
state, _ = env.reset(seed=42)
print("MODEL_LOADED")
print("--- FIRST 20 GREEDY ACTIONS ---")
for step in range(20):
    action = t.agent.select_action(
        state,
        training=False,
    )
    print(
        "STEP",
        step + 1,
        "ACTION",
        action.tolist(),
    )
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("EPISODE_ENDED_AT=", step + 1)
        break
