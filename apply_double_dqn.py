from pathlib import Path
path = Path(r".\algorithms\dqn_agent.py")
text = path.read_text()
start_marker = """        # --------------------------------------------------
        # Target Q-values
        # --------------------------------------------------
"""
end_marker = """        # --------------------------------------------------
        # Huber loss
        # --------------------------------------------------
"""
start = text.find(start_marker)
end = text.find(end_marker, start)
if start == -1:
    raise RuntimeError("TARGET_SECTION_START_NOT_FOUND")
if end == -1:
    raise RuntimeError("HUBER_LOSS_MARKER_NOT_FOUND")
if end <= start:
    raise RuntimeError("INVALID_TARGET_SECTION_RANGE")
new_section = """        # --------------------------------------------------
        # Target Q-values
        # --------------------------------------------------
        with torch.no_grad():
            # --------------------------------------------------
            # Double DQN target
            # --------------------------------------------------
            next_policy_q = (
                self.policy_network.action_values(
                    next_states_tensor
                )
            )
            if next_policy_q.shape != (
                self.batch_size,
                self.joint_action_size,
            ):
                raise RuntimeError(
                    "Invalid next policy Q shape: "
                    f"{next_policy_q.shape}. "
                    f"Expected "
                    f"({self.batch_size}, "
                    f"{self.joint_action_size})."
                )
            next_actions = (
                next_policy_q
                .argmax(
                    dim=1
                )
            )
            next_target_q = (
                self.target_network.action_values(
                    next_states_tensor
                )
            )
            if next_target_q.shape != (
                self.batch_size,
                self.joint_action_size,
            ):
                raise RuntimeError(
                    "Invalid next target Q shape: "
                    f"{next_target_q.shape}. "
                    f"Expected "
                    f"({self.batch_size}, "
                    f"{self.joint_action_size})."
                )
            selected_next_q = (
                next_target_q
                .gather(
                    1,
                    next_actions.unsqueeze(1),
                )
                .squeeze(1)
            )
            target_q = (
                rewards_tensor
                +
                self.gamma
                * selected_next_q
                *
                (
                    1.0
                    -
                    dones_tensor
                )
            )
"""
text = text[:start] + new_section + text[end:]
path.write_text(text)
print("DOUBLE_DQN_MARKER_EDIT_APPLIED")
