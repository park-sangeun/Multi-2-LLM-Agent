import os
import json
import argparse
from typing import List, Dict


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_high_low_containers(data):
    if isinstance(data, list):
        episodes = data
    else:
        episodes = [data]

    high_data_container = {
        'task_description': [],
        'obs': [],
        'subtask': [],
        'reward': [],
        'score': [],
        'done': []
    }

    low_data_container = {
        'subtask': [],
        'obs': [],
        'action': [],
        'reward': [],
        'score': [],
        'done': []
    }

    for ep_idx, episode in enumerate(episodes):
        conversations = episode.get("conversations", [])

        # --- 1️⃣ Task description 추출 ---
        task_description = ""
        for c in conversations:
            val = c.get("value", "")
            if "Your task is to:" in val:
                task_description = val.split("Your task is to:")[1].split("Task Knowledge")[0].strip()
                break

        # --- 2️⃣ 초기 Observation 추출 ---
        initial_obs = None
        for c in conversations:
            val = c.get("value", "")
            if "You are in the middle of a room" in val:
                initial_obs = val.split("Your task is to:")[0].strip()
                break

        obs_list = []
        actions_list = []
        subtasks = []

        if initial_obs:
            obs_list.append(initial_obs)

        # --- 3️⃣ Observation & Action 추출 ---
        for c in conversations:
            msg = c.get("value", "")

            # Observation
            if msg.startswith("Observation:"):
                obs_text = msg.replace("Observation:", "").strip()
                obs_list.append(obs_text)

            # Action
            if "Action:" in msg:
                act = msg.split("Action:")[-1].strip()

                # ⚠️ "<your next action>" 같은 placeholder 제거
                if "<" in act and ">" in act:
                    continue

                actions_list.append(act)

                # subtask 이름 생성
                subtask_name = " ".join(act.split()[:3]).strip()
                if not subtask_name or "your" in subtask_name.lower():
                    continue
                subtasks.append(subtask_name)

        groups = len(subtasks)

        # --- High-level ---
        high_data_container['task_description'].append(task_description)
        high_data_container['obs'].append(obs_list)
        high_data_container['subtask'].append(subtasks)
        high_data_container['reward'].append([0.0] * (groups - 1) + [1.0])  # ✅ 수정
        high_data_container['score'].append([0.0] * (groups - 1) + [1.0])
        high_data_container['done'].append([False] * (groups - 1) + [True])

        # --- Low-level ---
        for i, subtask in enumerate(subtasks):
            sub_obs = []
            sub_actions = []

            if len(obs_list) > 0:
                sub_obs.append(obs_list[0])

            for j, act in enumerate(actions_list):
                sub_actions.append(act)
                if j + 1 < len(obs_list):
                    sub_obs.append(obs_list[j + 1])

            steps = len(sub_actions)
            low_prompt = (
                "You are a low-level action executor. Based on the current subtask and observation, "
                "please generate an executable action and determine if the subtask is completed (true/false).\n"
                f"Subtask: {subtask}"
            )
            low_data_container['subtask'].append(low_prompt)
            low_data_container['obs'].append(sub_obs)
            low_data_container['action'].append(sub_actions)
            low_data_container['reward'].append([0.0] * (steps - 1) + [1.0])  # ✅ 수정
            low_data_container['score'].append([0.0] * (steps - 1) + [1.0])
            low_data_container['done'].append([False] * (steps - 1) + [True])

    return high_data_container, low_data_container




def main():
    parser = argparse.ArgumentParser(description="Convert ALFWorld conversation JSON to hierarchical containers")
    parser.add_argument("--input", type=str, default="./dataset/alfworld/expert.json", help="Input conversation JSON file")
    parser.add_argument("--out", type=str, default="./dataset/alfworld", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = load_json(args.input)
    high_c, low_c = build_high_low_containers(data)

    hi_out = os.path.join(args.out, "high_data/expert.json")
    lo_out = os.path.join(args.out, "low_data/expert.json")

    with open(hi_out, "w", encoding="utf-8") as f:
        json.dump(high_c, f, ensure_ascii=False, indent=2)
    with open(lo_out, "w", encoding="utf-8") as f:
        json.dump(low_c, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(f" - {hi_out}")
    print(f" - {lo_out}")
    print("\n✅ High-level summary:")
    print(json.dumps(high_c, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()