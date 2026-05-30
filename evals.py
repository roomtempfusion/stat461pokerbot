import json
import matplotlib.pyplot as plt
from collections import Counter
import os

print("Loading master simulation data...")


file_path = "master_simulation_results.json"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found")
    exit()

with open(file_path, "r") as f:
    data = json.load(f)


total_games = data.get('total_games', 0)
total_hands_played = 0
agent_wins = 0

all_logs = []


for game in data.get('games', []):
    total_hands_played += game['hands_played']
    
    
    if game['final_stacks'][0] > 0:
        agent_wins += 1
        
    all_logs.extend(game.get('agent_action_logs', []))

print(f"\n{'='*30}")
print(f"      MASTER SIMULATION RESULTS      ")
print(f"{'='*30}")
print(f"Games Played       : {total_games}")
print(f"Total Hands Played : {total_hands_played}")
win_percentage = (agent_wins / total_games) * 100 if total_games > 0 else 0
print(f"Agent Win Rate     : {agent_wins}/{total_games} ({win_percentage:.1f}%)")

if not all_logs:
    print("\nNo action logs found to analyze. Exiting.")
    exit()


actions = [log['action'] for log in all_logs]
action_counts = Counter(actions)

print("\n--- AGENT ACTION FREQUENCIES ---")
for action, count in action_counts.items():
    print(f"{action.capitalize():<6}: {count} times ({round((count/len(actions))*100, 1)}%)")


p2_scores = [log['p2_aggression'] for log in all_logs if log.get('p2_aggression') is not None]
p3_scores = [log['p3_aggression'] for log in all_logs if log.get('p3_aggression') is not None]

print(f"\n--- OPPONENT PROFILING ---")
if p2_scores:
    avg_p2 = sum(p2_scores) / len(p2_scores)
    print(f"P2 Average Aggression: {avg_p2:.1f}/10 (Sample size: {len(p2_scores)} post-flop decisions)")
if p3_scores:
    avg_p3 = sum(p3_scores) / len(p3_scores)
    print(f"P3 Average Aggression: {avg_p3:.1f}/10 (Sample size: {len(p3_scores)} post-flop decisions)")

# --- MATPLOTLIB VISUALIZATION ---
# Pre-define standard colors so 'Raise' is always green, 'Fold' is red, etc.
color_map = {'fold': '#ff9999', 'check': '#99ccff', 'call': '#ffe5cc', 'raise': '#99ff99'}

labels = list(action_counts.keys())
values = list(action_counts.values())
colors = [color_map.get(label, 'gray') for label in labels]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)

# Add exact counts on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), int(yval), ha='center', va='bottom', fontweight='bold')

plt.title(f"Agent Decision Distribution ({total_games} Games)", fontsize=14, pad=15)
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Action Taken", fontsize=12)

# Clean up the chart borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()