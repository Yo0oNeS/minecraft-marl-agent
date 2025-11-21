import pandas as pd
import matplotlib.pyplot as plt

def plot_training_data(log_file="training_log.csv"):
    try:
        # 1. Read the Data
        df = pd.read_csv(log_file)
        
        # 2. Calculate Smoothing (Moving Average)
        # RL data is noisy. We take the average of the last 10 episodes to show the trend.
        window_size = 10
        df['rolling_reward'] = df['total_reward'].rolling(window=window_size).mean()
        
        # 3. Setup the Plot
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Raw Data (Light Blue)
        plt.plot(df['episode'], df['total_reward'], label='Episode Reward', color='lightblue', alpha=0.6)
        
        # Plot 2: Trend Line (Dark Blue)
        plt.plot(df['episode'], df['rolling_reward'], label=f'{window_size}-Episode Moving Average', color='blue', linewidth=2)
        
        # Styling
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Multi-Agent PPO Training Progress (Handoff Task)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Success Line (Ideal Score)
        plt.axhline(y=190, color='green', linestyle='--', label='Consistently Solved (~190+)')
        
        # Save and Show
        plt.savefig("training_curve.png")
        print("Graph saved to 'training_curve.png'")
        plt.show()
        
    except FileNotFoundError:
        print(f"Could not find {log_file}. Wait for training to start!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    plot_training_data()