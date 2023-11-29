import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import time

# Function to extract data from TensorFlow tfevents files
def get_data_from_log(directory, tag="eval_return", q_value_tag="q_values"):
    eval_data = []
    q_value_data = []
    for event_file in os.listdir(directory):
        if 'tfevents' in event_file:
            path = os.path.join(directory, event_file)
            for event in summary_iterator(path):
                for value in event.summary.value:
                    if value.tag == tag:
                        eval_data.append((event.step, value.simple_value))
                        print("eval_return on step ",event.step, ": ", value.simple_value)
                    elif value.tag == q_value_tag:
                        q_value_data.append((event.step, value.simple_value))

    # Sort data by step
    eval_data.sort(key=lambda x: x[0])
    q_value_data.sort(key=lambda x: x[0])
    return eval_data, q_value_data

# Directory where tfevents are stored for the experiment
directory = r"C:\Users\yuche\Documents\homework_fall2023\hw5\data\hw5_offline_PointmassMedium-v0_cql10_28-11-2023_15-54-20"

# Extract data from logs
eval_data, q_value_data = get_data_from_log(directory, "eval_return", "q_values")

# Plotting
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

if eval_data and q_value_data:  # Check if data is not empty
    steps1, eval_values = zip(*eval_data)
    steps2, q_values = zip(*q_value_data)

    ax1.plot(steps1, eval_values, label="Eval", color='blue')
    ax2.plot(steps2, q_values, label="Q-Values", linestyle="--", color='orange')

ax1.set_xlabel("Number of Environment Steps")
ax1.set_ylabel("Average Eval Return", color='blue')
ax2.set_ylabel("Average Q-Values", color='orange')

# Caption
plt.figtext(0.5, -0.05, "Caption: Running on α = 8 "
                        "As α increases, the conservatism of Q-learning also increases, potentially leading to more robust, "
                        "but potentially less exploratory, policies.", wrap=True, horizontalalignment='center', fontsize=10)

plt.title("Evaluation Performance and Q-Values, α = 10")
fig.tight_layout()  # adjust the layout
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
