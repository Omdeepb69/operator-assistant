import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# --- Data Simulation (Replace with actual data loading/processing) ---

def _simulate_command_log(num_entries=200):
    """Generates sample command log data for demonstration."""
    commands = ['create_file', 'list_files', 'delete_file', 'web_search', 'unknown_intent', 'get_time']
    statuses = ['success', 'failure', 'cancelled', 'confirmed']
    intents = ['file_operation', 'web_query', 'utility', 'unknown']
    
    log_data = []
    start_time = datetime(2023, 10, 26, 9, 0, 0)
    time_deltas = np.random.exponential(scale=300, size=num_entries) # Time in seconds between commands
    current_time = start_time

    for i in range(num_entries):
        current_time += pd.Timedelta(seconds=time_deltas[i])
        command = np.random.choice(commands, p=[0.2, 0.2, 0.1, 0.35, 0.1, 0.05])
        
        status = 'success'
        confirmation = None
        query_length = None
        true_intent = 'unknown'
        predicted_intent = 'unknown'

        if command == 'delete_file':
            confirmation = np.random.choice(['confirmed', 'cancelled'], p=[0.7, 0.3])
            status = 'success' if confirmation == 'confirmed' else 'cancelled'
            true_intent = 'file_operation'
            predicted_intent = np.random.choice(['file_operation', 'web_query', 'unknown'], p=[0.9, 0.05, 0.05])
        elif command in ['create_file', 'list_files']:
             status = np.random.choice(['success', 'failure'], p=[0.95, 0.05])
             true_intent = 'file_operation'
             predicted_intent = np.random.choice(['file_operation', 'web_query', 'unknown'], p=[0.92, 0.05, 0.03])
        elif command == 'web_search':
            status = np.random.choice(['success', 'failure'], p=[0.9, 0.1])
            query_length = np.random.randint(10, 100)
            true_intent = 'web_query'
            predicted_intent = np.random.choice(['web_query', 'file_operation', 'unknown'], p=[0.88, 0.07, 0.05])
        elif command == 'get_time':
            status = 'success'
            true_intent = 'utility'
            predicted_intent = np.random.choice(['utility', 'web_query', 'unknown'], p=[0.95, 0.03, 0.02])
        else: # unknown_intent
            status = 'failure'
            true_intent = 'unknown' # Or the actual intent if known
            predicted_intent = 'unknown'


        log_data.append({
            'timestamp': current_time,
            'command_type': command,
            'status': status,
            'confirmation_status': confirmation,
            'query_length': query_length,
            'true_intent': true_intent,
            'predicted_intent': predicted_intent,
        })
        
    return pd.DataFrame(log_data)

# --- Visualization Functions ---

def plot_command_distribution(command_log: pd.DataFrame, title: str = "Command Type Distribution"):
    """
    Generates a bar chart showing the frequency of each command type.

    Args:
        command_log: Pandas DataFrame with a 'command_type' column.
        title: The title for the plot.
    """
    if command_log.empty or 'command_type' not in command_log.columns:
        print("Warning: Command log is empty or missing 'command_type' column.")
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(data=command_log, y='command_type', order=command_log['command_type'].value_counts().index, palette='viridis')
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Command Type")
    plt.tight_layout()
    plt.show()

def plot_command_status_distribution(command_log: pd.DataFrame, title: str = "Command Status Distribution"):
    """
    Generates a bar chart showing the frequency of each command status.

    Args:
        command_log: Pandas DataFrame with a 'status' column.
        title: The title for the plot.
    """
    if command_log.empty or 'status' not in command_log.columns:
        print("Warning: Command log is empty or missing 'status' column.")
        return

    plt.figure(figsize=(8, 5))
    sns.countplot(data=command_log, x='status', order=command_log['status'].value_counts().index, palette='magma')
    plt.title(title)
    plt.xlabel("Status")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_confirmation_outcomes(command_log: pd.DataFrame, title: str = "Deletion Confirmation Outcomes"):
    """
    Generates a pie chart showing the outcomes of confirmation prompts (e.g., for deletions).

    Args:
        command_log: Pandas DataFrame with a 'confirmation_status' column.
        title: The title for the plot.
    """
    if command_log.empty or 'confirmation_status' not in command_log.columns:
        print("Warning: Command log is empty or missing 'confirmation_status' column.")
        return
        
    confirmation_data = command_log['confirmation_status'].dropna().value_counts()
    
    if confirmation_data.empty:
        print("Warning: No confirmation status data found to plot.")
        return

    plt.figure(figsize=(7, 7))
    plt.pie(confirmation_data, labels=confirmation_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('coolwarm'))
    plt.title(title)
    plt.ylabel("") # Hide the default ylabel
    plt.show()

def plot_command_timeline(command_log: pd.DataFrame, freq: str = 'D', title: str = "Commands Over Time"):
    """
    Generates a line plot showing the number of commands executed over time.

    Args:
        command_log: Pandas DataFrame with a 'timestamp' column (datetime objects).
        freq: The frequency for resampling (e.g., 'H' for hourly, 'D' for daily).
        title: The title for the plot.
    """
    if command_log.empty or 'timestamp' not in command_log.columns:
        print("Warning: Command log is empty or missing 'timestamp' column.")
        return
        
    if not pd.api.types.is_datetime64_any_dtype(command_log['timestamp']):
         print("Warning: 'timestamp' column is not in datetime format. Attempting conversion.")
         try:
             command_log['timestamp'] = pd.to_datetime(command_log['timestamp'])
         except Exception as e:
             print(f"Error converting timestamp column: {e}")
             return

    command_log = command_log.set_index('timestamp')
    commands_over_time = command_log.resample(freq).size()

    if commands_over_time.empty:
        print("Warning: No data points after resampling.")
        return

    plt.figure(figsize=(12, 6))
    commands_over_time.plot(kind='line', marker='o', linestyle='-')
    plt.title(title + f" ({freq} Frequency)")
    plt.xlabel("Time")
    plt.ylabel("Number of Commands")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_interactive_command_timeline(command_log: pd.DataFrame, title: str = "Interactive Command Timeline"):
    """
    Generates an interactive scatter plot of commands over time using Plotly.

    Args:
        command_log: Pandas DataFrame with 'timestamp' and 'command_type' columns.
        title: The title for the plot.
    """
    if command_log.empty or 'timestamp' not in command_log.columns or 'command_type' not in command_log.columns:
        print("Warning: Command log is empty or missing 'timestamp'/'command_type' columns.")
        return
        
    if not pd.api.types.is_datetime64_any_dtype(command_log['timestamp']):
         print("Warning: 'timestamp' column is not in datetime format. Attempting conversion.")
         try:
             command_log['timestamp'] = pd.to_datetime(command_log['timestamp'])
         except Exception as e:
             print(f"Error converting timestamp column: {e}")
             return

    fig = px.scatter(command_log, x='timestamp', y='command_type', color='command_type',
                     hover_data=['status', 'confirmation_status'],
                     title=title)
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Command Type",
        legend_title="Command Type"
    )
    fig.show()


def plot_web_query_length_distribution(command_log: pd.DataFrame, title: str = "Web Search Query Length Distribution"):
    """
    Generates a histogram showing the distribution of web search query lengths.

    Args:
        command_log: Pandas DataFrame with a 'query_length' column.
        title: The title for the plot.
    """
    if command_log.empty or 'query_length' not in command_log.columns:
        print("Warning: Command log is empty or missing 'query_length' column.")
        return

    query_lengths = command_log['query_length'].dropna()

    if query_lengths.empty:
        print("Warning: No query length data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(query_lengths, kde=True, bins=20, color='skyblue')
    plt.title(title)
    plt.xlabel("Query Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_intent_confusion_matrix(command_log: pd.DataFrame, title: str = "Intent Recognition Confusion Matrix"):
    """
    Generates a heatmap visualization of the confusion matrix for intent recognition.

    Args:
        command_log: Pandas DataFrame with 'true_intent' and 'predicted_intent' columns.
        title: The title for the plot.
    """
    if command_log.empty or 'true_intent' not in command_log.columns or 'predicted_intent' not in command_log.columns:
        print("Warning: Command log is empty or missing 'true_intent'/'predicted_intent' columns.")
        return

    true_intents = command_log['true_intent'].astype(str)
    predicted_intents = command_log['predicted_intent'].astype(str)
    
    labels = sorted(list(set(true_intents) | set(predicted_intents)))

    if not labels:
        print("Warning: No intent labels found.")
        return

    cm = confusion_matrix(true_intents, predicted_intents, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Optional: Print classification report
    try:
        report = classification_report(true_intents, predicted_intents, labels=labels, zero_division=0)
        print("\nIntent Classification Report:\n", report)
    except Exception as e:
        print(f"\nCould not generate classification report: {e}")


def plot_success_failure_rate_per_command(command_log: pd.DataFrame, title: str = "Success/Failure Rate per Command Type"):
    """
    Generates a stacked bar chart showing success vs. failure/cancelled rates for each command type.

    Args:
        command_log: Pandas DataFrame with 'command_type' and 'status' columns.
        title: The title for the plot.
    """
    if command_log.empty or 'command_type' not in command_log.columns or 'status' not in command_log.columns:
        print("Warning: Command log is empty or missing 'command_type'/'status' columns.")
        return

    # Define success clearly
    command_log['outcome'] = command_log['status'].apply(lambda x: 'Success' if x in ['success', 'confirmed'] else 'Failure/Cancelled')

    outcome_counts = command_log.groupby(['command_type', 'outcome']).size().unstack(fill_value=0)
    
    if outcome_counts.empty:
        print("Warning: No outcome data to plot.")
        return

    # Ensure both columns exist even if one outcome type didn't occur
    if 'Success' not in outcome_counts.columns:
        outcome_counts['Success'] = 0
    if 'Failure/Cancelled' not in outcome_counts.columns:
        outcome_counts['Failure/Cancelled'] = 0
        
    outcome_counts = outcome_counts[['Success', 'Failure/Cancelled']] # Ensure order

    outcome_counts.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
    
    plt.title(title)
    plt.xlabel("Command Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.show()


# --- Main Execution Example (for testing) ---

if __name__ == "__main__":
    print("Generating sample data and running visualization functions...")

    # Generate sample data
    sample_log_df = _simulate_command_log(num_entries=300)

    # Run visualization functions
    print("\nPlotting Command Distribution...")
    plot_command_distribution(sample_log_df)

    print("\nPlotting Command Status Distribution...")
    plot_command_status_distribution(sample_log_df)

    print("\nPlotting Deletion Confirmation Outcomes...")
    plot_confirmation_outcomes(sample_log_df)

    print("\nPlotting Command Timeline (Daily)...")
    plot_command_timeline(sample_log_df.copy(), freq='D') # Pass copy to avoid modifying original df index

    print("\nPlotting Interactive Command Timeline...")
    plot_interactive_command_timeline(sample_log_df)

    print("\nPlotting Web Query Length Distribution...")
    plot_web_query_length_distribution(sample_log_df)

    print("\nPlotting Intent Confusion Matrix...")
    plot_intent_confusion_matrix(sample_log_df)

    print("\nPlotting Success/Failure Rate per Command...")
    plot_success_failure_rate_per_command(sample_log_df)

    print("\nVisualization script finished.")