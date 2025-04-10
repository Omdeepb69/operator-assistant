import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration Management ---

DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "log_level": "INFO",
    "log_file": "operator_assistant.log",
    "search_engine": "duckduckgo", # Example, replace with actual API if needed
    "tts_engine": "espeak", # Example, replace with actual TTS library/API
    "confirmation_required": True,
    "visualization_style": "seaborn-v0_8-darkgrid",
}

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Ensure all default keys are present
            for key, value in DEFAULT_CONFIG.items():
                config.setdefault(key, value)
            return config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Creating with defaults.")
        save_config(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError:
        print(f"Error: Config file '{config_path}' is corrupted. Using defaults.")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Saves configuration to a JSON file."""
    try:
        pathlib.Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config to '{config_path}': {e}")

# --- Logging Setup ---

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configures logging based on the loaded configuration."""
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file = config.get("log_file", "operator_assistant.log")

    logger = logging.getLogger("OperatorAssistant")
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if called again
    if not logger.handlers:
        # File Handler
        try:
            pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging to '{log_file}': {e}")


        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

# --- User Interaction & Confirmation ---

def get_confirmation(prompt: str) -> bool:
    """Asks the user for confirmation (yes/no)."""
    while True:
        try:
            response = input(f"{prompt} [y/N]: ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        except EOFError: # Handle cases where input stream is closed (e.g., piping)
             print("\nConfirmation cancelled due to EOF.")
             return False
        except KeyboardInterrupt:
             print("\nConfirmation cancelled by user.")
             return False


# --- File Operations ---

def create_file(file_path: str, content: str = "", overwrite: bool = False) -> Tuple[bool, str]:
    """Creates a file, optionally with content."""
    logger = logging.getLogger("OperatorAssistant")
    path = pathlib.Path(file_path)
    if path.exists() and not overwrite:
        msg = f"File '{file_path}' already exists. Use overwrite=True to replace."
        logger.warning(msg)
        return False, msg
    if path.exists() and path.is_dir():
        msg = f"Path '{file_path}' exists and is a directory, cannot create file."
        logger.error(msg)
        return False, msg

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        msg = f"File '{file_path}' created successfully."
        logger.info(msg)
        return True, msg
    except OSError as e:
        msg = f"Error creating file '{file_path}': {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred while creating file '{file_path}': {e}"
        logger.exception(msg) # Log full traceback for unexpected errors
        return False, msg


def list_directory(dir_path: str) -> Tuple[Optional[List[str]], str]:
    """Lists the contents of a directory."""
    logger = logging.getLogger("OperatorAssistant")
    path = pathlib.Path(dir_path)
    if not path.exists():
        msg = f"Directory '{dir_path}' not found."
        logger.error(msg)
        return None, msg
    if not path.is_dir():
        msg = f"Path '{dir_path}' is not a directory."
        logger.error(msg)
        return None, msg

    try:
        contents = [item.name for item in path.iterdir()]
        msg = f"Contents of '{dir_path}': {len(contents)} items."
        logger.info(msg)
        return contents, msg
    except OSError as e:
        msg = f"Error listing directory '{dir_path}': {e}"
        logger.error(msg)
        return None, msg
    except Exception as e:
        msg = f"An unexpected error occurred while listing directory '{dir_path}': {e}"
        logger.exception(msg)
        return None, msg


def delete_item(item_path: str, force: bool = False, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Deletes a file or directory, requiring confirmation unless force=True."""
    logger = logging.getLogger("OperatorAssistant")
    path = pathlib.Path(item_path)
    effective_config = config or load_config() # Load if not provided
    confirmation_required = effective_config.get("confirmation_required", True)

    if not path.exists():
        msg = f"Item '{item_path}' not found."
        logger.error(msg)
        return False, msg

    item_type = "directory" if path.is_dir() else "file"
    prompt = f"WARNING: Are you sure you want to permanently delete the {item_type} '{item_path}'?"

    if not force and confirmation_required:
        if not get_confirmation(prompt):
            msg = f"Deletion of '{item_path}' cancelled by user."
            logger.info(msg)
            return False, msg

    try:
        if path.is_dir():
            shutil.rmtree(path)
            msg = f"Directory '{item_path}' deleted successfully."
        else:
            path.unlink()
            msg = f"File '{item_path}' deleted successfully."
        logger.warning(f"DELETED: {item_type.capitalize()} '{item_path}'") # Log deletions clearly
        return True, msg
    except OSError as e:
        msg = f"Error deleting {item_type} '{item_path}': {e}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred while deleting {item_type} '{item_path}': {e}"
        logger.exception(msg)
        return False, msg

# --- Data Visualization ---

def plot_simple_bar_chart(data: Dict[str, Union[int, float]], title: str, xlabel: str, ylabel: str, filename: Optional[str] = None, style: str = "seaborn-v0_8-darkgrid") -> None:
    """Generates and optionally saves a simple bar chart."""
    logger = logging.getLogger("OperatorAssistant")
    try:
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(data.keys())
        values = list(data.values())
        sns.barplot(x=categories, y=values, ax=ax, palette="viridis")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if filename:
            pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename)
            logger.info(f"Bar chart saved to '{filename}'")
        else:
            plt.show()
        plt.close(fig) # Close the figure to free memory
    except ImportError:
        logger.error("Plotting libraries (matplotlib, seaborn) not installed. Cannot generate chart.")
    except Exception as e:
        logger.error(f"Error generating bar chart: {e}")


def plot_time_series(dates: List[Any], values: List[Union[int, float]], title: str, ylabel: str, filename: Optional[str] = None, style: str = "seaborn-v0_8-darkgrid") -> None:
    """Generates and optionally saves a time series line plot."""
    logger = logging.getLogger("OperatorAssistant")
    try:
        plt.style.use(style)
        df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Value': values})
        df = df.sort_values('Date')

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='Date', y='Value', data=df, ax=ax, marker='o')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if filename:
            pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename)
            logger.info(f"Time series plot saved to '{filename}'")
        else:
            plt.show()
        plt.close(fig) # Close the figure to free memory
    except ImportError:
        logger.error("Plotting libraries (matplotlib, seaborn, pandas) not installed. Cannot generate plot.")
    except Exception as e:
        logger.error(f"Error generating time series plot: {e}")


# --- Metrics Calculation ---

def calculate_accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculates simple accuracy."""
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")
    if not y_true:
        return 1.0 # Or 0.0, depending on definition for empty lists
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def calculate_mean_std(data: List[Union[int, float]]) -> Tuple[float, float]:
    """Calculates mean and standard deviation."""
    if not data:
        return 0.0, 0.0
    arr = np.array(data)
    return float(np.mean(arr)), float(np.std(arr))


# --- Placeholder/Integration Functions ---

def recognize_speech_from_mic() -> Optional[str]:
    """Placeholder for voice command recognition."""
    logger = logging.getLogger("OperatorAssistant")
    logger.info("Attempting to recognize speech (placeholder)...")
    # In a real implementation, use libraries like speech_recognition
    # Example:
    # import speech_recognition as sr
    # r = sr.Recognizer()
    # with sr.Microphone() as source:
    #     print("Say something!")
    #     audio = r.listen(source)
    # try:
    #     text = r.recognize_google(audio) # Requires internet and API key setup
    #     logger.info(f"Recognized speech: {text}")
    #     return text
    # except sr.UnknownValueError:
    #     logger.warning("Could not understand audio")
    #     return None
    # except sr.RequestError as e:
    #     logger.error(f"Could not request results from speech service; {e}")
    #     return None
    print("(Placeholder) Please type your command:")
    try:
        text = input("> ")
        return text if text else None
    except EOFError:
        return None
    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return None


def process_nlp_query(query: str, config: Dict[str, Any]) -> str:
    """Placeholder for NLP processing and web search."""
    logger = logging.getLogger("OperatorAssistant")
    logger.info(f"Processing NLP query (placeholder): '{query}'")
    # In a real implementation, use NLP libraries (spaCy, NLTK) and search APIs
    # (Google Search API, DuckDuckGo API, requests/BeautifulSoup for scraping)
    search_engine = config.get("search_engine", "duckduckgo")

    # Basic example: just return a search link
    if search_engine == "duckduckgo":
        search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
        result = f"I couldn't find a direct answer, but you can search on DuckDuckGo: {search_url}"
    elif search_engine == "google": # Requires API setup
         search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
         result = f"I couldn't find a direct answer, but you can search on Google: {search_url}"
    else:
        result = f"I cannot process web queries with the configured engine: {search_engine}."

    # A more advanced version would try to extract a concise answer.
    # Example (very basic scraping - needs requests/bs4 installed):
    # try:
    #     import requests
    #     from bs4 import BeautifulSoup
    #     headers = {'User-Agent': 'Mozilla/5.0'}
    #     response = requests.get(search_url, headers=headers, timeout=5)
    #     response.raise_for_status()
    #     soup = BeautifulSoup(response.text, 'html.parser')
    #     # Try to find a relevant snippet (this is highly site-dependent and fragile)
    #     # Look for definition, featured snippet, etc. This needs specific selectors.
    #     # E.g., find first paragraph <p> tag text
    #     first_p = soup.find('p')
    #     if first_p and first_p.get_text(strip=True):
    #         result = f"Here's a snippet I found: {first_p.get_text(strip=True)[:200]}..." # Limit length
    #     else:
    #          result = f"I searched online but couldn't extract a concise answer. Try this link: {search_url}"
    # except ImportError:
    #     logger.warning("Web scraping libraries (requests, beautifulsoup4) not installed.")
    # except requests.RequestException as e:
    #     logger.error(f"Web search failed: {e}")
    #     result = f"Sorry, I encountered an error trying to search online: {e}"
    # except Exception as e:
    #      logger.error(f"Error during web query processing: {e}")
    #      result = f"Sorry, an unexpected error occurred during the search."

    logger.info(f"NLP query result: {result[:100]}...") # Log truncated result
    return result


def speak_text(text: str, config: Dict[str, Any]) -> None:
    """Placeholder for text-to-speech output."""
    logger = logging.getLogger("OperatorAssistant")
    tts_engine = config.get("tts_engine", "print") # Default to printing if not specified
    logger.info(f"Speaking (TTS Engine: {tts_engine}): '{text}'")

    if not text:
        return

    try:
        if tts_engine == "espeak" and shutil.which("espeak"):
             # Basic espeak integration, works on Linux/macOS if installed
            subprocess.run(["espeak", text], check=True, capture_output=True)
        elif tts_engine == "say" and sys.platform == "darwin" and shutil.which("say"):
            # Basic macOS 'say' command
            subprocess.run(["say", text], check=True, capture_output=True)
        # Add other TTS engines here (e.g., pyttsx3, gTTS API)
        # elif tts_engine == "pyttsx3":
        #     import pyttsx3
        #     engine = pyttsx3.init()
        #     engine.say(text)
        #     engine.runAndWait()
        else:
            # Fallback to printing the text
            print(f"Assistant: {text}")
    except FileNotFoundError:
         logger.warning(f"TTS engine '{tts_engine}' command not found. Falling back to print.")
         print(f"Assistant: {text}")
    except subprocess.CalledProcessError as e:
        logger.error(f"TTS engine '{tts_engine}' failed: {e}. Stderr: {e.stderr.decode()}")
        print(f"Assistant: {text}") # Fallback on error
    except ImportError:
         logger.warning(f"Required library for TTS engine '{tts_engine}' not installed. Falling back to print.")
         print(f"Assistant: {text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during TTS: {e}")
        print(f"Assistant: {text}") # Fallback on error


def recognize_intent(command: str) -> Tuple[str, Dict[str, Any]]:
    """Basic intent recognition based on keywords."""
    logger = logging.getLogger("OperatorAssistant")
    command_lower = command.lower().strip()
    args = {}

    # File Operations
    create_match = re.match(r"(create|make) file (?:named |called )?['\"]?(.+?)['\"]?(?: with content ['\"]?(.*?)['\"]?)?$", command_lower)
    list_match = re.match(r"(list|show) files? (?:in|for) ['\"]?(.+?)['\"]?$", command_lower)
    delete_match = re.match(r"(delete|remove|rm) (file|directory|folder) ['\"]?(.+?)['\"]?$", command_lower)

    # Web Query
    query_match = re.match(r"(search for|what is|who is|tell me about|look up) ['\"]?(.+?)['\"]?$", command_lower)

    # Other potential commands
    stop_match = re.match(r"(stop|exit|quit|goodbye)", command_lower)

    intent = "unknown"

    if create_match:
        intent = "create_file"
        args['file_path'] = create_match.group(2).strip()
        args['content'] = create_match.group(3).strip() if create_match.group(3) else ""
        args['overwrite'] = "overwrite" in command_lower # Basic check
    elif list_match:
        intent = "list_directory"
        args['dir_path'] = list_match.group(2).strip()
    elif delete_match:
        intent = "delete_item"
        args['item_path'] = delete_match.group(3).strip()
        args['force'] = "force" in command_lower or "without confirmation" in command_lower # Basic check
    elif query_match:
        intent = "web_query"
        args['query'] = query_match.group(2).strip()
    elif stop_match:
        intent = "stop"
    else:
        # Default to web query if no specific keywords match file ops
        if len(command_lower) > 5: # Avoid triggering on very short inputs
             intent = "web_query"
             args['query'] = command # Use original casing for search query
        else:
             intent = "unknown"


    logger.debug(f"Recognized intent: {intent}, Args: {args}")
    return intent, args


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Example usage or tests when running utils.py directly
    config = load_config()
    logger = setup_logging(config)
    logger.info("Utils module loaded directly. Running example usage.")

    # Example: Speak welcome message
    speak_text("Utility module initialized.", config)

    # Example: File operation with confirmation
    test_file = "temp_test_file.txt"
    created, msg = create_file(test_file, "Hello from utils!")
    speak_text(msg, config)

    if created:
        listed_files, msg = list_directory(".")
        if listed_files:
             speak_text(f"Files in current directory: {', '.join(listed_files[:5])}...", config) # Show first 5
        else:
             speak_text(msg, config)

        deleted, msg = delete_item(test_file, config=config) # Will ask for confirmation
        speak_text(msg, config)

    # Example: Intent Recognition
    commands_to_test = [
        "create file 'my_document.txt' with content 'This is a test.'",
        "list files in /tmp",
        "delete file 'old_log.log'",
        "search for the capital of France",
        "what is the weather today",
        "stop"
    ]
    for cmd in commands_to_test:
        intent, args = recognize_intent(cmd)
        logger.info(f"Command: '{cmd}' -> Intent: {intent}, Args: {args}")
        speak_text(f"Recognized intent {intent} for command.", config)
        time.sleep(0.5) # Small delay for TTS

    # Example: Visualization (requires matplotlib/seaborn)
    try:
        sample_data = {'Category A': 25, 'Category B': 40, 'Category C': 15, 'Category D': 30}
        plot_simple_bar_chart(sample_data, "Sample Bar Chart", "Categories", "Values", filename="sample_bar_chart.png", style=config.get("visualization_style"))
        speak_text("Generated sample bar chart.", config)
    except Exception as e:
        logger.error(f"Could not generate sample plot: {e}")


    # Example: Metrics
    true_labels = ['A', 'B', 'A', 'C', 'B', 'A']
    pred_labels = ['A', 'B', 'B', 'C', 'B', 'A']
    acc = calculate_accuracy(true_labels, pred_labels)
    logger.info(f"Sample Accuracy: {acc:.2f}")
    speak_text(f"Calculated sample accuracy: {acc:.2f}", config)

    numeric_data = [1.2, 1.5, 1.1, 1.8, 2.0, 1.4, 1.6]
    mean, std = calculate_mean_std(numeric_data)
    logger.info(f"Sample Data Mean: {mean:.2f}, Std Dev: {std:.2f}")
    speak_text(f"Calculated sample mean {mean:.2f} and standard deviation {std:.2f}", config)

    logger.info("Example usage finished.")
    speak_text("Utility module examples complete.", config)