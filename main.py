import speech_recognition as sr
import pyttsx3
import os
import shutil
import sys
import time
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables if needed (e.g., for future API keys)
ASSISTANT_NAME = "Operator"
WORKING_DIRECTORY = os.getcwd() # Default to current working directory

# --- Initialization ---
try:
    # Speech Recognition
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    # Adjust for ambient noise once at the start
    with microphone as source:
        print("Operator Assistant: Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Operator Assistant: Ready.")

    # Text-to-Speech
    tts_engine = pyttsx3.init()
    # Optional: Configure voice properties (example)
    # voices = tts_engine.getProperty('voices')
    # tts_engine.setProperty('voice', voices[0].id) # Change index for different voices
    # tts_engine.setProperty('rate', 180) # Adjust speed

except Exception as e:
    print(f"Error during initialization: {e}")
    print("Please ensure microphone is connected and audio drivers are installed.")
    print("You might need to install 'portaudio' (e.g., 'sudo apt-get install portaudio19-dev' on Debian/Ubuntu or 'brew install portaudio' on macOS)")
    print("Also ensure 'pyaudio' is installed: 'pip install pyaudio'")
    sys.exit(1)

# --- Core Functions ---

def speak(text):
    """Converts text to speech."""
    print(f"{ASSISTANT_NAME}: {text}")
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in TTS: {e}")

def listen():
    """Listens for a command and transcribes it."""
    with microphone as source:
        print("Listening...")
        recognizer.pause_threshold = 1.0 # seconds of non-speaking audio before phrase is considered complete
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("No command detected.")
            return None
        except Exception as e:
            print(f"Error capturing audio: {e}")
            return None

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        speak(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error during speech recognition: {e}")
        return None

def confirm_action(action_description):
    """Asks for confirmation for a destructive action."""
    speak(f"Are you sure you want to {action_description}? Please say 'yes' to confirm or 'no' to cancel.")
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        confirmation = listen()
        if confirmation:
            if "yes" in confirmation:
                return True
            elif "no" in confirmation:
                speak("Action cancelled.")
                return False
            else:
                speak("Please respond with 'yes' or 'no'.")
        else:
            speak("I didn't catch that. Please say 'yes' or 'no'.")
        attempts += 1
    speak("Confirmation failed after multiple attempts. Action cancelled.")
    return False

# --- Command Handlers ---

def handle_file_creation(command):
    """Handles file creation commands."""
    try:
        # Basic parsing: assumes "create file filename.ext"
        parts = command.split()
        if len(parts) < 3 or parts[1] != "file":
             speak("Invalid create file command. Please say 'create file' followed by the filename.")
             return
        filename = parts[2]
        # Improve robustness for filenames potentially containing spaces later
        if len(parts) > 3:
            filename = " ".join(parts[2:]) # Basic handling for spaces in filename

        filepath = os.path.join(WORKING_DIRECTORY, filename)

        if os.path.exists(filepath):
            speak(f"File '{filename}' already exists.")
        else:
            with open(filepath, 'w') as f:
                f.write("") # Create an empty file
            speak(f"File '{filename}' created successfully in {WORKING_DIRECTORY}.")
    except IndexError:
         speak("Please specify a filename after 'create file'.")
    except OSError as e:
        speak(f"Error creating file: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during file creation: {e}")

def handle_directory_creation(command):
    """Handles directory creation commands."""
    try:
        parts = command.split()
        if len(parts) < 3 or parts[1] != "directory":
             speak("Invalid create directory command. Please say 'create directory' followed by the directory name.")
             return
        dirname = parts[2]
        if len(parts) > 3:
            dirname = " ".join(parts[2:])

        dirpath = os.path.join(WORKING_DIRECTORY, dirname)

        if os.path.exists(dirpath):
            speak(f"Directory '{dirname}' already exists.")
        else:
            os.makedirs(dirpath)
            speak(f"Directory '{dirname}' created successfully in {WORKING_DIRECTORY}.")
    except IndexError:
         speak("Please specify a directory name after 'create directory'.")
    except OSError as e:
        speak(f"Error creating directory: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during directory creation: {e}")


def handle_file_listing(command):
    """Handles file and directory listing commands."""
    try:
        target_dir = WORKING_DIRECTORY
        # Allow specifying directory: "list files in documents"
        if " in " in command:
            parts = command.split(" in ", 1)
            if len(parts) > 1:
                specified_dir = parts[1].strip()
                potential_path = os.path.join(WORKING_DIRECTORY, specified_dir)
                if os.path.isdir(potential_path):
                    target_dir = potential_path
                else:
                    speak(f"Directory '{specified_dir}' not found.")
                    return

        if not os.path.isdir(target_dir):
            speak(f"Error: Target path '{target_dir}' is not a valid directory.")
            return

        items = os.listdir(target_dir)
        if not items:
            speak(f"The directory '{os.path.basename(target_dir)}' is empty.")
        else:
            speak(f"Contents of '{os.path.basename(target_dir)}':")
            # Speak items with pauses, limit number spoken for brevity
            count = 0
            max_items_to_speak = 10
            item_list_str = ""
            for item in items:
                print(f"- {item}")
                if count < max_items_to_speak:
                    item_list_str += item + ", "
                count += 1

            if item_list_str:
                speak(item_list_str.rstrip(', '))
            if count > max_items_to_speak:
                speak(f"and {count - max_items_to_speak} more items.")

    except FileNotFoundError:
        speak("The specified directory was not found.")
    except OSError as e:
        speak(f"Error listing files: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during file listing: {e}")

def handle_file_deletion(command):
    """Handles file deletion commands with confirmation."""
    try:
        parts = command.split()
        if len(parts) < 3 or parts[1] != "file":
             speak("Invalid delete file command. Please say 'delete file' followed by the filename.")
             return
        filename = parts[2]
        if len(parts) > 3:
            filename = " ".join(parts[2:])

        filepath = os.path.join(WORKING_DIRECTORY, filename)

        if os.path.isfile(filepath):
            if confirm_action(f"delete the file '{filename}'"):
                os.remove(filepath)
                speak(f"File '{filename}' deleted successfully.")
        elif os.path.isdir(filepath):
             speak(f"'{filename}' is a directory, not a file. Use 'delete directory' instead.")
        else:
            speak(f"File '{filename}' not found.")
    except IndexError:
         speak("Please specify a filename after 'delete file'.")
    except OSError as e:
        speak(f"Error deleting file: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during file deletion: {e}")

def handle_directory_deletion(command):
    """Handles directory deletion commands with confirmation."""
    try:
        parts = command.split()
        if len(parts) < 3 or parts[1] != "directory":
             speak("Invalid delete directory command. Please say 'delete directory' followed by the directory name.")
             return
        dirname = parts[2]
        if len(parts) > 3:
            dirname = " ".join(parts[2:])

        dirpath = os.path.join(WORKING_DIRECTORY, dirname)

        if os.path.isdir(dirpath):
            if confirm_action(f"delete the directory '{dirname}' and all its contents"):
                shutil.rmtree(dirpath)
                speak(f"Directory '{dirname}' deleted successfully.")
        elif os.path.isfile(dirpath):
             speak(f"'{dirname}' is a file, not a directory. Use 'delete file' instead.")
        else:
            speak(f"Directory '{dirname}' not found.")
    except IndexError:
         speak("Please specify a directory name after 'delete directory'.")
    except OSError as e:
        speak(f"Error deleting directory: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during directory deletion: {e}")

def handle_web_query(command):
    """Handles web search queries using DuckDuckGo."""
    try:
        # Extract query: "search for X", "what is X", "look up X"
        query = ""
        if command.startswith("search for "):
            query = command[len("search for "):].strip()
        elif command.startswith("what is "):
            query = command[len("what is "):].strip()
        elif command.startswith("look up "):
            query = command[len("look up "):].strip()
        elif command.startswith("search "):
             query = command[len("search "):].strip()
        else:
            # Assume the whole command (after potential wake word) is the query if specific prefix not found
            query = command # This might need refinement based on usage patterns

        if not query:
            speak("What would you like me to search for?")
            query_response = listen()
            if query_response:
                query = query_response
            else:
                speak("Search cancelled.")
                return

        speak(f"Searching the web for '{query}'...")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3)) # Get top 3 text results

        if results:
            speak("Here's what I found:")
            # Speak the first result's body (snippet)
            first_result = results[0]
            speak(first_result.get('body', 'No summary available.'))

            # Optionally list titles of other results
            # if len(results) > 1:
            #     speak("Other results include:")
            #     for i, result in enumerate(results[1:], 1):
            #         speak(f"{i}. {result.get('title', 'No title')}")
        else:
            speak(f"Sorry, I couldn't find any results for '{query}'.")

    except Exception as e:
        speak(f"An error occurred during the web search: {e}")


# --- Main Loop ---

def run_assistant():
    """Main loop to listen for commands and execute them."""
    speak("Operator Assistant activated. How can I help you?")

    while True:
        command = listen()

        if command:
            # --- Intent Recognition ---
            if "create file" in command:
                handle_file_creation(command)
            elif "create directory" in command or "make directory" in command:
                handle_directory_creation(command)
            elif "list files" in command or "list directory" in command or "show files" in command:
                handle_file_listing(command)
            elif "delete file" in command or "remove file" in command:
                handle_file_deletion(command)
            elif "delete directory" in command or "remove directory" in command:
                handle_directory_deletion(command)
            elif "search for" in command or "what is" in command or "look up" in command or "search " in command:
                 # Check search triggers last as they are broad
                 handle_web_query(command)
            elif "stop" in command or "exit" in command or "quit" in command or "goodbye" in command:
                speak("Deactivating Operator Assistant. Goodbye!")
                break
            # Add more intents/commands here
            # elif "open application" in command:
            #    handle_app_open(command) # Example for future extension
            # elif "tell me a joke" in command:
            #    handle_joke(command) # Example for future extension
            else:
                # Basic fallback or clarification
                speak("Sorry, I didn't understand that command clearly. Can you please repeat or rephrase?")
        else:
            # Optional: Add a prompt if nothing was heard for a while
            # speak("Is there anything else?")
            pass # Continue listening silently if listen() returned None

        time.sleep(0.5) # Small delay to prevent tight looping issues


if __name__ == "__main__":
    try:
        run_assistant()
    except KeyboardInterrupt:
        print("\nOperator Assistant interrupted by user. Exiting.")
        speak("Assistant deactivated.")
    except Exception as e:
        print(f"\nAn unexpected critical error occurred: {e}")
        # Attempt to speak the error if TTS is still functional
        try:
            speak(f"A critical error occurred: {e}. Shutting down.")
        except:
            pass # Ignore TTS errors during shutdown
    finally:
        # Clean up resources if necessary
        if 'tts_engine' in locals() and tts_engine._inLoop:
             tts_engine.endLoop()