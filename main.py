import speech_recognition as sr
import pyttsx3
import os
import shutil
import sys
import time
import threading
import keyboard
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
ASSISTANT_NAME = "Operator"
WORKING_DIRECTORY = os.getcwd()  # Default to current working directory
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get API key from .env file
LISTENING_ACTIVE = False
GEMINI_MODEL = "gemini-2.5-pro-exp-03-25"

# --- Initialize Gemini AI API ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with GEMINI_API_KEY=your_api_key")

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

def listen(timeout=5):
    """Listens for a command and transcribes it."""
    global LISTENING_ACTIVE
    
    if not LISTENING_ACTIVE:
        return None
        
    with microphone as source:
        print("Listening...")
        recognizer.pause_threshold = 1.0  # seconds of non-speaking audio before phrase is considered complete
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
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
        
        # Check for "over and out" to stop listening
        if "over and out" in command.lower():
            LISTENING_ACTIVE = False
            speak("Listening mode deactivated.")
            return "over and out"
            
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

def ask_question(question):
    """Asks a question and returns the response."""
    speak(question)
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        response = listen(timeout=10)  # Longer timeout for detailed responses
        if response and response != "over and out":
            return response
        elif response == "over and out":
            return None
        else:
            speak("I didn't catch that. Could you please respond?")
        attempts += 1
    speak("Failed to get a response after multiple attempts.")
    return None

def generate_ai_content(prompt):
    """Generate content using Gemini AI."""
    if not GEMINI_API_KEY:
        speak("Gemini API key not configured. Please set up your API key in the .env file.")
        return None
        
    try:
        speak("Generating content with Gemini AI...")
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        speak(f"Error generating content with Gemini AI: {e}")
        return None

def keyboard_listener():
    """Listen for keyboard 'i' press to activate listening mode."""
    global LISTENING_ACTIVE
    
    def on_i_press(event):
        global LISTENING_ACTIVE
        if event.name == 'i' and not LISTENING_ACTIVE:
            LISTENING_ACTIVE = True
            print("\nListening mode activated (press 'i').")
            speak("Listening mode activated. Say 'over and out' to stop listening.")
    
    keyboard.on_press(on_i_press)
    
    # Keep the keyboard listener running
    keyboard.wait()

# --- Command Handlers ---

def handle_file_creation(command):
    """Handles file creation commands with more dynamic options."""
    try:
        # Extract path information
        # Pattern: "on my system create file [filename] in [directory]"
        parts = command.split()
        
        # Find the 'file' keyword position
        try:
            file_index = parts.index("file")
        except ValueError:
            speak("I didn't catch the file name. Please specify a file to create.")
            return
            
        # Check if directory is specified
        dir_path = WORKING_DIRECTORY
        if "in" in parts[file_index+1:]:
            in_index = parts.index("in", file_index)
            filename = " ".join(parts[file_index+1:in_index])
            dirname = " ".join(parts[in_index+1:])
            
            # Ask for directory clarification if needed
            if not dirname or dirname.isspace():
                dirname = ask_question("In which directory should I create this file?")
                if not dirname:
                    speak("File creation cancelled.")
                    return
                    
            dir_path = os.path.join(WORKING_DIRECTORY, dirname)
            
            # Create directory if it doesn't exist
            if not os.path.exists(dir_path):
                should_create = ask_question(f"Directory '{dirname}' doesn't exist. Should I create it? Say yes or no.")
                if should_create and "yes" in should_create:
                    try:
                        os.makedirs(dir_path)
                        speak(f"Created directory '{dirname}'.")
                    except Exception as e:
                        speak(f"Failed to create directory: {e}")
                        return
                else:
                    speak("File creation cancelled.")
                    return
        else:
            filename = " ".join(parts[file_index+1:])
        
        # Ask for filename if not provided
        if not filename or filename.isspace():
            filename = ask_question("What should be the name of the file?")
            if not filename:
                speak("File creation cancelled.")
                return
                
        filepath = os.path.join(dir_path, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            speak(f"File '{filename}' already exists.")
            should_overwrite = ask_question("Would you like to overwrite it? Say yes or no.")
            if not should_overwrite or "no" in should_overwrite:
                speak("File creation cancelled.")
                return
        
        # Ask if user wants to add content
        should_add_content = ask_question("Would you like me to generate content for this file using Gemini AI? Say yes or no.")
        
        content = ""
        if should_add_content and "yes" in should_add_content:
            content_prompt = ask_question("What would you like the content to be about?")
            if content_prompt:
                content = generate_ai_content(content_prompt)
                if not content:
                    speak("Failed to generate content. Creating empty file instead.")
                    content = ""
        
        # Create the file
        with open(filepath, 'w') as f:
            f.write(content or "")
            
        speak(f"File '{filename}' created successfully in {os.path.basename(dir_path)}.")
        
    except Exception as e:
        speak(f"An unexpected error occurred during file creation: {e}")

def handle_file_editing(command):
    """Handles file editing commands."""
    try:
        # Extract path information 
        parts = command.split()
        
        # Find the 'file' keyword position
        try:
            file_index = parts.index("file")
        except ValueError:
            speak("I didn't catch which file to edit. Please specify a file.")
            return
            
        # Get filename
        if len(parts) <= file_index + 1:
            filename = ask_question("Which file would you like to edit?")
            if not filename:
                speak("File editing cancelled.")
                return
        else:
            filename = " ".join(parts[file_index+1:])
            
        filepath = os.path.join(WORKING_DIRECTORY, filename)
        
        # Check if file exists
        if not os.path.isfile(filepath):
            # Check if it's in a subdirectory
            found = False
            for root, dirs, files in os.walk(WORKING_DIRECTORY):
                if filename in files:
                    filepath = os.path.join(root, filename)
                    found = True
                    break
                    
            if not found:
                speak(f"File '{filename}' not found. Would you like to create it instead?")
                create_instead = listen()
                if create_instead and "yes" in create_instead:
                    handle_file_creation(f"on my system create file {filename}")
                return
        
        # Ask what kind of edit to perform
        edit_type = ask_question("What would you like to do with this file? Options: append content, replace content, or view content.")
        
        if not edit_type:
            speak("File editing cancelled.")
            return
            
        # Handle different edit types
        if "append" in edit_type:
            content_prompt = ask_question("What content would you like to append? I can generate content with AI or you can dictate it.")
            if not content_prompt:
                speak("Editing cancelled.")
                return
                
            if "generate" in content_prompt or "ai" in content_prompt:
                prompt = ask_question("What should the generated content be about?")
                if prompt:
                    content = generate_ai_content(prompt)
                    if content:
                        with open(filepath, 'a') as f:
                            f.write("\n" + content)
                        speak(f"Content appended to '{filename}'.")
                    else:
                        speak("Failed to generate content. No changes made.")
            else:
                with open(filepath, 'a') as f:
                    f.write("\n" + content_prompt)
                speak(f"Content appended to '{filename}'.")
                
        elif "replace" in edit_type:
            content_prompt = ask_question("What content would you like to add? I can generate content with AI or you can dictate it.")
            if not content_prompt:
                speak("Editing cancelled.")
                return
                
            if "generate" in content_prompt or "ai" in content_prompt:
                prompt = ask_question("What should the generated content be about?")
                if prompt:
                    content = generate_ai_content(prompt)
                    if content:
                        with open(filepath, 'w') as f:
                            f.write(content)
                        speak(f"Content replaced in '{filename}'.")
                    else:
                        speak("Failed to generate content. No changes made.")
            else:
                with open(filepath, 'w') as f:
                    f.write(content_prompt)
                speak(f"Content replaced in '{filename}'.")
                
        elif "view" in edit_type:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                if content:
                    speak(f"Here's the content of '{filename}':")
                    print(f"\n--- Content of {filename} ---")
                    print(content)
                    print(f"--- End of {filename} ---\n")
                    speak("Content displayed in the console.")
                else:
                    speak(f"The file '{filename}' is empty.")
            except Exception as e:
                speak(f"Error reading file: {e}")
        else:
            speak("I didn't understand the edit type. Please try again with 'append', 'replace', or 'view'.")
            
    except Exception as e:
        speak(f"An unexpected error occurred during file editing: {e}")

def handle_directory_creation(command):
    """Handles directory creation commands with more dynamic options."""
    try:
        # Extract directory name
        parts = command.split()
        
        # Find the 'directory' keyword position
        try:
            dir_index = parts.index("directory")
        except ValueError:
            speak("I didn't catch the directory name. Please specify a directory to create.")
            return
            
        # Check if parent directory is specified with "in"
        parent_dir = WORKING_DIRECTORY
        if "in" in parts[dir_index+1:]:
            in_index = parts.index("in", dir_index)
            dirname = " ".join(parts[dir_index+1:in_index])
            parent_dirname = " ".join(parts[in_index+1:])
            
            # Ask for parent directory clarification if needed
            if not parent_dirname or parent_dirname.isspace():
                parent_dirname = ask_question("In which parent directory should I create this directory?")
                if not parent_dirname:
                    speak("Directory creation cancelled.")
                    return
                    
            parent_dir = os.path.join(WORKING_DIRECTORY, parent_dirname)
            
            # Create parent directory if it doesn't exist
            if not os.path.exists(parent_dir):
                should_create = ask_question(f"Parent directory '{parent_dirname}' doesn't exist. Should I create it? Say yes or no.")
                if should_create and "yes" in should_create:
                    try:
                        os.makedirs(parent_dir)
                        speak(f"Created parent directory '{parent_dirname}'.")
                    except Exception as e:
                        speak(f"Failed to create parent directory: {e}")
                        return
                else:
                    speak("Directory creation cancelled.")
                    return
        else:
            dirname = " ".join(parts[dir_index+1:])
        
        # Ask for directory name if not provided
        if not dirname or dirname.isspace():
            dirname = ask_question("What should be the name of the directory?")
            if not dirname:
                speak("Directory creation cancelled.")
                return
                
        dirpath = os.path.join(parent_dir, dirname)
        
        # Check if directory already exists
        if os.path.exists(dirpath):
            speak(f"Directory '{dirname}' already exists.")
            return
        
        # Create the directory
        os.makedirs(dirpath)
        speak(f"Directory '{dirname}' created successfully in {os.path.basename(parent_dir)}.")
        
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
        
        # Find the 'file' keyword position
        try:
            file_index = parts.index("file")
        except ValueError:
            speak("I didn't catch the file name. Please specify a file to delete.")
            return
            
        # Check if filename is provided
        if len(parts) <= file_index + 1:
            filename = ask_question("Which file would you like to delete?")
            if not filename:
                speak("File deletion cancelled.")
                return
        else:
            filename = " ".join(parts[file_index+1:])

        filepath = os.path.join(WORKING_DIRECTORY, filename)

        if os.path.isfile(filepath):
            if confirm_action(f"delete the file '{filename}'"):
                os.remove(filepath)
                speak(f"File '{filename}' deleted successfully.")
        elif os.path.isdir(filepath):
             speak(f"'{filename}' is a directory, not a file. Use 'delete directory' instead.")
        else:
            # Check if the file exists in any subdirectory
            found = False
            for root, dirs, files in os.walk(WORKING_DIRECTORY):
                if filename in files:
                    filepath = os.path.join(root, filename)
                    found = True
                    if confirm_action(f"delete the file '{filename}' located in {os.path.relpath(root, WORKING_DIRECTORY)}"):
                        os.remove(filepath)
                        speak(f"File '{filename}' deleted successfully.")
                    break
                    
            if not found:
                speak(f"File '{filename}' not found.")
    except OSError as e:
        speak(f"Error deleting file: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred during file deletion: {e}")

def handle_directory_deletion(command):
    """Handles directory deletion commands with confirmation."""
    try:
        parts = command.split()
        
        # Find the 'directory' keyword position
        try:
            dir_index = parts.index("directory")
        except ValueError:
            speak("I didn't catch the directory name. Please specify a directory to delete.")
            return
            
        # Check if directory name is provided
        if len(parts) <= dir_index + 1:
            dirname = ask_question("Which directory would you like to delete?")
            if not dirname:
                speak("Directory deletion cancelled.")
                return
        else:
            dirname = " ".join(parts[dir_index+1:])

        dirpath = os.path.join(WORKING_DIRECTORY, dirname)

        if os.path.isdir(dirpath):
            if confirm_action(f"delete the directory '{dirname}' and all its contents"):
                shutil.rmtree(dirpath)
                speak(f"Directory '{dirname}' deleted successfully.")
        elif os.path.isfile(dirpath):
             speak(f"'{dirname}' is a file, not a directory. Use 'delete file' instead.")
        else:
            speak(f"Directory '{dirname}' not found.")
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

def handle_ai_query(command):
    """Handles AI queries using Gemini."""
    try:
        # Extract query: "ask ai X", "tell me X", etc.
        query = command
        
        if not query:
            speak("What would you like to ask Gemini AI?")
            query_response = listen()
            if query_response:
                query = query_response
            else:
                speak("Query cancelled.")
                return

        speak(f"Asking Gemini AI about '{query}'...")
        response = generate_ai_content(query)
        
        if response:
            speak("Here's what Gemini AI says:")
            # Split long responses into chunks to avoid TTS issues
            max_chunk_size = 500
            for i in range(0, len(response), max_chunk_size):
                chunk = response[i:i+max_chunk_size]
                print(chunk)
                speak(chunk)
        else:
            speak("Sorry, I couldn't get a response from Gemini AI.")

    except Exception as e:
        speak(f"An error occurred while querying Gemini AI: {e}")

# --- Main Loop ---

def run_assistant():
    """Main loop to listen for commands and execute them."""
    global LISTENING_ACTIVE
    
    speak("Operator Assistant activated. Press 'i' to start listening. Say 'over and out' to stop listening.")
    
    # Start keyboard listener in a separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    while True:
        try:
            if LISTENING_ACTIVE:
                command = listen()
                
                if command == "over and out":
                    continue  # Already handled in listen()
                elif command:
                    # --- Intent Recognition ---
                    if "on my system" in command:
                        # Handle system commands
                        if "create file" in command or "make file" in command:
                            handle_file_creation(command)
                        elif "edit file" in command or "modify file" in command:
                            handle_file_editing(command)
                        elif "create directory" in command or "make directory" in command:
                            handle_directory_creation(command)
                        elif "list files" in command or "list directory" in command or "show files" in command:
                            handle_file_listing(command)
                        elif "delete file" in command or "remove file" in command:
                            handle_file_deletion(command)
                        elif "delete directory" in command or "remove directory" in command:
                            handle_directory_deletion(command)
                        else:
                            speak("Sorry, I didn't understand that system command. Please try again.")
                    elif "search for" in command or "what is" in command or "look up" in command or "search " in command:
                        handle_web_query(command)
                    elif "ask ai" in command or "ask gemini" in command or "tell me about" in command:
                        handle_ai_query(command)
                    elif "stop" in command or "exit" in command or "quit" in command or "goodbye" in command:
                        speak("Deactivating Operator Assistant. Goodbye!")
                        break
                    else:
                        # Forward to Gemini AI for general queries
                        handle_ai_query(command)
            else:
                # Wait a bit to avoid tight loop when not listening
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            speak("Assistant interrupted by user. Press 'i' to start listening again or Ctrl+C again to exit.")
            try:
                time.sleep(2)  # Give time to decide
            except KeyboardInterrupt:
                speak("Shutting down Operator Assistant. Goodbye!")
                break

if __name__ == "__main__":
    try:
        # Check for required modules
        required_modules = {
            "speech_recognition": sr,
            "pyttsx3": pyttsx3,
            "keyboard": keyboard,
            "duckduckgo_search": DDGS,
            "google.generativeai": genai,
            "dotenv": load_dotenv
        }
        
        missing_modules = []
        for name, module in required_modules.items():
            if module is None:
                missing_modules.append(name)
                
        if missing_modules:
            print("The following required modules are missing:")
            for module in missing_modules:
                print(f"- {module}")
            print("\nPlease install them using pip:")
            print(f"pip install {' '.join(missing_modules)}")
            sys.exit(1)
            
        # Check for API key
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not found!")
            print("Some features will not work without the Gemini API key.")
            print("Please create a .env file in the same directory with:")
            print("GEMINI_API_KEY=your_api_key_here")
            
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
        if 'tts_engine' in locals():
            try:
                if hasattr(tts_engine, '_inLoop') and tts_engine._inLoop:
                    tts_engine.endLoop()
            except:
                pass