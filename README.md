# Operator Assistant

## Description
A voice-activated desktop assistant designed for efficient task execution, capable of managing local files and retrieving web information on command, 'just like the simulations.' It acts as your personal AI operator for common desktop and information retrieval tasks.

## Features
- Voice command recognition for OS tasks (file creation/listing/deletion with mandatory user confirmation for destructive actions).
- Natural language query processing for targeted web searches using search APIs or scraping, delivering concise answers.
- Real-time text-to-speech feedback for responses, confirmations, and information delivery.
- Basic intent recognition to differentiate between file operations, web queries, and other potential commands.
- Secure handling of operations, ensuring no accidental data loss via strict confirmation prompts.

## Learning Benefits
Gain practical experience in integrating speech-to-text and text-to-speech APIs/libraries, implementing basic NLP for intent recognition, interacting with the operating system via Python, performing web requests/scraping for information retrieval, and building a multi-modal application pipeline. Enhances understanding of user interaction design for voice interfaces and the importance of safety checks in automated systems.

## Technologies Used
- SpeechRecognition (or consider Vosk for offline capability)
- pyttsx3 (or gTTS/elevenlabs for higher quality voices)
- os
- shutil
- requests
- beautifulsoup4 (or specific search engine APIs like duckduckgo_search)
- python-dotenv (for managing any potential API keys)
- sounddevice (or pyaudio for audio I/O)

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/Omdeepb69/operator-assistant.git
cd operator-assistant

# Install dependencies
pip install -r requirements.txt
```

## Usage
[Instructions on how to use the project]

## Project Structure
[Brief explanation of the project structure]

## License
MIT
