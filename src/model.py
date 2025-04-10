import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import NotFittedError

# Define intents
INTENT_FILE_CREATE = "file_create"
INTENT_FILE_LIST = "file_list"
INTENT_FILE_DELETE = "file_delete"
INTENT_WEB_SEARCH = "web_search"
INTENT_GREETING = "greeting"
INTENT_GOODBYE = "goodbye"
INTENT_UNKNOWN = "unknown" # Fallback intent

# Default model file path
DEFAULT_MODEL_PATH = "intent_classifier_model.joblib"

# Sample training data (replace with a larger, more diverse dataset for production)
TRAINING_DATA = {
    "text": [
        "create a file named report.txt", "make a new document called notes", "generate an empty file",
        "list all files in the current directory", "show me the files here", "what files are present",
        "delete the file temporary.log", "remove the document old_notes.doc", "erase config.bak",
        "search the web for python tutorials", "find information about large language models", "what is the weather today", "look up the capital of France",
        "hello assistant", "hi there", "good morning",
        "goodbye", "exit", "see you later",
        "tell me a joke", "what time is it", "open calculator" # Examples that might fall into unknown or need specific handling later
    ],
    "intent": [
        INTENT_FILE_CREATE, INTENT_FILE_CREATE, INTENT_FILE_CREATE,
        INTENT_FILE_LIST, INTENT_FILE_LIST, INTENT_FILE_LIST,
        INTENT_FILE_DELETE, INTENT_FILE_DELETE, INTENT_FILE_DELETE,
        INTENT_WEB_SEARCH, INTENT_WEB_SEARCH, INTENT_WEB_SEARCH, INTENT_WEB_SEARCH,
        INTENT_GREETING, INTENT_GREETING, INTENT_GREETING,
        INTENT_GOODBYE, INTENT_GOODBYE, INTENT_GOODBYE,
        INTENT_UNKNOWN, INTENT_UNKNOWN, INTENT_UNKNOWN
    ]
}

class IntentClassifier:
    """
    A classifier to determine the user's intent from text commands.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._pipeline = None
        self._is_fitted = False
        self._load_model() # Attempt to load existing model on initialization

    def _build_pipeline(self):
        """Builds the scikit-learn pipeline."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('clf', LogisticRegression(solver='liblinear', multi_class='auto', random_state=42))
        ])

    def train(self, data: pd.DataFrame = None, test_size: float = 0.2, tune_hyperparameters: bool = True):
        """
        Trains the intent classification model.

        Args:
            data (pd.DataFrame, optional): DataFrame with 'text' and 'intent' columns.
                                           Defaults to internal TRAINING_DATA.
            test_size (float): Proportion of the dataset to include in the test split.
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
        """
        if data is None:
            data = pd.DataFrame(TRAINING_DATA)

        if not all(col in data.columns for col in ['text', 'intent']):
            raise ValueError("Data must contain 'text' and 'intent' columns.")

        X = data['text']
        y = data['intent']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        self._pipeline = self._build_pipeline()

        if tune_hyperparameters:
            # Define parameter grid for GridSearchCV
            # Reduced grid for faster example execution
            param_grid = {
                'tfidf__max_df': [0.9, 1.0],
                'tfidf__min_df': [1, 2],
                'clf__C': [1, 10], # Regularization strength
                'clf__penalty': ['l1', 'l2']
            }

            print("Starting hyperparameter tuning...")
            grid_search = GridSearchCV(self._pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy') # Use cv=3 for small dataset
            grid_search.fit(X_train, y_train)

            print(f"Best parameters found: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            self._pipeline = grid_search.best_estimator_ # Use the best found pipeline
        else:
            print("Starting training with default parameters...")
            self._pipeline.fit(X_train, y_train)

        self._is_fitted = True
        print("Training complete.")
        self.evaluate(X_test, y_test)
        self.save_model()

    def evaluate(self, X_test: pd.Series, y_test: pd.Series):
        """
        Evaluates the trained model on the test set.

        Args:
            X_test (pd.Series): Test features (text commands).
            y_test (pd.Series): True test labels (intents).
        """
        if not self._is_fitted or self._pipeline is None:
            print("Model is not trained yet. Please train the model before evaluating.")
            return

        try:
            y_pred = self._pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)

            print("\nModel Evaluation Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
        except NotFittedError:
             print("Model is not fitted properly. Cannot evaluate.")
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")


    def predict(self, text: str) -> str:
        """
        Predicts the intent for a given text command.

        Args:
            text (str): The user's text command.

        Returns:
            str: The predicted intent label. Returns INTENT_UNKNOWN if model is not ready or prediction fails.
        """
        if not self._is_fitted or self._pipeline is None:
            print("Warning: Model not loaded or trained. Returning 'unknown' intent.")
            # Attempt to load if not fitted
            if not self._load_model():
                 return INTENT_UNKNOWN # Return unknown if load fails

        if not self._is_fitted: # Check again after load attempt
             print("Warning: Model could not be loaded. Returning 'unknown' intent.")
             return INTENT_UNKNOWN

        try:
            # Predict expects an iterable, so pass the text in a list
            prediction = self._pipeline.predict([text])
            return prediction[0]
        except NotFittedError:
             print("Error: Model seems not fitted correctly even after loading. Returning 'unknown'.")
             return INTENT_UNKNOWN
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return INTENT_UNKNOWN

    def save_model(self, path: str = None):
        """
        Saves the trained pipeline model to a file.

        Args:
            path (str, optional): Path to save the model. Defaults to self.model_path.
        """
        if not self._is_fitted or self._pipeline is None:
            print("Model is not trained. Nothing to save.")
            return

        save_path = path if path else self.model_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self._pipeline, save_path)
            print(f"Model saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")

    def _load_model(self, path: str = None) -> bool:
        """
        Loads the pipeline model from a file.

        Args:
            path (str, optional): Path to load the model from. Defaults to self.model_path.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        load_path = path if path else self.model_path
        if os.path.exists(load_path):
            try:
                self._pipeline = joblib.load(load_path)
                self._is_fitted = True # Assume loaded model is fitted
                print(f"Model loaded successfully from {load_path}")
                return True
            except (EOFError, ValueError, TypeError, ModuleNotFoundError) as e:
                 print(f"Error loading model from {load_path}: {e}. Model file might be corrupted or incompatible.")
                 self._pipeline = None
                 self._is_fitted = False
                 return False
            except Exception as e:
                print(f"An unexpected error occurred loading model from {load_path}: {e}")
                self._pipeline = None
                self._is_fitted = False
                return False
        else:
            print(f"Model file not found at {load_path}. Train the model first.")
            self._pipeline = None
            self._is_fitted = False
            return False

    def is_ready(self) -> bool:
        """Checks if the model is loaded and ready for prediction."""
        return self._is_fitted and self._pipeline is not None


# Main execution block for training and demonstration
if __name__ == "__main__":
    print("Initializing Intent Classifier...")
    classifier = IntentClassifier()

    # Check if a model already exists
    if not classifier.is_ready():
        print("\nNo pre-trained model found or loaded. Training a new model...")
        # Use the default sample data for training
        classifier.train(tune_hyperparameters=True) # Set to False for faster run without tuning
    else:
        print("\nPre-trained model loaded.")
        # Optionally re-train even if model exists:
        # print("\nRe-training the model...")
        # classifier.train(tune_hyperparameters=True)


    if classifier.is_ready():
        print("\n--- Testing Predictions ---")
        test_commands = [
            "make a shopping list file",
            "what files do I have?",
            "please remove the log file",
            "search for nearby restaurants",
            "hello",
            "bye bye",
            "what is 2 plus 2"
        ]
        for command in test_commands:
            predicted_intent = classifier.predict(command)
            print(f"Command: '{command}' -> Predicted Intent: '{predicted_intent}'")

        print("\n--- Model Ready ---")
    else:
        print("\n--- Model Not Ready ---")
        print("Could not train or load the model. Please check the data and environment.")