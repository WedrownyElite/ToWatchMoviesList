from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLineEdit,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QHBoxLayout,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QInputDialog,
    QMessageBox,
    QDialog,
    QCheckBox,
    QDialogButtonBox,
    QFormLayout,
    QPushButton,
    QFileDialog,
    QGridLayout,
    QShortcut,
)
from PyQt5.QtGui import QFont, QPixmap, QColor, QIcon, QIntValidator, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QSize
import requests
import re
import sys
import os
import json
import openai
import asyncio
import aiohttp
import configparser
from cryptography.fernet import Fernet
from queue import Queue
from urllib.parse import quote
from functools import lru_cache
from difflib import SequenceMatcher

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".MoviesList")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
API_KEY_FILE = os.path.join(CONFIG_DIR, "api_keys.secure")

omdb_key = None
openai_key = None
ENCRYPTION_KEY = b'PWnhKse5V-Vzrfv2gVZVyKgoP5490MNjDL9lds2J4jY='

API_KEY_DIR = os.path.dirname(API_KEY_FILE) or "."

max_suggestions = 10

if not os.path.exists(API_KEY_DIR):
    os.makedirs(API_KEY_DIR)

def resource_path(relative_path):
    """Get the absolute path to the resource, works for dev and PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def ensure_config():
    """Ensure the .MoviesList directory, config file, and saved data directory exist."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        print(f"[DEBUG] Created configuration directory: {CONFIG_DIR}")

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIG_FILE):
        print("[DEBUG] Config file not found. Creating new config file.")

        # Generate a new encryption key and save it in the config file
        encryption_key = Fernet.generate_key().decode()
        config['Settings'] = {
            'saved_data_dir': '',
            'encryption_key': encryption_key
        }
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        print(f"[DEBUG] Created config file with encryption key: {CONFIG_FILE}")
    else:
        config.read(CONFIG_FILE)

    # Verify settings in the config file
    if 'Settings' not in config or 'saved_data_dir' not in config['Settings'] or 'encryption_key' not in config['Settings']:
        raise RuntimeError("Invalid or missing configuration settings in config.ini")

    encryption_key = config['Settings']['encryption_key']
    saved_data_path = config['Settings']['saved_data_dir']

    # Prompt the user if the saved data directory is missing or doesn't exist
    if not saved_data_path or not os.path.exists(saved_data_path):
        print("[DEBUG] Saved data directory missing or invalid. Prompting user for a directory.")

        # Use a custom message dialog
        prompt_dialog = CustomMessageDialog(
            title="Saved Data Directory",
            message="The saved data directory is missing or invalid. Please select or create a directory.",
            button_options=[
                {"text": "Open Existing", "role": "open_existing"},
                {"text": "Create Directory", "role": "create_directory"},
                {"text": "Exit", "role": "exit"}
            ],
            text_color="white"
        )

        result = [None]  # Store the result from the dialog

        def handle_button(role):
            result[0] = role
            prompt_dialog.accept()

        prompt_dialog.button_pressed.connect(handle_button)
        prompt_dialog.exec_()

        if not result[0]:  # Handle case where the user closes the dialog
            CustomMessageDialog(
                "Exiting",
                "Closing Program",
                button_options=[{"text": "Exit", "role": "exit"}],
                text_color="red"
            ).exec_()
            sys.exit("[ERROR] User closed the dialog without selecting an option.")

        if result[0] == "open_existing":
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(None, "Select Existing Directory for MoviesList")
            if not dir_path:
                CustomMessageDialog(
                    "Configuration Error",
                    "You must select a directory to proceed.",
                    button_options=[{"text": "Exit", "role": "exit"}],
                    text_color="red"
                ).exec_()
                sys.exit(1)
            saved_data_path = dir_path

        elif result[0] == "create_directory":
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(None, "Select Parent Directory for New MoviesList")
            if not dir_path:
                CustomMessageDialog(
                    "Configuration Error",
                    "You must select a directory to proceed.",
                    button_options=[{"text": "Exit", "role": "exit"}],
                    text_color="red"
                ).exec_()
                sys.exit(1)

            # Create the MoviesList directory
            saved_data_path = os.path.join(dir_path, "MoviesList")
            os.makedirs(saved_data_path, exist_ok=True)

        elif result[0] == "exit":
            CustomMessageDialog(
                "Exiting",
                "Closing Program",
                button_options=[{"text": "Exit", "role": "exit"}],
                text_color="red"
            ).exec_()
            sys.exit(1)

        # Update and save the config file
        config['Settings']['saved_data_dir'] = saved_data_path
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        print(f"[DEBUG] Updated config file with saved_data_dir: {saved_data_path}")

    # Ensure the directory exists
    if not os.path.exists(saved_data_path):
        os.makedirs(saved_data_path)
        print(f"[DEBUG] Created missing saved_data directory: {saved_data_path}")

    config.read(CONFIG_FILE)

    # Load max suggestions from config
    global max_suggestions
    max_suggestions = int(config['Settings'].get('max_suggestions', '10'))

    return saved_data_path, encryption_key

def ensure_api_keys(encryption_key):
    """Ensure API keys are available, prompting the user if necessary."""
    global omdb_key, openai_key
    omdb_key, openai_key = None, None

    # Try loading existing API keys
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, "rb") as file:
                lines = file.readlines()
            omdb_key = decrypt_data(lines[0].strip(), encryption_key.encode())
            openai_key = decrypt_data(lines[1].strip(), encryption_key.encode()) if len(lines) > 1 else None
        except Exception as e:
            print(f"[ERROR] Failed to decrypt API keys: {e}")

    # State 1: First run, neither key is found
    if omdb_key is None and openai_key is None:
        # Prompt for OMDb key
        while not omdb_key or not validate_api_key_omdb(omdb_key):
            omdb_prompt = (
                '<span style="color:red;">The existing OMDb API Key is invalid. Please provide a valid one.</span>'
                if omdb_key and not validate_api_key_omdb(omdb_key)
                else "Please enter your OMDb API Key:"
            )
            omdb_dialog = CustomInputDialog("OMDb API Key", omdb_prompt, cancel_text="Exit")
            if omdb_dialog.exec_() == QDialog.Accepted:
                omdb_key = omdb_dialog.get_input().strip()
            else:
                CustomMessageDialog(
                    title="API Key Missing",
                    message="OMDb API Key is required to run the program.",
                    button_options=[{"text": "Exit", "role": "exit"}],
                    text_color="red"
                ).exec_()
                sys.exit("[ERROR] OMDb API Key is required.")

        # Prompt for OpenAI key
        while not openai_key or not validate_api_key_openai(openai_key):
            gpt_prompt = (
                '<span style="color:red;">The provided GPT API Key is invalid. Please re-enter the key.</span>'
                if openai_key and not validate_api_key_openai(openai_key)
                else "Enter your OpenAI GPT API Key (optional):"
            )
            gpt_dialog = CustomInputDialog("OpenAI GPT API Key", gpt_prompt, cancel_text="Skip")
            if gpt_dialog.exec_() == QDialog.Accepted:
                openai_key = gpt_dialog.get_input().strip()
            else:
                # If skipped, disable GPT suggestions
                openai_key = None
                QTimer.singleShot(500, lambda: CustomMessageDialog(
                    title="GPT Suggestions Disabled",
                    message="GPT-based suggestions are disabled because the OpenAI GPT API Key is missing.",
                    button_options=[{"text": "Ok", "role": "ok"}],
                    text_color="red"
                ).exec_())
                break

    # State 2: OMDb key valid, OpenAI key missing or empty
    elif omdb_key and (openai_key is None or openai_key.strip() == ""):
        # Do not prompt for OpenAI key, just disable GPT suggestions
        QTimer.singleShot(500, lambda: CustomMessageDialog(
            title="GPT Suggestions Disabled",
            message="GPT-based suggestions are disabled because the OpenAI GPT API Key is missing.",
            button_options=[{"text": "Ok", "role": "ok"}],
            text_color="red"
        ).exec_())
        openai_key = None

    # State 3: OMDb key valid, OpenAI key found but invalid
    elif omdb_key and openai_key and not validate_api_key_openai(openai_key):
        # Prompt to revalidate OpenAI key
        while not validate_api_key_openai(openai_key):
            gpt_prompt = (
                '<span style="color:red;">The provided GPT API Key is invalid. Please re-enter the key.</span>'
            )
            gpt_dialog = CustomInputDialog("OpenAI GPT API Key", gpt_prompt, cancel_text="Skip")
            if gpt_dialog.exec_() == QDialog.Accepted:
                openai_key = gpt_dialog.get_input().strip()
            else:
                # If skipped, disable GPT suggestions and remove invalid key
                openai_key = None
                QTimer.singleShot(500, lambda: CustomMessageDialog(
                    title="GPT Suggestions Disabled",
                    message="GPT-based suggestions are disabled.",
                    button_options=[{"text": "Ok", "role": "ok"}],
                    text_color="red"
                ).exec_())
                break

    # Save API keys if they are valid
    save_api_keys(omdb_key, openai_key, API_KEY_FILE, encryption_key)

    return omdb_key, openai_key

def encrypt_data(data, key):
    fernet = Fernet(key)
    return fernet.encrypt(data.encode())

def decrypt_data(data, key):
    fernet = Fernet(key)
    return fernet.decrypt(data).decode()

def save_api_keys(omdb_key, openai_key, api_key_file, encryption_key):
    """Save API keys securely to the specified file."""
    try:
        with open(api_key_file, "wb") as f:
            f.write(encrypt_data(omdb_key, encryption_key) + b"\n")
            # Only save the OpenAI key if it's not None
            if openai_key:
                f.write(encrypt_data(openai_key, encryption_key) + b"\n")
        print("[DEBUG] API keys saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save API keys: {e}")
        raise

def load_api_keys(api_key_file, encryption_key):
    """Load API keys securely from the specified file."""
    if not os.path.exists(api_key_file):
        print("[DEBUG] API key file not found.")
        return None, None
    try:
        with open(api_key_file, "rb") as f:
            lines = f.readlines()
        omdb_key = decrypt_data(lines[0].strip(), encryption_key)
        openai_key = decrypt_data(lines[1].strip(), encryption_key) if len(lines) > 1 else None
        print("[DEBUG] API keys loaded successfully.")
        return omdb_key, openai_key
    except Exception as e:
        print(f"[ERROR] Failed to load API keys: {e}")
        return None, None

def validate_api_key_omdb(api_key):
    try:
        response = requests.get(f"http://www.omdbapi.com/?apikey={api_key}&t=test")
        print(f"[DEBUG] OMDB response: {response.json()}")
        return response.status_code == 200 and response.json().get("Response") == "True"
    except Exception as e:
        print(f"[ERROR] OMDB API validation failed: {e}")
        return False

def validate_api_key_openai(api_key):
    """Test OpenAI API key by making a dummy request."""
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Test API key.",
            max_tokens=5
        )
        return "choices" in response
    except Exception:
        return False

def get_api_keys():
    omdb_key, openai_key = load_api_keys()
    print(f"[DEBUG] Loaded keys: OMDB={omdb_key}, OpenAI={openai_key}")

    if not omdb_key or not validate_api_key_omdb(omdb_key):
        print("[DEBUG] OMDB key missing or invalid. Prompting user.")
        while not omdb_key or not validate_api_key_omdb(omdb_key):
            omdb_key, ok = QInputDialog.getText(None, "OMDB API Key", "Enter your OMDB API Key:")
            if not ok or not omdb_key:
                sys.exit("[ERROR] OMDB API Key is required.")
            if not validate_api_key_omdb(omdb_key):
                QMessageBox.warning(None, "Invalid Key", "OMDB API Key is invalid.")

    if not openai_key or not validate_api_key_openai(openai_key):
        print("[DEBUG] OpenAI key missing or invalid. Prompting user.")
        while not openai_key or not validate_api_key_openai(openai_key):
            openai_key, ok = QInputDialog.getText(None, "OpenAI API Key", "Enter your OpenAI API Key:")
            if not ok or not openai_key:
                sys.exit("[ERROR] OpenAI API Key is required.")
            if not validate_api_key_openai(openai_key):
                QMessageBox.warning(None, "Invalid Key", "OpenAI API Key is invalid.")

    save_api_keys(omdb_key, openai_key)
    print("[DEBUG] Final keys: OMDB={omdb_key}, OpenAI={openai_key}")
    return omdb_key, openai_key

def validate_api_key_omdb(api_key):
    """Test OMDB API key by making a dummy request."""
    try:
        response = requests.get(f"http://www.omdbapi.com/?apikey={api_key}&t=test")
        return response.status_code == 200 and response.json().get("Response") == "True"
    except Exception:
        return False

def validate_api_key_openai(api_key):
    """Test OpenAI API key by fetching available models."""
    try:
        openai.api_key = api_key
        response = openai.Model.list()  # Fetch the list of available models
        return isinstance(response, dict) and "data" in response
    except Exception as e:
        print(f"[ERROR] OpenAI API key validation failed: {e}")
        return False

def get_api_keys():
    """Prompt the user for API keys if they are not found or invalid."""
    try:
        omdb_key, openai_key = load_api_keys()
    except Exception as e:
        print(f"[ERROR] Failed to load API keys: {e}")
        omdb_key, openai_key = None, None

    if not (omdb_key and openai_key):
        QMessageBox.warning(None, "API Keys Missing", "API keys not found. Please provide them.")

    while not omdb_key or not validate_api_key_omdb(omdb_key):
        try:
            omdb_key, ok = QInputDialog.getText(None, "OMDB API Key", "Enter your OMDB API Key:")
            if not ok or not omdb_key:
                sys.exit("OMDB API Key is required.")
            if not validate_api_key_omdb(omdb_key):
                QMessageBox.warning(None, "Invalid Key", "OMDB API Key is invalid.")
        except Exception as e:
            print(f"[ERROR] Exception during OMDB API Key input: {e}")
            sys.exit("OMDB API Key is required.")

    while not openai_key or not validate_api_key_openai(openai_key):
        try:
            openai_key, ok = QInputDialog.getText(None, "OpenAI API Key", "Enter your OpenAI API Key:")
            if not ok or not openai_key:
                sys.exit("OpenAI API Key is required.")
            if not validate_api_key_openai(openai_key):
                QMessageBox.warning(None, "Invalid Key", "OpenAI API Key is invalid.")
        except Exception as e:
            print(f"[ERROR] Exception during OpenAI API Key input: {e}")
            sys.exit("OpenAI API Key is required.")

    try:
        save_api_keys(omdb_key, openai_key)
    except Exception as e:
        print(f"[ERROR] Failed to save API keys: {e}")

    return omdb_key, openai_key

def compute_similarity(a, b):
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

@lru_cache(maxsize=100)
def get_cached_suggestions(query_text):
    """Fetch suggestions with caching."""
    # Combine OMDb and GPT requests with proper handling
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    omdb_results = loop.run_until_complete(FetchSuggestionsThread.fetch_omdb_suggestions(query_text, openai_key))
    gpt_results = loop.run_until_complete(query_llm_for_movie(query_text, omdb_key))
    return omdb_results + gpt_results
    raise ValueError("The OPENAI_API_KEY variable is not set. Please set it and try again.")

async def query_llm_for_movie(input_text):
    try:
        print(f"[DEBUG] Querying GPT for: {input_text}")
        
        # Extract title and optional year
        match = re.match(r"(.+?)\s+(\d{4})$", input_text.strip())
        title = match.group(1).strip() if match else input_text.strip()
        year = match.group(2).strip() if match else None

        # Refined query
        query = (
            f"List all movies in the '{title}' series/franchise along with their release year and genres. "
            f"If the title is from a series, list its sequels/prequels in chronological order."
            f"If the movie is standalone, list {max_suggestions} similar movies from the same genre or with a similar theme. "
            f"Only provide the titles and release years, maximum of 10"
            f"{'Include only movies from ' + year if year else ''}"
        )

        # Make the initial GPT query
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}],
            max_tokens=300,
            temperature=0.2,
        )

        # Capture initial response
        suggestions_raw = response['choices'][0]['message']['content']
        print(f"[DEBUG] Raw GPT suggestions: {suggestions_raw}")

        # Parse suggestions
        suggestions = []
        for line in suggestions_raw.split("\n"):
            match = re.match(r"(.*)\((\d{4})\)", line.strip())
            if match:
                suggestion_title, suggestion_year = match.groups()
                suggestions.append({
                    "title": suggestion_title.strip(),
                    "year": suggestion_year.strip(),
                    "poster_url": "",
                    "similarity": compute_similarity(input_text, suggestion_title.strip()),
                })
            elif line.strip():  # Log unhandled lines
                print(f"[DEBUG] Skipping unhandled line: {line.strip()}")

        # Log parsed suggestions
        print(f"[DEBUG] Parsed suggestions: {suggestions}")
        return suggestions

    except Exception as e:
        print(f"[ERROR] query_llm_for_movie: {e}")
        return []

class DataLoadingThread(QThread):
    dataLoaded = pyqtSignal(dict)

    def __init__(self, saved_movies):
        super().__init__()
        self.saved_movies = saved_movies

    def run(self):
        """Load data incrementally and emit data for each entry."""
        for title, data in self.saved_movies.items():
            print(f"[DEBUG] Emitting movie data: {data}")
            movie_data = {
                "title": title,
                "plot": data.get("plot", "N/A"),
                "rating": data.get("rating", "N/A"),
                "poster_path": data.get("poster_path", ""),
                "cast": data.get("cast", "N/A"),
                "list": data.get("list", "To Watch List"),
            }
            self.msleep(50)
            self.dataLoaded.emit(movie_data)  # Signal UI to add data

class ImportMoviesThread(QThread):
    movieAdded = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, movie_titles, queue):
        super().__init__()
        self.movie_titles = movie_titles
        self.queue = queue
        self._is_running = True

    def run(self):
        asyncio.run(self._process_movies())

    async def _process_movies(self):
        async with aiohttp.ClientSession() as session:
            while self._is_running and not self.queue.empty():
                try:
                    title = self.queue.get_nowait()
                    movie_data = await self._query_movie(title, session)
                    if movie_data:
                        self.movieAdded.emit(movie_data)
                    self.queue.task_done()
                except Exception as e:
                    print(f"[ERROR] Exception in _process_movies: {e}")
            self.finished.emit()

    async def _query_movie(self, title, session):
        """Query OMDb first, then fall back to GPT if necessary."""
        # Query OMDb for movie data
        movie_data = await self._query_omdb(title, session)
        if not movie_data:
            # If OMDb fails, query GPT for a suggestion and retry OMDb
            gpt_suggestion = await self._query_gpt_for_title(title)
            if gpt_suggestion:
                sanitized_title = re.sub(r'^"|"$', '', gpt_suggestion).strip()
                movie_data = await self._query_omdb(sanitized_title, session)
        return movie_data

    async def _query_omdb(self, title, session):
        """Query the OMDb API asynchronously."""
        try:
            url = f"http://www.omdbapi.com/?t={quote(title)}&apikey={omdb_key}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("Response") == "True":
                        # Combine title and year for consistency
                        title_with_year = f"{data.get('Title')} [{data.get('Year')}]"
                        return {
                            "title": title_with_year,
                            "plot": data.get("Plot", "N/A"),
                            "rating": data.get("imdbRating", "N/A"),
                            "poster_path": data.get("Poster", ""),
                            "cast": data.get("Actors", "N/A"),
                        }
        except Exception as e:
            print(f"[ERROR] Exception in _query_omdb for '{title}': {e}")
        return None

    async def _query_gpt_for_title(self, title):
        """Query GPT for the closest movie/show title and year."""
        try:
            prompt = f"Find the most accurate movie or show title based on '{title}'. Include the release year."
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5,
            )
            suggestion = response['choices'][0]['message']['content'].strip()
            print(f"[DEBUG] GPT suggestion for '{title}': {suggestion}")
            return suggestion
        except Exception as e:
            print(f"[ERROR] Exception in _query_gpt_for_title: {e}")
        return None

    def stop(self):
        """Stop the thread."""
        self._is_running = False

class CustomMessageDialog(QDialog):
    button_pressed = pyqtSignal(str)

    def __init__(self, title, message, button_options=None, text_color="white"):
        super().__init__()
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #1C1C1C; color: white;")

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Message label
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"color: {text_color}; font-size: 16px;")
        layout.addWidget(self.label)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.buttons = []
        button_options = button_options or [{"text": "Ok", "role": "accept"}]
        for option in button_options:
            button = self.create_button(option)
            button_layout.addWidget(button)
            self.buttons.append(button)

        layout.addLayout(button_layout)

        # Base and max dimensions
        self.base_width = 400
        self.base_height = 200
        self.max_width = 600
        self.max_height = 400

        self.adjust_size()

    def create_button(self, option):
        """Create a button with proper styling."""
        button = QPushButton(option["text"], self)
        button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        button.clicked.connect(lambda: self.handle_button_click(option["role"]))
        return button

    def handle_button_click(self, role):
        """Emit button role and close dialog."""
        self.button_pressed.emit(role)
        self.close()  # Ensure dialog closes when button is clicked

    def adjust_size(self):
        """Adjust dialog size dynamically based on content."""
        self.label.adjustSize()

        # Measure label and button dimensions
        label_width = self.label.sizeHint().width()
        label_height = self.label.sizeHint().height()
        button_widths = [button.sizeHint().width() for button in self.buttons]
        total_button_width = sum(button_widths) + 12 * (len(button_widths) - 1)
        max_button_width = max(button_widths)

        # Adjust width and height
        final_width = max(self.base_width, min(self.max_width, max(label_width, total_button_width)))
        final_height = max(self.base_height, min(self.max_height, label_height + 50 + max_button_width))

        self.setFixedSize(final_width, final_height)

class CustomInputDialog(QDialog):
    def __init__(self, title, message, cancel_text="Cancel"):
        super().__init__()
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #1C1C1C; color: white;")

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Message label
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 16px;")
        layout.addWidget(self.label)

        # Input field
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("Enter here...")
        self.line_edit.setStyleSheet("font-size: 16px; padding: 6px;")
        layout.addWidget(self.line_edit)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.button(QDialogButtonBox.Ok).setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        self.button_box.button(QDialogButtonBox.Cancel).setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                font-size: 16px;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.button_box.button(QDialogButtonBox.Cancel).setText(cancel_text)
        self.button_box.accepted.connect(self.accept)  # Connect OK button to accept
        self.button_box.rejected.connect(self.reject)  # Connect Cancel button to reject
        layout.addWidget(self.button_box)

        # Base and max dimensions
        self.base_width = 350
        self.base_height = 150
        self.max_width = 600
        self.max_height = 400

        self.adjust_size()

    def adjust_size(self):
        """Adjust dialog size dynamically based on content."""
        self.label.adjustSize()

        # Measure dimensions
        label_width = self.label.sizeHint().width()
        label_height = self.label.sizeHint().height()
        input_width = self.line_edit.sizeHint().width()
        input_height = self.line_edit.sizeHint().height()
        button_widths = [self.button_box.button(role).sizeHint().width() for role in [QDialogButtonBox.Ok, QDialogButtonBox.Cancel]]
        total_button_width = sum(button_widths) + 12
        max_button_width = max(button_widths)

        # Adjust width and height
        final_width = max(self.base_width, min(self.max_width, max(label_width, total_button_width, input_width)))
        final_height = max(self.base_height, min(self.max_height, label_height + input_height + max_button_width + 50))

        self.setFixedSize(final_width, final_height)

    def get_input(self):
        return self.line_edit.text()

    def set_message(self, message):
        """Update the dialog message dynamically and adjust the size."""
        self.label.setText(message)
        self.adjust_size()

class PosterLoaderThread(QThread):
    posterLoaded = pyqtSignal(QLabel, QPixmap)

    def __init__(self, label, poster_path):
        super().__init__()
        self.label = label
        self.poster_path = poster_path

    def run(self):
        try:
            if self.url.startswith("http"):
                response = requests.get(self.url, timeout=5)
                if response.status_code == 200:
                    pixmap = QPixmap()
                    pixmap.loadFromData(response.content)
                    scaled_pixmap = pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.imageLoaded.emit(self.label, scaled_pixmap)
            elif os.path.exists(self.url):
                scaled_pixmap = self.scale_poster(self.url)
                self.imageLoaded.emit(self.label, scaled_pixmap)
        except Exception as e:
            print(f"[ERROR] Failed to load poster: {e}")

class FetchSuggestionsThread(QThread):
    suggestionsFetched = pyqtSignal(dict)  # Emit a single suggestion at a time

    def __init__(self, text):
        super().__init__()
        self.text = text
        self._is_running = True

    def run(self):
        """Run the thread and fetch suggestions."""
        if not self._is_running:
            print("[DEBUG] Thread is not running.")
            return

        try:
            print(f"[DEBUG] Starting FetchSuggestionsThread for text: {self.text}")
            asyncio.run(self.fetch_and_emit_suggestions())  # Call the async method
        except Exception as e:
            print(f"[ERROR] Exception in FetchSuggestionsThread.run: {e}")

    async def fetch_and_emit_suggestions(self):
        """Fetch suggestions from OMDb and optionally GPT."""
        try:
            # Parse input text for title and year
            match = re.match(r"(.+?)\s+(\d{4})$", self.text.strip())
            title = match.group(1).strip() if match else self.text.strip()
            year = match.group(2).strip() if match else None

            print(f"[DEBUG] Fetching suggestions for: '{title}' ({year})")

            # First attempt: Fetch exact match with title and year
            omdb_result = await self.fetch_omdb_details(title, year)
            if omdb_result:
                print(f"[DEBUG] Exact match found in OMDb: {omdb_result}")
                self.suggestionsFetched.emit({
                    "title": omdb_result.get("Title", ""),
                    "year": omdb_result.get("Year", ""),
                    "poster_url": omdb_result.get("Poster", ""),
                    "actors": omdb_result.get("Actors", "N/A"),
                })

            # Fetch GPT-based suggestions
            gpt_suggestions = await query_llm_for_movie(self.text)
            if gpt_suggestions:
                for suggestion in gpt_suggestions[:max_suggestions]:
                    print(f"[DEBUG] Processing GPT suggestion: {suggestion}")
                    # Clean suggestion titles
                    suggestion_title = re.sub(r"^\d+\.\s*", "", suggestion["title"]).strip()
                    suggestion_year = suggestion.get("year", "")
                    omdb_result = await self.fetch_omdb_details(suggestion_title, suggestion_year)

                    # Use OMDb data if available; otherwise, use GPT suggestion directly
                    self.suggestionsFetched.emit({
                        "title": omdb_result.get("Title", suggestion_title) if omdb_result else suggestion_title,
                        "year": omdb_result.get("Year", suggestion_year) if omdb_result else suggestion_year,
                        "poster_url": omdb_result.get("Poster", "") if omdb_result else "",
                        "actors": omdb_result.get("Actors", "N/A") if omdb_result else "N/A",
                    })
                    print(f"[DEBUG] Emitted suggestion: {suggestion}")
            else:
                print("[DEBUG] No GPT suggestions returned.")

        except Exception as e:
            print(f"[ERROR] fetch_and_emit_suggestions: {e}")

    async def fetch_omdb_details(self, title, year=None):
        """Query the OMDb API asynchronously with optional year handling."""
        try:
            query_url = f"http://www.omdbapi.com/?t={quote(title)}"
            if year:
                query_url += f"&y={year}"  # Include the year in the query if provided
            query_url += f"&apikey={omdb_key}"

            print(f"[DEBUG] Fetching OMDb details for: {query_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(query_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("Response") == "True":
                            return data
                        else:
                            print(f"[DEBUG] OMDb fetch failed for '{title}' ({year}): {data.get('Error', 'Unknown error')}")
                    else:
                        print(f"[ERROR] OMDb request failed with status {response.status} for {query_url}")
            return None
        except Exception as e:
            print(f"[ERROR] fetch_omdb_details: {e}")
            return None

    def stop(self):
        """Stop the thread."""
        self._is_running = False

class ImageLoaderThread(QThread):
    imageLoaded = pyqtSignal(QLabel, QPixmap)

    def __init__(self, label, url):
        super().__init__()
        self.label = label
        self.url = url
        self._is_running = True

    def run(self):
        if not self._is_running:
            return
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(requests.get(self.url).content)
            self.imageLoaded.emit(self.label, pixmap)
        except Exception as e:
            print(f"Error loading image: {e}")

    def stop(self):
        """Stop the thread."""
        self._is_running = False

class AddMoviesThread(QThread):
    movieAdded = pyqtSignal(dict)

    def __init__(self, title):
        super().__init__()
        self.title = title
        self._is_running = True

    def run(self):
        """Fetch movie details from the API and prepare for addition."""
        if not self._is_running:
            return

        url = f"https://api.themoviedb.org/3/search/multi?api_key={omdb_key}&query={self.title}"
        try:
            response = requests.get(url)
            data = response.json()

            if data.get("results"):
                movie = data["results"][0]
                name = movie.get("title") or movie.get("name", "N/A")
                date = movie.get("release_date") or movie.get("first_air_date", "")
                year = date.split('-')[0] if date else "N/A"
                plot = movie.get("overview", "N/A")
                rating = "{:.1f}".format(float(movie.get("vote_average", 0.0)))
                poster_path = movie.get("poster_path", "")
                movie_data = {
                    "title": f"{name} [{year}]",
                    "plot": plot,
                    "rating": rating,
                    "poster_path": f"https://image.tmdb.org/t/p/w92{poster_path}" if poster_path else None,
                }
                self.movieAdded.emit(movie_data)
        except Exception as e:
            print(f"Error fetching movie details: {e}")

    def stop(self):
        """Stop the thread."""
        self._is_running = False

class FilterMoviesThread(QThread):
    filteringComplete = pyqtSignal(list)

    def __init__(self, filter_text, original_data):
        super().__init__()
        self.filter_text = filter_text.lower()
        self.original_data = original_data
        self._is_running = True

    def run(self):
        """Filter the table data based on the entered text."""
        if not self._is_running:
            return

        filtered_data = []
        for row_data in self.original_data:
            title_matches = self.filter_text in row_data.get("title", "").lower()
            cast_matches = self.filter_text in row_data.get("cast", "").lower()
            if title_matches or cast_matches:
                filtered_data.append(row_data)
        self.filteringComplete.emit(filtered_data)

    def stop(self):
        """Stop the thread."""
        self._is_running = False

class ResizableSuggestionBox(QListWidget):
    def __init__(self, parent=None, min_height=100, max_height=200):
        super().__init__(parent)
        self.min_height = min_height
        self.max_height = max_height
        self.resizing = False  # Track if the user is resizing

        # Add a draggable handle at the bottom
        self.handle_height = 15  # Increase the height of the draggable handle
        self.handle = QLabel(self)
        self.handle.setFixedHeight(self.handle_height)
        self.handle.setStyleSheet("background-color: #2E2E2E;")
        self.handle.setAlignment(Qt.AlignCenter)
        self.handle.setText("⇕")  # Add a resize symbol for clarity

        # Set the cursor for the handle to a resize icon
        self.handle.setCursor(Qt.SizeVerCursor)

        # Position the handle at the bottom of the suggestion box
        self.handle.move(0, self.height() - self.handle_height)

        # Track mouse events for resizing
        self.handle.setMouseTracking(True)
        self.handle.mousePressEvent = self.start_resizing
        self.handle.mouseMoveEvent = self.resize_box
        self.handle.mouseReleaseEvent = self.stop_resizing

    def resizeEvent(self, event):
        """Ensure the handle stays at the bottom of the suggestion box and spans the entire width."""
        super().resizeEvent(event)
        # Update handle width and position to match the width of the suggestion box
        self.handle.setFixedWidth(self.width())
        self.handle.move(0, self.height() - self.handle_height)

    def start_resizing(self, event):
        """Start resizing the suggestion box."""
        if event.button() == Qt.LeftButton:
            self.resizing = True
            self.start_y = event.globalPos().y()
            self.start_height = self.height()

    def resize_box(self, event):
        """Resize the suggestion box as the mouse moves."""
        if self.resizing:
            delta_y = event.globalPos().y() - self.start_y
            new_height = self.start_height + delta_y

            # Clamp the height within the allowed range
            if new_height > self.max_height:
                new_height = self.max_height  # Snap to max height
                self.resizing = False  # Stop further resizing

            elif new_height < self.min_height:
                new_height = self.min_height  # Snap to min height

            # Set the height and update the handle position
            self.setFixedHeight(new_height)
            self.handle.move(0, self.height() - self.handle_height)

    def stop_resizing(self, event):
        """Stop resizing the suggestion box."""
        self.resizing = False

class MovieApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movies List")
        self.setGeometry(100, 100, 800, 600)  # Initial window size
        self.original_table_data = {}
        self.current_suggestions = []
        self.suggestion_list_tab1 = None
        self.suggestion_list_tab2 = None
        self.query_in_progress = False

        self.suggestions_active = True
        self.is_busy = False
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
            
        # Thread management
        self.suggestion_thread = None
        self.image_threads = []

        # Set the background color of the entire window
        self.setStyleSheet("background-color: #1C1C1C;")

        self.column_weights = [0.3, 0.1, 0.4, 0.2]

        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add a layout for the revalidate button at the top
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)

        # Add a spacer to push the button to the right
        self.top_layout.addStretch()

        gear_icon_path = resource_path("resources/gear.png")

        # Add a settings button with the gear icon
        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon(gear_icon_path))
        self.settings_button.setIconSize(QSize(20, 20))
        self.settings_button.setFixedSize(30, 30)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;  /* Transparent background */
                border: none;  /* No border */
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);  /* Light hover effect */
                border-radius: 5px;  /* Optional: Rounded corners */
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.2);  /* Press effect */
            }
        """)
        self.settings_button.clicked.connect(self.open_settings_window)
        self.top_layout.addWidget(self.settings_button)

        # Add tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Initialize tabs before connecting signals
        self.tab1 = QWidget()
        self.setup_tab1()  # Ensure self.text_entry_tab1 is initialized
        self.tabs.addTab(self.tab1, "To Watch List")

        self.tab2 = QWidget()
        self.setup_tab2()  # Ensure self.text_entry_tab2 is initialized
        self.tabs.addTab(self.tab2, "Watched List")

        # Connect the tab change signal after initialization
        self.tabs.currentChanged.connect(self.hide_all_suggestion_lists)

        # Add a shortcut for Ctrl+Tab and Ctrl+Shift+Tab
        self.ctrl_tab_shortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)
        self.ctrl_tab_shortcut.activated.connect(self.next_sheet)

        self.ctrl_shift_tab_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Tab"), self)
        self.ctrl_shift_tab_shortcut.activated.connect(self.previous_sheet)

        # Listen for application-wide focus changes
        QApplication.instance().focusChanged.connect(self.on_focus_changed)

    # -----------------------------------
    # Layout Setup and Initialization
    # -----------------------------------
    def setup_tab1(self):
        """Set up the layout and widgets for Tab 1."""
        tab1_layout = QVBoxLayout()
        self.tab1.setLayout(tab1_layout)

        # Set margins for the layout to add padding around the table
        tab1_layout.setContentsMargins(10, 10, 10, 10)  # Left, Top, Right, Bottom

        # Add a title label for the tab
        self.tab1_title = QLabel("To Watch List")
        self.tab1_title.setFont(QFont("Arial", 16, QFont.Bold))  # Set font and size
        self.tab1_title.setAlignment(Qt.AlignCenter)  # Center align the title
        self.tab1_title.setStyleSheet("color: white;")  # Set text color to white
        tab1_layout.addWidget(self.tab1_title)  # Add the title to the layout

        # Import button container
        import_button_tab1_container = QWidget()
        import_button_tab1_layout = QHBoxLayout(import_button_tab1_container)
        import_button_tab1_layout.setAlignment(Qt.AlignCenter)

        # Import Button
        self.import_button_tab1 = QPushButton("Import")
        self.import_button_tab1.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        self.import_button_tab1.clicked.connect(lambda: self.import_movies(self.table_tab1 if self.tabs.currentIndex() == 0 else self.table_tab2))
        self.import_button_tab1.setMinimumWidth(50)
        self.import_button_tab1.setMaximumWidth(100)

        # Add import  button to import container
        import_button_tab1_layout.addWidget(self.import_button_tab1)
        # Add import container to layout
        tab1_layout.addWidget(import_button_tab1_container)

        # Add a container widget for the text entry
        search_container = QWidget()
        search_layout = QHBoxLayout()
        search_layout.setAlignment(Qt.AlignHCenter)  # Center the search elements horizontally
        search_container.setLayout(search_layout)
        tab1_layout.addWidget(search_container)

        # Calculate dimensions to match the filter box width exactly
        filter_box_width = 500  # Same as filter box's maximum width
        search_button_width = 100  # Width of the search button
        padding = 10  # Space between the text box and the button
        manualoffset = 7 # Shhh :)
        search_box_width = filter_box_width - search_button_width - padding - manualoffset

        # Add text entry box
        self.text_entry_tab1 = QLineEdit()
        self.text_entry_tab1.setFont(QFont("Arial", 12))
        self.text_entry_tab1.setAlignment(Qt.AlignCenter)  # Center the text within the box
        self.text_entry_tab1.setPlaceholderText("Enter movie details...")
        self.text_entry_tab1.setStyleSheet("""
            QLineEdit {
                background-color: #2C2C2C;  /* Background color */
                color: white;              /* Text color */
                border: 1px solid white;   /* White border */
                border-radius: 0px;        /* Rounded corners */
                padding: 0px;              /* Padding inside the box */
            }
        """)
        self.text_entry_tab1.setFixedWidth(search_box_width)  # Dynamically adjust width
        self.text_entry_tab1.returnPressed.connect(self.trigger_search)  # Trigger search on ENTER
        search_layout.addWidget(self.text_entry_tab1)

        # Add spacing between the text entry and the button
        search_layout.addSpacing(padding)

        # Add search button
        self.search_button = QPushButton("Search")
        self.search_button.setFont(QFont("Arial", 12))
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 0px;
                padding: 2.5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3C8C43;
            }
        """)
        self.search_button.setFixedWidth(search_button_width)
        self.search_button.clicked.connect(self.trigger_search)
        search_layout.addWidget(self.search_button)

        # Add filtering entry box
        self.filter_entry_tab1 = QLineEdit()
        self.filter_entry_tab1.setFont(QFont("Arial", 12))
        self.filter_entry_tab1.setAlignment(Qt.AlignCenter)
        self.filter_entry_tab1.setPlaceholderText("Filter by title...")
        self.filter_entry_tab1.setStyleSheet("background-color: #2C2C2C; color: white;")
        self.filter_entry_tab1.setFixedWidth(filter_box_width)  # Match the width of search + button + padding

        filter_container_tab1 = QWidget()
        filter_entry_tab1_layout = QVBoxLayout()
        filter_entry_tab1_layout.setAlignment(Qt.AlignHCenter)  # Center the filter box horizontally
        filter_container_tab1.setLayout(filter_entry_tab1_layout)
        tab1_layout.addWidget(filter_container_tab1)

        filter_entry_tab1_layout.addWidget(self.filter_entry_tab1)
        # Connect the filtering box to the filtering logic
        self.filter_entry_tab1.textChanged.connect(
            lambda text: self.filter_table_rows(self.table_tab1, text)
        )

        self.filter_entry_tab1.setMinimumWidth(300)  # Minimum width
        self.filter_entry_tab1.setMaximumWidth(500)  # Maximum width 

        # Add a horizontal container for the counter and filter entry
        filter_container = QWidget()
        filter_layout = QHBoxLayout()
        filter_container.setLayout(filter_layout)
        tab1_layout.addWidget(filter_container)

        # Add the total titles counter
        self.counter_tab1 = QLabel("Titles: 0")
        self.counter_tab1.setFont(QFont("Arial", 12))
        self.counter_tab1.setAlignment(Qt.AlignLeft)
        self.counter_tab1.setStyleSheet("color: white;")
        filter_layout.addWidget(self.counter_tab1)

        # Create a floating suggestion list
        self.suggestion_list_tab1 = ResizableSuggestionBox(self, min_height=100, max_height=800)
        self.suggestion_list_tab1.setWindowFlags(Qt.ToolTip)
        self.suggestion_list_tab1.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.suggestion_list_tab1.hide()
        self.suggestion_list_tab1.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.suggestion_list_tab1.verticalScrollBar().setSingleStep(8)

        # Connect the text entry to update suggestions
        self.suggestion_list_tab1.itemClicked.connect(
            lambda item: self.select_suggestion(item, self.suggestion_list_tab1)
        )

        # Add table
        self.table_tab1 = QTableWidget(0, 5)
        self.table_tab1.setHorizontalHeaderLabels(["Title", "Poster", "Plot", "Cast", "Rating"])
        self.table_tab1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_tab1.setStyleSheet("""
            QTableWidget {
                background-color: #1C1C1C; 
                color: white;
                gridline-color: white;  /* Set grid lines to white */
            }
            QHeaderView::section {
                background-color: #1C1C1C;
                color: white;
                border: 1px solid white;  /* Set header border to white */
            }
            QTableWidget::item {
                border: 1px solid white;  /* Set item border to white */
            }
            QTableWidget::item:selected {
                border: 2px solid #00FF00;  /* Bright green border for selected items */
                background-color: #1C1C1C;  /* Keep background color unchanged */
                color: white;  /* Keep text color white */
            }
        """)
        self.table_tab1.setFont(QFont("Arial", 12))
        self.table_tab1.verticalHeader().setVisible(False)  # Hide row numbers
        self.table_tab1.horizontalHeader().sectionClicked.connect(
        lambda col: self.sort_table(self.table_tab1, col)
        )
        self.populate_table()
        tab1_layout.addWidget(self.table_tab1)

        self.table_tab1.verticalScrollBar().setSingleStep(1)

        # Connect right-click to context menu
        self.table_tab1.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_tab1.customContextMenuRequested.connect(self.show_context_menu)

        self.timer_tab1 = QTimer()
        self.timer_tab1.setSingleShot(True)
        self.timer_tab1.timeout.connect(self.fetch_suggestions)
        self.table_tab1.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_tab1.verticalScrollBar().setSingleStep(15)

        self.thread = None
        
    def setup_tab2(self):
        """Set up the layout and widgets for Tab 2."""
        tab2_layout = QVBoxLayout()
        self.tab2.setLayout(tab2_layout)

        # Set margins for the layout to add padding around the table
        tab2_layout.setContentsMargins(10, 10, 10, 10)

        # Add a title label for the tab
        self.tab2_title = QLabel("Watched List")
        self.tab2_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.tab2_title.setAlignment(Qt.AlignCenter)
        self.tab2_title.setStyleSheet("color: white;")
        tab2_layout.addWidget(self.tab2_title)

        # Import button container
        import_button_tab2_container = QWidget()
        import_button_tab2_layout = QHBoxLayout(import_button_tab2_container)
        import_button_tab2_layout.setAlignment(Qt.AlignCenter)

        # Import Button
        self.import_button_tab2 = QPushButton("Import")
        self.import_button_tab2.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        self.import_button_tab2.clicked.connect(lambda: self.import_movies(self.table_tab1 if self.tabs.currentIndex() == 0 else self.table_tab2))
        self.import_button_tab2.setMinimumWidth(50)
        self.import_button_tab2.setMaximumWidth(100)

        # Add import  button to import container
        import_button_tab2_layout.addWidget(self.import_button_tab2)
        # Add import container to layout
        tab2_layout.addWidget(import_button_tab2_container)
        
        # Add a container widget for the text entry
        search_container = QWidget()
        search_layout = QHBoxLayout()
        search_layout.setAlignment(Qt.AlignHCenter)  # Center the search elements horizontally
        search_container.setLayout(search_layout)
        tab2_layout.addWidget(search_container)

        # Calculate dimensions to match the filter box width exactly
        filter_box_width = 500  # Same as filter box's maximum width
        search_button_width = 100  # Width of the search button
        padding = 10  # Space between the text box and the button
        manualoffset = 7 # Shhh :)
        search_box_width = filter_box_width - search_button_width - padding - manualoffset

        # Add text entry box
        self.text_entry_tab2 = QLineEdit()
        self.text_entry_tab2.setFont(QFont("Arial", 12))
        self.text_entry_tab2.setAlignment(Qt.AlignCenter)  # Center the text within the box
        self.text_entry_tab2.setPlaceholderText("Enter movie details...")
        self.text_entry_tab2.setStyleSheet("""
            QLineEdit {
                background-color: #2C2C2C;  /* Background color */
                color: white;              /* Text color */
                border: 1px solid white;   /* White border */
                border-radius: 0px;        /* Rounded corners */
                padding: 0px;              /* Padding inside the box */
            }
        """)
        self.text_entry_tab2.setFixedWidth(search_box_width)  # Dynamically adjust width
        self.text_entry_tab2.returnPressed.connect(self.trigger_search)  # Trigger search on ENTER
        search_layout.addWidget(self.text_entry_tab2)

        # Add spacing between the text entry and the button
        search_layout.addSpacing(padding)

        # Add search button
        self.search_button = QPushButton("Search")
        self.search_button.setFont(QFont("Arial", 12))
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 0px;
                padding: 2.5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3C8C43;
            }
        """)
        self.search_button.setFixedWidth(search_button_width)
        self.search_button.clicked.connect(self.trigger_search)
        search_layout.addWidget(self.search_button)

        # Add filtering entry box
        self.filter_entry_tab2 = QLineEdit()
        self.filter_entry_tab2.setFont(QFont("Arial", 12))
        self.filter_entry_tab2.setAlignment(Qt.AlignCenter)
        self.filter_entry_tab2.setPlaceholderText("Filter by title...")
        self.filter_entry_tab2.setStyleSheet("background-color: #2C2C2C; color: white;")
        self.filter_entry_tab2.setFixedWidth(filter_box_width)  # Match the width of search + button + padding

        filter_container_tab2 = QWidget()
        filter_entry_tab2_layout = QVBoxLayout()
        filter_entry_tab2_layout.setAlignment(Qt.AlignHCenter)  # Center the filter box horizontally
        filter_container_tab2.setLayout(filter_entry_tab2_layout)
        tab2_layout.addWidget(filter_container_tab2)

        filter_entry_tab2_layout.addWidget(self.filter_entry_tab2)
        # Connect the filtering box to the filtering logic
        self.filter_entry_tab2.textChanged.connect(
            lambda text: self.filter_table_rows(self.table_tab2, text)
        )

        self.filter_entry_tab2.setMinimumWidth(300)  # Minimum width
        self.filter_entry_tab2.setMaximumWidth(500)  # Maximum width 

        # Add a horizontal container for the counter and filter entry
        filter_container = QWidget()
        filter_layout = QHBoxLayout()
        filter_container.setLayout(filter_layout)
        tab2_layout.addWidget(filter_container)

        # Add the total titles counter
        self.counter_tab2 = QLabel("Titles: 0")
        self.counter_tab2.setFont(QFont("Arial", 12))
        self.counter_tab2.setAlignment(Qt.AlignLeft)
        self.counter_tab2.setStyleSheet("color: white;")
        filter_layout.addWidget(self.counter_tab2)

        # Create a floating suggestion list
        self.suggestion_list_tab2 = ResizableSuggestionBox(self, min_height=100, max_height=800)
        self.suggestion_list_tab2.setWindowFlags(Qt.ToolTip)
        self.suggestion_list_tab2.setStyleSheet("background-color: #1C1C1C; color: white;")
        self.suggestion_list_tab2.hide()
        self.suggestion_list_tab2.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.suggestion_list_tab2.verticalScrollBar().setSingleStep(8)


        # Connect the text entry to update suggestions
        self.text_entry_tab2.textChanged.connect(
            lambda text: self.on_text_changed(text, self.text_entry_tab2, self.suggestion_list_tab2)
        )
        self.suggestion_list_tab2.itemClicked.connect(
            lambda item: self.select_suggestion(item, self.suggestion_list_tab2)
        )

        # Add table
        self.table_tab2 = QTableWidget(0, 5)
        self.table_tab2.setHorizontalHeaderLabels(["Title", "Poster", "Plot", "Cast", "Rating"])
        self.table_tab2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_tab2.setStyleSheet("""
            QTableWidget {
                background-color: #1C1C1C; 
                color: white;
                gridline-color: white;  /* Set grid lines to white */
            }
            QHeaderView::section {
                background-color: #1C1C1C;
                color: white;
                border: 1px solid white;  /* Set header border to white */
            }
            QTableWidget::item {
                border: 1px solid white;  /* Set item border to white */
            }
            QTableWidget::item:selected {
                border: 2px solid #00FF00;  /* Bright green border for selected items */
                background-color: #1C1C1C;  /* Keep background color unchanged */
                color: white;  /* Keep text color white */
            }
        """)
        self.table_tab2.setFont(QFont("Arial", 12))
        self.table_tab2.verticalHeader().setVisible(False)
        self.table_tab2.horizontalHeader().sectionClicked.connect(
        lambda col: self.sort_table(self.table_tab2, col)
        )
        self.populate_table()
        tab2_layout.addWidget(self.table_tab2)

        self.table_tab2.verticalScrollBar().setSingleStep(1)

        # Connect right-click to context menu
        self.table_tab2.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_tab2.customContextMenuRequested.connect(self.show_context_menu)

        self.timer_tab2 = QTimer()
        self.timer_tab2.setSingleShot(True)
        self.timer_tab2.timeout.connect(self.fetch_suggestions)
        self.table_tab2.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table_tab2.verticalScrollBar().setSingleStep(15)

        self.thread = None

    def open_settings_window(self):
        """Open the settings window."""
        settings_window = QDialog(self)
        settings_window.setWindowTitle("Settings")
        settings_window.setFixedSize(400, 300)
        settings_window.setStyleSheet("background-color: #1C1C1C; color: white;")

        layout = QGridLayout(settings_window)
        layout.setSpacing(5)

        # Add Switch API Keys button
        switch_api_keys_button = QPushButton("Switch API Keys")
        switch_api_keys_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        switch_api_keys_button.clicked.connect(self.open_revalidate_dialog)
        layout.addWidget(switch_api_keys_button, 0, 0, 1, 3)

        # Add Move Data Directory button
        move_data_dir_button = QPushButton("Move Data Directory")
        move_data_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        move_data_dir_button.clicked.connect(self.move_data_directory)
        layout.addWidget(move_data_dir_button, 1, 0, 1, 3)

        # Add Max Suggestions entry and Save button in the next row
        max_suggestions_label = QLabel("Maximum Suggestions:")
        max_suggestions_label.setStyleSheet("color: white;")
        layout.addWidget(max_suggestions_label, 3, 0)  # Adjusted to row 3

        max_suggestions_entry = QLineEdit(str(max_suggestions))
        max_suggestions_entry.setValidator(QIntValidator(1, 100))
        layout.addWidget(max_suggestions_entry, 3, 1)  # Adjusted to row 3

        save_button = QPushButton("Save")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        save_button.clicked.connect(lambda: self.save_settings(max_suggestions_entry.text(), settings_window))
        layout.addWidget(save_button, 3, 2)  # Adjusted to row 3

        # Add an empty row
        empty_widget1 = QWidget()  # This will be an empty row
        layout.addWidget(empty_widget1, 4, 0)

        # Add another empty row if needed
        empty_widget2 = QWidget()  # Another empty row
        layout.addWidget(empty_widget2, 5, 0)

        settings_window.setLayout(layout)
        settings_window.exec_()

    def center_window(self):
            """Center the main window on the screen."""
            screen_geometry = QApplication.desktop().screenGeometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())

    def showEvent(self, event):
        """Trigger the resizeEvent explicitly when the window is shown."""
        super().showEvent(event)
        self.resizeEvent(None)

    def resizeEvent(self, event):
        """Update the suggestion list's position and size on window resize."""
        super().resizeEvent(event)
        active_tab_index = self.tabs.currentIndex()
        if active_tab_index == 0:  # Tab 1
            self.position_suggestion_list(self.text_entry_tab1, self.suggestion_list_tab1)
        elif active_tab_index == 1:  # Tab 2
            self.position_suggestion_list(self.text_entry_tab2, self.suggestion_list_tab2)

    def moveEvent(self, event):
        """Update the suggestion list's position and size on window move."""
        super().moveEvent(event)
        active_tab_index = self.tabs.currentIndex()
        if active_tab_index == 0:  # Tab 1
            self.position_suggestion_list(self.text_entry_tab1, self.suggestion_list_tab1)
        elif active_tab_index == 1:  # Tab 2
            self.position_suggestion_list(self.text_entry_tab2, self.suggestion_list_tab2)

    def highlight_row(self, row):
        """Highlight the entire row in orange."""
        for column in range(self.table.columnCount()):
            item = self.table.item(row, column)
            if item:
                item.setBackground(QColor("#FFA500"))

    # -----------------------------------
    # User Interaction Management
    # -----------------------------------
    def enable_user_interaction(self):
        """Re-enable search, filter, and table interactions."""
        self.text_entry_tab1.setEnabled(True)
        self.text_entry_tab2.setEnabled(True)
        self.filter_entry_tab1.setEnabled(True)
        self.filter_entry_tab2.setEnabled(True)
        self.table_tab1.setEnabled(True)
        self.table_tab2.setEnabled(True)
        self.import_button_tab1.setEnabled(True)
        self.import_button_tab2.setEnabled(True)

    def disable_user_interaction(self):
        """Disable search, filter, and table interactions."""
        self.text_entry_tab1.setEnabled(False)
        self.text_entry_tab2.setEnabled(False)
        self.filter_entry_tab1.setEnabled(False)
        self.filter_entry_tab2.setEnabled(False)
        self.table_tab1.setEnabled(False)
        self.table_tab2.setEnabled(False)
        self.import_button_tab1.setEnabled(False)
        self.import_button_tab2.setEnabled(False)
    
    def on_focus_changed(self, old_widget, new_widget):
        """Handle focus changes in the application."""
        if new_widget in [self.text_entry_tab1, self.text_entry_tab2]:
            # Show the relevant suggestion list if the new widget is a text entry
            if new_widget == self.text_entry_tab1 and self.suggestion_list_tab1.count() > 0:
                self.suggestion_list_tab1.show()
            elif new_widget == self.text_entry_tab2 and self.suggestion_list_tab2.count() > 0:
                self.suggestion_list_tab2.show()
        else:
            # Hide both suggestion lists if focus is elsewhere
            self.suggestion_list_tab1.hide()
            self.suggestion_list_tab2.hide()

    def mousePressEvent(self, event):
        """Handle mouse clicks anywhere in the main widget."""
        if not (self.text_entry_tab1.geometry().contains(event.pos()) or 
                self.text_entry_tab2.geometry().contains(event.pos())):
            # Clear focus from both text entry boxes
            self.text_entry_tab1.clearFocus()
            self.text_entry_tab2.clearFocus()
            # Hide suggestion lists if not focused
            self.suggestion_list_tab1.hide()
            self.suggestion_list_tab2.hide()

        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Backslash:  # Check for the "\" key
            self.show_popup("This is a test popup!", color="blue")
        super().keyPressEvent(event)  # Call the parent class method for other key events

    def show_context_menu(self, position):
        """Show context menu on right-click."""
        if self.is_busy:
            self.show_popup("Cannot delete or move titles while data is being loaded or imported.", color="red")
            return

        current_table = self.table_tab1 if self.tabs.currentIndex() == 0 else self.table_tab2
        target_table = self.table_tab2 if current_table == self.table_tab1 else self.table_tab1
        current_tab = self.tabs.tabText(self.tabs.currentIndex())

        index = current_table.indexAt(position)
        if index.isValid():
            row = index.row()

            menu = QMenu()
            delete_action = menu.addAction("Delete Entry")
            move_action = menu.addAction(
                "Move to Watched List" if current_table == self.table_tab1 else "Move to To Watch List"
            )

            action = menu.exec_(current_table.viewport().mapToGlobal(position))
            if action == delete_action:
                # Get the title from the row
                title_item = current_table.item(row, 0)
                if title_item:
                    title = title_item.text().strip()
                    # Remove from saved data
                    self.delete_movie_data(title, current_table)
                    # Remove the row from the table
                    current_table.removeRow(row)
                    # Update the table's row counter
                    self.update_counter(
                        self.counter_tab1 if current_table == self.table_tab1 else self.counter_tab2, current_table
                    )
            elif action == move_action:
                self.move_entry_to_other_tab(row, current_table, target_table)

    def reset_search_button(self):
        """Reset the search button to its original state."""
        self.search_button.setText("Search")
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 0px;
                padding: 2.5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3C8C43;
            }
        """)
        self.search_button.clicked.disconnect()
        self.search_button.clicked.connect(self.trigger_search)
    
    def cancel_search(self):
        """Cancel the current search."""
        if self.suggestion_thread and self.suggestion_thread.isRunning():
            print("[DEBUG] Canceling the search thread.")
            self.query_in_progress = False
            self.suggestion_thread.stop()
            self.suggestion_thread.wait()

        self.reset_search_button()
    
    def trigger_search(self):
        """Start querying suggestions."""
        text = self.text_entry_tab1.text().strip()
        if not text:
            print("[DEBUG] Search text is empty. Aborting search.")
            return

        # Change button text
        self.search_button.setText("Cancel")
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                color: white;
                border-radius: 0px;
                padding: 2.5px;
            }
            QPushButton:hover {
                background-color: #FF4500;
            }
            QPushButton:pressed {
                background-color: #CC3700;
            }
        """)
        self.search_button.clicked.disconnect()
        self.search_button.clicked.connect(self.cancel_search)

        # Start the fetch suggestions logic
        self.fetch_suggestions(self.text_entry_tab1, self.suggestion_list_tab1)

    # -----------------------------------
    # Tab Switching
    # -----------------------------------
    def next_sheet(self):
        """Move to the next sheet (tab) dynamically."""
        current_index = self.tabs.currentIndex()
        next_index = (current_index + 1) % self.tabs.count()  # Wrap around to the first tab
        self.tabs.setCurrentIndex(next_index)

    def previous_sheet(self):
        """Move to the previous sheet (tab) dynamically."""
        current_index = self.tabs.currentIndex()
        previous_index = (current_index - 1) % self.tabs.count()  # Wrap around to the last tab
        self.tabs.setCurrentIndex(previous_index)

    def hide_all_suggestion_lists(self):
        """Hide both suggestion lists and update the active tab's suggestion list."""
        # Hide both suggestion lists
        if self.suggestion_list_tab1:
            self.suggestion_list_tab1.hide()
        if self.suggestion_list_tab2:
            self.suggestion_list_tab2.hide()

        # Update position and size of the active tab's suggestion list
        active_tab_index = self.tabs.currentIndex()
        if active_tab_index == 0:  # Tab 1
            self.position_suggestion_list(self.text_entry_tab1, self.suggestion_list_tab1)
        elif active_tab_index == 1:  # Tab 2
            self.position_suggestion_list(self.text_entry_tab2, self.suggestion_list_tab2)

    def move_data_directory(self):
        """Allow the user to move the saved_data directory to a new location."""
        print("[DEBUG] User initiated move of the saved_data directory.")
        
        # Create a custom dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Move Save Directory")
        dialog.setStyleSheet("background-color: #1C1C1C; color: white;")
        layout = QVBoxLayout(dialog)

        # Add instructions
        instructions = QLabel(
            "Would you like to open an existing save directory or create a new one?"
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)  # Center align the text
        instructions.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(instructions)

        # Add buttons for choices
        open_button = QPushButton("Open Existing Directory")
        open_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        layout.addWidget(open_button)

        create_button = QPushButton("Create New Directory")
        create_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        layout.addWidget(create_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #2E2E2E;
                color: white;
                border: 1px solid white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
        """)
        layout.addWidget(cancel_button)

        def handle_open():
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(dialog, "Select Existing Directory for MoviesList")
            if dir_path:
                self.move_to_selected_directory(dir_path)
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Invalid Directory", "You must select a directory.")

        def handle_create():
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(dialog, "Select Parent Directory for New MoviesList")
            if dir_path:
                new_movies_list_path = os.path.join(dir_path, "MoviesList")
                os.makedirs(new_movies_list_path, exist_ok=True)
                self.move_to_selected_directory(new_movies_list_path)
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Invalid Directory", "You must select a directory.")

        def handle_cancel():
            dialog.reject()

        open_button.clicked.connect(handle_open)
        create_button.clicked.connect(handle_create)
        cancel_button.clicked.connect(handle_cancel)

        dialog.exec_()

    def move_to_selected_directory(self, new_dir_path):
        """Move the save data to the selected directory."""
        try:
            old_dir_path = MovieApp.SAVE_DIR

            # Move files to the new directory
            for item in os.listdir(old_dir_path):
                old_item_path = os.path.join(old_dir_path, item)
                new_item_path = os.path.join(new_dir_path, item)
                if os.path.isdir(old_item_path):
                    os.rename(old_item_path, new_item_path)
                else:
                    os.replace(old_item_path, new_item_path)

            print(f"[DEBUG] Data moved successfully from {old_dir_path} to {new_dir_path}.")

            # Update the config file
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE)
            config['Settings']['saved_data_dir'] = new_dir_path
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            print(f"[DEBUG] Updated config.ini with new directory: {new_dir_path}")

            # Delete the old directory
            os.rmdir(old_dir_path)
            print(f"[DEBUG] Deleted old directory: {old_dir_path}")

            # Update the application state
            MovieApp.SAVE_DIR = new_dir_path
            self.show_popup("Save directory moved successfully.", color="green")
        except Exception as e:
            print(f"[ERROR] Failed to move save directory: {e}")
            QMessageBox.critical(self, "Move Failed", f"An error occurred while moving the directory:\n{e}")

    # -----------------------------------
    # Table Management
    # -----------------------------------
    def populate_table(self):
        """Populate the table with initial data."""
        data = []
        for row_idx, row_data in enumerate(data):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, col_idx, item)

    def populate_table_with_saved_data(self):
        """Populate the tables with saved movies and posters from saved data."""
        to_watch_movies = []
        watched_movies = []

        # Separate movies by their target table
        for title, data in self.saved_movies.items():
            movie_data = {
                "title": title,
                "plot": data.get("plot", ""),
                "rating": data.get("rating", ""),
                "poster_path": data.get("poster_path", ""),
            }
            if data.get("list") == "To Watch List":
                to_watch_movies.append(movie_data)
            else:
                watched_movies.append(movie_data)

        # Add movies in bulk to the tables
        self.add_movies_to_table_in_bulk(to_watch_movies, self.table_tab1)
        self.add_movies_to_table_in_bulk(watched_movies, self.table_tab2)

    def add_movie_to_table(self, title, plot=None, rating=None, poster_path=None, cast=None, target_table=None):
        print(f"[DEBUG] Starting to add movie: '{title}'")
        
        if target_table is None:
            target_table = self.table_tab1

        # Parse the title and year range
        match = re.match(r"(.*?)(?:\s\[(\d{4}(?:–\d{4}|–)?)\])?$", title)
        if match:
            title_only = match.group(1).strip()
            year_range = match.group(2)
            year = year_range.split("–")[0] if year_range else None
        else:
            title_only = title
            year = None

        # Check for duplicates
        existing_list = self.get_existing_list(title)
        if existing_list:
            print(f"[WARNING] Movie '{title}' is already in '{existing_list}'. Skipping addition.")
            self.show_popup(f"'{title}' is already in '{existing_list}'.", color="red")
            return

        print(f"[DEBUG] No duplicates found. Proceeding to add '{title}'.")

        # Fetch additional details if plot, rating, or cast is missing
        if not plot or not rating or not cast:
            url = f"http://www.omdbapi.com/?t={quote(title_only)}&y={year}&apikey={omdb_key}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("Response") == "True":
                        plot = data.get("Plot", "N/A")
                        rating = data.get("imdbRating", "N/A")
                        cast = data.get("Actors", "N/A")
                        poster_path = data.get("Poster", "")
            except Exception as e:
                print(f"[ERROR] Failed to fetch details from OMDb API: {e}")

        # Validate or download the poster
        poster_path = self.get_or_download_poster(poster_path, title)

        # Add the movie to the table
        current_row_count = target_table.rowCount()
        target_table.insertRow(current_row_count)

        # Add title to the table
        title_item = QTableWidgetItem(title)
        title_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 0, title_item)

        # Add poster to the table
        image_label = QLabel()
        if poster_path and os.path.exists(poster_path):
            scaled_pixmap = self.scale_poster(poster_path)
            if scaled_pixmap:
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
        else:
            image_label.setText("No Image")
        image_label.setAlignment(Qt.AlignCenter)
        target_table.setCellWidget(current_row_count, 1, image_label)
        target_table.setRowHeight(current_row_count, 138)

        # Add plot to the table
        plot_item = QTableWidgetItem(plot if plot else "N/A")
        plot_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 2, plot_item)

        # Add cast/actors to the table
        cast_item = QTableWidgetItem(cast if cast else "N/A")
        cast_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 3, cast_item)

        # Add rating to the table
        rating_item = QTableWidgetItem(rating if rating else "N/A")
        rating_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 4, rating_item)

        # Save to saved_movies
        self.saved_movies[title] = {
            "plot": plot if plot else "N/A",
            "rating": rating if rating else "N/A",
            "cast": cast if cast else "N/A",
            "poster_path": poster_path,
            "list": "To Watch List" if target_table == self.table_tab1 else "Watched List",
        }
        self.save_movies()
        print(f"[DEBUG] Movie '{title}' added successfully to the table.")

        # Update original table data
        row_data = {
            "title": title,
            "plot": plot if plot else "N/A",
            "rating": rating if rating else "N/A",
            "cast": cast if cast else "N/A",
            "pixmap": image_label.pixmap().copy() if image_label.pixmap() else None,
        }
        if target_table not in self.original_table_data:
            self.original_table_data[target_table] = []
        self.original_table_data[target_table].append(row_data)

        # Update counter after adding the row
        if target_table == self.table_tab1:
            self.update_counter(self.counter_tab1, self.table_tab1)
        elif target_table == self.table_tab2:
            self.update_counter(self.counter_tab2, self.table_tab2)

    def add_movie_to_table_incrementally(self, movie_data, target_table):
        """Add movies to the specified table incrementally and check for poster."""
        current_row_count = target_table.rowCount()
        target_table.insertRow(current_row_count)

        # Check if the poster exists, otherwise fetch it
        title = movie_data.get("title", "Unknown Title")
        poster_path = movie_data.get("poster_path", "")
        poster_path = self.get_or_download_poster(poster_path, title)

        # Add title to the table
        title_item = QTableWidgetItem(title)
        title_item.setFlags(title_item.flags() & ~Qt.ItemIsEditable)
        title_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 0, title_item)

        # Add poster to the table
        image_label = QLabel()
        if poster_path and os.path.exists(poster_path):
            scaled_pixmap = self.scale_poster(poster_path)
            if scaled_pixmap:
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("No Image")
        else:
            image_label.setText("No Image")
        image_label.setAlignment(Qt.AlignCenter)
        target_table.setCellWidget(current_row_count, 1, image_label)
        target_table.setRowHeight(current_row_count, 138)

        # Add plot, cast, and rating
        plot = movie_data.get("plot", "N/A")
        rating = movie_data.get("rating", "N/A")
        cast = movie_data.get("cast", "N/A")
        plot_item = QTableWidgetItem(plot)
        plot_item.setFlags(plot_item.flags() & ~Qt.ItemIsEditable)
        plot_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 2, plot_item)
        cast_item = QTableWidgetItem(cast)
        cast_item.setFlags(cast_item.flags() & ~Qt.ItemIsEditable)
        cast_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 3, cast_item)
        rating_item = QTableWidgetItem(rating)
        rating_item.setFlags(rating_item.flags() & ~Qt.ItemIsEditable)
        rating_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(current_row_count, 4, rating_item)

        # Update saved_movies
        self.saved_movies[title] = {
            "plot": plot,
            "rating": rating,
            "cast": cast,
            "poster_path": poster_path,
            "list": "To Watch List" if target_table == self.table_tab1 else "Watched List",
        }
        self.save_movies()

        # Add to original_table_data for filtering
        row_data = {
            "title": title,
            "plot": plot,
            "rating": rating,
            "cast": cast,
            "pixmap": image_label.pixmap().copy() if image_label.pixmap() else None,
        }
        if target_table not in self.original_table_data:
            self.original_table_data[target_table] = []
        self.original_table_data[target_table].append(row_data)

        # Update counter
        if target_table == self.table_tab1:
            self.update_counter(self.counter_tab1, self.table_tab1)
        elif target_table == self.table_tab2:
            self.update_counter(self.counter_tab2, self.table_tab2)

    def add_movies_to_table_in_bulk(self, movies, target_table):
        """Add multiple movies to a table."""
        for movie in movies:
            self.add_movie_to_table(
                title=movie["title"],
                plot=movie["plot"],
                rating=movie["rating"],
                poster_path=movie["poster_path"],
                target_table=target_table,
            )
        target_table.update()  # Refresh the table UI once after bulk addition

    def sort_table(self, table, column):
        """Sort the table based on the clicked column with four sorting states."""
        # Define the column-specific sorting state keys
        sort_states = table.property("sort_states") or {0: 0, 4: 0}
        current_state = sort_states.get(column, 0)

        # Get all row data and store it in a list
        row_data = []
        for row in range(table.rowCount()):
            title = table.item(row, 0).text() if table.item(row, 0) else ""
            rating = float(table.item(row, 4).text()) if table.item(row, 4) and table.item(row, 4).text().replace('.', '', 1).isdigit() else 0.0
            cast = table.item(row, 3).text() if table.item(row, 3) else ""
            plot = table.item(row, 2).text() if table.item(row, 2) else ""
            widget = table.cellWidget(row, 1)
            pixmap = widget.pixmap() if widget and widget.pixmap() else None

            # Store the row data
            row_data.append({
                "title": title,
                "rating": rating,
                "cast": cast,
                "plot": plot,
                "pixmap": pixmap,
            })

        # Determine sorting order based on current state
        if column == 0:
            if current_state == 0:
                row_data.sort(key=lambda x: x["title"], reverse=False)  # A-Z
            elif current_state == 1:
                row_data.sort(key=lambda x: x["title"], reverse=True)  # Z-A
        elif column == 4:
            if current_state == 0:
                row_data.sort(key=lambda x: x["rating"], reverse=True)  # Highest to Lowest
            elif current_state == 1:
                row_data.sort(key=lambda x: x["rating"], reverse=False)  # Lowest to Highest

        # Cycle to the next state
        sort_states[column] = (current_state + 1) % 2
        table.setProperty("sort_states", sort_states)  # Save the updated states

        # Clear and repopulate the table
        table.setRowCount(0)
        for row in row_data:
            row_idx = table.rowCount()
            table.insertRow(row_idx)

            # Title
            title_item = QTableWidgetItem(row["title"])
            title_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 0, title_item)

            # Poster
            new_label = QLabel()
            if row["pixmap"]:
                new_label.setPixmap(row["pixmap"].scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                new_label.setText("No Image")
            new_label.setAlignment(Qt.AlignCenter)
            table.setCellWidget(row_idx, 1, new_label)

            # Plot
            plot_item = QTableWidgetItem(row["plot"])
            plot_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 2, plot_item)

            # Cast
            cast_item = QTableWidgetItem(row["cast"])
            cast_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, cast_item)

            # Rating
            rating_item = QTableWidgetItem(f"{row['rating']:.1f}" if row['rating'] else "N/A")
            rating_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 4, rating_item)

            # Set consistent row height
            table.setRowHeight(row_idx, 138)

        # Update sort indicator for visual feedback
        table.horizontalHeader().setSortIndicator(column, Qt.AscendingOrder if current_state == 0 else Qt.DescendingOrder)

    def delete_movie_data(self, title, current_table):
        """Delete movie data, including removing from saved_movies and poster."""
        print(f"[DEBUG] Attempting to delete movie: {title}")
        normalized_title = title.strip().lower()

        # Find an exact match in saved_movies
        matching_key = None
        for saved_title in self.saved_movies.keys():
            if saved_title.strip().lower() == normalized_title:
                matching_key = saved_title
                break

        if matching_key:
            print(f"[DEBUG] Found exact match: {matching_key}")
            movie_data = self.saved_movies.pop(matching_key, None)
            if movie_data:
                print(f"[DEBUG] Found movie in saved_movies: {movie_data}")

                # Delete poster if it exists
                poster_path = movie_data.get("poster_path", "")
                if poster_path and os.path.exists(poster_path):
                    try:
                        os.remove(poster_path)
                        print(f"[DEBUG] Deleted poster at: {poster_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete poster at {poster_path}: {e}")
                else:
                    print(f"[DEBUG] No poster to delete for: {title}")
            else:
                print(f"[ERROR] Movie data for {title} is None. Skipping deletion.")

            # Save the updated movies to file
            self.save_movies()
            print("[DEBUG] Updated saved_movies and saved to file.")
        else:
            print(f"[WARNING] Movie '{title}' not found in saved_movies. Skipping deletion.")

        # Verify saved_movies after deletion
        print(f"[DEBUG] Current saved_movies: {json.dumps(self.saved_movies, indent=4)}")

        if current_table == self.table_tab1:
            self.update_counter(self.counter_tab1, self.table_tab1)
        elif current_table == self.table_tab2:
            self.update_counter(self.counter_tab2, self.table_tab2)

    def move_entry_to_other_tab(self, row, current_table, target_table):
        """Move a row from one table to the other and update original_table_data."""
        # Extract data from the current table
        title = current_table.item(row, 0).text()
        plot = current_table.item(row, 2).text()
        cast = current_table.item(row, 3).text()
        rating = current_table.item(row, 4).text()
        poster_widget = current_table.cellWidget(row, 1)
        pixmap = poster_widget.pixmap() if poster_widget and poster_widget.pixmap() else None

        # Add the row to the target table
        target_row = target_table.rowCount()
        target_table.insertRow(target_row)

        # Title
        title_item = QTableWidgetItem(title)
        title_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(target_row, 0, title_item)

        # Poster
        image_label = QLabel()
        if pixmap:
            scaled_pixmap = pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
        else:
            image_label.setText("No Image")
        image_label.setAlignment(Qt.AlignCenter)
        target_table.setCellWidget(target_row, 1, image_label)
        target_table.setRowHeight(target_row, 138)

        # Plot
        plot_item = QTableWidgetItem(plot)
        plot_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(target_row, 2, plot_item)

        # Cast
        cast_item = QTableWidgetItem(cast)
        cast_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(target_row, 3, cast_item)

        # Rating
        rating_item = QTableWidgetItem(rating)
        rating_item.setTextAlignment(Qt.AlignCenter)
        target_table.setItem(target_row, 4, rating_item)

        # Update saved_movies
        if title in self.saved_movies:
            self.saved_movies[title]["list"] = (
                "To Watch List" if target_table == self.table_tab1 else "Watched List"
            )
            self.save_movies()

        # Update original_table_data
        if current_table in self.original_table_data:
            # Remove from current_table's original_table_data
            for idx, row_data in enumerate(self.original_table_data[current_table]):
                if row_data.get("title") == title:
                    # Move to target_table's original_table_data
                    if target_table not in self.original_table_data:
                        self.original_table_data[target_table] = []
                    self.original_table_data[target_table].append(row_data)
                    del self.original_table_data[current_table][idx]
                    break

        # Remove the row from the current table
        current_table.removeRow(row)

        # Update counters
        self.update_counter(
            self.counter_tab1 if current_table == self.table_tab1 else self.counter_tab2, current_table
        )
        self.update_counter(
            self.counter_tab1 if target_table == self.table_tab1 else self.counter_tab2, target_table
        )

    def is_duplicate(self, title_with_year, table):
        """Check if a title with the year already exists in the given table."""
        for row in range(table.rowCount()):
            existing_title = table.item(row, 0)  # Column 0 contains titles
            if existing_title and existing_title.text().strip().lower() == title_with_year.strip().lower():
                return True
        return False

    # -----------------------------------
    # Suggestions and Filtering
    # -----------------------------------
    def create_suggestion_item(self, title, year, poster_url, actors):
        """Create a suggestion list item."""
        widget = QWidget()
        layout = QHBoxLayout()

        # Add poster image
        image_label = QLabel()
        image_label.setFixedSize(92, 138)
        if poster_url and poster_url != "N/A":
            try:
                pixmap = QPixmap()
                pixmap.loadFromData(requests.get(poster_url).content)
                image_label.setPixmap(
                    pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            except Exception as e:
                print(f"[ERROR] Failed to load poster: {e}")
                image_label.setText("No Image")
        else:
            image_label.setText("No Image")
        layout.addWidget(image_label)

        # Add movie details
        title_label = QLabel(f"{title} ({year})")
        title_label.setStyleSheet("color: white;")
        title_label.setWordWrap(True)

        actors_label = QLabel(f"Actors: {actors}")
        actors_label.setStyleSheet("color: gray; font-size: 12px;")
        actors_label.setWordWrap(True)

        text_layout = QVBoxLayout()
        text_layout.addWidget(title_label)
        text_layout.addWidget(actors_label)

        layout.addLayout(text_layout)
        widget.setLayout(layout)
        return widget

    def fetch_suggestions(self, text_entry, suggestion_list):
        """Fetch movie suggestions."""
        text = text_entry.text().strip()
        if not text or not self.suggestions_active:
            print("[DEBUG] No text entered or suggestions disabled. Skipping fetch suggestions.")
            self.reset_search_button()
            return

        # Clear the suggestion list before fetching new suggestions
        suggestion_list.clear()
        self.current_suggestions = []

        # Start a new suggestion thread
        self.query_in_progress = True
        self.suggestion_thread = FetchSuggestionsThread(text=text)
        self.suggestion_thread.suggestionsFetched.connect(
            lambda suggestion: self.add_unique_suggestion_to_list(suggestion, suggestion_list)
        )
        self.suggestion_thread.finished.connect(self.cleanup_suggestion_thread)

        self.suggestion_thread.start()
        print("[DEBUG] FetchSuggestionsThread started.")

    def fetch_suggestions(self, text_entry, suggestion_list):
        """Fetch movie suggestions."""
        text = text_entry.text().strip()
        if not text or not self.suggestions_active:
            print("[DEBUG] No text entered or suggestions disabled. Skipping fetch suggestions.")
            self.reset_search_button()
            self.query_in_progress = False  # Ensure it's reset
            return

        # Clear the suggestion list before fetching new suggestions
        suggestion_list.clear()
        self.current_suggestions = []
        self.query_in_progress = True  # Set the query in progress flag

        # Start a new suggestion thread
        self.suggestion_thread = FetchSuggestionsThread(text=text)
        self.suggestion_thread.suggestionsFetched.connect(
            lambda suggestion: self.add_unique_suggestion_to_list(suggestion, suggestion_list)
        )
        self.suggestion_thread.finished.connect(self.cleanup_suggestion_thread)

        self.suggestion_thread.start()
        print("[DEBUG] FetchSuggestionsThread started.")

    def add_suggestion_to_list(self, suggestion, suggestion_list):
        """Dynamically add suggestions to the suggestion list."""
        if not suggestion:
            print("[ERROR] Empty suggestion passed.")
            return

        # Validate fields
        title = suggestion.get("title", "Unknown Title")
        year = suggestion.get("year", "Unknown Year")
        poster_url = suggestion.get("poster_url", "")
        actors = suggestion.get("actors", "N/A")

        # Add to current_suggestions for later matching
        self.current_suggestions.append(suggestion)
        print(f"[DEBUG] Adding suggestion: {suggestion}")

        # Create the list item
        item_widget = self.create_suggestion_item(title, year, poster_url, actors)
        item = QListWidgetItem(suggestion_list)
        item.setSizeHint(item_widget.sizeHint())
        suggestion_list.addItem(item)
        suggestion_list.setItemWidget(item, item_widget)

        # Show the suggestion list
        if not suggestion_list.isVisible():
            suggestion_list.show()

    def add_unique_suggestion_to_list(self, suggestion, suggestion_list):
        """Add suggestions to the list while ensuring uniqueness."""
        if not self.suggestions_active or not self.query_in_progress:
            print("[DEBUG] Suggestions fetching disabled. Ignoring suggestion.")
            return

        # Check for duplicates
        existing_titles = {(s["title"].strip().lower(), s["year"].strip()) for s in self.current_suggestions}
        title_key = (suggestion["title"].strip().lower(), suggestion["year"].strip())

        if title_key in existing_titles:
            print(f"[DEBUG] Duplicate suggestion ignored: {suggestion}")
            return

        # Respect max_suggestions (excluding the OMDb suggestion)
        if len(self.current_suggestions) >= max_suggestions + 1:  # +1 for the OMDb suggestion
            print(f"[DEBUG] Max suggestions reached. Ignoring suggestion: {suggestion}")
            return

        # Add suggestion to the current list
        self.current_suggestions.append(suggestion)
        print(f"[DEBUG] Added to current_suggestions: {suggestion}")

        # Add to the suggestion list UI
        item_widget = self.create_suggestion_item(
            suggestion["title"], suggestion["year"], suggestion["poster_url"], suggestion["actors"]
        )
        item = QListWidgetItem(suggestion_list)
        item.setSizeHint(item_widget.sizeHint())
        suggestion_list.addItem(item)
        suggestion_list.setItemWidget(item, item_widget)

        if not suggestion_list.isVisible():
            suggestion_list.show()

    def select_suggestion(self, item, suggestion_list):
        """Handle the selection of a suggestion."""
        # Disable further querying temporarily
        self.suggestions_active = False

        # Stop any ongoing suggestion thread
        if self.suggestion_thread is not None and self.suggestion_thread.isRunning():
            self.suggestion_thread.stop()
            self.suggestion_thread.wait()

        active_tab_index = self.tabs.currentIndex()
        table = self.table_tab1 if active_tab_index == 0 else self.table_tab2
        text_entry = self.text_entry_tab1 if active_tab_index == 0 else self.text_entry_tab2

        widget = suggestion_list.itemWidget(item)
        if widget:
            layout = widget.layout()
            title_label = None
            for i in range(layout.count()):
                child_layout = layout.itemAt(i)
                if isinstance(child_layout, QVBoxLayout):
                    for j in range(child_layout.count()):
                        child = child_layout.itemAt(j).widget()
                        if isinstance(child, QLabel) and "(" in child.text():
                            title_label = child
                            break
                    if title_label:
                        break

            if title_label:
                suggestion_text = title_label.text().strip()
                match = re.match(r"(.+?)\s\((\d{4}(?:–\d{4}|–)?)\)", suggestion_text)
                if match:
                    title = match.group(1).strip()
                    year = match.group(2).strip()

                    # Handle multi-year formats correctly
                    title_with_year = f"{title} [{year}]"

                    # Add the selected suggestion to the table
                    self.add_movie_to_table(
                        title=title_with_year,
                        plot=None,
                        rating=None,
                        poster_path=None,
                        cast=None,
                        target_table=table,
                    )

        suggestion_list.hide()
        text_entry.clear()

        # Reactivate suggestions after selection
        self.suggestions_active = True

    def position_suggestion_list(self, text_entry, suggestion_list):
        """Position the suggestion list directly below the given text entry."""
        print("[DEBUG] Positioning suggestion list.")
        text_entry_global_pos = text_entry.mapToGlobal(QPoint(0, text_entry.height()))
        suggestion_list.move(text_entry_global_pos)
        suggestion_list.setFixedWidth(text_entry.width())
        print(f"[DEBUG] Suggestion list positioned at: {text_entry_global_pos}")

    def filter_table_rows(self, table, filter_text):
        """Filter rows in the table by title or cast."""
        filter_text = filter_text.strip().lower()
        if not filter_text:
            self.restore_original_table_data(table)
            return

        if table not in self.original_table_data or not self.original_table_data[table]:
            self.save_original_table_data(table)

        table.setRowCount(0)
        for row_data in self.original_table_data[table]:
            title = row_data.get("title", "").lower()
            cast = row_data.get("cast", "N/A").lower()
            print(f"[DEBUG] Filtering row - Title: {title}, Cast: {cast}")

            if filter_text in title or filter_text in cast:
                self.add_filtered_row(table, row_data)

    def save_original_table_data(self, table):
        """Save the current data of the table to original_table_data."""
        self.original_table_data[table] = []
        for row in range(table.rowCount()):
            title = table.item(row, 0).text() if table.item(row, 0) else ""
            plot = table.item(row, 2).text() if table.item(row, 2) else ""
            cast = table.item(row, 3).text() if table.item(row, 3) else "N/A"
            rating = table.item(row, 4).text() if table.item(row, 4) else "N/A"
            poster_widget = table.cellWidget(row, 1)
            pixmap = poster_widget.pixmap() if poster_widget and poster_widget.pixmap() else None
            self.original_table_data[table].append({
                "title": title,
                "plot": plot,
                "cast": cast,
                "rating": rating,
                "pixmap": pixmap.copy() if pixmap else None,
            })

    def restore_original_table_data(self, table):
        """Restore the table data from original_table_data."""
        if table in self.original_table_data:
            table.setRowCount(0)
            for row_data in self.original_table_data[table]:
                self.add_filtered_row(table, row_data)

    def add_filtered_row(self, table, row_data):
        try:
            row_idx = table.rowCount()
            table.insertRow(row_idx)

            # Title
            title_item = QTableWidgetItem(row_data.get("title", ""))
            title_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 0, title_item)

            # Poster
            new_label = QLabel()
            pixmap = row_data.get("pixmap", None)
            if pixmap and not pixmap.isNull():
                new_label.setPixmap(pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                new_label.setText("No Image")
            new_label.setAlignment(Qt.AlignCenter)
            table.setCellWidget(row_idx, 1, new_label)

            # Plot
            plot_item = QTableWidgetItem(row_data.get("plot", "N/A"))
            plot_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 2, plot_item)

            # Cast
            cast_item = QTableWidgetItem(row_data.get("cast", "N/A"))
            cast_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 3, cast_item)

            # Rating
            rating_item = QTableWidgetItem(row_data.get("rating", "N/A"))
            rating_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row_idx, 4, rating_item)

            table.setRowHeight(row_idx, 138)
        except Exception as e:
            print("Error while adding filtered row:", str(e))

    def cleanup_suggestion_thread(self):
        """Cleanup suggestion thread."""
        self.query_in_progress = False  # Reset the query in progress flag
        self.suggestion_thread = None
        self.reset_search_button()  # Ensure button is reset after query

    def load_image_async(self, label, url):
        """Load image asynchronously."""
        thread = ImageLoaderThread(label, url)
        thread.imageLoaded.connect(self.set_image)
        thread.finished.connect(lambda: self.image_threads.remove(thread))
        self.image_threads.append(thread)
        thread.start()

    def set_image(self, label, pixmap):
        """Set the loaded image on the label with proper scaling."""
        if pixmap and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        else:
            label.setText("No Image")

    # -----------------------------------
    # Movie Data Management
    # -----------------------------------
    def save_movies(self):
        """Save movies to the JSON file in the user-defined directory."""
        try:
            file_path = os.path.join(self.SAVE_DIR, "movies.json")
            with open(file_path, "w") as file:
                json.dump(self.saved_movies, file, indent=4)
            print(f"[DEBUG] Movies saved to file at: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save movies: {e}")
    
    def load_saved_movies(self):
        """Load saved movies from the JSON file in the user-defined directory."""
        try:
            file_path = os.path.join(self.SAVE_DIR, "movies.json")
            print(f"[DEBUG] Attempting to load movies from: {file_path}")
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    saved_movies = json.load(file)
                    print(f"[DEBUG] Successfully loaded movies: {json.dumps(saved_movies, indent=4)}")
                    return saved_movies
            else:
                print("[DEBUG] No saved movies file found. Returning empty dictionary.")
                return {}
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse saved movies JSON: {e}")
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load saved movies: {e}")
            return {}
    
    def get_existing_list(self, title_with_year):
        """Check if a movie or show already exists in either list."""
        match = re.match(r"(.*?)(?:\s\[(\d{4}(?:–\d{4}|–)?)\])?$", title_with_year)
        if match:
            title_only = match.group(1).strip()
            input_year_range = match.group(2)

            # Extract start year for input comparison
            input_start_year = None
            if input_year_range:
                input_start_year = input_year_range.split("–")[0]

        for table, list_name in [(self.table_tab1, "To Watch List"), (self.table_tab2, "Watched List")]:
            for row in range(table.rowCount()):
                existing_title_with_year = table.item(row, 0).text() if table.item(row, 0) else ""
                existing_match = re.match(r"(.*?)(?:\s\[(\d{4}(?:–\d{4}|–)?)\])?$", existing_title_with_year)
                if existing_match:
                    existing_title = existing_match.group(1).strip()
                    existing_year_range = existing_match.group(2)

                    # Extract start year for existing comparison
                    existing_start_year = None
                    if existing_year_range:
                        existing_start_year = existing_year_range.split("–")[0]

                    # Compare the title and start year
                    if title_only.lower() == existing_title.lower() and input_start_year == existing_start_year:
                        return list_name
        return None

    def import_movies(self, target_table):
        """Handle importing movies from a text file using threading."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Movie List File", "", "Text Files (*.txt)")

        if not file_path:
            return  # User canceled the dialog

        try:
            with open(file_path, "r") as file:
                movie_titles = [line.strip() for line in file if line.strip()]

            if not movie_titles:
                QMessageBox.warning(self, "No Movies Found", "The selected file contains no movie titles.")
                return

            # Start the import thread
            self.start_import_thread(movie_titles, target_table)
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"An error occurred while reading the file: {e}")

    def start_import_thread(self, movie_titles, target_table):
        """Start a threaded movie import with a concurrent queue."""
        self.is_busy = True
        self.queue = Queue()
        for title in movie_titles:
            self.queue.put(title)

        self.import_thread = ImportMoviesThread(movie_titles, self.queue)
        self.import_thread.movieAdded.connect(
            lambda movie_data: self.add_movie_to_table_incrementally(movie_data, target_table)
        )
        self.import_thread.finished.connect(self.on_import_finished)
        self.import_thread.start()
        QMessageBox.information(self, "Import Started", "Movies are being imported using a queue.")

    def on_import_finished(self):
        """Handle the end of the import process."""
        self.is_busy = False
        self.show_popup("Import complete. You can now move or delete titles.", color="green")
        self.save_movies()

    def start_querying_movies(self, movie_titles, target_table):
        """Start querying movies using a thread."""
        self.import_thread = ImportMoviesThread(movie_titles, target_table)
        self.import_thread.movieAdded.connect(lambda movie_data: self.add_movie_to_table(
            movie_data["title"],
            movie_data["plot"],
            movie_data["rating"],
            movie_data["poster_path"],
            movie_data["cast"],
            target_table
        ))
        self.import_thread.finished.connect(lambda: QMessageBox.information(self, "Import Complete", "All movies have been processed."))
        self.import_thread.start()

    def scale_poster(self, poster_path):
        """Scale poster images to fit within table entries."""
        try:
            pixmap = QPixmap(poster_path)
            if not pixmap.isNull():
                return pixmap.scaled(92, 138, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                print(f"[ERROR] Pixmap is null for: {poster_path}")
        except Exception as e:
            print(f"[ERROR] Exception while scaling poster: {e}")
        return None

    def download_poster(self, url, title):
        """Download and save the poster image locally in the user-defined directory."""
        poster_path = os.path.join(self.SAVE_DIR, f"{title}.jpg")
        if os.path.exists(poster_path):
            return poster_path  # Return if already exists

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(poster_path, "wb") as file:
                    file.write(response.content)
                return poster_path
            else:
                print(f"[ERROR] Failed to download poster. HTTP Status: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Exception while downloading poster: {e}")

        return None  # Return None if download fails

    def get_or_download_poster(self, poster_url, title):
        """Get the poster from saved data or use a placeholder if unavailable."""
        if not poster_url or poster_url == "N/A":  # Handle missing or invalid URLs
            print(f"[DEBUG] Poster URL is invalid for title: {title}")
            return "path/to/placeholder/image.jpg"  # Replace with an actual path to a placeholder image

        poster_path = os.path.join(self.SAVE_DIR, f"{title}.jpg")
        if os.path.exists(poster_path):
            return poster_path

        try:
            response = requests.get(poster_url, timeout=5)
            if response.status_code == 200:
                with open(poster_path, "wb") as file:
                    file.write(response.content)
                return poster_path
            else:
                print(f"[ERROR] Failed to download poster. HTTP Status: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Exception while downloading poster: {e}")

        # Use the placeholder if download fails
        return "path/to/placeholder/image.jpg"
    
    def _load_and_populate_data_in_thread(self):
        """Load and populate data incrementally in a background thread."""
        self.is_busy = True  # Disable interactions
        self.disable_user_interaction()
        self.saved_movies = self.load_saved_movies()

        self.loading_thread = DataLoadingThread(self.saved_movies)
        self.loading_thread.dataLoaded.connect(
            lambda movie_data: self.add_movie_to_table_incrementally(
                movie_data,
                self.table_tab1 if movie_data.get("list") == "To Watch List" else self.table_tab2
            )
        )
        self.loading_thread.finished.connect(self.on_data_loaded)
        self.loading_thread.start()

    def on_data_loaded(self):
        """Handle the end of the data loading process."""
        self.is_busy = False  # Re-enable interactions
        self.enable_user_interaction()
        self.show_popup("Data loading complete. You can now move or delete titles.", color="green")

        # Update tab names with the current counts
        self.update_counter(self.counter_tab1, self.table_tab1)
        self.update_counter(self.counter_tab2, self.table_tab2)

    def update_counter(self, counter_label, table):
        """Update the counter label with the total number of rows in the table, considering original_table_data."""
        # Check if we have original data stored for the table
        total_rows = len(self.original_table_data.get(table, [])) if table in self.original_table_data else table.rowCount()
        counter_label.setText(f"Titles: {total_rows}")

        # Update the tab name with the total row count
        if table == self.table_tab1:
            self.tabs.setTabText(0, f"To Watch List ({total_rows})")
        elif table == self.table_tab2:
            self.tabs.setTabText(1, f"Watched List ({total_rows})")

    def update_table_with_filtered_data(self, table, filtered_data):
        """Update the table with filtered rows."""
        table.setRowCount(0)
        for row_data in filtered_data:
            self.add_filtered_row(table, row_data)

    # -----------------------------------
    # Popup Notifications
    # -----------------------------------
    def show_popup(self, message, color="red"):
        """Show a non-blocking popup with a vertically centered message, close 'X' button, and timer."""
        popup = QWidget(self)
        popup_layout = QHBoxLayout(popup)
        popup_layout.setContentsMargins(10, 0, 10, 0)  # Add padding inside the popup
        popup_layout.setSpacing(10)  # Space between elements

        popup.setStyleSheet(f"""
            QWidget {{
                background-color: {color};
                border-radius: 5px;
            }}
            QLabel {{
                color: white;
                font: bold 14px;
            }}
            QPushButton {{
                background-color: transparent;
                border: none;
                color: white;
                font: bold 14px;
            }}
            QPushButton:hover {{
                color: lightgray;
            }}
        """)
        popup.setFixedHeight(50)
        popup.setFixedWidth(self.width() - 40)

        # Message label
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        popup_layout.addWidget(message_label, stretch=1)

        # Timer label
        remaining_time = 3
        timer_label = QLabel(f"{remaining_time}s")
        timer_label.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        timer_label.setFixedWidth(30)
        popup_layout.addWidget(timer_label, alignment=Qt.AlignVCenter)

        # Close button
        close_button = QPushButton("X")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(lambda: self.remove_popup(popup))  # Close the popup on click
        popup_layout.addWidget(close_button, alignment=Qt.AlignVCenter)

        popup.setLayout(popup_layout)

        # Add popup to the active popups list
        if not hasattr(self, "active_popups"):
            self.active_popups = []
        self.active_popups.append(popup)

        # Position popups
        self.adjust_popups()

        popup.show()

        # Timer logic
        def update_timer():
            nonlocal remaining_time
            remaining_time -= 1
            timer_label.setText(f"{remaining_time}s")
            if remaining_time <= 0:
                self.remove_popup(popup)

        # Create a QTimer to update the timer label every second
        popup_timer = QTimer(popup)
        popup_timer.timeout.connect(update_timer)
        popup_timer.start(1000)

    def adjust_popups(self):
        """Reposition popups so they stack from the top."""
        y_offset = 10
        for popup in reversed(self.active_popups):
            popup.move(10, y_offset)
            y_offset += popup.height() + 10

    def remove_popup(self, popup):
        """Remove popup and adjust positions."""
        if popup in self.active_popups:
            self.active_popups.remove(popup)
            popup.deleteLater()  # Safely delete the popup
            self.adjust_popups()

    # -----------------------------------
    # API Key Management
    # -----------------------------------
    def open_revalidate_dialog(self):
        """Open a dialog to allow revalidating one or both API keys."""
        print("[DEBUG] Opening revalidate API keys dialog.")
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Switch API Keys")
        dialog.setStyleSheet("background-color: #1C1C1C; color: white;")
        layout = QFormLayout(dialog)

        # Add checkboxes for selecting keys to revalidate
        self.omdb_checkbox = QCheckBox("Switch OMDB API Key")
        self.omdb_checkbox.setStyleSheet("color: white;")
        layout.addRow(self.omdb_checkbox)

        self.openai_checkbox = QCheckBox("Switch OpenAI API Key")
        self.openai_checkbox.setStyleSheet("color: white;")
        layout.addRow(self.openai_checkbox)

        # Add dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        # Connect the dialog buttons
        buttons.accepted.connect(lambda: self.handle_revalidate_keys(dialog))
        buttons.rejected.connect(dialog.reject)

        # Show dialog
        dialog.exec_()
        print("[DEBUG] Revalidate API keys dialog closed.")

    def handle_revalidate_keys(self, dialog):
        print("[DEBUG] Revalidate API keys dialog confirmed.")
        omdb_revalidate = self.omdb_checkbox.isChecked()
        openai_revalidate = self.openai_checkbox.isChecked()

        print(f"[DEBUG] Selected revalidation - OMDB: {omdb_revalidate}, OpenAI: {openai_revalidate}")

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        encryption_key = config['Settings']['encryption_key']
        api_key_file = os.path.join(CONFIG_DIR, "api_keys.secure")

        try:
            current_omdb_key, current_openai_key = load_api_keys(api_key_file, encryption_key)
        except Exception as e:
            print(f"[ERROR] Failed to load existing API keys: {e}")
            current_omdb_key, current_openai_key = None, None

        # Revalidate OMDB API Key
        if omdb_revalidate:
            while True:
                omdb_dialog = CustomInputDialog("Switch OMDB API Key", "Enter new OMDB API Key:", cancel_text="Cancel")
                if omdb_dialog.exec_() == QDialog.Accepted:
                    omdb_key = omdb_dialog.get_input()
                    if not omdb_key:
                        QMessageBox.warning(self, "Invalid Key", "OMDB API Key cannot be empty.")
                        continue
                    if validate_api_key_omdb(omdb_key):
                        print("[DEBUG] OMDB API Key validated successfully.")
                        break
                    else:
                        QMessageBox.warning(self, "Invalid Key", "OMDB API Key is invalid.")
                        print("[DEBUG] OMDB API Key validation failed.")
                else:
                    print("[DEBUG] User canceled OMDB API Key revalidation.")
                    omdb_key = current_omdb_key  # Fallback to the current key
                    break
        else:
            omdb_key = current_omdb_key  # Use the existing key if not revalidated

        # Revalidate OpenAI API Key
        if openai_revalidate:
            while True:
                openai_dialog = CustomInputDialog("Switch OpenAI API Key", "Enter new OpenAI API Key:", cancel_text="Cancel")
                if openai_dialog.exec_() == QDialog.Accepted:
                    new_openai_key = openai_dialog.get_input()
                    if not new_openai_key:
                        QMessageBox.warning(self, "Invalid Key", "OpenAI API Key cannot be empty.")
                        continue
                    if validate_api_key_openai(new_openai_key):
                        print("[DEBUG] OpenAI API Key validated successfully.")
                        global openai_key  # Update the global variable
                        openai_key = new_openai_key
                        openai.api_key = new_openai_key  # Update OpenAI library with the new key
                        break
                    else:
                        QMessageBox.warning(self, "Invalid Key", "OpenAI API Key is invalid.")
                        print("[DEBUG] OpenAI API Key validation failed.")
                else:
                    print("[DEBUG] User canceled OpenAI API Key revalidation.")
                    new_openai_key = current_openai_key  # Fallback to the current key
                    break
        else:
            new_openai_key = current_openai_key  # Use the existing key if not revalidated

        # Save the revalidated keys
        try:
            save_api_keys(omdb_key, new_openai_key, api_key_file, encryption_key)
            print("[DEBUG] API keys saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save API keys: {e}")
            QMessageBox.critical(self, "Save Error", "Failed to save API keys. Please try again.")

        # Refresh threads
        if self.suggestion_thread and self.suggestion_thread.isRunning():
            print("[DEBUG] Terminating active suggestion thread.")
            self.suggestion_thread.terminate()
            self.suggestion_thread.wait()
            self.suggestion_thread = None

        dialog.accept()
        print("[DEBUG] Revalidate API keys dialog closed.")

    def save_settings(self, max_suggestions_value, settings_window):
        global max_suggestions
        try:
            max_suggestions = int(max_suggestions_value)
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE)
            config['Settings']['max_suggestions'] = str(max_suggestions)
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            settings_window.accept()
            self.show_popup("Settings saved successfully.", color="green")
        except ValueError:
            self.show_popup("Invalid value for maximum suggestions.", color="red")

        # Misc.
    
    # -----------------------------------
    # Miscellaneous
    # -----------------------------------
    def cleanup_thread(self):
        """Cleanup the thread after it finishes."""
        self.thread = None

    def closeEvent(self, event):
        """Handle window close event."""

        # Stop the suggestion thread
        if self.suggestion_thread is not None and self.suggestion_thread.isRunning():
            self.suggestion_thread.stop()
            self.suggestion_thread.wait()

        # Stop all image threads
        for thread in self.image_threads:
            thread.terminate()
        self.image_threads.clear()

        event.accept()

def main():
    global omdb_key, openai_key
    app = QApplication(sys.argv)
    try:
        saved_data_dir, encryption_key = ensure_config()
        print(f"[DEBUG] Using saved_data directory: {saved_data_dir}")

        # Update global variables for saved data paths
        MovieApp.SAVE_DIR = saved_data_dir

        # Ensure API keys are loaded
        omdb_key, openai_key = ensure_api_keys(encryption_key)

        # Initialize and show the main application window
        window = MovieApp()
        window.center_window()
        window.show()

        # Process UI events immediately to prevent a blank screen
        QApplication.processEvents()

        # Load saved data
        window._load_and_populate_data_in_thread()

        # Enter the application's event loop
        sys.exit(app.exec_())
    except Exception as e:
        print(f"[ERROR] Application failed to start: {e}")
        QMessageBox.critical(None, "Application Error", f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()