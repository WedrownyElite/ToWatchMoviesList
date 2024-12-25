Movies List Application | Krzysztof Sliwinski / WedrownyElite

This program is not perfect, and has a lot of querks, but it works! Lol

Everyone is free to use it and add onto it if need be. 

This repository contains:

1. script.py – The Python script for the program.
2. EXEs – A folder containing script.exe, the pre-built executable for users who don’t want to run the Python script.
3. Resources - A folder containg png's and any other resources used
The program was built to save and encrypt API keys for GPT and OMDb. The program will not run without an OMDb API key, but will run without a GPT API key.

Key Features:
 - Search for any movie title using OMDb and GPT APIs.
 - Add movies to "To Watch" or "Watched" lists.
 - Download and display posters for movies using OMDb API.
 - Save and load lists between sessions.
 - Revalidate API keys using a built-in dialog.

How to Use the Program:

Option 1: Use the Pre-Built Executable
1. Download the .exe file: Navigate to the EXEs folder in this repository and download script.exe.

2. Run the .exe:
 - Double-click the script.exe file.
 - Follow the prompts to provide your OMDb and (optional) GPT API keys.

3. Start Adding Movies:
 - Use the search box to find movies.
 - Add them to your "To Watch" or "Watched" list.
 - Downloaded posters will be stored locally for quick access.

Option 2: Run the Python Script
1. Install Python 3.9+ Make sure Python is installed on your system. You can download it from python.org.

2. Clone or Download the Repository:
git clone https://github.com/your-repo/movies-list.git
cd movies-list

3. Install Dependencies: Install the required Python packages using the following command:
pip install -r requirements.txt

4. Run the Script: Run the program using the command:
python script.py

5. Provide API Keys:
 - Enter your OMDb API key when prompted (mandatory).
 - Optionally provide your GPT API key for enhanced suggestions.

Notes:

User-Defined Save Directory: Movie data and posters are saved in a directory that the user specifies during the program setup. This allows flexibility in organizing your files.
Config Directory (~/.MoviesList): The encrypted API keys and configuration file are stored in this directory. These files are used to securely save and manage your API keys.
OMDb API Key: You can get a free API key from OMDb. This is required for the program to run.
GPT API Key: GPT functionality is optional. Without it, you will only see an exact match suggestion from OMDb. The GPT API key is not free
Feel free to reach out or open an issue if you encounter any problems!
wedrownyelite@gmail.com

![MoviesList-Base](https://github.com/user-attachments/assets/d77622c6-0bb2-4e2d-9a7f-eaedf6aafa08)
MoviesList
![MoviesList-RightClick](https://github.com/user-attachments/assets/c571b346-604b-48cb-a209-74bacb64a753)
Right Click options
![MoviesList-Settings](https://github.com/user-attachments/assets/13ecd970-6667-43af-8606-168acdce2ad3)
Settings
![MoviesList-Suggestions](https://github.com/user-attachments/assets/21191bf6-4900-4544-8151-61b5ca3755c7)
Suggestions (With GPT API key)
