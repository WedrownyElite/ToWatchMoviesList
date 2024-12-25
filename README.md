Movies List Application | Krzysztof Sliwinski / WedrownyElite

This program is not perfect, and has a lot of querks, but it works! Lol

Everyone is free to use it and add onto it if need be. 

![MoviesList-BaseImage](https://github.com/user-attachments/assets/9a0e64c9-faa1-4a29-ac89-3d915a7289c8)

This repo contains a script.py which can be edited for your own preferences, it also contains a fodler "EXEs" which contains a script.exe which was built by pyinstaller.
This script was built to save and encrypt API keys for GPT and OMDb. The OMDb key is required but the GPT key is optional

You can search for any title that OMDb or GPT is aware of and add it to the list, most titles will have posters that can be fetched by OMDb's API.
I've attempted the thread the suggestion population but as you'll probably notice when using, you have to wait for all the suggestions to populate before you select one or else the program will freeze for a few seconds (it won't break)

![MoviesList-Suggestions](https://github.com/user-attachments/assets/3e5722e8-4b2e-4bbe-986c-424c7c1a2c5c)

You can edit the maximum suggestions that will populate in the update_suggestions method, and within the GPT query in query_llm_for_movies method. (I will probably make a global variable for maximum suggestions for easier modification)
Any titles added to your list is saved and will load on program startup. Title posters are also downloaded and saved within the defined saved data directory for quicker loading.

You can also revalidate either OMDb or GPT API key if you ever need to by using the button in the top right of the program. It will automatically encrypt the new API key and save it within the config directory.
You can right click entries within a table to delete the entry or easily move the entry to the other table.

![MoviesList-Revalidate API Keys](https://github.com/user-attachments/assets/b88d8455-864b-4b28-b687-84aa2d0a6a3b)

You can also filter entered titles by title name, top 3 cast members, and IMDb rating

![MoviesList-Filtering](https://github.com/user-attachments/assets/e8959b44-ce3f-4cff-a990-5498ccd23331)

This is my first actual python project that revolves around API querying.

12/22/2024 - Use of the GPT API is now optional but the suggestion box will only be populated with 1 suggestion if you spell the movie you're looking for correctly. It currently does not account for release years so if you're looking for a title that has multiple releases, you will not be able to add it yet.
OMDb API key is still required (OMDb API Keys are free, no query limits)

This program is not perfect, and has a lot of quirks, but it works! Everyone is free to use it and add onto it if needed.

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