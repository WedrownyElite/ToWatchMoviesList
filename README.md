+ Krzysztof Sliwinski | WedrownyElite +

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
