<h1 align="center">🎥 Movies List Application | Krzysztof Sliwinski / WedrownyElite 🎥</h1>

<p align="center" style="text-align: center; font-size: 1.2em;">
 This program is not perfect and has a lot of quirks, but it works! 😂<br>
 Everyone is free to use it and add onto it if need be.
</p>

---

<h2 align="center">📂 This Repository Contains:</h2>

<table align="center" style="text-align: center;">
  <tr>
    <td align="right">1.</td>
    <td><strong>script.py</strong> – The Python script for the program.</td>
  </tr>
  <tr>
    <td align="right">2.</td>
    <td><strong>EXEs</strong> – A folder containing <code>script.exe</code>, the pre-built executable for users who don’t want to run the Python script.</td>
  </tr>
  <tr>
    <td align="right">3.</td>
    <td><strong>Resources</strong> – A folder containing PNGs and any other resources used.</td>
  </tr>
</table>

<p align="center" style="text-align: center;">
The program was built to save and encrypt API keys for GPT and OMDb.<br>
The program <strong>will not run without an OMDb API key</strong>, but will run without a GPT API key.
</p>

---

<h2 align="center">✨ Key Features:</h2>

<table align="center" style="text-align: center;">
  <tr>
    <td align="right">✔️</td>
    <td>Search for any movie title using OMDb and GPT APIs.</td>
  </tr>
  <tr>
    <td align="right">✔️</td>
    <td>Add movies to "To Watch" or "Watched" lists.</td>
  </tr>
  <tr>
    <td align="right">✔️</td>
    <td>Download and display posters for movies using OMDb API.</td>
  </tr>
  <tr>
    <td align="right">✔️</td>
    <td>Save and load lists between sessions.</td>
  </tr>
  <tr>
    <td align="right">✔️</td>
    <td>Revalidate API keys using a built-in dialog.</td>
  </tr>
   <tr>
    <td align="right">✔️</td>
    <td>Easily move titles to a different lists</td>
  </tr>
</table>

---

<h2 align="center">🚀 How to Use the Program:</h2>

### Option 1: Use the Pre-Built Executable
1. **Download the .exe file**: Navigate to the `EXEs` folder in this repository and download <code>script.exe</code>.
2. **Run the .exe**:
   - Double-click the `script.exe` file.
   - Follow the prompts to provide your OMDb and (optional) GPT API keys.
3. **Start Adding Movies**:
   - Use the search box to find movies.
   - Add them to your "To Watch" or "Watched" list.
   - Downloaded posters will be stored locally for quick access.

### Option 2: Run the Python Script
1. **Install Python 3.9+**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/).
2. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/your-repo/movies-list.git
   cd movies-list
3. **Install Dependecies**:
   ```bash
   pip install -r requirements.txt
4. **Run the script**:
   ```bash
   python script.py
5. **Provide API Keys**:
   - Enter your OMDb API key when prompted (mandatory).
   - Optionally provide your GPT API key for enhanced suggestions.
  
<h2 align="center">💾 Notes:</h2>

<p align="center">
<ul>
  <li><b>User-Defined Save Directory:</b> Movie data and posters are saved in a directory that the user specifies during the program setup. This allows flexibility in organizing your files.</li>
  <li><b>Config Directory (~/.MoviesList):</b> The encrypted API keys and configuration file are stored in this directory. These files are used to securely save and manage your API keys.</li>
  <li><b>OMDb API Key:</b> You can get a free API key from OMDb. This is required for the program to run.</li>
  <li><b>GPT API Key:</b> GPT functionality is optional. Without it, you will only see an exact match suggestion from OMDb. The GPT API key is not free.</li>
  <li><b>Filter search bar for titles and top 3 cast members</li>
  <li><b>Interactable table headers for quick sorting (Titles: A-Z, Z-A. IMDb Rating: Highest-Lowest, Lowest-Highest)</li>
</ul>
Feel free to reach out or open an issue if you encounter any problems!<br>
📧 <b>Email:</b> <a href="mailto:wedrownyelite@gmail.com">wedrownyelite@gmail.com</a>
</p>


<h2 align="center">📸 Screenshots:</h2>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1d3b9bf-d1c6-418f-a72f-ce07f967ab09" alt="MoviesList-Base" style="width:50%;">
  <br>
  <em>Figure 1: Base view of the MoviesList application</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08a1e43f-57b5-4646-aaae-f83b2b1ac1ef" alt="MoviesList-RightClick" style="width:50%;">
  <br>
  <em>Figure 2: Right-click context menu in the MoviesList application</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7d6b333-ea9b-43b3-a20c-1623fbf39d13" alt="MoviesList-Filtering" style="width:50%;">
  <br>
  <em>Figure 3: Filtering movies in the MoviesList application</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/29a7a1c4-845c-4f67-8e60-901bfa45d2fc" alt="MoviesList-Settings" style="width:50%;">
  <br>
  <em>Figure 4: Settings menu in the MoviesList application</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e1a41e7-cbf9-48c8-9c08-f90712647e16" alt="MoviesList-RevalidateAPIKeys" style="width:50%;">
  <br>
  <em>Figure 5: Revalidating API keys in the MoviesList application</em>
</p>
