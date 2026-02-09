# **Setup Guide for RaceGPT Chatbot**

This guide will walk you through setting up the environment, installing dependencies, and running the RaceGPT chatbot.

## **1. Setup the Zip Folder**

1. **Download or Clone the Project:**
   - Unzip the folder containing the RaceGPT chatbot files (like `multiapp.py`, transcript data, etc.).

2. **Folder Structure:**
   Ensure your folder structure looks like this:
   ```
   racegpt_chatbot/
   ├── .env
   ├── multiapp.py
   ├── transcripts/
   │   ├── Race1/
   │   │   ├── driver1.txt
   │   │   ├── driver2.txt
   │   ├── Race2/
   │   │   ├── driver1.txt
   ├── requirements.txt
   ```

   - The **transcripts** folder contains the race transcript text files categorized by races.
   - The **`.env`** file stores your environment variables (API keys, etc.).

## **2. Install Python and Virtual Environment (Optional)**

Make sure you have Python 3.7+ installed on your system. Then set up a virtual environment to isolate your dependencies.

### **Create Virtual Environment (Optional but Recommended):**
```bash
# Navigate to your project folder
cd racegpt_chatbot

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

## **3. Install Dependencies**

### **Install the Required Packages:**

Inside the `racegpt_chatbot` folder, install the required Python libraries using the `requirements.txt` file:

```bash
# Make sure you are inside the racegpt_chatbot folder
cd racegpt_chatbot

# Install the necessary Python packages
pip install -r requirements.txt
```

The `requirements.txt` file contains dependencies like `openai`, `chainlit`, `tqdm`, `sentence_transformers`, etc.


## **4. Configure Environment Variables**

You need to set up an `.env` file in the project folder containing your Azure OpenAI credentials.

### **Example `.env` File:**
```
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_BASE=https://your_openai_resource_name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2023-03-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=RaceGPT4oMini
```

- **AZURE_OPENAI_API_KEY:** Your Azure OpenAI API Key.
- **AZURE_OPENAI_API_BASE:** The base URL for your Azure OpenAI instance.
- **AZURE_OPENAI_API_VERSION:** The API version.
- **AZURE_OPENAI_DEPLOYMENT_NAME:** The name of your Azure OpenAI deployment.

## **5. Run the Chatbot**

Once everything is set up, you can run the chatbot using Chainlit.

```bash
# Make sure you are inside the racegpt_chatbot folder
cd racegpt_chatbot

# Run the Chainlit app
chainlit run multiapp.py
```

This will start the Chainlit server and provide you with a local URL (usually `http://localhost:8000`), where you can interact with the RaceGPT chatbot.

---

### **Troubleshooting:**
- If the app fails to start, ensure all dependencies are correctly installed.
- If any API calls fail, double-check that your `.env` file is properly configured with valid Azure OpenAI credentials.

---
