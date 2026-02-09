# ======================================
# Import Necessary Libraries
# ======================================
import os
import ast
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from dotenv import load_dotenv
import chainlit as cl
import joblib
import hashlib
import json
import warnings
import atexit

# ======================================
# Suppress FutureWarnings
# ======================================
# Set environment variables to suppress specific warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# ======================================
# Load Environment Variables
# ======================================
load_dotenv()  # Load variables from .env file into environment

# ======================================
# Define Paths and Directories
# ======================================
# Base directory path (assuming the script and transcripts are in the same folder)
base_path = os.path.dirname(os.path.abspath(__file__))

# Directory containing all race transcripts
transcripts_dir = os.path.join(base_path, "transcripts")

# Directory for output results (optional)
output_folder = os.path.join(base_path, "openai_output")

# Directory for storing cached embeddings
cache_folder = os.path.join(base_path, "embedding_cache")

# Create the output and cache directories if they don't exist
for folder in [output_folder, cache_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")

# ======================================
# Configure Logging
# ======================================
logging.basicConfig(
    filename='interaction_logs.txt',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# ======================================
# Load the Embedding Model
# ======================================
# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-mpnet-base-v2')

# ======================================
# Define Helper Functions
# ======================================

def preprocess_text(text):
    """
    Preprocesses the text by converting to lowercase, removing special characters, and extra spaces.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def extract_driver_name(file_name):
    """
    Extracts the driver name from the file name.
    Assumes the file name format is 'driver_name.txt' or similar.
    """
    return os.path.splitext(file_name)[0]

def generate_file_hash(file_path):
    """
    Generates a SHA256 hash for a given file to detect changes.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in 64KB chunks
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def load_cached_embeddings(cache_file):
    """
    Loads cached embeddings and their corresponding file hashes from a JSON file.
    Handles empty or invalid JSON files gracefully.

    Args:
        cache_file (str): Path to the cache JSON file.

    Returns:
        dict: Cached embeddings data.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            # Convert embedding lists back to numpy arrays
            for file_hash, data in cache_data.items():
                for conv in data['conversations']:
                    conv['embedding'] = np.array(conv['embedding'])
            return cache_data
        except json.JSONDecodeError:
            logging.warning(f"Cache file {cache_file} is invalid or empty. Starting with an empty cache.")
            return {}
    return {}

def save_cached_embeddings(cache_file, cache_data):
    """
    Saves cached embeddings and their corresponding file hashes to a JSON file.
    Converts numpy arrays to lists for JSON serialization.

    Args:
        cache_file (str): Path to the cache JSON file.
        cache_data (dict): Cached embeddings data.
    """
    # Convert embeddings to lists
    serializable_cache = {}
    for file_hash, data in cache_data.items():
        serializable_cache[file_hash] = {
            'conversations': [
                {**conv, 'embedding': conv['embedding'].tolist()} for conv in data['conversations']
            ]
        }

    with open(cache_file, 'w') as f:
        json.dump(serializable_cache, f)
    print("Saved updated embeddings cache.")

def process_transcript(transcript_file_path, race_name, driver_name, cache_data):
    """
    Processes the transcript file and extracts conversations.
    Utilizes caching to avoid redundant embedding computations.

    Args:
        transcript_file_path (str): Path to the transcript file.
        race_name (str): Name of the race.
        driver_name (str): Name of the driver.
        cache_data (dict): Cached embeddings data.

    Returns:
        list: A list of conversation dictionaries with metadata.
    """
    conversations = []
    full_path = os.path.join(base_path, transcript_file_path)

    # Generate a unique hash for the file to detect changes
    file_hash = generate_file_hash(full_path)

    # Open the transcript file
    with open(full_path, 'r') as file:
        content = file.read().strip()

        try:
            # Parse the content as a list of lists
            transcript_data = ast.literal_eval(content)

            if isinstance(transcript_data, list):
                for entry in transcript_data:
                    if isinstance(entry, list) and len(entry) == 4:
                        start_time = entry[0]
                        end_time = entry[1]
                        text = entry[2].strip()
                        speaker = entry[3]
                        # Skip empty lines
                        if text:
                            conversations.append({
                                "race_name": race_name,
                                "driver_name": driver_name,
                                "file_name": os.path.basename(transcript_file_path),
                                "start_time": start_time,
                                "end_time": end_time,
                                "speaker": speaker,
                                "text": text
                            })
        except (ValueError, SyntaxError):
            logging.error(f"Error parsing the transcript file: {transcript_file_path}")
            print(f"Error parsing the transcript file: {transcript_file_path}")

    # Check if embeddings for this file are already cached and up-to-date
    if file_hash in cache_data:
        conversations_with_embeddings = cache_data[file_hash]['conversations']
        print(f"Loaded cached embeddings for {driver_name} in {race_name}")
    else:
        # Generate embeddings for each conversation
        for conv in conversations:
            preprocessed_text = preprocess_text(conv['text'])
            conv['embedding'] = model.encode(preprocessed_text, convert_to_numpy=True)
        conversations_with_embeddings = conversations

        # Convert embeddings to lists for JSON serialization
        for conv in conversations_with_embeddings:
            conv['embedding'] = conv['embedding'].tolist()

        # Update cache
        cache_data[file_hash] = {
            'conversations': conversations_with_embeddings
        }
        print(f"Generated and cached embeddings for {driver_name} in {race_name}")

    return conversations_with_embeddings

def generate_sample_questions(user_query, num_samples=5):
    """
    Generates sample questions based on the user's vague query using a language model.
    This approach provides more contextually relevant questions compared to heuristic methods.

    Args:
        user_query (str): The original user query.
        num_samples (int): Number of sample questions to generate.

    Returns:
        list: A list of sample questions.
    """
    prompt = f"Generate {num_samples} diverse and contextually relevant questions based on the following query: '{user_query}'"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose a different engine if preferred
            prompt=prompt,
            max_tokens=100,
            n=num_samples,
            stop=None,
            temperature=0.7,
        )
        samples = [choice['text'].strip() for choice in response['choices']]
    except Exception as e:
        logging.error(f"Error generating sample questions: {e}")
        # Fallback to heuristic-based samples
        samples = []
        question_words = ["what", "how", "why", "when", "where", "tell me about"]
        for i in range(num_samples):
            prefix = np.random.choice(question_words)
            samples.append(f"{prefix} {user_query}")
    return samples

def build_prompt(context: str, question: str) -> str:
    """
    Builds the prompt for the AI model, incorporating context and the user's question.

    Args:
        context (str): Relevant text chunks from the transcript.
        question (str): The user's query.

    Returns:
        str: The constructed prompt.
    """
    prompt = (
        "You are an AI assistant knowledgeable about the content of the provided race transcripts. "
        "Based on the following excerpts, answer the user's question accurately. "
        "If the answer is not present in the excerpts, indicate that the information is not available.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:\n"
    )
    return prompt

# ======================================
# Azure OpenAI Configuration
# ======================================
# Load Azure OpenAI configurations from environment variables
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")  # Ensure this is set in your .env file
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")  # Ensure this is set in your .env file
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview")  # API version (verify with your Azure OpenAI resource)

# Deployment name for the RaceGPT4oMini model
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "RaceGPT4oMini")  # Ensure this is set in your .env file

# ======================================
# Load or Initialize Cached Embeddings
# ======================================
cache_file = os.path.join(cache_folder, 'embeddings_cache.json')
cached_embeddings = load_cached_embeddings(cache_file)

# ======================================
# Process Transcripts and Collect Conversations
# ======================================
all_conversations = []

# Iterate through each race folder in transcripts directory
for race_folder in os.listdir(transcripts_dir):
    race_path = os.path.join(transcripts_dir, race_folder)
    if os.path.isdir(race_path):
        # Iterate through each driver file in the race folder
        for driver_file in os.listdir(race_path):
            if driver_file.endswith('.txt'):
                driver_path = os.path.join(race_path, driver_file)
                driver_name = extract_driver_name(driver_file)
                print(f"Processing Transcript: {driver_name}")
                conversations = process_transcript(driver_path, race_folder, driver_name, cached_embeddings)
                all_conversations.extend(conversations)

# ======================================
# Define Chainlit Event Handlers
# ======================================

@cl.on_chat_start
async def on_chat_start():
    """
    Event handler that runs when a new chat starts.
    Sends a welcome message to the user.
    """
    await cl.Message(content="🤖 Hello! I'm RaceGPT, your assistant for race information. Ask me anything about the race based on the transcripts.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handles incoming user messages (queries) and generates responses based on transcript content.

    Args:
        message (cl.Message): The user's message.
    """
    global all_conversations, model

    user_query = message.content
    if not user_query.strip():
        await cl.Message(content="❌ Please enter a valid question.").send()
        return

    # Generate sample questions to enhance retrieval
    sample_questions = generate_sample_questions(user_query)
    all_queries = [preprocess_text(q) for q in [user_query] + sample_questions]

    # Generate embeddings for all queries
    try:
        query_embeddings = model.encode(all_queries, convert_to_numpy=True)
    except Exception as e:
        logging.error(f"Error generating query embeddings: {e}")
        await cl.Message(content=f"❌ Error processing your query: {e}").send()
        return

    # Compute similarities and aggregate scores
    scores = np.zeros(len(all_conversations))
    for qe in query_embeddings:
        scores += cosine_similarity([qe], [conv['embedding'] for conv in all_conversations]).flatten()

    # Normalize scores
    scores = scores / len(query_embeddings)

    # Log the scores for debugging
    logging.debug(f"User Query: {user_query}")
    for i in range(len(scores)):
        logging.debug(f"Score: {scores[i]:.4f}, Race: {all_conversations[i]['race_name']}, Driver: {all_conversations[i]['driver_name']}, File: {all_conversations[i]['file_name']}, Conversation: {all_conversations[i]['text']}")

    # Get top 10 most relevant conversations
    top_indices = np.argsort(scores)[-10:][::-1]
    relevant_conversations = [all_conversations[i] for i in top_indices if scores[i] > 0.3]

    # Define keywords for additional matching (optional enhancement)
    keywords = ['tire', 'caution', 'engine', 'race', 'issue', 'problem']

    # Keyword matching to retrieve additional relevant conversations
    keyword_matches = []
    for conv in all_conversations:
        for keyword in keywords:
            if keyword in preprocess_text(conv['text']):
                keyword_matches.append(conv)
                break  # Stop after the first keyword match

    # Merge and remove duplicates using a set based on unique text
    combined_conversations_dict = {conv['text']: conv for conv in relevant_conversations + keyword_matches}
    combined_conversations = list(combined_conversations_dict.values())

    if combined_conversations:
        # Prepare the context for the model
        context = ""
        citations = []
        for conv in combined_conversations:
            # Format: Driver, Transcript, Race, Time, Speaker, Conversation
            context += f"Driver: {conv['driver_name']}, Transcript: {conv['file_name']}, Race: {conv['race_name']}, Time: {conv['start_time']}-{conv['end_time']}, Speaker: {conv['speaker']}\n"
            context += f"Conversation: {conv['text']}\n\n"
            citations.append({
                "race_name": conv['race_name'],
                "driver_name": conv['driver_name'],
                "file_name": conv['file_name'],
                "start_time": conv['start_time'],
                "end_time": conv['end_time'],
                "speaker": conv['speaker'],
                "text": conv['text']
            })

        # Generate refined search queries using LLM to maximize information retrieval
        refined_queries = generate_sample_questions(user_query, num_samples=3)  # Generate fewer samples for brevity
        refined_queries = [preprocess_text(q) for q in refined_queries]

        # Generate embeddings for refined queries
        try:
            refined_query_embeddings = model.encode(refined_queries, convert_to_numpy=True)
        except Exception as e:
            logging.error(f"Error generating refined query embeddings: {e}")
            await cl.Message(content=f"❌ Error processing your refined queries: {e}").send()
            return

        # Compute similarities for refined queries
        refined_scores = np.zeros(len(all_conversations))
        for qe in refined_query_embeddings:
            refined_scores += cosine_similarity([qe], [conv['embedding'] for conv in all_conversations]).flatten()

        # Normalize refined scores
        refined_scores = refined_scores / len(refined_query_embeddings)

        # Combine original and refined scores
        combined_scores = scores + refined_scores

        # Re-sort and select top conversations based on combined scores
        top_indices_combined = np.argsort(combined_scores)[-10:][::-1]
        final_relevant_conversations = [all_conversations[i] for i in top_indices_combined if combined_scores[i] > 0.3]

        # Remove duplicates
        final_conversations_dict = {conv['text']: conv for conv in final_relevant_conversations}
        final_conversations = list(final_conversations_dict.values())

        # Re-prepare context and citations based on final conversations
        context = ""
        citations = []
        for conv in final_conversations:
            context += f"Driver: {conv['driver_name']}, Transcript: {conv['file_name']}, Race: {conv['race_name']}, Time: {conv['start_time']}-{conv['end_time']}, Speaker: {conv['speaker']}\n"
            context += f"Conversation: {conv['text']}\n\n"
            citations.append({
                "race_name": conv['race_name'],
                "driver_name": conv['driver_name'],
                "file_name": conv['file_name'],
                "start_time": conv['start_time'],
                "end_time": conv['end_time'],
                "speaker": conv['speaker'],
                "text": conv['text']
            })

        # Build the prompt without embedding-specific details
        prompt = build_prompt(context, user_query)

        # Call the GPT model to generate the answer
        try:
            response = openai.ChatCompletion.create(
                deployment_id=deployment_name,  # Use deployment_id for Azure OpenAI
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.7
            )
            answer = response['choices'][0]['message']['content'].strip()

            # Structure the response with selective citations
            structured_citations = ""
            for citation in citations:
                structured_citations += (
                    f"- **Race:** {citation['race_name']}, **Driver:** {citation['driver_name']}, "
                    f"**File:** {citation['file_name']}, **Time:** {citation['start_time']}-{citation['end_time']}, "
                    f"**Speaker:** {citation['speaker']}\n"
                    f"  **Conversation:** {citation['text']}\n"
                )

            structured_answer = f"{answer}\n\n**Citations:**\n{structured_citations}"

            await cl.Message(content=structured_answer).send()
            logging.debug(f"Assistant Answer: {structured_answer}")
        except openai.error.OpenAIError as e:
            error_message = f"❌ Error during OpenAI API call: {e}"
            await cl.Message(content=error_message).send()
            logging.error(error_message)
    else:
        await cl.Message(content="🤷‍♂️ I'm sorry, I couldn't find any relevant information in the transcripts.").send()

# ======================================
# Save Updated Cached Embeddings
# ======================================
def save_embeddings_cache():
    """
    Saves the cached embeddings to the cache file.
    """
    # Convert embeddings to lists
    serializable_cache = {}
    for file_hash, data in cached_embeddings.items():
        serializable_cache[file_hash] = {
            'conversations': [
                {**conv, 'embedding': conv['embedding'].tolist()} for conv in data['conversations']
            ]
        }

    with open(cache_file, 'w') as f:
        json.dump(serializable_cache, f)
    print("Saved updated embeddings cache.")

# Register a cleanup handler to save cache on exit
atexit.register(save_embeddings_cache)

# ======================================
# Run the Chainlit app
# ======================================
if __name__ == "__main__":
    # Start the Chainlit server
    cl.run()
