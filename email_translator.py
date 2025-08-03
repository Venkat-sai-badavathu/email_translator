import os
import pickle
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# --- Configuration ---
# If modifying these scopes, delete the file token.pickle.
# 'https://www.googleapis.com/auth/gmail.readonly' allows reading emails.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
TOKEN_PICKLE_FILE = 'token.pickle'
CREDENTIALS_JSON_FILE = 'credentials.json'

# Hugging Face model for English to Telugu translation
# Changed model to a Facebook NLLB model for broader language support
HUGGINGFACE_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# --- Gmail API Functions ---

def gmail_authenticate():
    """
    Authenticates with Gmail API using OAuth 2.0.
    It will try to load credentials from token.pickle. If not found or expired,
    it will prompt the user to authorize via a web browser and save the token.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_PICKLE_FILE):
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_JSON_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PICKLE_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def get_email_body(msg):
    """
    Extracts the plain text body from a Gmail message.
    Handles multipart messages and decodes base64 content.
    """
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body']['data']
                decoded_data = base64.urlsafe_b64decode(data).decode('utf-8')
                return decoded_data
            elif part['mimeType'] == 'text/html':
                data = part['body']['data']
                decoded_data = base64.urlsafe_b64decode(data).decode('utf-8')
                # Use BeautifulSoup to strip HTML tags
                soup = BeautifulSoup(decoded_data, 'html.parser')
                return soup.get_text()
            elif part['mimeType'].startswith('multipart/'):
                # Recursively call for multipart parts
                return get_email_body({'payload': part})
    elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
        data = msg['payload']['body']['data']
        decoded_data = base64.urlsafe_b64decode(data).decode('utf-8')
        return decoded_data
    return ""

def fetch_latest_emails(service, num_emails=1, query='is:unread'):
    """
    Fetches the latest emails from the user's inbox.
    By default, fetches unread emails.
    """
    try:
        # Request a list of all messages
        results = service.users().messages().list(userId='me', q=query, maxResults=num_emails).execute()
        messages = results.get('messages', [])

        if not messages:
            print("No emails found.")
            return []

        email_details = []
        for msg_id in messages:
            msg = service.users().messages().get(userId='me', id=msg_id['id'], format='full').execute()
            
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown Sender')
            
            body = get_email_body(msg)
            
            email_details.append({
                'id': msg_id['id'],
                'subject': subject,
                'sender': sender,
                'body': body
            })
        return email_details

    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

# --- Hugging Face Translation Functions ---

def load_translation_model(model_name):
    """Loads the Hugging Face translation model and tokenizer."""
    print(f"Loading Hugging Face model: {model_name}...")
    try:
        # Using AutoModel and AutoTokenizer for broader compatibility with different model architectures
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Check for GPU and move model if available
        if torch.cuda.is_available():
            print("Using GPU for translation.")
            model.to('cuda')
        else:
            print("Using CPU for translation.")
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Hugging Face model: {e}")
        print("Please ensure you have 'transformers' and 'torch' installed.")
        print("You might also need to download the model files if this is the first run.")
        return None, None

def translate_text(text, model, tokenizer, source_lang="eng_Latn", target_lang="tel_Telu"):
    """Translates text using the loaded Hugging Face model."""
    if not text.strip():
        return "No text to translate."
    if model is None or tokenizer is None:
        return "Translation model not loaded."

    try:
        # Set the source language for NLLB models
        tokenizer.src_lang = source_lang
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Move inputs to GPU if model is on GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Generate translation, forcing the beginning of sentence token for the target language
        # Use tokenizer.convert_tokens_to_ids to get the ID for the target language token
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
        
        # Decode the translated tokens
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation failed."

# --- Main Logic ---

def main():
    print("Starting AI Email Assistant...")
    
    # 1. Authenticate with Gmail
    gmail_service = gmail_authenticate()
    if not gmail_service:
        print("Gmail authentication failed. Exiting.")
        return

    # 2. Load Hugging Face translation model
    translation_model, translation_tokenizer = load_translation_model(HUGGINGFACE_MODEL_NAME)
    if translation_model is None:
        print("Hugging Face model loading failed. Exiting.")
        return

    print("\nFetching the latest unread email...")
    # Fetch only the latest unread email
    emails = fetch_latest_emails(gmail_service, num_emails=1, query='is:unread')

    if emails:
        # Process only the first (and only) email in the list
        email = emails[0]
        print(f"\n--- Latest Email ---")
        print(f"From: {email['sender']}")
        print(f"Subject: {email['subject']}")
        print(f"Original Body:\n{email['body'][:500]}...") # Print first 500 chars

        # Translate the email body
        # Pass explicit source and target language codes for NLLB model
        translated_body = translate_text(email['body'], translation_model, translation_tokenizer, 
                                         source_lang="eng_Latn", target_lang="tel_Telu")
        print(f"\nTranslated Body (Telugu):\n{translated_body}")
        print("-" * 30)
    else:
        print("No unread emails to process.")

if __name__ == '__main__':
    # Install necessary packages if they are not already installed
    try:
        import google_auth_oauthlib
        import googleapiclient
        import transformers
        import torch
        import bs4 # BeautifulSoup
    except ImportError:
        print("Installing required Python packages...")
        os.system("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib transformers torch beautifulsoup4")
        print("Packages installed. Please run the script again.")
        exit()
    
    main()
