import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import asyncio
from agent import root_agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import pandas as pd
import csv
import time
import re
from threading import Thread

# --- Environment & API Key ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "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")

# --- CSV Setup for leads ---
CSV_FILE = "leads.csv"
CSV_COLUMNS = ["lead_id", "name", "age", "country", "interest", "status"]
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

# --- Flask app ---
app = Flask(__name__)

# --- ADK Session & Runner Setup ---
APP_NAME = "sales_agent_app"
USER_ID = "user_1"
session_service = InMemorySessionService()
# We'll create sessions dynamically per lead

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- Helper: Sync wrapper around async runner ---
async def _run_agent_async(conversation_id: str, text: str) -> str:
    content = types.Content(role='user', parts=[types.Part(text=text)])
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=conversation_id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_text = f"Agent escalated: {event.error_message or 'No message'}"
            break
    return final_text

def call_agent_sync(conversation_id: str, text: str) -> str:
    # Create session if not exists
    try:
        session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=conversation_id)
    except KeyError:
        session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=conversation_id)
    return asyncio.run(_run_agent_async(conversation_id, text))

# --- Memory to store conversation details ---
conversation_details = {}  # Will store lead details by lead_id

# --- Helper to save lead data ---
def save_to_csv(lead_id, name, age, country, interest, status):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({
            "lead_id": lead_id,
            "name": name,
            "age": age,
            "country": country,
            "interest": interest,
            "status": status
        })

# --- Helper to extract user details from a structured response ---
def extract_user_details(message):
    details = {
        'name': None,
        'age': None,
        'country': None,
        'interest': None
    }
    
    # Extract details using regex patterns
    name_match = re.search(r'name:?\s*([^\n]+)', message, re.IGNORECASE)
    age_match = re.search(r'age:?\s*([^\n]+)', message, re.IGNORECASE)
    country_match = re.search(r'country:?\s*([^\n]+)', message, re.IGNORECASE)
    interest_match = re.search(r'(?:product )?interest:?\s*([^\n]+)', message, re.IGNORECASE)
    
    if name_match:
        details['name'] = name_match.group(1).strip()
    if age_match:
        details['age'] = age_match.group(1).strip()
    if country_match:
        details['country'] = country_match.group(1).strip()
    if interest_match:
        details['interest'] = interest_match.group(1).strip()
    
    return details

# --- Check if all required details are present ---
def are_details_complete(details):
    return all(details.values())

# --- Format details for confirmation ---
def format_details_for_confirmation(details):
    return (
        f"Great! Let's review the details you've provided:\n\n"
        f"Your name: {details['name']}\n"
        f"Age: {details['age']}\n"
        f"Country: {details['country']}\n"
        f"Product interest: {details['interest']}\n\n"
        f"Please confirm if the above details are correct by typing 'confirm'."
    )

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    lead_id = request.args.get('lead_id')
    if not lead_id:
        return "Missing lead_id parameter", 400
    
    message = request.json.get('message')
    if not message:
        return "Missing message in request body", 400
    
    # Check if this is a confirmation message
    if message.lower().strip() == 'confirm':
        if lead_id in conversation_details and 'last_details_message' in conversation_details[lead_id]:
            previous_response = conversation_details[lead_id]['last_details_message']
            details = extract_user_details(previous_response)
            if are_details_complete(details):
                save_to_csv(
                    lead_id, 
                    details['name'], 
                    details['age'], 
                    details['country'], 
                    details['interest'], 
                    'confirmed'
                )
                agent_reply = "Thank you for confirming your details! Your information has been saved successfully. How else can I assist you today?"
                return {"response": agent_reply}
    
    # Send message to agent for normal conversation
    agent_reply = call_agent_sync(lead_id, message)
    
    # Check if the agent response contains user details in a structured format
    details = extract_user_details(agent_reply)
    if are_details_complete(details):
        # Store the formatted details for later confirmation
        if lead_id not in conversation_details:
            conversation_details[lead_id] = {}
        conversation_details[lead_id]['last_details_message'] = agent_reply
        
        # Replace agent response with formatted confirmation prompt
        agent_reply = format_details_for_confirmation(details)
    
    return {"response": agent_reply}

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    lead_id = request.form.get('lead_id')
    name = request.form.get('name')
    if lead_id and name:
        # Initialize session
        session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=lead_id
        )
        
        # Initialize conversation details
        conversation_details[lead_id] = {
            'last_details_message': ''
        }
        
        # Greet lead via agent
        prompt = f"Hello {name}, thank you for filling out the form. I'd like to gather some information from you including your name, age, country, and product interest. Is that okay?"
        response = call_agent_sync(lead_id, prompt)
        # Optionally store initial status
        save_to_csv(lead_id, name, '', '', '', 'started')
        return redirect(url_for('conversation', lead_id=lead_id))
    return render_template('index.html', message="Please provide valid details.")

@app.route('/conversation/<lead_id>', methods=['GET', 'POST'])
def conversation(lead_id):
    if request.method == 'POST':
        message = request.form.get('message')
        
        # Check if this is a confirmation message
        if message.lower().strip() == 'confirm':
            if lead_id in conversation_details and 'last_details_message' in conversation_details[lead_id]:
                previous_response = conversation_details[lead_id]['last_details_message']
                details = extract_user_details(previous_response)
                if are_details_complete(details):
                    save_to_csv(
                        lead_id, 
                        details['name'], 
                        details['age'], 
                        details['country'], 
                        details['interest'], 
                        'confirmed'
                    )
                    agent_reply = "Thank you for confirming your details! Your information has been saved successfully. How else can I assist you today?"
                    return render_template('conversation.html', lead_id=lead_id, response=agent_reply)
        
        # Send lead message to agent
        agent_reply = call_agent_sync(lead_id, message)
        
        # Check if the agent response contains user details in a structured format
        details = extract_user_details(agent_reply)
        if are_details_complete(details):
            # Store the formatted details for later confirmation
            if lead_id not in conversation_details:
                conversation_details[lead_id] = {}
            conversation_details[lead_id]['last_details_message'] = agent_reply
            
            # Replace agent response with formatted confirmation prompt
            agent_reply = format_details_for_confirmation(details)
        
        return render_template('conversation.html', lead_id=lead_id, response=agent_reply)
    
    # GET
    welcome = "Hello! I'm ready to assist you. What would you like to know about our products?"
    return render_template('conversation.html', lead_id=lead_id, response=welcome)

# --- Follow-up thread (24h) ---
def follow_up_checker():
    while True:
        time.sleep(86400)
        print("Checking for unresponsive leads...")
        # implement follow-up

def start_follow_up_thread():
    t = Thread(target=follow_up_checker, daemon=True)
    t.start()


def save_csv_complete(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Drop rows where any of 'age', 'country', or 'interest' are missing (NaN or empty string)
    df_cleaned = df.dropna(subset=['age', 'country', 'interest'])

    # Also remove rows where fields are just empty strings after stripping spaces
    df_cleaned = df_cleaned[
        (df_cleaned['age'].astype(str).str.strip() != '') &
        (df_cleaned['country'].astype(str).str.strip() != '') &
        (df_cleaned['interest'].astype(str).str.strip() != '')
    ]
    
    # Save the cleaned DataFrame back to the same CSV
    df_cleaned.to_csv(csv_file_path, index=False)

    print("CSV saved!")



if __name__ == '__main__':
    start_follow_up_thread()
    app.run(debug=True)
    save_csv_complete("leads.csv")
