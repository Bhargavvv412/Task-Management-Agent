import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime, date, timedelta
import re
import os
import json
from dotenv import load_dotenv
from bson.objectid import ObjectId # For better MongoDB ID handling

# -----------------------------
# CONFIG: Gemini + MongoDB
# -----------------------------
# Load environment variables (API Key and Mongo URI)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") # Use a default for local dev

if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize Gemini Client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    st.error(f"Gemini Configuration Error: {e}")
    st.stop()

# Initialize MongoDB Client
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Check connection (will raise exception if connection fails)
    client.admin.command('ping') 
    db = client["meeting_planner"]
    collection = db["meetings"]
except Exception as e:
    st.error(f"MongoDB Connection Error: Could not connect to database. Ensure MongoDB is running. Details: {e}")
    st.stop()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_priority_color(priority):
    """Returns a color code based on priority for better visual distinction."""
    return {
        "High": "red",
        "Medium": "orange",
        "Low": "blue",
    }.get(priority, "gray")

def resolve_priority(priority_str):
    """Maps various priority strings from the AI to 'High', 'Medium', or 'Low'."""
    if not priority_str:
        return "Medium"
    
    lower_pri = priority_str.lower()
    if any(word in lower_pri for word in ["high", "urgent", "critical", "immediate"]):
        return "High"
    elif any(word in lower_pri for word in ["medium", "normal", "standard"]):
        return "Medium"
    elif any(word in lower_pri for word in ["low", "optional", "later"]):
        return "Low"
    return "Medium" # Default fallback

def resolve_date(date_str):
    """Attempts to resolve relative dates (like 'tomorrow') to YYYY-MM-DD."""
    today = date.today()
    
    if "tomorrow" in date_str.lower():
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Simple regex to catch common YYYY-MM-DD or MM/DD/YYYY formats
    # Note: Gemini is instructed to use YYYY-MM-DD, but this adds robustness
    date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', date_str)
    if date_match:
        # Simple extraction assumes Gemini returns a valid format close to ISO
        return date_match.group(0).split('T')[0] 
        
    # If no date found, return today's date
    return today.strftime("%Y-%m-%d")

def resolve_time(time_str):
    """Attempts to ensure the time is in a simple HH:MM format."""
    if not time_str:
        return "12:00" # Default to noon if no time is provided
    
    # Simple regex to find HH:MM or HH:MM:SS
    time_match = re.search(r'(\d{1,2}:\d{2}(:\d{2})?)', time_str)
    if time_match:
        # Split and take only HH:MM
        return ":".join(time_match.group(1).split(':')[:2])
        
    # Fallback to a default time
    return "12:00"

# -----------------------------
# STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(page_title="AI Meeting Planner", page_icon="🤖", layout="centered")
st.title("🤖 AI-Powered Meeting Planner v3 (Auto-Adjusted)")
st.markdown("Plan, prioritize, and organize meetings automatically using **Gemini 2.5 Flash** ⚡ and **MongoDB**. The AI now determines the **Date**, **Time**, and **Priority**.")
st.divider()

# -----------------------------
# USER INPUT SECTION
# -----------------------------
with st.container(border=True):
    st.subheader("Plan a New Meeting")
    
    # Only the main text input is needed now!
    user_input = st.text_input(
        "🗣️ Describe your meeting (e.g., 'Urgent sync with Alex tomorrow at 9 AM about the Q3 budget review'):",
        placeholder="e.g. Schedule a meeting with Alex next Monday at 10 AM about the Q3 budget review.",
        key="user_meeting_desc"
    )

# -----------------------------
# GEMINI PARSER (SAFE JSON)
# -----------------------------
@st.cache_data(show_spinner="🧠 AI is parsing and prioritizing your request...")
def extract_meeting_info(text):
    """
    Uses Gemini to extract structured meeting data, including priority and time.
    """
    if not text.strip():
        return None
        
    prompt = f"""
    You are an expert data extractor and planner. Extract structured meeting information, including the time and a calculated priority, from the following user request.
    
    USER REQUEST:
    "{text}"

    Return ONLY valid JSON (no markdown, no explanations, no prefix).
    Fields must be: 
    "title" (a concise name), 
    "participants" (a comma-separated list of names/teams), 
    "date" (in YYYY-MM-DD format if possible, otherwise use the extracted text), 
    "time" (in HH:MM 24-hour format), 
    "priority" (select one of 'High', 'Medium', or 'Low'), 
    "topic" (a brief description of the meeting content).
    
    Example Output:
    {{"title": "Q3 Budget Review", "participants": "Alex, Finance Team", "date": "2025-10-28", "time": "10:00", "priority": "High", "topic": "Reviewing quarterly financials"}}
    """
    try:
        response = model.generate_content(prompt)
        clean = response.text.strip()
        # Robust cleaning of markdown and potential quotes
        clean = re.sub(r"```json|```|\\n", "", clean, flags=re.MULTILINE).strip()
        
        # Replace 'null' with 'None' for proper Python JSON loading
        clean = clean.replace("null", "None").replace("'", '"')

        # Use json.loads for safe parsing
        data = json.loads(clean)
        
        # Final validation and cleanup
        if isinstance(data, dict):
            # Resolve relative dates or clean up the date format
            data['date'] = resolve_date(data.get('date', date.today().strftime("%Y-%m-%d")))
            data['time'] = resolve_time(data.get('time', '12:00'))
            data['priority'] = resolve_priority(data.get('priority'))
            data['title'] = data.get('title', 'Untitled Meeting').strip()
            data['participants'] = data.get('participants', 'Unspecified').strip()
            data['topic'] = data.get('topic', 'No topic provided').strip()
            
            return data
            
    except json.JSONDecodeError:
        st.error("Gemini Parsing Error: Could not decode AI output as JSON. Please try refining your description.")
    except Exception as e:
        st.error(f"An unexpected Gemini error occurred: {e}")
        
    return None

# -----------------------------
# ADD MEETING BUTTON
# -----------------------------
if st.button("🚀 Confirm & Add Meeting (AI Auto-Adjusted)", use_container_width=True, type="primary"):
    if user_input.strip():
        info = extract_meeting_info(user_input)
        
        if info:
            # The document now uses the AI-extracted values
            final_doc = {
                "title": info['title'],
                "participants": info['participants'],
                "topic": info['topic'],
                "date": info['date'],
                "time": info['time'], # AI-extracted time
                "priority": info['priority'], # AI-extracted priority
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Insert into MongoDB
            collection.insert_one(final_doc)
            st.toast(f"✅ Meeting '{final_doc['title']}' added! Priority: {final_doc['priority']}", icon='🎉')
            
            # Clear input and rerun
            if "user_meeting_desc" in st.session_state:
                del st.session_state.user_meeting_desc 
                
            st.rerun()
            
    else:
        st.warning("Please describe your meeting first!")

# ---
# DISPLAY MEETINGS (SORTED)
# ---
st.subheader("📋 Scheduled Meetings")

priority_order = {"High": 1, "Medium": 2, "Low": 3}

try:
    # Retrieve all meetings
    meetings = list(collection.find())
    
    if meetings:
        # Robust sorting by Priority (High-Low) then by Date/Time
        meetings.sort(
            key=lambda x: (
                # 1. Priority (Safe access)
                priority_order.get(x.get("priority", "Medium"), 99),
                
                # 2. Date: Sort by YYYY-MM-DD
                x.get("date") if x.get("date") is not None else "9999-01-01", 
                
                # 3. Time: Sort by HH:MM
                x.get("time") if x.get("time") is not None else "23:59",
            )
        )

        for m in meetings:
            meeting_id_str = str(m["_id"]) 
            
            # Use columns for a clean card-like layout
            col_left, col_right = st.columns([5, 1])
            
            with col_left:
                st.markdown(
                    f"""
                    #### :blue[{m.get('title', 'Untitled Meeting')}]
                    **📅 Date & 🕒 Time:** {m.get('date', 'N/A')} at {m.get('time', 'N/A')}
                    **👥 Participants:** {m.get('participants', 'Unspecified')}
                    **💬 Topic:** {m.get('topic', 'No topic provided')}
                    """
                )
            with col_right:
                # Use markdown with color for priority
                st.markdown(
                    f"**<p style='color:{get_priority_color(m.get('priority'))}'>{m.get('priority', 'Medium')} Priority</p>**", 
                    unsafe_allow_html=True
                )
                st.write("\n") # Add a small spacer
                
                # Delete button
                if st.button("❌ Delete", key=f"delete_{meeting_id_str}"):
                    collection.delete_one({"_id": m["_id"]})
                    st.toast(f"Meeting deleted.", icon='🗑️')
                    st.rerun()
            
            st.divider()
            
    else:
        st.info("No meetings yet! Use the planner above to schedule your first meeting. 👆")

except Exception as e:
    st.error(f"Error displaying meetings: {e}")

# ---
# CLEAR ALL BUTTON
# ---
# Check if 'meetings' is defined and not empty before checking the button
if 'meetings' in locals() and meetings and st.sidebar.button("🧹 Clear All Meetings", use_container_width=True):
    collection.delete_many({})
    st.toast("🗑️ All meetings cleared!", icon='🔥')
    st.rerun()

st.sidebar.caption("Built with Gemini, Streamlit, and MongoDB 🚀")