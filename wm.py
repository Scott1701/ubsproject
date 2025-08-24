import os
from typing import List, Dict, Any
from uuid import uuid4
import base64

import streamlit as st
from openai import OpenAI

# -------------------- Config --------------------
st.set_page_config(page_title="UBS Orion | Wealth Management Assistant", page_icon="💬", layout="centered")

model = "gpt-4.1-nano"
temperature = 0.7
MAX_REPLY_CHARS = 1900

# Dark & Light Mode
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Dict: {chat_id: {"name": ..., "messages": [...] }}
if "active_chat" not in st.session_state:
    new_id = str(uuid4())
    st.session_state.chats[new_id] = {"name": "Chat #1", "messages": []}
    st.session_state.active_chat = new_id

SYSTEM_PROMPT = (
    "You are an AI Wealth Manager Assistant called Orion, you will assist the user of the chatbot who is a dedicated UBS Wealth Manager Executive"
    "in his journey to ensure that the clients under his portfolio are well taken care of and with ample details to support them in their journey"
    "include: (1) identifying and proactively recommending exclusive investment or lifestyle opportunities triggered by significant life milestones—"
    "such as tailored education planning for children entering elite institutions or orchestrating bespoke celebratory events like milestone anniversaries—"
    "with Relationship Managers (RMs) empowered to coordinate logistics upon client approval; and (2) delivering high-touch concierge recommendations for upcoming travel, "
    "taking into account personal health requirements, family composition (e.g., infant care or elder support), and cultural preferences, "
    "all informed by RM-provided natural language input and the client’s comprehensive wealth and lifestyle profile."


    "Before proceeding, confirm with me who will I be assisting or ask me if I would like a list of my current portfolio."

    # Alexandra Wu-Chan
    "Here is Alexandra Wu-Chan’s details: Alexandra Wu-Chan, age 38, is based in Hong Kong and has recently been appointed as CEO of her family’s multinational conglomerate—a major career milestone. "
    "She is planning an opulent 40th birthday celebration in January 2026 at a château in the Loire Valley. "
    "Her two children attend elite international schools, prompting opportunities for summer academic programmes at institutions such as Oxford and Cambridge. "
    "She is a private wine collector and a lover of fine art photography. "
    "Recommendations should include bespoke birthday experiences, private vineyard tastings, and early-access academic residencies, with full event coordination offered upon her approval. "

    # Luca Bianchi
    "Here is Luca Bianchi’s details: Luca Bianchi, age 52, is based in Milan, Italy and is preparing to celebrate his 25th wedding anniversary with a vow renewal in Santorini in April 2026—a significant milestone. "
    "His wife favours Mediterranean décor and holistic spa retreats. "
    "The couple follow a vegan lifestyle and prioritise sustainability. "
    "Luca is also an avid collector of rare timepieces. "
    "This occasion should trigger recommendations including luxury villa bookings, Grecian-style bespoke ceremony planning, and wellness-focused itineraries. "
    "In addition, real-time alerts should surface for exclusive watch auctions during his May 2026 visit to Geneva. "

    # Charles Montgomery Iv
    "Here is Charles Montgomery IV’s details: Charles Montgomery IV, age 60, is based in London and is entering retirement in early 2026—a key transition point. "
    "He intends to establish a philanthropic foundation focused on climate resilience and education equity. "
    "He frequently travels to Monaco and the Maldives and has a deep appreciation for classical music, antiques, and private yacht excursions. "
    "High-touch suggestions should include tailored introductions to philanthropic consultants, curated donor forum invitations, and Mediterranean yacht cruises aligned with major cultural and auction events. "
    "Special attention should be given to coordinating these opportunities with his post-retirement wellness plans. "

    # Noor Al-Fulan
    "Here is Noor Al-Fulan’s details: Noor Al-Fulan, age 29, is based in Dubai and recently got engaged—her wedding is planned for December 2025 at a private riad in Marrakesh. "
    "Noor is a luxury influencer with over 2 million followers, and her interests include bespoke fashion, wellness retreats, and high jewellery. "
    "She travels regularly to Paris, Los Angeles, and Tokyo for brand collaborations. "
    "High-touch recommendations should focus on honeymoon destinations offering privacy and wellness (such as Bhutan or Bali), custom jewellery consultations, and exclusive access to Paris Fashion Week in October 2025. "
    "All travel and event arrangements should prioritise discretion and VIP access. "

    # Kenji Tanaka
    "Here is Kenji Tanaka’s details: Kenji Tanaka, age 44, is based in Tokyo and has recently exited his tech startup. "
    "He is planning a family sabbatical across Europe in Summer 2026—a major lifestyle transition. "
    "He is married with three children (ages 4 to 11), and values education, culinary experiences, and cultural immersion. "
    "Recommendations should include multi-country luxury itineraries with interactive museum access, private cooking classes, and premium scenic rail journeys. "
    "All bookings must be tailored for families with young children and provide Japanese-speaking guides. "
    "Investment briefings and venture capital summit alerts may also be surfaced during his sabbatical period. "


)

# -------------------- Helpers --------------------

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") # type: ignore[attr-defined]
    if not api_key:
        st.stop()
    return OpenAI(api_key=api_key)

def _truncate(text: str, max_chars: int = MAX_REPLY_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

def _build_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history

# -------------------- Sidebar --------------------

# Initialize state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    new_id = str(uuid4())
    st.session_state.chats[new_id] = {"name": "Chat #1", "messages": []}
    st.session_state.active_chat = new_id

with st.sidebar:

    # UBS Logo from URL
    st.markdown(
        """
        <div style="text-align: center; margin-top: -20px; margin-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/UBS_Logo.png/960px-UBS_Logo.png" 
                 style="width: 120px; display: block; margin: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ➕ New Chat
    if st.button("➕ New Chat", key="new_chat_btn"):
        new_id = str(uuid4())
        name = f"Chat #{len(st.session_state.chats) + 1}"
        st.session_state.chats[new_id] = {"name": name, "messages": []}
        st.session_state.active_chat = new_id
        st.rerun()

    # Chat list
    for chat_id, chat in st.session_state.chats.items():
        is_active = chat_id == st.session_state.active_chat

        chat_cols = st.columns([0.8, 0.2])
        if chat_cols[0].button(chat["name"], key=f"chat_btn_{chat_id}"):
            st.session_state.active_chat = chat_id
            st.rerun()

        if chat_cols[1].button("🗑", key=f"del_btn_{chat_id}"):
            del st.session_state.chats[chat_id]
            if chat_id == st.session_state.active_chat:
                if st.session_state.chats:
                    st.session_state.active_chat = next(iter(st.session_state.chats))
                else:
                    new_id = str(uuid4())
                    st.session_state.chats[new_id] = {"name": "Chat #1", "messages": []}
                    st.session_state.active_chat = new_id
            st.rerun()

    # Style everything
    st.markdown(f"""
        <style>
            /* Reset Streamlit button styles */
            button[kind="secondary"], button[kind="primary"] {{
                all: unset !important;
                font-family: inherit !important;
                cursor: pointer !important;
                display: block;
                width: 100%;
            }}

            /* New Chat button */
            button[data-testid="baseButton-new_chat_btn"] {{
                color: red !important;
                font-size: 1rem !important;
                font-weight: bold !important;
                text-align: left !important;
                margin-bottom: 1rem !important;
                padding: 8px 12px !important;
            }}
            button[data-testid="baseButton-new_chat_btn"]:hover {{
                text-decoration: underline;
            }}

            /* Chat buttons */
            button[data-testid^="baseButton-chat_btn_"] {{
                color: black !important;
                font-size: 0.95rem !important;
                padding: 10px 14px !important;
                border-radius: 10px !important;
                text-align: left !important;
                background-color: transparent !important;
                transition: background-color 0.2s;
            }}
            button[data-testid^="baseButton-chat_btn_"]:hover {{
                background-color: rgba(0, 0, 0, 0.05) !important;
            }}

            /* ACTIVE CHAT HIGHLIGHT */
            button[data-testid="baseButton-chat_btn_{st.session_state.active_chat}"] {{
            background-color: #d0d0d0 !important;
            color: black !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            border: 2px solid #888 !important;  /* <-- dark grey border */
            }}

            /* Delete icons */
            button[data-testid^="baseButton-del_btn_"] {{
                font-size: 1.2rem !important;
                color: grey !important;
                padding: 6px !important;
                background: none !important;
                border: none !important;
            }}
            button[data-testid^="baseButton-del_btn_"]:hover {{
                color: red !important;
            }}
        </style>
    """, unsafe_allow_html=True)

# -------------------- Main UI --------------------
active_chat_id = st.session_state.active_chat
active_chat = st.session_state.chats[active_chat_id]

st.title("UBS Wealth Manager Assistant 💬")
st.caption(f"Chat Name: {active_chat['name']}")

# Render previous messages
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_input = st.chat_input("Type your message…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    active_chat["messages"].append({"role": "user", "content": user_input})

    # Example response
    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=_build_messages(active_chat["messages"]),
            temperature=0.7,
        )
        reply = response.choices[0].message.content or ""
        reply = _truncate(reply, 2000)
    except Exception as e:
        reply = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(reply)
    active_chat["messages"].append({"role": "assistant", "content": reply})