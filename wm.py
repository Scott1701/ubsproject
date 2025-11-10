import os
from typing import List, Dict, Any
from uuid import uuid4

import requests
import streamlit as st
from openai import OpenAI

# ==================== Config ====================
st.set_page_config(page_title="Orion | Wealth Management Assistant", page_icon="💬", layout="centered")

MODEL = "gpt-4o"
TEMPERATURE = 0.9
MAX_REPLY_CHARS = 4000

# ==================== Helpers ====================
@st.cache_resource(show_spinner=False)
def _get_openai_client() -> OpenAI:
    """Create a single OpenAI client for the app lifecycle."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to environment or st.secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


def _truncate(text: str, max_chars: int = MAX_REPLY_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _build_messages(system_prompt: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build messages for the API, stripping UI-only metadata keys."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        msgs.append({"role": role, "content": content})
    return msgs


def get_realtime_context(query: str, max_results: int = 3) -> str:
    """Fetch short summaries of live data using DuckDuckGo's Instant Answer API.
    This is optional context and will fail closed (returns a note) if the request fails.
    """
    try:
        res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=8,
        )
        data = res.json()
        snippets = []

        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"])

        if snippets:
            return "\n".join(snippets)
    except Exception as e:  # noqa: BLE001
        return f"(Live data unavailable: {e})"

    return "(No live results found.)"


# ==================== Base System Prompt ====================
BASE_SYSTEM_PROMPT = (
    "You are an AI Wealth Manager Assistant called Orion. You assist a dedicated Wealth Manager Executive "
    "in ensuring clients under their portfolio are well taken care of with ample details. "
    "You (1) proactively recommend exclusive investment or lifestyle opportunities triggered by significant life milestones—"
    "such as tailored education planning for children entering elite institutions or orchestrating bespoke celebratory events like milestone anniversaries—"
    "with Relationship Managers (RMs) empowered to coordinate logistics upon client approval; and (2) deliver high-touch concierge recommendations for upcoming travel, "
    "considering personal health requirements, family composition (e.g., infant care or elder support), and cultural preferences, "
    "informed by RM-provided natural language input and the client’s comprehensive wealth and lifestyle profile. "
    "Before proceeding, confirm who you will be assisting or offer to list the current portfolio. "

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

    # Charles Montgomery IV
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

    # Additional Notation
    "Lastly, when asked for recommendations, be direct and straightforward and give exact locations or services."
)

# ==================== Seeded Opportunities Prompt ====================
OPP_PROMPT = (
    "Identify and summarize recent exclusive opportunities that may be of strong interest to any clients in your portfolio, based on their profiles, life milestones, and stated interests."
    "Use the following criteria:"
    "• For each client, find 1–2 high-relevance opportunities that match their current context (e.g. major life events, lifestyle preferences, travel plans, or investment themes)."
    "• Focus on ultra-high-net-worth-appropriate experiences, investments, or partnerships (e.g. private placements, art or watch auctions, bespoke retreats, cultural events, or philanthropic forums)."
    "• Prioritize relevance and recency — reference opportunities or events within the past 3 months."
    "• Be concrete: specify names of events, locations, institutions, or offerings, not generic suggestions."
    "• Keep each recommendation under 3 sentences."
    "You will consider all client profiles and curate a recommendation for each one"
    "Output format:"
    "Client Name — [Short title of opportunity]"
    "[Concise 2–3 sentence description with timing, location, and why it fits their interests.]"
    "Example:"
    "Noor Al-Fulan — “Van Cleef & Arpels Haute Joaillerie Private Preview, Paris (Oct 2025)”"
    "An invitation-only showing of Van Cleef’s newest bridal haute jewellery line, with VIP fittings arranged through Maison representatives. Perfectly aligned with Noor’s engagement and brand collaborations."
    "If no relevant opportunities are found for a client, state no opportunities as of now."
)

# ==================== Session State Init ====================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "client_profiles" not in st.session_state:
    st.session_state.client_profiles = []
if "client_profile_hashes" not in st.session_state:   # prevent duplicates
    st.session_state.client_profile_hashes = set()
if "uploader_version" not in st.session_state:        # lets us clear uploader selection
    st.session_state.uploader_version = 0

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    new_id = str(uuid4())
    st.session_state.chats[new_id] = {"name": "Chat #1", "messages": []}
    st.session_state.active_chat = new_id

# Ensure a persistent, non-deletable Opportunities tab exists
if "opps_chat_id" not in st.session_state:
    opps_id = str(uuid4())
    st.session_state.opps_chat_id = opps_id
    st.session_state.chats[opps_id] = {
        "name": "Recommendations",
        "messages": [],
        "meta": {"pinned": True, "system": "opportunities"},
    }

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-top: -20px; margin-bottom: 20px;">
            <img src="https://i.gyazo.com/737ba90e6e261129b45c099fa1b68c52.png"
                 style="width: 120px; display: block; margin: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🔎 Recommendations (navigate to pinned tab; does not create a new chat)
    if st.button("🔎 Recommendations", key="opps_btn"):
    # Remove the old Recommendations tab if it exists
        if "opps_chat_id" in st.session_state:
            old_id = st.session_state.opps_chat_id
            if old_id in st.session_state.chats:
                del st.session_state.chats[old_id]

        # Create a new Recommendations chat
        new_id = str(uuid4())
        st.session_state.opps_chat_id = new_id
        st.session_state.chats[new_id] = {
            "name": "Recommendations",
            "messages": [],
            "meta": {"pinned": True, "system": "opportunities"},
        }

        # Set it active and trigger auto-generation
        st.session_state.active_chat = new_id
        st.session_state.autorun = True
        st.rerun()

    # ➕ New Chat
    if st.button("➕ New Chat", key="new_chat_btn"):
        new_id = str(uuid4())
        name = f"Chat #{len(st.session_state.chats) + 1}"
        st.session_state.chats[new_id] = {"name": name, "messages": []}
        st.session_state.active_chat = new_id
        st.rerun()

    # Chat list (pinned Opportunities first, non-deletable)
    opps_id = st.session_state.get("opps_chat_id")

    # Render the rest (deletable)
    to_delete: List[str] = []
    for chat_id, chat in list(st.session_state.chats.items()):
        if chat_id == opps_id:
            continue
        cols = st.columns([0.8, 0.2])
        if cols[0].button(chat["name"], key=f"chat_btn_{chat_id}"):
            st.session_state.active_chat = chat_id
            st.rerun()
        if cols[1].button("🗑", key=f"del_btn_{chat_id}"):
            to_delete.append(chat_id)

    # Delete selected (never delete the pinned tab)
    for chat_id in to_delete:
        if chat_id == opps_id:
            continue
        del st.session_state.chats[chat_id]
        if chat_id == st.session_state.active_chat:
            # fallback to Opportunities or first remaining
            next_ids = [cid for cid in st.session_state.chats.keys() if cid != opps_id]
            if next_ids:
                st.session_state.active_chat = next_ids[0]
            else:
                new_id = str(uuid4())
                st.session_state.chats[new_id] = {"name": "Chat #1", "messages": []}
                st.session_state.active_chat = new_id
        st.rerun()

    # Styling (avoid brittle testid selectors where possible)
    st.markdown(
        f"""
        <style>
            .new-chat-btn {{
                color: red; font-size: 1rem; font-weight: bold; text-align: left;
                display: block; width: 100%; margin-bottom: 1rem; padding: 8px 12px;
            }}
            /* Minimal highlight for active chat via data-key attribute */
            [data-chat-id="{st.session_state.active_chat}"] {{
                background-color: #d0d0d0 !important; color: black !important;
                font-weight: 600 !important; border-radius: 10px !important; border: 2px solid #888 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("📎 Add Client Profile")

    # Custom wrapper to visually simplify the uploader
    st.markdown(
        """
        <style>
            /* Hide the file name and clear (x) button */
            .uploadedFile, .stUploadedFile {display: none !important;}
            .stFileUploader label div[data-testid="stFileUploaderDropzone"] {
                border: 2px dashed #bbb !important;
                border-radius: 8px !important;
                padding: 10px !important;
                text-align: center !important;
                color: #555 !important;
            }
            .stFileUploader label div[data-testid="stFileUploaderDropzone"]::before {
                content: "📄 Drop or click to add client profile PDFs";
                display: block;
                font-weight: 500;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Always-visible uploader (multi-file)
    uploaded_pdfs = st.file_uploader(
        label="Upload client PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"client_pdf_{st.session_state.uploader_version}",
        label_visibility="collapsed",  # hide label to rely on CSS placeholder
    )

    system_prompt = BASE_SYSTEM_PROMPT

    if uploaded_pdfs:
        import hashlib, fitz
        for f in uploaded_pdfs:
            content = f.read()
            h = hashlib.md5(content).hexdigest()
            if h in st.session_state.client_profile_hashes:
                continue
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                text = "".join(page.get_text("text") for page in doc).strip()
                if not text:
                    continue
                sanitized = text.replace("\x00", "")
                st.session_state.client_profiles.append(
                    f"# New Client Profile Added\n---\n{sanitized}\n"
                )
                st.session_state.client_profile_hashes.add(h)
            except Exception as e:
                st.error(f"Failed to process {getattr(f, 'name', 'a file')}: {e}")
        st.session_state.uploader_version += 1
        st.rerun()

        if added:
            st.success(f"✅ Added {added} profile(s).")

    # Controls: wipe all or just clear the current selection UI
    if st.button("🧹 Wipe New Client Data", key="wipe_clients_btn", use_container_width=True):
        st.session_state.client_profiles = []
        st.session_state.client_profile_hashes = set()
        st.session_state.uploader_version += 1
        st.rerun()

    st.caption(f"Profiles in session: {len(st.session_state.client_profiles)}")

    if st.session_state.client_profiles:
        MAX_PROFILE_CHARS = 6000
        joined = "\n\n".join(st.session_state.client_profiles)
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n# Additional Client Profiles (session)\n{joined[:MAX_PROFILE_CHARS]}"
    else:
        system_prompt = BASE_SYSTEM_PROMPT

MAX_PROFILE_CHARS = 12000  # raise if you like
def _build_system_prompt() -> str:
    base = BASE_SYSTEM_PROMPT
    profs = st.session_state.get("client_profiles", [])
    if profs:
        joined = "\n\n".join(profs)
        return f"{base}\n\n# Additional Client Profiles (session)\n{joined[:MAX_PROFILE_CHARS]}"
    return base

system_prompt = _build_system_prompt()

# ==================== Main UI ====================
active_chat_id = st.session_state.active_chat
active_chat = st.session_state.chats[active_chat_id]

st.title("Orion | Wealth Assistant 💬")
st.caption(f"Chat Name: {active_chat['name']}")

# Render previous messages (hide seeded/hidden messages)
for msg in active_chat.get("messages", []):
    # Hide UI-only seeded messages
    if msg.get("meta", {}).get("hidden"):
        continue
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", "")) 

# Auto-run the Opportunities prompt only when switching to the pinned tab
if st.session_state.get("autorun") and active_chat_id == st.session_state.get("opps_chat_id"):
    st.session_state.autorun = False
    client = _get_openai_client()
    messages = _build_messages(system_prompt, [{"role": "user", "content": OPP_PROMPT}])
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        reply = _truncate(response.choices[0].message.content or "", MAX_REPLY_CHARS)
    except Exception as e:  # noqa: BLE001
        reply = f"Error: {e}"
    with st.chat_message("assistant"):
        st.markdown(reply)
    # Persist only the assistant's reply so the tab looks like a regular chat of reports
    active_chat["messages"].append({"role": "assistant", "content": reply})

# Input
user_input = st.chat_input("Type your message…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    active_chat["messages"].append({"role": "user", "content": user_input})

    client = _get_openai_client()

    # Compose request (optionally enrich with real-time context if desired)
    messages = _build_messages(system_prompt, active_chat["messages"])  # system + history

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        reply = response.choices[0].message.content or ""
        reply = _truncate(reply, MAX_REPLY_CHARS)
    except Exception as e:  # noqa: BLE001
        reply = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(reply)
    active_chat["messages"].append({"role": "assistant", "content": reply})
