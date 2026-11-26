import streamlit as st

import agent_core

st.set_page_config(page_title="My Agent", page_icon="🧭")
st.title("🧭 My Agent")
st.caption("Bring your own Gemini key. Your key stays in this browser session and is never saved.")

api_key = st.sidebar.text_input("Your Gemini API key", type="password", help="Get a free key at aistudio.google.com.")
if st.sidebar.button("Clear chat"):
    st.session_state.pop("chat", None)
    st.session_state.pop("history", None)

if not api_key:
    st.info("Paste your Gemini API key in the sidebar to start.")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []      # what the page shows: (role, text, steps, stopped)
    st.session_state.history = []   # what the agent remembers: message objects
if st.session_state.get("graph_key") != api_key:
    st.session_state.graph = agent_core.build_graph(agent_core.make_llm(api_key))
    st.session_state.graph_key = api_key

for role, text, steps, stopped in st.session_state.chat:
    with st.chat_message(role):
        st.write(text)
        if steps:
            with st.expander(f"Agent steps ({len(steps)} tool call{'s' if len(steps) != 1 else ''})"):
                for number, s in enumerate(steps, 1):
                    st.markdown(f"**{number}. {s['tool']}** with `{s['args']}`")
                    st.caption(f"Result: {s['result'][:300]}")
        if stopped:
            st.warning("The step limit stopped this run.")

question = st.chat_input("Ask your agent")
if question:
    st.session_state.chat.append(("user", question, None, False))
    try:
        answer, steps, stopped, st.session_state.history = agent_core.run_agent(st.session_state.graph, st.session_state.history, question)
    except Exception as e:
        answer, steps, stopped = f"Something went wrong ({type(e).__name__}). Check your key and try again in a minute.", [], False
    st.session_state.chat.append(("assistant", answer, steps, stopped))
    st.rerun()