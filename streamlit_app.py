import streamlit as st
import pandas as pd
import time
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

try:
    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from src.semantic_router import SemanticRouter, SemanticRouterError
    from src.specialist_clients import query_specialist, LLMClientError
    try:
        from config import config
    except ImportError:
        from config import config

except ImportError as e:
    st.error(f"Failed to import components: {e}.")
    st.error(f"Current sys.path: {sys.path}")
    st.error("Please ensure 'semantic_router.py', 'specialist_clients.py', and 'config.py' are inside the 'src' directory, and the 'src' directory is in the same folder as this Streamlit script OR adjust the sys.path append logic.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during imports: {e}")
    st.error(traceback.format_exc())
    st.stop()

# /////////////////////////////////////////////////////////////
st.set_page_config(
    page_title="Router",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# /////////////////////////////////////////////////////////////
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        /* Optional: Simplified gradient or solid color */
        color: #1f77b4;
        /* background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; */
        margin-bottom: 2rem;
    }
    /* .metric-card can be removed if not used */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724; /* Ensure text is readable */
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24; /* Ensure text is readable */
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460; /* Ensure text is readable */
    }
    /* Ensure code blocks are readable */
    pre {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto; /* Allow horizontal scroll for long code */
        color: #333; /* Default text color */
    }
    code {
         color: #333; /* Ensure inline code is also readable */
    }

</style>
""", unsafe_allow_html=True)

if 'router' not in st.session_state:
    st.session_state.router = None
if 'routing_history' not in st.session_state:
    st.session_state.routing_history = []
if 'router_initialized' not in st.session_state:
     st.session_state.router_initialized = False 


def display_header():
    st.markdown('<h1 class="main-header">🎯 Prompt Router</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #666;">
            Route prompts to specialist models.
        </p>
    </div>
    """, unsafe_allow_html=True)

def initialize_router():
    if not st.session_state.router_initialized and st.session_state.router is None:
        try:
            with st.spinner("Initializing SS-GER Router... This may take a moment."):
                time.sleep(0.5)
                if config:
                    st.session_state.router = SemanticRouter()
                    st.session_state.router_initialized = True 
                    st.success("✅ Router initialized successfully!")
                    time.sleep(1) 
                    return True
                else:
                    st.error("❌ Configuration not loaded. Cannot initialize router.")
                    return False
        except SemanticRouterError as e:
            st.error(f"❌ Failed to initialize router: {e}")
            st.error("Please ensure the expertise database exists and is accessible (check CHROMADB_PATH in config). You might need to build it first.")
            st.session_state.router_initialized = False 
            return False
        except Exception as e: 
             st.error(f"❌ An unexpected error occurred during router initialization: {e}")
             st.error(traceback.format_exc()) 
             st.session_state.router_initialized = False
             return False
    elif st.session_state.router is not None:
         st.session_state.router_initialized = True 
         return True
    else: 
         return False


def routing_interface():
    if not st.session_state.router_initialized:
        if not initialize_router():
             st.warning("Router initialization failed. Routing is unavailable.")
             if st.button("Retry Initialization"):
                  st.session_state.router = None 
                  st.session_state.router_initialized = False
                  st.rerun()
             return 

    st.subheader("🎯 Enter Prompt")

    prompt = st.text_area(
        "Your prompt:",
        height=150,
        placeholder="e.g., Explain the concept of recursion in Python with an example."
    )

    col1, col2 = st.columns([3, 1]) 
   
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True) 
        submit_button = st.button("🚀 Route ", type="primary", key="route_button", use_container_width=True)

    if submit_button:
        if not prompt.strip():
            st.error("Please enter a prompt.")
            return

        if not st.session_state.router:
             st.error("Router is not available. Initialization might have failed.")
             if st.button("Retry Router Initialization"):
                  st.session_state.router = None
                  st.session_state.router_initialized = False
                  st.rerun()
             return

        try:
            result_placeholder = st.empty()
            response_placeholder = st.empty()

            with result_placeholder.status("Processing...", expanded=True):
                st.write("🔄 Routing prompt...")
                routing_start_time = time.time()
                routing_result = st.session_state.router.route(prompt)
                routing_end_time = time.time()
                routing_result['routing_time'] = routing_result.get('routing_time', routing_end_time - routing_start_time)
                st.write(f"✅ Routed to: **{routing_result.get('category', 'N/A')}** (Confidence: {routing_result.get('confidence', 0):.3f})")

                st.write("🗣️ Querying specialist model...")
                response_result = query_specialist(
                    routing_result.get('category', 'general_knowledge'), 
                    prompt, 
                )
                st.write("✅ Received response.")

            MAX_HISTORY_SIZE = 50 
            routing_entry = {
                'timestamp': datetime.now(),
                'prompt': prompt,
                'routing': routing_result,
                'response': response_result
            }
            st.session_state.routing_history.insert(0, routing_entry)
            st.session_state.routing_history = st.session_state.routing_history[:MAX_HISTORY_SIZE] 

            result_placeholder.empty()
            response_placeholder.empty()
            display_routing_results(routing_result, response_result, prompt)

        except SemanticRouterError as sre:
             st.error(f"Routing Error: {sre}")
        except LLMClientError as llme:
             st.error(f"LLM Client Error: {llme}")
        except Exception as e:
            st.error(f"An unexpected error occurred during routing: {e}")
            st.error(traceback.format_exc()) 

def display_routing_results(routing_result: Dict, response_result: Dict, prompt: str):
    st.markdown("---")
    st.subheader("📊 Routing Decision")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Category", routing_result.get('category', 'N/A'))
    with col2:
        st.metric("Confidence", f"{routing_result.get('confidence', 0):.3f}")
    with col3:
        st.metric("Routing Time", f"{routing_result.get('routing_time', 0):.3f}s")
    with col4:
        st.metric("From Cache", "Yes" if routing_result.get('from_cache', False) else "No")

    with st.expander("🔍 Show Detailed Routing Info", expanded=False):
        routing_info = routing_result.get('routing_info', {})
        st.write(f"**Reasoning:** {routing_result.get('reasoning', 'N/A')}")
        if not routing_result.get('from_cache', False) and routing_info:
            st.write(f"**Closest DB Example:** `{routing_info.get('closest_example', 'N/A')[:150]}...`")
            st.write(f"**Similarity Score:** {routing_info.get('similarity_score', 'N/A'):.4f}")
            st.write(f"**L2 Distance:** {routing_info.get('l2_distance', 'N/A'):.4f}")
            st.write(f"**Source:** {routing_info.get('source', 'N/A')}")
        elif routing_result.get('from_cache', False):
             st.write("*(Retrieved from cache, detailed info not applicable)*")

    st.subheader("🤖 Specialist Response")

    col1a, col2a, col3a = st.columns(3)

    with col1a:
        st.metric("Specialist Used", response_result.get('specialist', 'N/A'))
    with col2a:
        st.metric("Model Used", response_result.get('model', 'N/A'))
    with col3a:
        st.metric("Response Time", f"{response_result.get('response_time', 0):.3f}s")

    st.markdown("**Response:**")
    response_text = response_result.get('response', 'No response received.').strip() 
    success = response_result.get('success', False)
    category = routing_result.get('category', 'general_knowledge')

    is_code_like = any(kw in response_text for kw in ["def ", "class ", "import ", "public static", "console.log", "{", "}", ";", "=>", "```"]) or response_text.strip().startswith("#")

    if success:
        if category == "math":
            try:
                
                st.latex(response_text)
            except Exception as latex_error:
                st.warning(f"Could not render response as LaTeX: {latex_error}")
                st.info(response_text)
        elif is_code_like:
            lang_guess = 'python'
            if 'java' in prompt.lower() or 'Java' in response_text: lang_guess = 'java'
            elif 'javascript' in prompt.lower() or 'JavaScript' in response_text: lang_guess = 'javascript'
            elif 'c++' in prompt.lower() or ' C++' in response_text or '#include' in response_text: lang_guess = 'cpp'
            elif 'sql' in prompt.lower() or 'SELECT ' in response_text.upper(): lang_guess = 'sql'
            st.code(response_text, language=lang_guess, line_numbers=True)
        else:
            st.info(response_text)
    else:
        st.error(f"Error generating response:\n{response_text}")

def routing_history_interface():
    """Routing history interface"""
    st.subheader("📚 Routing History")

    if not st.session_state.routing_history:
        st.info("No routing history available. Route some prompts first!")
        return

    history_data = []

    for i, entry in enumerate(st.session_state.routing_history): 
        routing_info = entry.get('routing', {})
        response_info = entry.get('response', {})
        history_data.append({
            'Timestamp': entry.get('timestamp', datetime.min).strftime('%Y-%m-%d %H:%M:%S'),
            'Prompt': entry.get('prompt', 'N/A')[:100] + ("..." if len(entry.get('prompt', 'N/A')) > 100 else ""),
            'Routed Category': routing_info.get('category', 'N/A'),
            'Confidence': f"{routing_info.get('confidence', 0):.3f}",
            'Route Time (s)': f"{routing_info.get('routing_time', 0):.3f}",
            'Cache Hit': "Yes" if routing_info.get('from_cache', False) else "No",
            'Specialist': response_info.get('specialist', 'N/A'),
            'Resp. Time (s)': f"{response_info.get('response_time', 0):.3f}", 
            'Success': "✅" if response_info.get('success', False) else "❌ Error"
        })

    df = pd.DataFrame(history_data)

    st.dataframe(df, use_container_width=True, height=400) 

    if not df.empty:
         options = list(range(len(df)))
         def format_hist_option(idx):
             ts = df.iloc[idx]['Timestamp']
             pr = df.iloc[idx]['Prompt'][:30]
             return f"{idx}: {ts} - '{pr}...'"

         selected_index = st.selectbox(
             "View full details for entry:",
             options=options,
             format_func=format_hist_option,    
             index=None, 
             placeholder="Select an entry..."
             )
         if selected_index is not None:
             
             history_actual_index = selected_index 
             with st.expander(f"Details for Entry at Index {selected_index}", expanded=True):
                  st.markdown("**Prompt:**")
                  st.text(st.session_state.routing_history[history_actual_index]['prompt'])
                  st.markdown("**Routing Result:**")
                  st.json(st.session_state.routing_history[history_actual_index]['routing'])
                  st.markdown("**Response Result:**")
                  st.json(st.session_state.routing_history[history_actual_index]['response'])

    st.markdown("---")
    if st.button("🗑️ Clear History", key="clear_history_button"):
        st.session_state.routing_history = []
        st.rerun()

def main():
    display_header()

    st.sidebar.title("📄 Navigation")
    page = st.sidebar.radio( 
        "Select Page",
        [
            "🎯 Prompt Routing", 
            "📚 History"
        ],
        key="page_selector"
    )


    if page == "🎯 Prompt Routing":
        routing_interface()
    elif page == "📚 History":
        routing_history_interface()


if __name__ == "__main__":
    main()