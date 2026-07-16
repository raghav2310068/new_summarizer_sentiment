import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import importlib

# Make sure we import utils correctly and force reload for Streamlit hot-reloading stability
import utils
importlib.reload(utils)
from sample_article import SAMPLE_ARTICLE_TEXT

# Set page config for premium look
st.set_page_config(
    page_title="Intelligent News Summarizer & Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Set up custom premium CSS
st.markdown("""
<style>
    /* CSS for custom premium layout */
    .stApp {
        background-color: #0F172A; /* Slate 900 */
        color: #F8FAFC; /* Slate 50 */
    }
    
    /* Main Header styling */
    .gradient-header {
        background: linear-gradient(135deg, #60A5FA 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .subtitle {
        color: #94A3B8; /* Slate 400 */
        text-align: center;
        font-size: 1.25rem;
        margin-bottom: 30px;
    }
    
    /* Premium visual container card */
    .premium-card, div[data-testid="stVerticalBlockBorderDiv"] {
        background: rgba(30, 41, 59, 0.7) !important; /* Slate 800 with transparency */
        border: 1px solid rgba(148, 163, 184, 0.15) !important; /* Slate 400 */
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 24px !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* Custom tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1E293B;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        background-color: transparent;
        color: #94A3B8;
        font-weight: 600;
        padding: 0 16px;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(148, 163, 184, 0.1);
        color: #F8FAFC;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important; /* Blue 500 */
        color: #FFFFFF !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
    }
    
    /* Styled metric components */
    .stat-box {
        background: #1E293B;
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .stat-val {
        font-size: 2.25rem;
        font-weight: 700;
        color: #3B82F6;
    }
    
    .stat-lbl {
        font-size: 0.875rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Style blockquotes and results */
    blockquote {
        border-left: 4px solid #8B5CF6 !important; /* Violet 500 */
        background-color: rgba(139, 92, 246, 0.1);
        color: #E2E8F0;
        padding: 10px 20px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Description
st.markdown('<h1 class="gradient-header">Intelligent News Summarizer & Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Initialize Session States
if "article_text" not in st.session_state:
    st.session_state.article_text = ""
if "url" not in st.session_state:
    st.session_state.url = ""
if "nlp_data" not in st.session_state:
    st.session_state.nlp_data = None
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "llm_sentiment" not in st.session_state:
    st.session_state.llm_sentiment = ""
if "entities" not in st.session_state:
    st.session_state.entities = ""
if "quotes" not in st.session_state:
    st.session_state.quotes = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bias_analysis" not in st.session_state:
    st.session_state.bias_analysis = ""
if "translated_content" not in st.session_state:
    st.session_state.translated_content = ""
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "Spanish 🇪🇸"
if "jargon_glossary" not in st.session_state:
    st.session_state.jargon_glossary = ""
if "briefing_script" not in st.session_state:
    st.session_state.briefing_script = ""
if "style_transformer" not in st.session_state:
    st.session_state.style_transformer = ""
if "factcheck_queries" not in st.session_state:
    st.session_state.factcheck_queries = ""
if "compare_article_text" not in st.session_state:
    st.session_state.compare_article_text = ""
if "compare_url" not in st.session_state:
    st.session_state.compare_url = ""
if "compare_analysis" not in st.session_state:
    st.session_state.compare_analysis = ""
if "highlighted_html" not in st.session_state:
    st.session_state.highlighted_html = ""
if "entity_relation_map" not in st.session_state:
    st.session_state.entity_relation_map = ""
if "cultural_neutralizer" not in st.session_state:
    st.session_state.cultural_neutralizer = ""
if "echo_chamber_rewrite" not in st.session_state:
    st.session_state.echo_chamber_rewrite = ""
if "selected_echo_style" not in st.session_state:
    st.session_state.selected_echo_style = "Sensationalist Red-Top Tabloid 📰"

# SIDEBAR: API configuration loaded silently
env_api_key = os.environ.get("NARAROUTER_API_KEY", "")
env_base_url = os.environ.get("NARAROUTER_BASE_URL", "https://router.bynara.id/v1")
env_model_name = os.environ.get("NARAROUTER_MODEL_NAME", "mistral-medium-3-5")
api_configured = bool(env_api_key and env_base_url and env_model_name)

# Add quick status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Quick Start Demo Mode")
st.sidebar.markdown("Don't have a news link or text? Load our preloaded Space Exploration article immediately:")
if st.sidebar.button("💡 Load Preloaded News Article", type="primary", width="stretch"):
    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
    st.session_state.url = "NASA Space Exploration"
    st.session_state.nlp_data = None
    st.session_state.summary = ""
    st.session_state.llm_sentiment = ""
    st.session_state.entities = ""
    st.session_state.quotes = ""
    st.session_state.translated_content = ""
    st.session_state.jargon_glossary = ""
    st.session_state.briefing_script = ""
    st.session_state.bias_analysis = ""
    st.session_state.style_transformer = ""
    st.session_state.factcheck_queries = ""
    st.session_state.compare_article_text = ""
    st.session_state.compare_url = ""
    st.session_state.compare_analysis = ""
    st.session_state.highlighted_html = ""
    st.session_state.entity_relation_map = ""
    st.session_state.cultural_neutralizer = ""
    st.session_state.echo_chamber_rewrite = ""
    st.session_state.chat_history = []
    utils.log_activity("Loaded Preloaded Article", "Loaded preloaded space exploration news article (NASA Space Exploration) via sidebar shortcut")
    st.sidebar.success("✅ Space Discovery Article loaded!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Student Project Details")
st.sidebar.markdown("**Course:** Natural Language Processing (NLP)")
st.sidebar.markdown("**Project:** Intelligent News Summarizer & Sentiment Analysis System")
st.sidebar.markdown("**Technologies Used:**")
st.sidebar.markdown("- Python & Streamlit\n- NLTK Tokenizer & VADER\n- Plotly Data Visualization\n- Mistral Medium 3.5 (Nararouter)")

# Activity Log Auditing Section
st.sidebar.markdown("---")
with st.sidebar.expander("📜 Session Audit & Activity Log", expanded=False):
    st.markdown("This panel displays a persistent audit trail of user and AI activities, saved securely to a local JSON file.")
    
    # Refresh logs on display
    history = utils.get_activity_history()
    
    if not history:
        st.info("No activity logged in this session yet.")
    else:
        df_history = pd.DataFrame(history)
        df_display = df_history.rename(columns={
            "timestamp": "Timestamp",
            "activity_type": "Activity Type",
            "details": "Details"
        })
        st.dataframe(df_display, width='stretch', hide_index=True)
        
        # Download action for history log json
        try:
            with open("activity_history.json", "r", encoding="utf-8") as f:
                json_data = f.read()
            st.download_button(
                label="📥 Export Audit History (JSON)",
                data=json_data,
                file_name="news_analysis_audit_history.json",
                mime="application/json",
                width="stretch"
            )
        except Exception:
            pass
            
        if st.button("🧹 Clear Audit Logs", key="clear_logs_project1"):
            utils.clear_activity_history()
            st.success("Audit history cleared successfully!")
            st.rerun()

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📰 Article Input", 
    "⚙️ NLP Preprocessing", 
    "📊 Summary & Sentiment", 
    "🔑 Key Insights & Entities",
    "💬 Interactive Q&A",
    "⚖️ Bias & Framing",
    "🌐 Translation Hub",
    "🆚 Dual-Article Comparison"
])

# ==================== TAB 1: ARTICLE INPUT ====================
with tab1:
    with st.container(border=True):
        st.markdown("### Provide News Article Source")
        st.markdown("You can either enter a direct **news article URL** to scrape its content, or **manually paste** the article text below.")
        
        input_method = st.radio("Choose Input Method:", ["Fetch via URL", "Paste Text Manually", "Use Preloaded Demo Article"], horizontal=True)
        
        if input_method == "Fetch via URL":
            url_input = st.text_input("Enter News Article URL:", value=st.session_state.url if st.session_state.url != "NASA Space Exploration" else "", placeholder="https://news.ycombinator.com/item?id=...")
            if st.button("Scrape & Import Article", type="primary"):
                if url_input.strip():
                    with st.spinner("🕷️ Crawling article text and stripping unwanted tags..."):
                        try:
                            scraped_text = utils.fetch_article_text(url_input)
                            st.session_state.article_text = scraped_text
                            st.session_state.url = url_input
                            # Clear old analyses on new input
                            st.session_state.nlp_data = None
                            st.session_state.summary = ""
                            st.session_state.llm_sentiment = ""
                            st.session_state.entities = ""
                            st.session_state.quotes = ""
                            st.session_state.translated_content = ""
                            st.session_state.jargon_glossary = ""
                            st.session_state.briefing_script = ""
                            st.session_state.bias_analysis = ""
                            st.session_state.style_transformer = ""
                            st.session_state.factcheck_queries = ""
                            st.session_state.compare_article_text = ""
                            st.session_state.compare_url = ""
                            st.session_state.compare_analysis = ""
                            st.session_state.highlighted_html = ""
                            st.session_state.entity_relation_map = ""
                            st.session_state.cultural_neutralizer = ""
                            st.session_state.echo_chamber_rewrite = ""
                            st.session_state.chat_history = []
                            utils.log_activity("Scraped Article URL", f"Successfully scraped URL: {url_input}")
                            st.success("✅ Scraped successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a valid URL.")
                    
        elif input_method == "Paste Text Manually":
            text_input = st.text_area("Paste News Text Here (min. 100 words):", value=st.session_state.article_text if st.session_state.url != "NASA Space Exploration" else "", height=300)
            if st.button("Save & Process Text", type="primary"):
                if len(text_input.strip()) > 100:
                    st.session_state.article_text = text_input
                    st.session_state.url = ""
                    # Clear old analyses
                    st.session_state.nlp_data = None
                    st.session_state.summary = ""
                    st.session_state.llm_sentiment = ""
                    st.session_state.entities = ""
                    st.session_state.quotes = ""
                    st.session_state.translated_content = ""
                    st.session_state.jargon_glossary = ""
                    st.session_state.briefing_script = ""
                    st.session_state.bias_analysis = ""
                    st.session_state.style_transformer = ""
                    st.session_state.factcheck_queries = ""
                    st.session_state.compare_article_text = ""
                    st.session_state.compare_url = ""
                    st.session_state.compare_analysis = ""
                    st.session_state.highlighted_html = ""
                    st.session_state.entity_relation_map = ""
                    st.session_state.cultural_neutralizer = ""
                    st.session_state.echo_chamber_rewrite = ""
                    st.session_state.chat_history = []
                    utils.log_activity("Manually Pasted Text", f"Imported manual article text ({len(text_input.strip().split())} words)")
                    st.success("✅ Text saved successfully!")
                    st.rerun()
                else:
                    st.warning("Please paste a longer text to analyze.")
                    
        else:
            st.markdown("#### NASA Space Exploration: JWST Star-Forming Discovery")
            st.markdown("Load a rich, high-quality, preloaded news report detailing NASA's latest James Webb Space Telescope observations in the Ophiuchus cloud complex.")
            if st.button("🚀 Load Preloaded Space Discovery Article", type="primary", width="stretch"):
                st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                st.session_state.url = "NASA Space Exploration"
                st.session_state.nlp_data = None
                st.session_state.summary = ""
                st.session_state.llm_sentiment = ""
                st.session_state.entities = ""
                st.session_state.quotes = ""
                st.session_state.bias_analysis = ""
                st.session_state.chat_history = []
                utils.log_activity("Loaded Preloaded Article", "Loaded preloaded space exploration news article (NASA Space Exploration) from Tab 1 screen")
                st.success("✅ Space Discovery Article loaded successfully!")
                st.rerun()
    
    # Document Preview
    # Document Preview & Automated Pipeline
    if st.session_state.article_text:
        with st.container(border=True):
            st.subheader("⚡ Automated Analysis Pipeline")
            st.markdown("Run the complete NLP and LLM processing pipeline at once. This will crawl, pre-process, analyze, summarize, and compute sentiment classifications in a single step with real-time feedback.")
            
            if st.button("🚀 Execute Complete AI Pipeline", type="primary", width="stretch"):
                if not api_configured:
                    st.error("❌ Cannot run. API keys are missing in your .env file.")
                else:
                    with st.status("🛠️ Initiating News AI Analysis Pipeline...", expanded=True) as status:
                        st.write("📥 **Phase 1:** Normalizing raw text input...")
                        time.sleep(1.0)
                        
                        st.write("⚙️ **Phase 2:** Running classical NLTK sentence and word tokenization...")
                        st.session_state.nlp_data = utils.perform_nlp_preprocessing(st.session_state.article_text)
                        time.sleep(1.0)
                        
                        st.write("📊 **Phase 3:** Computing NLTK VADER compound sentiment polarity score...")
                        vader_res = utils.get_vader_sentiment(st.session_state.article_text)
                        time.sleep(0.8)
                        
                        st.write("🧠 **Phase 4:** Connecting to Nararouter & executing Mistral 3.5 abstractive summarization...")
                        summary_prompt = (
                            f"Please read the following news article and provide a highly professional, "
                            f"concise, and objective bulleted summary of 3-5 bullets, followed by a 'Key Takeaway' single sentence.\n\n"
                            f"Article Content:\n{st.session_state.article_text}"
                        )
                        st.session_state.summary = utils.query_nararouter(
                            summary_prompt, 
                            system_prompt="You are a professional research and media summarizer. Be objective and concise."
                        )
                        
                        st.write("🎭 **Phase 5:** Querying Mistral 3.5 for full semantic sentiment breakdown & reasoning...")
                        sentiment_prompt = (
                            f"Perform an in-depth sentiment analysis on this news article. Determine whether the article is "
                            f"POSITIVE, NEGATIVE, or NEUTRAL. Explain your reasoning in a clear, concise bulleted breakdown "
                            f"pointing to emotional word choices, the author's tone, and overall implications. "
                            f"Your output should follow this strict markdown format:\n"
                            f"**Overall LLM Sentiment:** [POSITIVE/NEGATIVE/NEUTRAL]\n"
                            f"**Analysis & Key Indicators:** [Detailed reasoning bullet points]\n\n"
                            f"Article Content:\n{st.session_state.article_text}"
                        )
                        st.session_state.llm_sentiment = utils.query_nararouter(
                            sentiment_prompt,
                            system_prompt="You are an expert NLP Sentiment Analyst. Analyze tones, adjectives, and overall bias."
                        )
                        
                        st.write("🔑 **Phase 6:** Extracting Named Entities (People, Orgs, Locations, Key Concepts)...")
                        entity_prompt = (
                            f"Analyze the following news article and extract the main named entities. "
                            f"Identify up to 5 entries for each category. Keep it clear, concise, and structured in clean markdown list groups. "
                            f"Categories to extract:\n"
                            f"- 👤 **Key People Mentioned**:\n"
                            f"- 🏢 **Organizations/Companies**:\n"
                            f"- 📍 **Locations/Places**:\n"
                            f"- 🛠️ **Technologies, Products, or Key Concepts**:\n\n"
                            f"Article Content:\n{st.session_state.article_text}"
                        )
                        st.session_state.entities = utils.query_nararouter(
                            entity_prompt,
                            system_prompt="You are an expert at Named Entity Recognition (NER). Produce clean, structured markdown list blocks."
                        )
                        
                        st.write("💬 **Phase 7:** Mining direct citations and critical quotes...")
                        quotes_prompt = (
                            f"Identify and extract up to 4 most significant direct quotes, assertions, or key statements from the following news article. "
                            f"Provide the quote, who said it (if available), and a single sentence explaining why it is critical to the article's narrative. "
                            f"Structure the output as beautiful styled blockquotes `> Quote` followed by the citation and context in bullet points.\n\n"
                            f"Article Content:\n{st.session_state.article_text}"
                        )
                        st.session_state.quotes = utils.query_nararouter(
                            quotes_prompt,
                            system_prompt="You are a professional quotes miner. Structure quotes beautifully in markdown blockquotes."
                        )
                        
                        status.update(label="✅ News AI Pipeline Processing Successful!", state="complete", expanded=True)
                    utils.log_activity("Executed Complete AI Pipeline", "Triggered automated execution of NLTK preprocessing, summaries, hybrid sentiments, entities, and citations extraction")
                    st.success("🎉 **Success!** Entire article pipeline executed. Click on the other tabs above to view your fully generated interactive charts, reports, entities, and Q&A chat.")
                    st.rerun()

        with st.container(border=True):
            st.subheader("📝 Loaded Article Preview")
            st.caption(f"Character Count: {len(st.session_state.article_text)} | Estimated Word Count: {len(st.session_state.article_text.split())}")
            st.text_area("Content Preview", st.session_state.article_text, height=200, disabled=True)

# ==================== TAB 2: NLP PREPROCESSING ====================
with tab2:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please choose one of the options below to begin analysis:")
            col_load1, col_load2 = st.columns([1, 1])
            with col_load1:
                st.markdown("#### Option 1: Import News Article")
                st.markdown("Use the **'📰 Article Input'** tab to scrape any online article URL or paste your own raw text.")
            with col_load2:
                st.markdown("#### Option 2: Run Immediate Demo")
                st.markdown("Don't have an article ready? Load a preloaded copy of NASA's James Webb Space Telescope discovery article immediately.")
                if st.button("🚀 Load Preloaded Article", type="primary", width="stretch", key="quick_load_news_tab2"):
                    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                    st.session_state.url = "NASA Space Exploration"
                    st.session_state.nlp_data = None
                    st.session_state.summary = ""
                    st.session_state.llm_sentiment = ""
                    st.session_state.entities = ""
                    st.session_state.quotes = ""
                    st.session_state.chat_history = []
                    st.rerun()
    else:
        st.markdown("### Classical Natural Language Processing Preprocessing Pipeline")
        st.markdown("Before sending text to a LLM, standard NLP techniques isolate key terms. Here we run sentence boundary detection, tokenization, clean-up, and compute word densities.")
        
        # Calculate statistics
        if st.session_state.nlp_data is None:
            with st.spinner("Processing NLP tokens..."):
                st.session_state.nlp_data = utils.perform_nlp_preprocessing(st.session_state.article_text)
                utils.log_activity("Ran NLP Tokenizer & Stats", f"Analyzed character count, word counts, NLTK sentence boundary splits, and word densities ({st.session_state.nlp_data['total_words']} tokens)")
                
        nlp = st.session_state.nlp_data
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{nlp["total_words"]}</div>
                <div class="stat-lbl">Raw Word Tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{nlp["total_sentences"]}</div>
                <div class="stat-lbl">Detected Sentences</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-val">{len(nlp["cleaned_tokens"])}</div>
                <div class="stat-lbl">Filtered Content Tokens</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Advanced stats calculations
        cleaned_words_count = len(nlp["cleaned_tokens"])
        unique_cleaned_words = len(set(nlp["cleaned_tokens"]))
        lexical_diversity = (unique_cleaned_words / cleaned_words_count * 100) if cleaned_words_count > 0 else 0
        
        avg_sentence_len = (nlp["total_words"] / nlp["total_sentences"]) if nlp["total_sentences"] > 0 else 0
        
        reading_time_seconds = int((nlp["total_words"] / 200) * 60)
        if reading_time_seconds < 60:
            reading_time_str = f"{reading_time_seconds} seconds"
        else:
            mins = reading_time_seconds // 60
            secs = reading_time_seconds % 60
            reading_time_str = f"{mins} min {secs}s"
            
        # Metrics Row 2
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        with col_adv1:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 3px solid #10B981;">
                <div class="stat-val">{lexical_diversity:.1f}%</div>
                <div class="stat-lbl">Lexical Diversity Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with col_adv2:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 3px solid #3B82F6;">
                <div class="stat-val">{avg_sentence_len:.1f}</div>
                <div class="stat-lbl">Avg Words per Sentence</div>
            </div>
            """, unsafe_allow_html=True)
        with col_adv3:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 3px solid #F59E0B;">
                <div class="stat-val">{reading_time_str}</div>
                <div class="stat-lbl">Est. Reading Duration</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two columns for text comparison and plotting
        c1, col2 = st.columns([1, 1])
        with c1:
            with st.container(height=480):
                st.subheader("🔬 Text Transformation & Cleaning")
                
                st.markdown("**Original Snippet:**")
                st.text(st.session_state.article_text[:300] + "...")
                
                st.markdown("**Normalized Text Snippet:**")
                normalized_sample = utils.clean_and_normalize_text(st.session_state.article_text[:300])
                st.code(normalized_sample + "...")
                
                st.markdown("**NLTK Stopwords Filtered Tokens:**")
                st.write(nlp["cleaned_tokens"][:40])
                
        with col2:
            with st.container(height=480):
                st.subheader("📊 Key Term Frequency Chart")
                
                # Show Plotly interactive barchart
                freq_df = pd.DataFrame(nlp["word_frequencies"].most_common(15), columns=["Keyword", "Frequency"])
                
                fig = px.bar(
                    freq_df, 
                    x="Frequency", 
                    y="Keyword", 
                    orientation='h',
                    title="Top 15 Most Frequent Content Words",
                    labels={"Frequency": "Occurrences", "Keyword": "Word"},
                    color="Frequency",
                    color_continuous_scale="Purples",
                    template="plotly_dark"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=350
                )
                st.plotly_chart(fig, width="stretch")

# ==================== TAB 3: SUMMARY & SENTIMENT ====================
with tab3:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please choose one of the options below to begin analysis:")
            col_load1, col_load2 = st.columns([1, 1])
            with col_load1:
                st.markdown("#### Option 1: Import News Article")
                st.markdown("Use the **'📰 Article Input'** tab to scrape any online article URL or paste your own raw text.")
            with col_load2:
                st.markdown("#### Option 2: Run Immediate Demo")
                st.markdown("Don't have an article ready? Load a preloaded copy of NASA's James Webb Space Telescope discovery article immediately.")
                if st.button("🚀 Load Preloaded Article", type="primary", width="stretch", key="quick_load_news_tab3"):
                    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                    st.session_state.url = "NASA Space Exploration"
                    st.session_state.nlp_data = None
                    st.session_state.summary = ""
                    st.session_state.llm_sentiment = ""
                    st.session_state.entities = ""
                    st.session_state.quotes = ""
                    st.session_state.chat_history = []
                    st.rerun()
    elif not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### 🧠 AI Core: Abstractive Summarization & Hybrid Sentiment Analysis")
        st.markdown("We combine a structured prompt with the **Mistral Medium 3.5** model to abstractly synthesize findings, and perform side-by-side sentiment analysis comparisons.")
        
        # Generation Button
        if not st.session_state.summary:
            run_btn = st.button("🚀 Execute LLM Analysis", type="primary")
        else:
            run_btn = st.button("🔄 Re-run LLM Analysis")
            
        if run_btn:
            with st.spinner("Processing summarization & sentiment analysis with Mistral via Nararouter..."):
                # 1. Summary prompt
                summary_prompt = (
                    f"Please read the following news article and provide a highly professional, "
                    f"concise, and objective bulleted summary of 3-5 bullets, followed by a 'Key Takeaway' single sentence.\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.summary = utils.query_nararouter(
                    summary_prompt, 
                    system_prompt="You are a professional research and media summarizer. Be objective and concise."
                )
                
                # 2. Sentiment prompt
                sentiment_prompt = (
                    f"Perform an in-depth sentiment analysis on this news article. Determine whether the article is "
                    f"POSITIVE, NEGATIVE, or NEUTRAL. Explain your reasoning in a clear, concise bulleted breakdown "
                    f"pointing to emotional word choices, the author's tone, and overall implications. "
                    f"Your output should follow this strict markdown format:\n"
                    f"**Overall LLM Sentiment:** [POSITIVE/NEGATIVE/NEUTRAL]\n"
                    f"**Analysis & Key Indicators:** [Detailed reasoning bullet points]\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.llm_sentiment = utils.query_nararouter(
                    sentiment_prompt,
                    system_prompt="You are an expert NLP Sentiment Analyst. Analyze tones, adjectives, and overall bias."
                )
                utils.log_activity("Generated Summary & Sentiment", "Synthesized news abstractive summaries and side-by-side hybrid sentiment models via Mistral 3.5")
                st.rerun()
                
        # If summaries have been generated, display them
        if st.session_state.summary:
            # Two column display for side-by-side Summary and Sentiment
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                with st.container(border=True):
                    st.subheader("📝 Concised Article Summary")
                    st.markdown(st.session_state.summary)
                    
            with col_right:
                with st.container(border=True):
                    st.subheader("🎭 Sentiment Analysis Comparison")
                    
                    # VADER baseline calculation
                    vader = utils.get_vader_sentiment(st.session_state.article_text)
                    
                    # Layout VADER in small card
                    st.markdown("#### 1. Classical NLP Baseline: NLTK VADER")
                    st.markdown(
                        f"NLTK VADER compound polarity calculation checks individual word-valence ratings. "
                    )
                    st.markdown(
                        f"<div style='background: {vader['color']}1A; border-left: 5px solid {vader['color']}; padding: 12px; border-radius: 4px; margin-bottom: 20px;'>"
                        f"<strong>Computed Label:</strong> {vader['label']} <br>"
                        f"<strong>Compound Polarity Score:</strong> {vader['scores']['compound']:.4f} "
                        f"(Pos: {vader['scores']['pos']:.2f}, Neu: {vader['scores']['neu']:.2f}, Neg: {vader['scores']['neg']:.2f})"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("---")
                    
                    # Layout LLM Sentiment
                    st.markdown("#### 2. Advanced LLM Sentiment (Mistral Medium 3.5)")
                    st.markdown(st.session_state.llm_sentiment)
            
            # Dynamic Sentiment Progression Trend Line
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader("📈 Dynamic Sentiment Progression & Emotional Arc")
                st.markdown(
                    "This chart plots the rolling sentiment score per sentence across the article. "
                    "It exposes the emotional narrative flow—tracing how the tone peaks and valleys from start to finish."
                )
                prog_data = utils.get_sentiment_progression(st.session_state.article_text)
                if prog_data:
                    df_prog = pd.DataFrame(prog_data)
                    
                    # Modern styling for line chart
                    fig_prog = px.line(
                        df_prog,
                        x="Sentence Index",
                        y="Rolling Sentiment",
                        hover_data=["Raw Sentiment", "Snippet"],
                        title="Narrative Sentiment Arc (Sliding Window Moving Average)",
                        labels={"Rolling Sentiment": "Rolling Tone Score", "Sentence Index": "Narrative Progression (Sentence #)"},
                        template="plotly_dark"
                    )
                    
                    # Customize look
                    fig_prog.update_traces(
                        line=dict(color="#A78BFA", width=3), # Sleek violet line
                        mode="lines+markers",
                        marker=dict(size=6, color="#EC4899") # Pink markers
                    )
                    
                    fig_prog.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        yaxis=dict(range=[-1.1, 1.1], gridcolor="#334155"),
                        xaxis=dict(gridcolor="#334155")
                    )
                    
                    st.plotly_chart(fig_prog, width="stretch")
                else:
                    st.info("Could not calculate narrative segments.")
                    
            # Sentiment-Triggered Visual Highlighter
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader("🟢 Sentiment-Triggered Visual Word Highlighter")
                st.markdown(
                    "Expose the lexical items carrying the strongest sentiment weight. "
                    "Positive emotional signals are highlighted in **soft green**, while negative emotional tones are highlighted in **soft red**."
                )
                
                if not st.session_state.highlighted_html:
                    run_highlight = st.button("🟢 Generate Visual Sentiment Highlights", type="primary", key="run_visual_highlight_btn")
                else:
                    run_highlight = st.button("🔄 Refresh Highlights", type="secondary", key="run_visual_highlight_btn_redo")
                    
                if run_highlight:
                    with st.spinner("Synthesizing lexical sentiment weights..."):
                        highlight_prompt = (
                            f"Read the following news article and wrap the key emotional, polarized, or sentiment-carrying words or short phrases "
                            f"in specific HTML span tags.\n\n"
                            f"Use exactly these tags:\n"
                            f"- For highly POSITIVE words or phrases: `<span style='background-color: rgba(34, 197, 94, 0.25); border-bottom: 2px solid rgb(34, 197, 94); padding: 2px 4px; border-radius: 4px; color: #4ADE80; font-weight: bold;'>[word]</span>`\n"
                            f"- For highly NEGATIVE words or phrases: `<span style='background-color: rgba(239, 68, 68, 0.25); border-bottom: 2px solid rgb(239, 68, 68); padding: 2px 4px; border-radius: 4px; color: #FCA5A5; font-weight: bold;'>[word]</span>`\n\n"
                            f"Do not modify the original text or structure of the article, just wrap the key emotional adjectives or loaded nouns in those tags.\n\n"
                            f"Article Content:\n{st.session_state.article_text[:3000]}"
                        )
                        st.session_state.highlighted_html = utils.query_nararouter(
                            highlight_prompt,
                            system_prompt="You are an expert NLP developer. Output the article text with specific HTML spans injected around positive and negative words."
                        )
                        utils.log_activity("Generated Sentiment Highlights", "Injected CSS highlights around positive and negative terms inside the news text")
                        st.rerun()
                        
                if st.session_state.highlighted_html:
                    st.markdown("---")
                    st.markdown("🔍 **Linguistic Sentiment Map:**")
                    st.markdown(
                        f"<div style='background: #0F172A; padding: 20px; border-radius: 8px; border: 1px solid #1E293B; line-height: 1.8; color: #E2E8F0; max-height: 400px; overflow-y: auto;'>"
                        f"{st.session_state.highlighted_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
            # AI Daily Briefing Script Generator
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader("🎙️ AI Morning Briefing Podcast & Radio Script")
                st.markdown(
                    "Transform static news articles into an interactive, high-energy conversational "
                    "audio briefing script. Perfect for listening or rehearsing as a daily podcast briefing."
                )
                
                if not st.session_state.briefing_script:
                    run_briefing = st.button("🎙️ Generate Conversational Podcast Script", type="primary", key="run_briefing_btn")
                else:
                    run_briefing = st.button("🔄 Re-generate Podcast Script", type="secondary", key="run_briefing_btn_redo")
                    
                if run_briefing:
                    with st.spinner("Drafting witty, dual-host morning news briefing dialogue..."):
                        briefing_prompt = (
                            f"Generate a professional, witty, and highly engaging morning news radio/podcast script based on the following article.\n"
                            f"Create a lively dialogue between two hosts:\n"
                            f"- **Alex (🎙️ Main Anchor):** Highly professional, analytical, and structured.\n"
                            f"- **Sam (🗣️ Co-Host):** Inquisitive, charismatic, and loves asking practical questions.\n\n"
                            f"Format the dialogue using exactly these prefix tags so we can style them programmatically:\n"
                            f"ALEX: [Alex's lines, informative and narrative]\n"
                            f"SAM: [Sam's lines, asking questions or adding excitement]\n\n"
                            f"Include stage directions in brackets like *[laughs]* or *[shocked]* to keep it alive. Write a script of around 6-8 exchanges that covers the core facts, sentiment, and significance of the article.\n\n"
                            f"Article Content:\n{st.session_state.article_text}"
                        )
                        st.session_state.briefing_script = utils.query_nararouter(
                            briefing_prompt,
                            system_prompt="You are a brilliant morning radio show producer and conversational scriptwriter."
                        )
                        utils.log_activity("Generated Podcast Script", f"Drafted a two-host conversational morning briefing script for '{st.session_state.url[:30]}'")
                        st.rerun()
                        
                if st.session_state.briefing_script:
                    st.markdown("---")
                    st.markdown("🗣 **Alex & Sam's Morning Briefing Dialogue Show:**")
                    
                    script_lines = st.session_state.briefing_script.split("\n")
                    for line in script_lines:
                        if line.strip().startswith("ALEX:"):
                            text = line.replace("ALEX:", "").strip()
                            st.markdown(
                                f"<div style='background: #1E293B; border-left: 5px solid #6366F1; padding: 15px; border-radius: 8px; margin-bottom: 12px;'>"
                                f"🎙️ <strong>Alex (Main Anchor):</strong> {text}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        elif line.strip().startswith("SAM:"):
                            text = line.replace("SAM:", "").strip()
                            st.markdown(
                                f"<div style='background: #1E293B; border-left: 5px solid #EC4899; padding: 15px; border-radius: 8px; margin-bottom: 12px;'>"
                                f"🗣️ <strong>Sam (Co-Host):</strong> {text}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        elif line.strip():
                            st.markdown(line)
                            
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Daily Briefing Script (.md)",
                        data=st.session_state.briefing_script,
                        file_name=f"daily_briefing_podcast_script.md",
                        mime="text/markdown"
                    )

# ==================== TAB 4: KEY INSIGHTS & ENTITIES ====================
with tab4:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please choose one of the options below to begin analysis:")
            col_load1, col_load2 = st.columns([1, 1])
            with col_load1:
                st.markdown("#### Option 1: Import News Article")
                st.markdown("Use the **'📰 Article Input'** tab to scrape any online article URL or paste your own raw text.")
            with col_load2:
                st.markdown("#### Option 2: Run Immediate Demo")
                st.markdown("Don't have an article ready? Load a preloaded copy of NASA's James Webb Space Telescope discovery article immediately.")
                if st.button("🚀 Load Preloaded Article", type="primary", width="stretch", key="quick_load_news_tab4"):
                    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                    st.session_state.url = "NASA Space Exploration"
                    st.session_state.nlp_data = None
                    st.session_state.summary = ""
                    st.session_state.llm_sentiment = ""
                    st.session_state.entities = ""
                    st.session_state.quotes = ""
                    st.session_state.chat_history = []
                    st.rerun()
    elif not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### 🔑 Key Insights, Entities, and Quotes Extraction")
        st.markdown(
            "Natural Language Processing (NLP) is not just summarization. It involves separating facts, people, places, and quotes. "
            "Below, we utilize advanced LLM reasoning to extract and group critical entities and citations."
        )
        
        # Generation Button
        if not st.session_state.entities:
            run_btn = st.button("🚀 Extract Named Entities & Quotes", type="primary", width="stretch")
        else:
            run_btn = st.button("🔄 Re-run Entity & Quote Extraction", type="secondary", width="stretch")
            
        if run_btn:
            with st.status("🔍 Extracting critical indicators and structures...", expanded=True) as status:
                st.write("👤 **Step 1:** Parsing sentences and classifying Named Entities (People, Orgs, Places, Concepts)...")
                entity_prompt = (
                    f"Analyze the following news article and extract the main named entities. "
                    f"Identify up to 5 entries for each category. Keep it clear, concise, and structured in clean markdown list groups. "
                    f"Categories to extract:\n"
                    f"- 👤 **Key People Mentioned**:\n"
                    f"- 🏢 **Organizations/Companies**:\n"
                    f"- 📍 **Locations/Places**:\n"
                    f"- 🛠️ **Technologies, Products, or Key Concepts**:\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.entities = utils.query_nararouter(
                    entity_prompt,
                    system_prompt="You are an expert at Named Entity Recognition (NER). Produce clean, structured markdown list blocks."
                )
                
                st.write("💬 **Step 2:** Isolating direct quotes and statements with critical narrative value...")
                quotes_prompt = (
                    f"Identify and extract up to 4 most significant direct quotes, assertions, or key statements from the following news article. "
                    f"Provide the quote, who said it (if available), and a single sentence explaining why it is critical to the article's narrative. "
                    f"Structure the output as beautiful styled blockquotes `> Quote` followed by the citation and context in bullet points.\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.quotes = utils.query_nararouter(
                    quotes_prompt,
                    system_prompt="You are a professional quotes miner. Structure quotes beautifully in markdown blockquotes."
                )
                
                st.write("📖 **Step 3:** Analyzing article vocabulary for specialized jargon and acronyms...")
                jargon_prompt = (
                    f"Scan the following news article and extract any specialized acronyms or dense industry jargon. "
                    f"Create a clean, styled markdown dictionary definitions list (glossary) with up to 5 terms and definitions. "
                    f"If no specific acronyms exist, explain up to 5 of the most complex topical concepts in simple terms.\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.jargon_glossary = utils.query_nararouter(
                    jargon_prompt,
                    system_prompt="You are an expert computational lexicographer. Create clean, beautiful markdown glossary definitions."
                )
                
                st.write("🕸️ **Step 4:** Constructing interactive Entity Relationship Flowchart...")
                map_prompt = (
                    f"Analyze the key connections, interactions, and relationships between the extracted people, organizations, and concepts in this article.\n"
                    f"Create a beautiful, vertically aligned Mermaid.js flowchart mapping these relationships. Keep it simple and clear (4-6 nodes maximum).\n"
                    f"Use exactly this format block:\n"
                    f"```mermaid\n"
                    f"graph TD\n"
                    f"  nodeA[\"Node A\"] -- \"Relationship\" --> nodeB[\"Node B\"]\n"
                    f"```\n"
                    f"Never use parentheses, brackets, or braces inside node names. Do not add any conversational text outside the mermaid block.\n\n"
                    f"Article Content:\n{st.session_state.article_text[:2000]}"
                )
                st.session_state.entity_relation_map = utils.query_nararouter(
                    map_prompt,
                    system_prompt="You are a brilliant computational linguist. Create clean, error-free Mermaid.js flowcharts representing text entity relations."
                )
                
                utils.log_activity("Extracted Key Insights & Glossary", "Extracted people, organizations, locations, direct citations, relationship maps, and a jargon glossary via Mistral 3.5")
                status.update(label="✅ Extraction Complete!", state="complete")
                st.rerun()
                
        # Display results if present
        if st.session_state.entities:
            col_ent, col_quo = st.columns([1, 1])
            with col_ent:
                with st.container(border=True):
                    st.subheader("🔑 Extracted Named Entities")
                    st.markdown(st.session_state.entities)
            with col_quo:
                with st.container(border=True):
                    st.subheader("💬 Key Quotes & Claims")
                    st.markdown(st.session_state.quotes)
                    
            if st.session_state.jargon_glossary:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.subheader("📖 Specialized Jargon & Acronym Glossary")
                    st.markdown("This dictionary lists specialized terms, acronyms, or complex technical concepts extracted from the article and decodes them into plain English:")
                    st.markdown(st.session_state.jargon_glossary)
                    
            if st.session_state.entity_relation_map:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.subheader("🕸️ Interactive Journalistic Entity Relationship Map")
                    st.markdown(
                        "Visualize the critical connections and interactive flow between politicians, organizations, "
                        "locations, and key concepts extracted from this news story."
                    )
                    st.markdown(st.session_state.entity_relation_map)
                    
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download center
            with st.container(border=True):
                st.subheader("📥 Intelligent News AI Download Center")
                st.markdown(
                    "You can download the comprehensive analysis of this article (including pre-processing metrics, "
                    "the abstractive summary, hybrid sentiment analysis, key named entities, and citations) as a "
                    "formatted Executive Report."
                )
                
                # Vader sentiment for compiling report
                v_res = utils.get_vader_sentiment(st.session_state.article_text)
                
                report_md = f"""# 📰 Executive News AI Analysis Report
**Analyzed Article Source:** {st.session_state.url if st.session_state.url else "Manually Pasted Text Block"}

---

## 📝 1. Abstractive Executive Summary
{st.session_state.summary}

---

## ⚙️ 2. Lexical & Preprocessing Metrics
- **Total Raw Words:** {st.session_state.nlp_data["total_words"]} words
- **Detected Sentences:** {st.session_state.nlp_data["total_sentences"]} sentences
- **Lexical Diversity Ratio:** {(len(set(st.session_state.nlp_data["cleaned_tokens"])) / len(st.session_state.nlp_data["cleaned_tokens"]) * 100) if len(st.session_state.nlp_data["cleaned_tokens"]) > 0 else 0:.1f}% unique words

---

## 🎭 3. Hybrid Sentiment Analysis Report
### Classical NLP (NLTK VADER Baseline)
- **Computed Sentiment Label:** {v_res['label']}
- **Compound Polarity Score:** {v_res['scores']['compound']:.4f}
- **VADER Score Breakdown:** Pos: {v_res['scores']['pos']:.2f}, Neu: {v_res['scores']['neu']:.2f}, Neg: {v_res['scores']['neg']:.2f}

### Advanced LLM Insights (Mistral Medium 3.5)
{st.session_state.llm_sentiment}

---

## 👤 4. Identified Named Entities
{st.session_state.entities}

---

## 💬 5. Key Direct Quotes & Statements
{st.session_state.quotes}

---
*Generated by the Intelligent News Summarizer & Sentiment Analysis System.*
"""
                st.download_button(
                    label="📥 Download Professional Executive Report (.md)",
                    data=report_md,
                    file_name="news_ai_executive_report.md",
                    mime="text/markdown",
                    width="stretch"
                )

# ==================== TAB 5: INTERACTIVE Q&A ====================
with tab5:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please choose one of the options below to begin analysis:")
            col_load1, col_load2 = st.columns([1, 1])
            with col_load1:
                st.markdown("#### Option 1: Import News Article")
                st.markdown("Use the **'📰 Article Input'** tab to scrape any online article URL or paste your own raw text.")
            with col_load2:
                st.markdown("#### Option 2: Run Immediate Demo")
                st.markdown("Don't have an article ready? Load a preloaded copy of NASA's James Webb Space Telescope discovery article immediately.")
                if st.button("🚀 Load Preloaded Article", type="primary", width="stretch", key="quick_load_news_tab5"):
                    st.session_state.article_text = SAMPLE_ARTICLE_TEXT
                    st.session_state.url = "NASA Space Exploration"
                    st.session_state.nlp_data = None
                    st.session_state.summary = ""
                    st.session_state.llm_sentiment = ""
                    st.session_state.entities = ""
                    st.session_state.quotes = ""
                    st.session_state.chat_history = []
                    st.rerun()
    elif not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### 💬 Article Q&A Chat Session")
        st.markdown("Engage with the document through standard Prompt Engineering. Ask questions about the timeline, quotes, entities, or implications. The LLM is forced via grounding instructions to stay within the boundaries of the article.")
        
        # Display conversation history
        chat_container = st.container()
        
        with chat_container:
            for speaker, text in st.session_state.chat_history:
                if speaker == "User":
                    st.markdown(f"**👤 You:** {text}")
                else:
                    st.markdown(f"**🤖 Mistral 3.5:**")
                    st.info(text)
                    st.markdown("---")
                    
        # Chat input box
        user_query = st.text_input("Ask a question about this article:", key="news_query_input", placeholder="e.g., What are the main limitations discussed? When did this event occur?")
        
        if st.button("Send Query", type="primary"):
            if user_query.strip():
                # Add to chat history immediately
                st.session_state.chat_history.append(("User", user_query))
                utils.log_activity("Article Q&A Chat Query", f"User asked: '{user_query}'")
                
                # Grounded Prompt Engineering setup
                qa_prompt = (
                    f"You are a helpful and strictly accurate Q&A system. Your task is to answer the user's question "
                    f"accurately using ONLY the facts present in the provided article text. "
                    f"If the answer cannot be found or inferred from the article, say 'I am sorry, but the article does not provide enough information to answer that question.' "
                    f"Do not hallucinate or use any outside knowledge.\n\n"
                    f"Article Text:\n{st.session_state.article_text}\n\n"
                    f"User Question: {user_query}\n\n"
                    f"Answer:"
                )
                
                with st.spinner("Formulating grounded response..."):
                    response = utils.query_nararouter(
                        qa_prompt,
                        system_prompt="You are a strict, grounded reading-comprehension assistant."
                    )
                st.session_state.chat_history.append(("AI", response))
                st.rerun()
                
        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

# ==================== TAB 6: JOURNALISTIC BIAS & FRAMING ====================
with tab6:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please load an article first.")
    elif not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### ⚖️ Cognitive Journalistic Bias & Framing Guardrails")
        st.markdown(
            "Isolate political framing, emotional manipulation, loaded vocabulary, and sensationalism in news writing. "
            "Our system leverages advanced language model heuristics to dissect structural journalism."
        )
        
        if not st.session_state.bias_analysis:
            run_bias_btn = st.button("🚀 Analyze Article Bias & Framing", type="primary", width="stretch")
        else:
            run_bias_btn = st.button("🔄 Re-run Bias & Framing Analysis", type="secondary", width="stretch")
            
        if run_bias_btn:
            with st.spinner("Deconstructing writing style and evaluating journalistic integrity..."):
                bias_prompt = (
                    f"Perform a comprehensive Journalistic Bias and Linguistic Framing analysis on the following article.\n"
                    f"Provide your output strictly in these four sections:\n\n"
                    f"### 1. 📢 Sensationalism & Hyperbole Score (0-10)\n"
                    f"[Provide a numerical rating and a 2-sentence explanation of clickbait patterns, emotional adjectives, or exaggeration density.]\n\n"
                    f"### 2. 🎒 Loaded Vocabulary Tracker\n"
                    f"[Isolate 3 to 5 highly subjective or loaded words/phrases used in the text and explain what they imply or try to lead the reader to believe.]\n\n"
                    f"### 3. ⚖️ Framing Perspective Analysis\n"
                    f"[Explain the narrative framing used. Is the topic presented from a single dominant angle? Are other valid perspectives or counter-arguments ignored?]\n\n"
                    f"### 4. 📝 Recommendations for Neutral Re-Writing\n"
                    f"[Provide a brief neutral version of the headline and the opening paragraph as an objective reporting model.]\n\n"
                    f"Article Content:\n{st.session_state.article_text}"
                )
                st.session_state.bias_analysis = utils.query_nararouter(
                    bias_prompt,
                    system_prompt="You are an expert computational linguist and media bias auditor. Be objective, strict, and precise."
                )
                utils.log_activity("Analyzed Journalistic Bias", f"Evaluated sensationalism, loaded language, and structural narrative framing for '{st.session_state.url[:30]}'")
                st.rerun()
                
        if st.session_state.bias_analysis:
            with st.container(border=True):
                st.markdown("### 🎓 Academic Media Literacy Report")
                st.markdown(st.session_state.bias_analysis)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                with st.container(border=True):
                    st.subheader("✍️ Headline Clickbait Style Transformer")
                    st.markdown("Transform sensational headlines into distinct academic and objective frames:")
                    
                    if not st.session_state.style_transformer:
                        run_style = st.button("✍️ De-Sensationalize Headline", type="primary", key="run_style_transformer")
                    else:
                        run_style = st.button("🔄 Re-Transform Style", type="secondary", key="run_style_transformer_redo")
                        
                    if run_style:
                        with st.spinner("Rewriting journalism headlines..."):
                            style_prompt = (
                                f"Read this article text and identify its core news topic. "
                                f"Draft 3 alternate versions of the headline matching these exact archetypes:\n\n"
                                f"- **🔬 Objective Academic:** Completely passive, data-driven, neutral.\n"
                                f"- **📰 Conservative Neutral:** Traditional objective AP/Reuters journalism standard.\n"
                                f"- **🧐 Skeptical Investigator:** Questioning the premises, looking for hidden motives.\n\n"
                                f"Article Content:\n{st.session_state.article_text}"
                            )
                            st.session_state.style_transformer = utils.query_nararouter(
                                style_prompt,
                                system_prompt="You are an expert editor who hates sensationalism. Transform headlines objectively."
                            )
                            utils.log_activity("Transformed Headline Styles", "Re-wrote clickbait headlines into 3 neutral academic archetypes")
                            st.rerun()
                            
                    if st.session_state.style_transformer:
                        st.markdown("---")
                        st.markdown(st.session_state.style_transformer)
                        
            with col_b2:
                with st.container(border=True):
                    st.subheader("🔍 Google Fact-Check Query Assistant")
                    st.markdown("Extract the core factual claims and construct query strings with advanced operators:")
                    
                    if not st.session_state.factcheck_queries:
                        run_factcheck = st.button("🔍 Construct Fact-Check Queries", type="primary", key="run_factcheck_queries")
                    else:
                        run_factcheck = st.button("🔄 Re-build Queries", type="secondary", key="run_factcheck_queries_redo")
                        
                    if run_factcheck:
                        with st.spinner("Constructing advanced search operator queries..."):
                            fact_prompt = (
                                f"Identify 3 core factual claims or numbers asserted in this article. "
                                f"For each claim, write a brief summary of the claim, followed by an advanced Google Search operator query designed to verify it.\n"
                                f"Example format:\n"
                                f"**Claim 1:** [Claim description]\n"
                                f"*Suggested Search Operator:* `\"Claim key phrase\" site:apnews.com OR site:reuters.com`\n\n"
                                f"Article Content:\n{st.session_state.article_text}"
                            )
                            st.session_state.factcheck_queries = utils.query_nararouter(
                                fact_prompt,
                                system_prompt="You are an expert fact-checker. Provide advanced search strings with specific query parameters."
                            )
                            utils.log_activity("Created Fact-Check Queries", "Generated 3 factual claims and corresponding verification search queries")
                            st.rerun()
                            
                    if st.session_state.factcheck_queries:
                        st.markdown("---")
                        st.markdown(st.session_state.factcheck_queries)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.subheader("🎭 The Stylistic Rewriter & Echo-Chamber Simulator")
                st.markdown(
                    "Observe how the exact same factual event can be framed to create completely distinct narratives. "
                    "Select a target journalistic style below to run the real-time simulation."
                )
                
                selected_style = st.selectbox(
                    "Select Target Editorial Style:",
                    [
                        "Sensationalist Red-Top Tabloid 📰", 
                        "Dense Intellectual Journal 🎓", 
                        "Wall Street Corporate Wire 💼", 
                        "Satirical News Comedy 🎭"
                    ],
                    key="news_style_rewrite_dropdown"
                )
                
                run_rewrite = st.button("🎭 Simulate Stylistic Rewrite", type="primary", key="run_style_rewrite_btn")
                
                if run_rewrite:
                    st.session_state.selected_echo_style = selected_style
                    with st.spinner(f"Drafting style simulation for: {selected_style}..."):
                        rewrite_prompt = (
                            f"Rewrite the following news article to match exactly this editorial style archetype:\n"
                            f"Style: {selected_style}\n\n"
                            f"Guidelines:\n"
                            f"- Tabloid: Exaggerate everything, use capitalizations, heavy emotional trigger words, sensational quotes.\n"
                            f"- Academic Journal: Extremely dry, passive voice, highly structured with research methodology phrasing, completely unemotional.\n"
                            f"- Corporate Wire: Frames every aspect around share price, capital risk, mergers, asset metrics, and legal compliance.\n"
                            f"- Satirical Comedy: Highly witty, deadpan political humor, poking fun at the primary subjects or entities in the article.\n\n"
                            f"Draft a compelling, beautifully formatted news summary in that exact style. Keep it under 250 words.\n\n"
                            f"Article Content:\n{st.session_state.article_text[:2000]}"
                        )
                        st.session_state.echo_chamber_rewrite = utils.query_nararouter(
                            rewrite_prompt,
                            system_prompt=f"You are a master stylistic editor capable of replicating distinct journalistic style templates perfectly."
                        )
                        utils.log_activity("Executed Stylistic Rewrite", f"Simulated editorial framing for '{selected_style}'")
                        st.rerun()
                        
                if st.session_state.echo_chamber_rewrite:
                    st.markdown("---")
                    st.markdown(f"🎭 **Simulated Editorial Outcome ({st.session_state.selected_echo_style}):**")
                    st.markdown(st.session_state.echo_chamber_rewrite)

# ==================== TAB 7: TRANSLATION HUB ====================
with tab7:
    if not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### 🌐 Cross-Lingual Academic Translation Hub")
        st.markdown(
            "Translate the scraped article content and the generated summaries into your desired academic and regional languages. "
            "Our system leverages Mistral 3.5 to construct high-fidelity translations that preserve context, technical jargon, and sentiment tone."
        )
        
        col_translate_1, col_translate_2 = st.columns([1, 2])
        
        with col_translate_1:
            with st.container(border=True):
                st.markdown("##### ⚙️ Translation Settings")
                target_lang = st.selectbox(
                    "Select Target Language:",
                    [
                        "Spanish 🇪🇸", 
                        "French 🇫🇷", 
                        "German 🇩🇪", 
                        "Japanese 🇯🇵", 
                        "Chinese (Simplified) 🇨🇳", 
                        "Arabic 🇸🇦", 
                        "Hindi 🇮🇳", 
                        "Russian 🇷🇺", 
                        "Portuguese 🇵🇹", 
                        "Italian 🇮🇹"
                    ],
                    index=[
                        "Spanish 🇪🇸", 
                        "French 🇫🇷", 
                        "German 🇩🇪", 
                        "Japanese 🇯🇵", 
                        "Chinese (Simplified) 🇨🇳", 
                        "Arabic 🇸🇦", 
                        "Hindi 🇮🇳", 
                        "Russian 🇷🇺", 
                        "Portuguese 🇵🇹", 
                        "Italian 🇮🇹"
                    ].index(st.session_state.selected_lang)
                )
                
                content_to_translate = st.radio(
                    "Select Content to Translate:",
                    ["Executive Summary only", "Complete Scraped Article"]
                )
                
                st.session_state.selected_lang = target_lang
                
                translate_btn = st.button("🌐 Generate High-Fidelity Translation", type="primary", key="translate_btn_run")
                
                if translate_btn:
                    source_text = ""
                    if content_to_translate == "Executive Summary only":
                        if not st.session_state.summary:
                            st.warning("⚠️ No Executive Summary has been generated yet. Please run the Summary pipeline under the 'Summary & Sentiment' tab first!")
                        else:
                            source_text = st.session_state.summary
                    else:
                        source_text = st.session_state.article_text
                        
                    if source_text:
                        with st.spinner(f"Translating into {target_lang} while preserving tone, sentiment, and layout..."):
                            translation_prompt = (
                                f"Translate the following text into {target_lang}. Maintain any markdown headings, list structures, and paragraph layouts. "
                                f"Be natural, accurate, and ensure professional terminology matches the style of quality journalism. Do not add any conversational meta-commentary, return only the translated text.\n\n"
                                f"Text to Translate:\n{source_text}"
                            )
                            st.session_state.translated_content = utils.query_nararouter(
                                translation_prompt,
                                system_prompt=f"You are an expert translator specializing in high-fidelity cross-lingual translation into {target_lang}."
                            )
                            utils.log_activity("Translated Content", f"Translated {content_to_translate} into {target_lang}")
                            st.rerun()
                            
        with col_translate_2:
            with st.container(border=True):
                st.markdown("##### 📄 Translated Output Panel")
                if not st.session_state.translated_content:
                    st.info("Your translation will appear here. Choose your target language in the settings panel and click the generate button.")
                else:
                    st.markdown(f"**Target Language:** {st.session_state.selected_lang}")
                    st.markdown("---")
                    st.markdown(st.session_state.translated_content)
                    st.markdown("---")
                    st.download_button(
                        label=f"📥 Download Translated Output ({st.session_state.selected_lang.split()[0]})",
                        data=st.session_state.translated_content,
                        file_name=f"translated_article_{st.session_state.selected_lang.split()[0].lower()}.md",
                        mime="text/markdown"
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("🌍 Geopolitical Cross-Cultural Framing Neutralizer")
        st.markdown(
            "Geopolitical narratives are often framed very differently by regional media outlets across the globe. "
            "Our engine analyzes localized biases and constructs a fully balanced, 'neutralized' international synthesis."
        )
        
        if not st.session_state.cultural_neutralizer:
            run_neutral = st.button("🌍 Execute Cross-Cultural Bias Analysis", type="primary", key="run_neutralizer_btn")
        else:
            run_neutral = st.button("🔄 Re-neutralize Frame Bias", type="secondary", key="run_neutralizer_btn_redo")
            
        if run_neutral:
            with st.spinner("Simulating regional press covers and extracting objective cross-cultural truth..."):
                neutral_prompt = (
                    f"Analyze the following news article from three major localized journalistic viewpoints:\n"
                    f"1. **Western/North-American Outlets:** (Emphasizing democratic procedures, free market, or corporate/legal angles)\n"
                    f"2. **European/UK Outlets:** (Emphasizing social policy, environmental impact, or public institutional trust)\n"
                    f"3. **Asian/Global-South Outlets:** (Emphasizing regional sovereignty, national growth, or community stability)\n\n"
                    f"Contrast these three simulated framing angles, identify localized media biases, and finally construct a 3-sentence **Global Neutralized coverage Synthesis** that represents the most objective, balanced version of the news story.\n\n"
                    f"Article Content:\n{st.session_state.article_text[:2500]}"
                )
                st.session_state.cultural_neutralizer = utils.query_nararouter(
                    neutral_prompt,
                    system_prompt="You are a global media editor and international press analyst specializing in framing bias neutralization."
                )
                utils.log_activity("Executed Cross-Cultural Frame Neutralizer", "Simulated multi-regional frames and compiled a neutralized media coverage synthesis.")
                st.rerun()
                
        if st.session_state.cultural_neutralizer:
            st.markdown("---")
            st.markdown("🌍 **Geopolitical Frame Audit & Neutralized Synthesis:**")
            st.markdown(st.session_state.cultural_neutralizer)

# ==================== TAB 8: DUAL-ARTICLE COMPARISON ====================
with tab8:
    if not st.session_state.article_text:
        with st.container(border=True):
            st.markdown("### 📰 News AI Analysis Suite: Quick Start Guide")
            st.info("ℹ️ No news article has been loaded yet. Please load a primary article first.")
    elif not api_configured:
        st.error("❌ Please supply your Nararouter API Credentials in the sidebar config panel.")
    else:
        st.markdown("### 🆚 Side-by-Side Dual Article Comparison Workspace")
        st.markdown(
            "Compare how two different publications frame, report, and analyze the same news story. "
            "Our system extracts factual divergences, structural framing variations, and sensationalism ratings side-by-side."
        )
        
        # Inputs for second article
        col_comp1, col_comp2 = st.columns([1, 1])
        with col_comp1:
            st.markdown("#### Article 1 (Primary Loaded)")
            st.info(f"📍 **Source:** {st.session_state.url or 'Manual Paste'}\n\n🏷️ **Length:** {len(st.session_state.article_text.split())} words")
            with st.expander("👁️ View Primary Text Snippet"):
                st.markdown(st.session_state.article_text[:600] + "...")
                
        with col_comp2:
            st.markdown("#### Article 2 (Comparison Target)")
            comp_input_method = st.radio("Choose Input Method for Article 2:", ["Paste Text Manually", "Enter News URL to Scrape"], key="comp_input_choice")
            
            comp_url = ""
            comp_text = ""
            if comp_input_method == "Enter News URL to Scrape":
                comp_url_input = st.text_input("Article 2 URL:", key="comp_url_str")
                if st.button("🕸️ Scrape Comparison Article", type="secondary", key="comp_url_scrape_btn"):
                    if comp_url_input.strip():
                        with st.spinner("🕷️ Crawling Comparison Article..."):
                            try:
                                comp_text = utils.fetch_article_text(comp_url_input)
                                st.session_state.compare_article_text = comp_text
                                st.session_state.compare_url = comp_url_input
                                st.session_state.compare_analysis = "" # clear old
                                utils.log_activity("Scraped Comparison Article", f"Crawled comparison article URL: {comp_url_input}")
                                st.success("✅ Comparison article scraped successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error scraping comparison article: {str(e)}")
            else:
                comp_text_input = st.text_area("Paste Article 2 Content:", key="comp_text_area", height=150)
                if st.button("💾 Save Comparison Article", type="secondary", key="comp_text_save_btn"):
                    if len(comp_text_input.strip()) > 100:
                        st.session_state.compare_article_text = comp_text_input
                        st.session_state.compare_url = "Manual Paste"
                        st.session_state.compare_analysis = "" # clear old
                        utils.log_activity("Saved Comparison Article", f"Saved manual comparison article text ({len(comp_text_input.strip().split())} words)")
                        st.success("✅ Comparison article saved!")
                        st.rerun()
                        
            if st.session_state.compare_article_text:
                st.success(f"📍 **Source:** {st.session_state.compare_url}\n\n🏷️ **Length:** {len(st.session_state.compare_article_text.split())} words")
                
        # Run Comparison Button
        if st.session_state.compare_article_text:
            st.markdown("---")
            if not st.session_state.compare_analysis:
                run_comp = st.button("🚀 Analyze & Compare Articles", type="primary", use_container_width=True)
            else:
                run_comp = st.button("🔄 Re-run Comparison Analysis", type="secondary", use_container_width=True)
                
            if run_comp:
                with st.spinner("Synthesizing, deconstructing, and framing coverage divergence side-by-side..."):
                    comp_prompt = (
                        f"Perform a comprehensive side-by-side journalistic comparison of these two news articles covering the same event.\n"
                        f"Deconstruct differences in fact-reporting, loaded terminology, narrative framing, and media sensationalism.\n\n"
                        f"Structure your response strictly with these sections:\n\n"
                        f"### 📊 Comparative Overview Table\n"
                        f"Provide a Markdown comparative table comparing: Headline, Word Count, Primary Tone, Sensationalism Rating (0-10), and Main Target Audience.\n\n"
                        f"### 🔍 Factual & Data Coverage Divergence\n"
                        f"[Identify any crucial factual assertions, statistics, or sources present in one article but completely omitted or downplayed in the other.]\n\n"
                        f"### ⚖️ Framing & Rhetorical Bias Contrast\n"
                        f"[Contrast how each article frames the story. For example, does one focus on economic impact while the other focuses on environmental consequences? Is one sympathetic to a specific actor?]\n\n"
                        f"### 📝 Key Editorial Recommendation\n"
                        f"[Provide a 2-sentence synthesis recommending how a critical reader can combine both coverages to form a completely neutral, objective opinion.]\n\n"
                        f"Article 1 (Primary):\n{st.session_state.article_text[:2500]}\n\n"
                        f"Article 2 (Comparison Target):\n{st.session_state.compare_article_text[:2500]}"
                    )
                    st.session_state.compare_analysis = utils.query_nararouter(
                        comp_prompt,
                        system_prompt="You are a senior computational linguist and media bias comparison specialist."
                    )
                    utils.log_activity("Executed Dual Article Comparison", f"Compared framing, bias, and coverage divergence side-by-side.")
                    st.rerun()
                    
            if st.session_state.compare_analysis:
                with st.container(border=True):
                    st.markdown("### 🎓 Peer-Review Dual-Article Comparison Report")
                    st.markdown(st.session_state.compare_analysis)
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download Comparative Report (.md)",
                        data=st.session_state.compare_analysis,
                        file_name="dual_article_comparative_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
