import sys
import os
import hashlib
import streamlit as st
import pandas as pd
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_csv, get_data_summary
from agent.agent import DataMindAgent

st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .user-msg {
        background: rgba(102, 126, 234, 0.15);
        border-left: 4px solid #667eea;
    }
    .assistant-msg {
        background: rgba(118, 75, 162, 0.15);
        border-left: 4px solid #764ba2;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #0e1117;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "eda_done" not in st.session_state:
        st.session_state.eda_done = False
    if "eda_result" not in st.session_state:
        st.session_state.eda_result = None
    if "eda_figures" not in st.session_state:
        st.session_state.eda_figures = []
    if "ml_done" not in st.session_state:
        st.session_state.ml_done = False
    if "ml_result" not in st.session_state:
        st.session_state.ml_result = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "csv_hash" not in st.session_state:
        st.session_state.csv_hash = None


def render_header():
    st.markdown(
        """
    <div class="main-header">
        <h1>DataMind AI</h1>
        <p>Votre analyste de donnees propulse par l'IA</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")

        api_key = st.text_input(
            "Cle API (z.Ai / OpenAI)",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Votre cle API z.Ai ou OpenAI",
        )
        base_url = st.text_input(
            "URL de base",
            value=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
        )
        model = st.text_input(
            "Modele",
            value=os.getenv("OPENAI_MODEL", "glm-5.1"),
        )

        if api_key:
            config_changed = (
                os.getenv("OPENAI_API_KEY") != api_key
                or os.getenv("OPENAI_BASE_URL") != base_url
                or os.getenv("OPENAI_MODEL") != model
            )

            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_MODEL"] = model

            if config_changed and st.session_state.agent is not None:
                st.session_state.agent.reload_config()

            if st.session_state.df is not None and st.session_state.agent is None:
                st.session_state.agent = DataMindAgent()
                st.session_state.agent.set_data(st.session_state.df)

        st.markdown("---")
        st.markdown("### Jeu de donnees")

        uploaded_file = st.file_uploader(
            "Charger un fichier CSV",
            type=["csv"],
            help="Chargez votre fichier CSV pour l'analyse",
        )

        if uploaded_file:
            try:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()

                if file_hash != st.session_state.get("csv_hash"):
                    st.session_state.csv_hash = file_hash
                    df = load_csv(uploaded_file)
                    st.session_state.df = df

                    if st.session_state.agent is None:
                        st.session_state.agent = DataMindAgent()
                    st.session_state.agent.set_data(df)

                    st.session_state.chat_history = []
                    st.session_state.eda_done = False
                    st.session_state.eda_result = None
                    st.session_state.eda_figures = []
                    st.session_state.ml_done = False
                    st.session_state.ml_result = None

                if st.session_state.df is not None:
                    df = st.session_state.df
                    summary = get_data_summary(df)

                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <strong>{summary["shape"][0]:,}</strong> lignes<br>
                        <strong>{summary["shape"][1]}</strong> colonnes<br>
                        <strong>{summary["memory_mb"]}</strong> Mo
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Colonnes"):
                        for col in df.columns:
                            dtype = str(df[col].dtype)
                            missing = summary["missing_pct"].get(col, 0)
                            badge_color = (
                                "#2ecc71"
                                if missing == 0
                                else "#e74c3c"
                                if missing > 20
                                else "#f39c12"
                            )
                            st.markdown(
                                f"**{col}** `{dtype}` "
                                f'<span class="status-badge" style="background:{badge_color};color:white">'
                                f"{missing:.1f}% manquants</span>",
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")

        st.markdown("---")
        st.markdown(
            """
        <div style="text-align:center;opacity:0.5;font-size:0.8rem;">
            DataMind AI v1.0<br>
            Propulse par z.Ai
        </div>
        """,
            unsafe_allow_html=True,
        )

        return uploaded_file


def render_data_preview():
    if st.session_state.df is not None:
        tab1, tab2 = st.tabs(["Apercu des donnees", "Statistiques"])
        with tab1:
            df = st.session_state.df
            st.dataframe(
                df.head(20),
                width="stretch",
                height=300,
            )
        with tab2:
            df = st.session_state.df
            numeric_df = df.describe()
            st.dataframe(
                numeric_df,
                width="stretch",
            )


def render_auto_eda():
    if st.session_state.df is not None and st.session_state.agent is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Analyse exploratoire automatique des donnees")
        with col2:
            if st.button("Lancer l'EDA", type="primary", width="stretch"):
                with st.spinner(
                    "L'agent analyse vos donnees... Cela peut prendre 30 a 60 secondes."
                ):
                    try:
                        result = st.session_state.agent.auto_eda()
                        st.session_state.eda_result = result["message"]
                        st.session_state.eda_figures = result["figures"]
                        st.session_state.eda_done = True
                    except Exception as e:
                        st.error(f"Erreur lors de l'EDA : {e}")
                        return

        if st.session_state.eda_done and st.session_state.eda_result:
            st.markdown("---")
            st.markdown(st.session_state.eda_result)

            if st.session_state.eda_figures:
                for fig in st.session_state.eda_figures:
                    st.plotly_chart(fig, width="stretch")


def render_ml_suggestion():
    if st.session_state.df is not None and st.session_state.agent is not None:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Suggestion de pipeline Machine Learning")
        with col2:
            if st.button("Generer le rapport ML", type="primary", width="stretch"):
                with st.spinner(
                    "Analyse du dataset et generation des recommandations ML..."
                ):
                    try:
                        from agent.tools import execute_tool

                        result = execute_tool(
                            "suggest_ml_pipeline", {}, st.session_state.df
                        )
                        if result["success"]:
                            st.session_state.ml_result = result["text"]
                            st.session_state.ml_done = True
                        else:
                            st.error(f"Erreur : {result['text']}")
                    except Exception as e:
                        st.error(f"Erreur lors de la suggestion ML : {e}")
                        return

        if st.session_state.ml_done and st.session_state.ml_result:
            st.markdown(st.session_state.ml_result)


def render_chat():
    if st.session_state.df is not None and st.session_state.agent is not None:
        st.markdown("---")
        st.markdown("### Discutez avec vos donnees")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                <div class="chat-message user-msg">
                    <strong>Vous :</strong> {msg["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="chat-message assistant-msg">
                    <strong>DataMind :</strong><br>
                    {msg["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
                if msg.get("figures"):
                    for fig in msg["figures"]:
                        st.plotly_chart(fig, width="stretch")

        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Posez une question sur vos donnees :",
                    placeholder="Ex : Quelles sont les variables les plus correlees ?",
                    label_visibility="collapsed",
                )
            with col2:
                send = st.form_submit_button("Envoyer", type="primary")

        quick_actions = st.columns(5)
        prompts = [
            "Decrire le jeu de donnees",
            "Afficher la carte de correlations",
            "Detecter les valeurs aberrantes",
            "Trouver les 5 motifs les plus interessants",
            "Sugg\u00e9rer une pipeline ML",
        ]
        for i, (col, prompt) in enumerate(zip(quick_actions, prompts)):
            with col:
                if st.button(prompt, key=f"quick_{i}", width="stretch"):
                    st.session_state.pending_prompt = prompt

        if st.session_state.get("pending_prompt"):
            user_input = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            send = True

        if send and user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Reflexion..."):
                try:
                    result = st.session_state.agent.chat(user_input)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["message"],
                            "figures": result.get("figures", []),
                        }
                    )
                except Exception as e:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"Erreur : {e}",
                            "figures": [],
                        }
                    )

            st.rerun()


def render_welcome():
    if st.session_state.df is None:
        st.markdown(
            """
        <div style="text-align:center;padding:3rem;">
            <h2>Bienvenue sur DataMind AI</h2>
            <p style="font-size:1.1rem;opacity:0.7;">
                Chargez un fichier CSV dans la barre laterale pour commencer.
            </p>
            <p style="font-size:1rem;opacity:0.5;">
                Votre agent IA va analyser automatiquement vos donnees,<br>
                generer des visualisations et repondre a vos questions.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    init_session_state()
    render_header()
    render_sidebar()

    if st.session_state.df is not None:
        render_data_preview()
        st.markdown("---")
        render_auto_eda()
        render_ml_suggestion()
        render_chat()
    else:
        render_welcome()


if __name__ == "__main__":
    main()
