import streamlit as st
import os
import fitz  # PyMuPDF
import requests
import re
from dotenv import load_dotenv, find_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import tempfile
import base64
from typing import List, Dict, Any, Tuple

# Libellés de navigation (évite les divergences de chaînes)
MENU_PAGE_GENERATE = "Générer des exercices"
MENU_PAGE_RESPOND = "Répondre aux exercices"
MENU_PAGE_CHAT = "Chatbot PDF"
MENU_PAGE_DIAG = "🔧 Diagnostic API"

AVAILABLE_MODELS = [
    "anthropic/claude-3-opus",
    "openai/gpt-4-turbo",
    "mistralai/mistral-large",
    "cohere/command-r-plus",
    "google/gemini-pro",
    "anthropic/claude-3-haiku",
]

st.set_page_config(
    page_title="Générateur d'Exercices PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des variables d'environnement
env_path = find_dotenv(filename=".env", usecwd=True)
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("OPENROUTER_API_KEY")

# Fonction pour lire le PDF
def lire_pdf(uploaded_file):
    """Lit le contenu d'un PDF uploadé"""
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Lire le PDF
        doc = fitz.open(tmp_file_path)
        texte = ""
        for i, page in enumerate(doc, start=1):
            texte_page = page.get_text()
            texte += f"\n\n=== [PAGE {i}] ===\n" + texte_page.strip()
        
        # Nettoyer le fichier temporaire
        os.unlink(tmp_file_path)
        return texte
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF: {str(e)}")
        return None

# Fonction pour appeler l'API OpenRouter
def reponse(consignes, texte):
    """Appelle l'API OpenRouter pour générer du contenu"""
    if not api_key:
        st.error("❌ Clé API manquante. Vérifiez votre fichier .env")
        st.info("💡 Assurez-vous que votre fichier .env contient : OPENROUTER_API_KEY=votre_cle_ici")
        return None
    
    # Vérifier le format de la clé API
    if not api_key.startswith('sk-'):
        st.warning("⚠️ Format de clé API suspect. Les clés OpenRouter commencent généralement par 'sk-'")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8501/",
        "Content-Type": "application/json"
    }
    
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    # Sélection du modèle depuis la session si défini
    modele = st.session_state.get("selected_model", AVAILABLE_MODELS[0])
    
    payload = {
        "model": modele,
        "messages": [
            {"role": "system", "content": consignes},
            {"role": "user", "content": texte}
        ],
        "max_tokens": 4000,  # Augmenté pour le nouveau modèle
        "temperature": 0.7   # Ajouté pour une meilleure génération
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        # Vérifier le code de statut HTTP
        if response.status_code != 200:
            # Essayer un modèle de secours si disponible
            fallback_models = [m for m in AVAILABLE_MODELS if m != modele]
            if fallback_models:
                st.warning(f"Le modèle '{modele}' a échoué ({response.status_code}). Essai avec '{fallback_models[0]}'...")
                st.session_state["selected_model"] = fallback_models[0]
                return reponse(consignes, texte)
            st.error(f"Erreur HTTP {response.status_code}: {response.text}")
            return None
        
        response_json = response.json()
        
        # Debug: afficher la réponse complète en cas d'erreur
        if 'error' in response_json:
            error_msg = response_json['error'].get('message', 'Erreur inconnue')
            error_type = response_json['error'].get('type', 'Unknown')
            st.error(f"Erreur API ({error_type}): {error_msg}")
            
            # Afficher plus de détails pour le debug
            if st.checkbox("🔍 Afficher les détails de l'erreur"):
                st.json(response_json)
            return None
        
        # Vérifier si la réponse contient des choix
        if 'choices' not in response_json or not response_json['choices']:
            st.error("Aucune réponse générée par l'API")
            if st.checkbox("🔍 Afficher la réponse complète"):
                st.json(response_json)
            return None
        
        return response_json['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        st.error("Vérifiez votre connexion internet et votre clé API")
        return None
    except ValueError as e:
        st.error(f"Erreur de parsing JSON: {str(e)}")
        st.error(f"Réponse brute: {response.text[:500]}...")
        return None
    except KeyError as e:
        st.error(f"Erreur de format de réponse API: {str(e)}")
        if st.checkbox("🔍 Afficher la réponse reçue"):
            st.json(response_json)
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel API: {str(e)}")
        return None

# Fonction pour créer un PDF QCM
def create_proper_pdf(resume, nom_de_la_matiere):
    """Crée un PDF QCM"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        doc = SimpleDocTemplate(tmp_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Styles personnalisés
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            leading=14
        )
        
        option_style = ParagraphStyle(
            'Option',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20,
            spaceAfter=2,
            textColor=colors.darkblue,
            leading=12
        )
        
        story = []
        
        # Titre principal
        story.append(Paragraph("Epreuve de " + nom_de_la_matiere, title_style))
        story.append(Spacer(1, 15))
        story.append(Paragraph(" ", styles['Normal']))
        story.append(Spacer(1, 25))
        
        # Traiter le contenu
        lines = resume.strip().split('\n')
        current_question = ""
        current_options = []
        question_number = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Détecter si c'est une question
            if line[0].isdigit() and '.' in line:
                if current_question:
                    story.append(Paragraph(current_question, question_style))
                    story.append(Spacer(1, 5))
                    
                    for option in current_options:
                        story.append(Paragraph(option, option_style))
                    
                    story.append(Spacer(1, 15))
                    
                    if question_number % 5 == 0:
                        story.append(PageBreak())
                
                current_question = line
                current_options = []
                question_number += 1
                
            # Détecter si c'est une option
            elif line.startswith('A)') or line.startswith('B)') or line.startswith('C)') or line.startswith('D)'):
                current_options.append(line)
            else:
                if current_options:
                    current_options[-1] += " " + line
                else:
                    current_question += " " + line
        
        # Ajouter la dernière question
        if current_question:
            story.append(Paragraph(current_question, question_style))
            story.append(Spacer(1, 5))
            
            for option in current_options:
                story.append(Paragraph(option, option_style))
        
        doc.build(story)
        return tmp_file.name

# Fonction pour créer un PDF d'épreuves
def create_proper_pdf_epreuves(resume, nom_de_la_matiere):
    """Crée un PDF d'épreuves"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        doc = SimpleDocTemplate(tmp_file.name, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Styles personnalisés
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        exercise_title_style = ParagraphStyle(
            'ExerciseTitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=25,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            leading=14,
            fontName='Helvetica-Bold'
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20,
            spaceAfter=12,
            textColor=colors.grey,
            leading=12
        )
        
        context_style = ParagraphStyle(
            'Context',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leading=13,
            fontName='Helvetica'
        )
        
        story = []
        
        # Titre principal
        story.append(Paragraph("Épreuves " + nom_de_la_matiere, title_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(" ", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Traiter le contenu
        lines = resume.strip().split('\n')
        current_exercise = ""
        current_context = ""
        current_question = ""
        current_table_lines = []
        exercise_number = 0
        question_number = 0
        in_table = False
        
        for line in lines:
            if not line.strip():
                continue
                
            # Détecter le début d'un nouvel exercice
            if line.lower().startswith('exercice') or line.startswith('## Exercice'):
                if current_exercise:
                    if current_question:
                        story.append(Paragraph(current_question, question_style))
                        for i in range(6):
                            story.append(Paragraph("_______________________________________________________", answer_style))
                        story.append(Spacer(1, 15))
                        current_question = ""
                    story.append(Spacer(1, 20))
                    
                current_exercise = line.replace('## ', '').replace('# ', '')
                exercise_number += 1
                question_number = 0
                story.append(Paragraph(current_exercise, exercise_title_style))
                
            # Détecter un tableau
            elif '|' in line:
                if not in_table:
                    current_table_lines = []
                    in_table = True
                current_table_lines.append(line)
                
            # Fin d'un tableau
            elif in_table and '|' not in line:
                in_table = False
                if current_table_lines:
                    table_data = []
                    for table_line in current_table_lines:
                        row = [cell.strip() for cell in table_line.split('|') if cell.strip() != ""]
                        table_data.append(row)
                    
                    tbl = Table(table_data, hAlign='LEFT')
                    tbl.setStyle(TableStyle([
                        ('FONTNAME', (0,0), (-1,-1), 'Courier'),
                        ('FONTSIZE', (0,0), (-1,-1), 9),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ]))
                    
                    story.append(tbl)
                    story.append(Spacer(1, 10))
                    current_table_lines = []
                
            # Détecter le contexte/énoncé
            elif (
                not any(char.isdigit() for char in line)
                and not line.lower().startswith(("calculez", "déterminez", "concluez"))
                and not line.startswith("Réponse")
                and not line.startswith("##")
                and current_exercise
                and not current_question
            ):
                if "Analyse de" in line or "Échantillon de" in line:
                    current_context += line + " "

            # Détecter une sous-question
            elif any([line.lower().startswith('calculez'), line.lower().startswith('déterminez'), line.lower().startswith('concluez'), (line[0].isdigit() and ')' in line)]):
                if current_context and question_number == 0:
                    story.append(Paragraph(current_context.strip(), context_style))
                    story.append(Spacer(1, 10))
                    current_context = ""
                    
                if current_question:
                    story.append(Paragraph(current_question, question_style))
                    for i in range(6):
                        story.append(Paragraph("_______________________________________________________", answer_style))
                    story.append(Spacer(1, 15))
                
                current_question = line
                question_number += 1
                
            else:
                if current_question and not in_table:
                    if '|' not in line:
                        current_question += " " + line
                elif current_exercise and not in_table and current_context:
                    current_context += " " + line
        
        # Ajouter le dernier tableau s'il existe
        if in_table and current_table_lines:
            table_data = []
            for table_line in current_table_lines:
                row = [cell.strip() for cell in table_line.split('|') if cell.strip() != ""]
                table_data.append(row)
            tbl = Table(table_data, hAlign='LEFT')
            tbl.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,-1), 'Courier'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 10))
        
        if current_question:
            story.append(Paragraph(current_question, question_style))
            for i in range(6):
                story.append(Paragraph("_______________________________________________________", answer_style))
        
        if current_context and question_number == 0:
            story.append(Paragraph(current_context.strip(), context_style))
        
        doc.build(story)
        return tmp_file.name

# Interface principale
def main():
    st.title("📚 Générateur d'Exercices PDF")
    st.markdown("---")
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = MENU_PAGE_GENERATE
    page = st.sidebar.selectbox(
        "Choisir une fonctionnalité",
        [MENU_PAGE_GENERATE, MENU_PAGE_CHAT, MENU_PAGE_DIAG],
        key="nav_page"
    )

    # ===== Chatbot PDF rapide dans la sidebar (AVEC historique) =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Chatbot PDF rapide")
    if "sidebar_chatbot_history" not in st.session_state:
        st.session_state["sidebar_chatbot_history"] = []
    chatbot_prompt = st.sidebar.text_input("Posez une question sur le PDF ici...", key="sidebar_chatbot_prompt")
    if chatbot_prompt:
        texte_pdf = st.session_state.get("source_pdf_text", "")
        if texte_pdf:
            nom_matiere = st.session_state.get("generated_meta", {}).get("matiere", "Mathématiques")
            niveau = st.session_state.get("generated_meta", {}).get("niveau", "L1")
            filiere = st.session_state.get("generated_meta", {}).get("filiere", "")
            consignes_questions = generate_chatbot_instructions(nom_matiere, niveau, filiere)
            with st.sidebar:
                with st.spinner("Recherche dans le PDF..."):
                    response = reponse(consignes_questions, f"Question : {chatbot_prompt}\n\nTexte PDF :\n{texte_pdf}")
                    if response:
                        st.session_state["sidebar_chatbot_history"].append({"question": chatbot_prompt, "response": response})
                        st.sidebar.success("Réponse :")
                        st.sidebar.markdown(response)
                    else:
                        st.sidebar.error("❌ Erreur lors de la génération de la réponse")
        else:
            st.sidebar.info("Chargez d'abord un PDF dans l'application.")

    # Affichage de l'historique dans la sidebar
    if st.session_state["sidebar_chatbot_history"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕑 Historique Chatbot")
        for item in st.session_state["sidebar_chatbot_history"]:
            st.sidebar.markdown(f"**Question :** {item['question']}")
            st.sidebar.markdown(f"**Réponse :** {item['response']}")
            st.sidebar.markdown("---")

    # Section diagnostic API
    if page == "🔧 Diagnostic API":
        diagnostic_page()
        return
    
    if page == MENU_PAGE_GENERATE:
        generate_exercises_page()
    elif page == MENU_PAGE_CHAT:
        chatbot_page()
    else:
        pass

# ...existing code...

def generate_exercises_page():
    st.header("🎯 Génération d'Exercices")
    if "show_interactive" not in st.session_state:
        st.session_state.show_interactive = False

    # Contrôles sous la barre de menu pour accéder au mode réponse ou au chatbot
    mode = st.radio(
        "Mode d'affichage",
        ["Génération", "Répondre aux exercices", "Chatbot PDF"],
        horizontal=True,
        key="mode_affichage_radio"
    )

    if mode == "Répondre aux exercices":
        respond_page()
        st.markdown("---")
        st.stop()
    elif mode == "Chatbot PDF":
        chatbot_page()
        st.markdown("---")
        st.stop()

    # Sélecteur de modèle (raccourci) sur la page de génération
    st.caption("Modèle utilisé pour la génération")
    current_model = st.session_state.get("selected_model", AVAILABLE_MODELS[0])
    st.session_state["selected_model"] = st.selectbox(
        "Modèle",
        AVAILABLE_MODELS,
        index=max(0, AVAILABLE_MODELS.index(current_model)),
        key="model_select_generate"
    )
    
    # Upload du PDF
    uploaded_file = st.file_uploader("📄 Téléchargez votre PDF", type=['pdf'])
    use_previous_pdf = False
    if not uploaded_file and st.session_state.get("source_pdf_text"):
        use_previous_pdf = st.checkbox("Utiliser le PDF déjà chargé précédemment", value=True)
    
    if uploaded_file is not None or use_previous_pdf:
        # Lecture du PDF
        with st.spinner("Lecture du PDF en cours..."):
            if uploaded_file is not None:
                texte = lire_pdf(uploaded_file)
            else:
                texte = st.session_state.get("source_pdf_text")
        
        if texte:
            st.success("✅ PDF lu avec succès !")
            
            # Configuration des paramètres
            col1, col2 = st.columns(2)
            
            with col1:
                regen_params = st.session_state.get("regen_params", {})
                default_matiere = regen_params.get("matiere", "Mathématiques")
                default_niveau = regen_params.get("niveau", "L1")
                nom_matiere = st.text_input("📚 Nom de la matière", value=default_matiere)
                niveau_options = ["6ème", "5ème", "4ème", "3ème", "2nde", "1ère", "Terminale", 
                                  "L1", "L2", "L3", "M1", "M2", "Doctorat"]
                try:
                    default_niveau_index = max(0, niveau_options.index(default_niveau))
                except ValueError:
                    default_niveau_index = 0
                niveau = st.selectbox("🎓 Niveau scolaire", niveau_options, index=default_niveau_index)
                default_filiere = regen_params.get("filiere", "")
                filiere = ""
                if niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
                    filiere = st.text_input("🎯 Filière/Spécialité", 
                                          value=default_filiere,
                                          placeholder="Ex: Informatique, Mathématiques, Physique, etc.",
                                          help="Précisez votre filière pour des exercices plus adaptés")
            
            with col2:
                type_options = ["QCM", "QRO", "Épreuves"]
                regen_params = st.session_state.get("regen_params", {})
                default_type = regen_params.get("type_exercice", "QCM")
                try:
                    default_type_index = max(0, type_options.index(default_type))
                except ValueError:
                    default_type_index = 0
                type_exercice = st.selectbox("📝 Type d'exercice", type_options, index=default_type_index)
                default_nb = int(regen_params.get("nb_questions", 20))
                nb_questions = st.slider("🔢 Nombre de questions", 5, 30, default_nb)
            
            # Génération automatique si demandé
            if st.session_state.get("auto_generate"):
                with st.spinner("Génération en cours..."):
                    if type_exercice == "QCM":
                        consignes = generate_qcm_instructions(nom_matiere, niveau, nb_questions, filiere)
                    elif type_exercice == "QRO":
                        consignes = generate_qro_instructions(nom_matiere, niveau, nb_questions, filiere)
                    else:
                        consignes = generate_epreuves_instructions(nom_matiere, niveau, filiere)
                    resume = reponse(consignes, texte)
                st.session_state["auto_generate"] = False
                if resume:
                    st.session_state["generated_text"] = resume
                    st.session_state["generated_type"] = type_exercice
                    st.session_state["generated_meta"] = {
                        "matiere": nom_matiere,
                        "niveau": niveau,
                        "filiere": filiere,
                    }
                    st.session_state["source_pdf_text"] = texte
                    st.session_state["answers"] = {}
                    st.session_state["persisted_questions_text"] = resume
                    st.session_state["persisted_questions_type"] = type_exercice
                    if type_exercice == "Épreuves":
                        pdf_path = create_proper_pdf_epreuves(resume, nom_matiere)
                    else:
                        pdf_path = create_proper_pdf(resume, nom_matiere)
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📥 Télécharger le PDF",
                        data=pdf_data,
                        file_name=f"Sujet_{nom_matiere}_{type_exercice}.pdf",
                        mime="application/pdf"
                    )
                    os.unlink(pdf_path)
                    st.subheader("📋 Aperçu du contenu généré")
                    st.text_area("Contenu", resume, height=300)
                else:
                    st.error("❌ Erreur lors de la génération")

            # Bouton de génération
            if st.button("🚀 Générer les exercices", type="primary"):
                with st.spinner("Génération en cours..."):
                    if type_exercice == "QCM":
                        consignes = generate_qcm_instructions(nom_matiere, niveau, nb_questions, filiere)
                    elif type_exercice == "QRO":
                        consignes = generate_qro_instructions(nom_matiere, niveau, nb_questions, filiere)
                    else:
                        consignes = generate_epreuves_instructions(nom_matiere, niveau, filiere)
                    
                    resume = reponse(consignes, texte)
                    
                    if resume:
                        st.session_state["generated_text"] = resume
                        st.session_state["generated_type"] = type_exercice
                        st.session_state["generated_meta"] = {
                            "matiere": nom_matiere,
                            "niveau": niveau,
                            "filiere": filiere,
                        }
                        st.session_state["source_pdf_text"] = texte
                        st.session_state["answers"] = {}
                        st.session_state["persisted_questions_text"] = resume
                        st.session_state["persisted_questions_type"] = type_exercice
                        if type_exercice == "Épreuves":
                            pdf_path = create_proper_pdf_epreuves(resume, nom_matiere)
                        else:
                            pdf_path = create_proper_pdf(resume, nom_matiere)
                        with open(pdf_path, "rb") as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="📥 Télécharger le PDF",
                            data=pdf_data,
                            file_name=f"Sujet_{nom_matiere}_{type_exercice}.pdf",
                            mime="application/pdf"
                        )
                        os.unlink(pdf_path)
                        st.subheader("📋 Aperçu du contenu généré")
                        st.text_area("Contenu", resume, height=300)
                    else:
                        st.error("❌ Erreur lors de la génération")

    # Le mode de réponse se fait désormais uniquement via l'onglet dédié

def respond_page():
    st.header("✍️ Répondre aux exercices")
    # Récupérer prioritairement les questions persistées si présentes
    if st.session_state.get("persisted_questions_text") and not st.session_state.get("generated_text"):
        st.session_state["generated_text"] = st.session_state["persisted_questions_text"]
        st.session_state["generated_type"] = st.session_state.get("persisted_questions_type", st.session_state.get("generated_type", ""))

    if not st.session_state.get("generated_text"):
        st.info("Aucun exercice en mémoire. Vous pouvez soit générer un sujet, soit coller un texte généré ici.")
        with st.form("manual_load_form"):
            manual_type = st.selectbox("Type d'exercice", ["QCM", "QRO", "Épreuves"], key="manual_type")
            manual_text = st.text_area("Collez ici le contenu généré (questions)", height=250, key="manual_text")
            col1, col2 = st.columns(2)
            with col1:
                mat = st.text_input("Matière (optionnel)", key="manual_mat")
            with col2:
                niv = st.text_input("Niveau (optionnel)", key="manual_niv")
            submitted = st.form_submit_button("Charger ce sujet")
        if submitted and manual_text.strip():
            st.session_state["generated_text"] = manual_text.strip()
            st.session_state["generated_type"] = manual_type
            st.session_state["generated_meta"] = {"matiere": mat, "niveau": niv, "filiere": ""}
            if "answers" not in st.session_state:
                st.session_state["answers"] = {}
            st.success("Sujet chargé. Vous pouvez répondre ci-dessous.")
            st.rerun()
        else:
            return
    render_interactive_exercises()

def chatbot_page():
    st.header("🤖 Chatbot PDF")
    # Sélecteur de modèle rapide
    st.caption("Modèle utilisé pour le chatbot")
    current_model = st.session_state.get("selected_model", AVAILABLE_MODELS[0])
    st.session_state["selected_model"] = st.selectbox(
        "Modèle",
        AVAILABLE_MODELS,
        index=max(0, AVAILABLE_MODELS.index(current_model)),
        key="model_select_chat"
    )
    
    # Si un PDF est déjà chargé depuis "Générer des exercices", l'utiliser directement
    if st.session_state.get("source_pdf_text"):
        st.success("✅ PDF déjà chargé depuis 'Générer des exercices'. Vous pouvez poser vos questions.")
        texte = st.session_state["source_pdf_text"]
        
        col1, col2 = st.columns(2)
        with col1:
            nom_matiere = st.text_input("📚 Matière", value=(st.session_state.get("generated_meta", {}).get("matiere", "Mathématiques")), key="chatbot_matiere")
            niveau = st.selectbox("🎓 Niveau", 
                                ["6ème", "5ème", "4ème", "3ème", "2nde", "1ère", "Terminale", 
                                 "L1", "L2", "L3", "M1", "M2", "Doctorat"],
                                 index= ["6ème","5ème","4ème","3ème","2nde","1ère","Terminale","L1","L2","L3","M1","M2","Doctorat"].index(st.session_state.get("generated_meta", {}).get("niveau", "L1")),
                                 key="chatbot_niveau")
        with col2:
            filiere_chatbot = ""
            if niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
                filiere_chatbot = st.text_input("🎯 Filière/Spécialité", 
                                              value=st.session_state.get("generated_meta", {}).get("filiere", ""),
                                              placeholder="Ex: Informatique, Mathématiques, Physique, etc.",
                                              help="Précisez votre filière pour des réponses plus adaptées",
                                              key="chatbot_filiere")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Posez votre question sur le PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Recherche dans le PDF..."):
                    consignes_questions = generate_chatbot_instructions(nom_matiere, niveau, filiere_chatbot)
                    response = reponse(consignes_questions, f"Question : {prompt}\n\nTexte PDF :\n{texte}")
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("❌ Erreur lors de la génération de la réponse")
        return

    # Sinon, proposer l’upload (fallback)
    uploaded_file = st.file_uploader("📄 Téléchargez votre PDF", type=['pdf'], key="chatbot_upload")
    if uploaded_file is not None:
        with st.spinner("Lecture du PDF en cours..."):
            texte = lire_pdf(uploaded_file)
        if texte:
            st.success("✅ PDF chargé ! Vous pouvez maintenant poser vos questions.")
            st.session_state.source_pdf_text = texte
            
            col1, col2 = st.columns(2)
            with col1:
                nom_matiere = st.text_input("📚 Matière", value="Mathématiques", key="chatbot_matiere")
                niveau = st.selectbox("🎓 Niveau", 
                                    ["6ème", "5ème", "4ème", "3ème", "2nde", "1ère", "Terminale", 
                                     "L1", "L2", "L3", "M1", "M2", "Doctorat"], key="chatbot_niveau")
            with col2:
                filiere_chatbot = ""
                if niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
                    filiere_chatbot = st.text_input("🎯 Filière/Spécialité", 
                                                  placeholder="Ex: Informatique, Mathématiques, Physique, etc.",
                                                  help="Précisez votre filière pour des réponses plus adaptées",
                                                  key="chatbot_filiere")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Posez votre question sur le PDF..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Recherche dans le PDF..."):
                        consignes_questions = generate_chatbot_instructions(nom_matiere, niveau, filiere_chatbot)
                        response = reponse(consignes_questions, f"Question : {prompt}\n\nTexte PDF :\n{texte}")
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.error("❌ Erreur lors de la génération de la réponse")

# Fonctions de génération des consignes
def generate_qcm_instructions(nom_matiere, niveau, nb_questions, filiere=""):
    niveau_description = f"{niveau}"
    if filiere and niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
        niveau_description = f"{niveau} en {filiere}"
    
    return (
        f"Tu es un enseignant de {nom_matiere} et tu donnes cours à des étudiants de niveau {niveau_description}.\n"
        f"À partir du texte fourni, génère EXCLUSIVEMENT des QCM de qualité professionnelle, "
        f"parfaitement adaptés à {niveau_description}, dans un format prêt à être utilisé dans un examen écrit.\n\n"

        "### EXIGENCES SPÉCIFIQUES POUR QCM ###\n"
        f"• Produire exactement {nb_questions} questions QCM (pas plus, pas moins)\n"
        "• Chaque question doit tenir sur une ou deux lignes maximum\n"
        "• Une seule question par bloc, numérotée clairement\n"
        "• Chaque option doit apparaître sur UNE LIGNE DISTINCTE (A), B), C), D))\n"
        "• Saut de ligne entre la question et les options\n"
        "• Saut de 2 lignes vides après la dernière option\n"
        "• Aucune réponse ou correction ne doit apparaître\n\n"

        "### FORMATAGE EXIGÉ ###\n"
        "[Numéro]. [Question complète]\n"
        "(ligne vide)\n"
        "A) [Option A]\n"
        "B) [Option B]\n"
        "C) [Option C]\n"
        "D) [Option D]\n"
        "(laisser 2 lignes vides)\n\n"

        "### RÈGLES COMMUNES ###\n"
        "• Ne pas inventer de contenu extérieur au texte fourni\n"
        f"• Les questions doivent rester adaptées au niveau {niveau_description}\n"
        "• Style professionnel et lisible\n\n"

        "GÉNÈRE UNIQUEMENT des QCM dans le format indiqué ci-dessus.\n"
        "Respecte strictement et formellement le format demandé."
    )

def generate_qro_instructions(nom_matiere, niveau, nb_questions, filiere=""):
    niveau_description = f"{niveau}"
    if filiere and niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
        niveau_description = f"{niveau} en {filiere}"
    
    return (
        f"Tu es un enseignant de {nom_matiere} et tu donnes cours à des étudiants de niveau {niveau_description}.\n"
        f"À partir du texte fourni, génère EXCLUSIVEMENT des QRO de qualité professionnelle, "
        f"parfaitement adaptés à {niveau_description}.\n\n"

        "### EXIGENCES SPÉCIFIQUES POUR QUESTIONS À RÉPONSE OUVERTE (QRO) ###\n"
        f"• Produire exactement {nb_questions} questions QRO\n"
        "• Mélanger questions conceptuelles, techniques, interprétatives et de calcul\n"
        "• Chaque question doit être claire, précise et sans ambiguïté\n"
        "• AUCUNE réponse, correction ou indice ne doit apparaître\n"
        "• Prévoir 2 à 3 lignes vides pour la réponse\n\n"

        "### FORMATAGE EXIGÉ ###\n"
        "[Numéro]. [Question complète]\n"
        "(saut de ligne)\n"
        "Réponse : _______________________________________________________\n"
        "_______________________________________________________\n"
        "(laisser 1 seule ligne vide avant la question suivante)\n\n"

        "### RÈGLES COMMUNES ###\n"
        "• Ne pas inventer de contenu extérieur au texte fourni\n"
        f"• Les questions doivent rester adaptées au niveau {niveau_description}\n"
        "• Style professionnel avec des questions bien séparées\n\n"

        "GÉNÈRE UNIQUEMENT des QRO dans le format indiqué ci-dessus."
    )

def generate_epreuves_instructions(nom_matiere, niveau, filiere=""):
    niveau_description = f"{niveau}"
    if filiere and niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
        niveau_description = f"{niveau} en {filiere}"
    
    return (
        "Respecte STRICTEMENT les consignes suivantes pour générer des épreuves complètes.\n\n"
        f"Tu es un enseignant de {nom_matiere} et tu donnes cours à des étudiants de {niveau_description}.\n"
        f"À partir de ce texte, génère exclusivement des épreuves de qualité professionnelle adapté à {niveau_description}.\n\n"

        "## EN CAS D'ÉPREUVES ##\n"
        "EXIGENCES SPÉCIFIQUES POUR ÉPREUVES :\n"
        "• Produire Minimum 1 exercice de calculs complets\n"
        "• Chaque exercice doit contenir 5 sous-questions progressives\n"
        "• Espace de réponse suffisant (2 lignes par sous-question)\n"
        "• Contextualisation claire avec données numériques complètes\n"
        "• Instructions précises pour chaque calcul demandé\n"
        "• Formatage PROFESSIONNEL avec tableaux parfaitement alignés\n"
        "• Aucun QCM - uniquement des exercices de calcul\n\n"
        
        "EXEMPLES DE FORMATAGE PARFAIT POUR UN TABLEAU SI UTILE POUR L'EXERCICE:\n"
        "Exercice [numéro]: [Titre contextuel descriptif]\n"
        "\n"
        "[Énoncé détaillé avec contexte et objectif de l'analyse - 2-3 lignes maximum]\n"
        "TABLEAU\n"
        "\n"
        "| [Nom colonne 1]            | [Nom colonne 2]               | [Nom colonne 3]                     | [Nom colonne 4]              |\n"
        "| [Valeur colonne 1 ligne 1] | [Valeur colonne 2 ligne 1]    | [Valeur colonne 3 ligne 1]          | [Valeur colonne 4 ligne 1]   |\n"
        "| [Valeur colonne 1 ligne 2] | [Valeur colonne 2 ligne 2]    | [Valeur colonne 3 ligne 2]          | [Valeur colonne 4 ligne 2]   |\n"
        "| [Valeur colonne 1 ligne 3] | [Valeur colonne 2 ligne 3]    | [Valeur colonne 3 ligne 3]          | [Valeur colonne 4 ligne 3]   |\n"
        "| [Valeur colonne 1 ligne 4] | [Valeur colonne 2 ligne 4]    | [Valeur colonne 3 ligne 4]          | [Valeur colonne 4 ligne 4]   |\n"
        "\n"

        "STRUCTURE DÉTAILLÉE EXIGÉE POUR CHAQUE EXERCICE :\n"
        "Exercice [numéro]: [Titre contextuel descriptif]\n"
        "\n"
        "[Énoncé détaillé avec contexte et objectif de l'analyse - 2-3 lignes maximum]\n"
        "\n"
        "[TABLEAU - Formatage EXACT avec pipes]\n"
        "\n"
        "1) [Instruction claire et précise sur une seule ligne]\n"
        "   Réponse : \n"
        "   _______________________________________________________\n"
        "   _______________________________________________________\n"
        "\n"
        "2) [Instruction claire et précise sur une seule ligne]\n"
        "   Réponse : \n"
        "   _______________________________________________________\n"
        "   _______________________________________________________\n"
        "\n"
        "3) [Instruction claire et précise sur une seule ligne]\n"
        "   Réponse : \n"
        "   _______________________________________________________\n"
        "   _______________________________________________________\n"
        "\n"
        "4) [Instruction claire et précise sur une seule ligne]\n"
        "   Réponse : \n"
        "   _______________________________________________________\n"
        "   _______________________________________________________\n"
        "\n"
        "5) [Instruction claire et précise sur une seule ligne]\n"
        "   Réponse : \n"
        "   _______________________________________________________\n"
        "   _______________________________________________________\n\n"

        "RÈGLES DE FORMATAGE ABSOLUMENT OBLIGATOIRES POUR LES TABLEAUX :\n"
        "• FORMAT PIPE : Utiliser EXACTEMENT le format '|' pour les bordures\n"
        "• ALIGNEMENT VERTICAL : Toutes les colonnes doivent être parfaitement alignées\n"
        "• SAUTS DE LIGNE : Chaque ligne du tableau doit être sur une NOUVELLE ligne\n"
        "• ESPACEMENT : Une ligne vide AVANT et APRÈS chaque tableau\n\n"

        "GÉNÈRE UNIQUEMENT des exercices de calcul dans ce format EXACT."
    )

def generate_chatbot_instructions(nom_matiere, niveau, filiere=""):
    niveau_description = f"{niveau}"
    if filiere and niveau in ["L1", "L2", "L3", "M1", "M2", "Doctorat"]:
        niveau_description = f"{niveau} en {filiere}"
    
    return (
        f"Tu es un enseignant de {nom_matiere} pour des étudiants de niveau {niveau_description}.\n"
        "Tu DOIS répondre UNIQUEMENT en utilisant le texte fourni.\n\n"

        "### INSTRUCTIONS STRICTES ###\n"
        "1) Vérifie si le texte contient le mot ou sujet exact de la question.\n"
        "2) Si le mot/sujet EXACT est présent :\n"
        "   - Réponds EXACTEMENT :\n"
        "     Réponse : [passage exact]\n"
        "     Page : [numéro de page]\n"
        "3) Si le mot/sujet EXACT n'est PAS présent :\n"
        "   - Réponds STRICTEMENT :\n"
        "     Réponse : non précisé\n"
        "     Page : non précisé\n"
        "4) TU NE DOIS JAMAIS :\n"
        "   • Ajouter des lignes supplémentaires\n"
        "   • Introduire, expliquer, résumer, reformuler ou modifier le texte\n"
        "   • Changer le mot 'Page' ou l'ordre des lignes\n"
        "5) Toute sortie différente de ce format EXACTEMENT à 2 lignes est considérée comme FAUSSE."
    )

def diagnostic_page():
    """Page de diagnostic pour l'API"""
    st.header("🔧 Diagnostic API")
    # Sélecteur de modèle global
    st.subheader("Modèle à utiliser")
    current_model = st.session_state.get("selected_model", AVAILABLE_MODELS[0])
    selected = st.selectbox("Choisir un modèle", AVAILABLE_MODELS, index=max(0, AVAILABLE_MODELS.index(current_model)))
    st.session_state["selected_model"] = selected
    
    # Vérification de la clé API
    st.subheader("1. Vérification de la clé API")
    if not api_key:
        st.error("❌ Clé API non trouvée")
        st.info("💡 Créez un fichier .env avec : OPENROUTER_API_KEY=votre_cle_ici")
        return
    else:
        st.success(f"✅ Clé API trouvée : {api_key[:8]}...{api_key[-4:]}")
        
        # Vérifier le format
        if api_key.startswith('sk-'):
            st.success("✅ Format de clé API correct")
        else:
            st.warning("⚠️ Format de clé API suspect")
    
    # Test de connexion
    st.subheader("2. Test de connexion API")
    if st.button("🧪 Tester la connexion API"):
        with st.spinner("Test en cours..."):
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:8501/",
                "Content-Type": "application/json"
            }
            
            # Test simple
            payload = {
                "model": st.session_state.get("selected_model", AVAILABLE_MODELS[0]),
                "messages": [
                    {"role": "user", "content": "Bonjour, ceci est un test."}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                st.write(f"**Code de statut HTTP :** {response.status_code}")
                
                if response.status_code == 200:
                    response_json = response.json()
                    if 'choices' in response_json:
                        st.success("✅ Connexion API réussie !")
                        st.write("**Réponse de test :**")
                        st.write(response_json['choices'][0]['message']['content'])
                    else:
                        st.error("❌ Réponse API invalide")
                        st.json(response_json)
                else:
                    st.error(f"❌ Erreur HTTP {response.status_code}")
                    st.write("**Réponse d'erreur :**")
                    st.text(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("❌ Timeout - L'API met trop de temps à répondre")
            except requests.exceptions.ConnectionError:
                st.error("❌ Erreur de connexion - Vérifiez votre internet")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {str(e)}")
    
    # Informations de debug
    st.subheader("3. Informations de debug")
    with st.expander("🔍 Détails techniques"):
        st.write(f"**URL API :** https://openrouter.ai/api/v1/chat/completions")
        st.write(f"**Modèle utilisé :** mistralai/ministral-8b")
        
        # Headers pour l'affichage
        debug_headers = {
            "Authorization": f"Bearer {api_key[:8]}...{api_key[-4:]}",
            "HTTP-Referer": "http://localhost:8501/",
            "Content-Type": "application/json"
        }
        st.write(f"**Headers :** {debug_headers}")
    
    # Suggestions de dépannage
    st.subheader("4. Solutions courantes")
    st.markdown("""
    **Si vous obtenez "Provider returned error" :**
    - ✅ Vérifiez que votre clé API est valide sur [OpenRouter](https://openrouter.ai/)
    - ✅ Assurez-vous d'avoir des crédits disponibles
    - ✅ Vérifiez que le modèle ministral-8b est disponible
    - ✅ Essayez de redémarrer l'application
    
    **Si vous obtenez des erreurs de connexion :**
    - ✅ Vérifiez votre connexion internet
    - ✅ Désactivez temporairement votre antivirus/firewall
    - ✅ Essayez un autre réseau (mobile, par exemple)
    
    **Optimisations pour ministral-8b :**
    - ✅ Le modèle est plus rapide et plus efficace
    - ✅ Meilleure qualité de génération de texte
    - ✅ Support amélioré pour les instructions complexes
    """)

# ====== INTERACTIF: parsing et rendu ======
def parse_qcm(text: str) -> List[Dict[str, Any]]:
    questions = []
    current = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Accepte 1. ou [1]. ou 1)
        if re.match(r"^(\[\d+\]\.|\d+\.)", s):
            if current:
                questions.append(current)
            current = {"question": s, "options": []}
        elif re.match(r"^[ABCD]\)", s):
            if current is not None:
                current["options"].append(s)
        else:
            if current is not None and current["options"]:
                current["options"][-1] += " " + s
            elif current is not None:
                current["question"] += " " + s
    if current:
        questions.append(current)
    return questions

def parse_qro(text: str) -> List[Dict[str, Any]]:
    questions = []
    current = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+\.", s):
            if current:
                questions.append(current)
            current = {"question": s}
        else:
            if current is not None:
                current["question"] += " " + s
    if current:
        questions.append(current)
    return questions

def parse_epreuves(text: str) -> List[Dict[str, Any]]:
    blocs: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Début d'exercice
        if s.lower().startswith("exercice") or s.startswith("## Exercice"):
            # cloturer le précédent
            if current.get("title"):
                blocs.append(current)
            # extraire numéro et titre
            num_match = re.search(r"(\d+)", s)
            number = int(num_match.group(1)) if num_match else len(blocs) + 1
            # nettoyer préfixes
            title = s
            title = title.replace("## ", "").replace("# ", "")
            title = re.sub(r"(?i)^exercice\s*\d*:?\s*", "", title).strip() or f"Exercice {number}"
            current = {
                "number": number,
                "title": title,
                "subquestions": [],
                "context": []
            }
            continue
        # Sous-questions (1) 2) ...)
        if re.match(r"^\d+\)", s):
            current.setdefault("subquestions", []).append(s)
            continue
        # Contexte / autres lignes
        current.setdefault("context", []).append(s)
    if current.get("title"):
        blocs.append(current)
    return blocs

def render_interactive_exercises():
    generated_text = st.session_state.get("generated_text", "")
    generated_type = st.session_state.get("generated_type", "")
    meta = st.session_state.get("generated_meta", {})
    if not generated_text or not generated_type:
        st.info("Aucun exercice en mémoire. Générez d'abord un contenu.")
        st.write({
            "has_generated_text": bool(generated_text),
            "generated_type": generated_type,
        })
        return

    st.subheader("📝 Répondez aux exercices")
    st.caption(f"Type: {generated_type} — Matière: {meta.get('matiere','')} — Niveau: {meta.get('niveau','')}")

    if generated_type == "QCM":
        items = parse_qcm(generated_text)
        for idx, q in enumerate(items, 1):
            st.markdown(f"**{q['question']}**")
            key = f"qcm_{idx}"
            choice = st.radio("Votre réponse", options=["A", "B", "C", "D"], horizontal=True, key=key)
            st.session_state.answers[key] = choice
            with st.expander("Afficher les options"):
                for opt in q["options"]:
                    st.write(opt)
            st.markdown("---")
    elif generated_type == "QRO":
        items = parse_qro(generated_text)
        for idx, q in enumerate(items, 1):
            st.markdown(f"**{q['question']}**")
            key = f"qro_{idx}"
            ans = st.text_area("Votre réponse", key=key, height=100)
            st.session_state.answers[key] = ans
            st.markdown("---")
    elif generated_type == "Épreuves":
        blocs = parse_epreuves(generated_text)
        for bidx, b in enumerate(blocs, 1):
            st.markdown(f"#### Exercice {b.get('number', bidx)} : {b.get('title','')}")
            context_lines = b.get("context", [])
            table_lines = []
            normal_lines = []
            in_table = False
            for line in context_lines:
                s = line.strip()
                # Ignore les lignes "Réponse", vides et les lignes de tirets
                if (
                    not s or
                    s.lower().startswith("réponse") or
                    re.match(r"^_+$", s)  # ignore les lignes composées uniquement de tirets
                ):
                    continue
                if "|" in s:
                    table_lines.append(s)
                    in_table = True
                else:
                    if in_table and table_lines:
                        # Nettoie les tirets dans les cases du tableau
                        table_data = []
                        for row in table_lines:
                            cells = [cell.strip() for cell in row.split("|")]
                            # Ignore les cellules qui ne sont que des tirets
                            cleaned_cells = [cell if not re.match(r"^-+$", cell) else "" for cell in cells]
                            # Ignore les lignes qui sont uniquement des tirets (Markdown separator)
                            if any(cleaned_cells) and not all(re.match(r"^-+$", cell) or cell == "" for cell in cells):
                                table_data.append(cleaned_cells[1:-1] if row.startswith("|") else cleaned_cells)
                        if table_data:
                            st.table(table_data)
                        table_lines = []
                        in_table = False
                    normal_lines.append(s)
            # Affiche le dernier tableau si présent
            if table_lines:
                table_data = []
                for row in table_lines:
                    cells = [cell.strip() for cell in row.split("|")]
                    cleaned_cells = [cell if not re.match(r"^-+$", cell) else "" for cell in cells]
                    if any(cleaned_cells) and not all(re.match(r"^-+$", cell) or cell == "" for cell in cells):
                        table_data.append(cleaned_cells[1:-1] if row.startswith("|") else cleaned_cells)
                if table_data:
                    st.table(table_data)
            # Affiche le reste du contexte (sans "Réponse", espaces ni tirets)
            for line in normal_lines:
                if line:
                    st.markdown(line)
            # Affichage des sous-questions
            sub = b.get("subquestions", [])
            if not sub:
                st.write("(Aucune sous-question détectée — zone de réponse libre)")
                key = f"ep_{bidx}_libre"
                st.session_state.answers[key] = st.text_area("Votre réponse", key=key)
            else:
                for sidx, s in enumerate(sub, 1):
                    st.markdown(f"**{s}**")
                    key = f"ep_{bidx}_{sidx}"
                    st.session_state.answers[key] = st.text_area("Votre réponse", key=key, height=80)
            st.markdown("---")

    if st.button("✅ Valider et corriger", key="btn_validate_correct"):
        correct_with_model()

def correct_with_model():
    texte_pdf = st.session_state.get("source_pdf_text", "")
    generated_text = st.session_state.get("generated_text", "")
    answers = st.session_state.get("answers", {})
    if not texte_pdf or not generated_text or not answers:
        st.error("Données insuffisantes pour corriger.")
        return

    consignes = (
        "Respects strictement et formellement les formats et instructions donnés.\n"
        "Corrige chaque QCM en comparant la réponse de l'étudiant (option complète choisie) à la bonne option parmi celles proposées.\n"
        "Tu es un correcteur impartial. Corrige les réponses de l'étudiant en te basant STRICTEMENT sur le texte PDF fourni.\n"
        "- Pour chaque question, affiche:\n"
        "Question: [Numero et texte de la question]\n"
        "Réponse de l'étudiant: [réponse avec mention correcte ou pas]\n"
        "Correction: [corrige et explique en 2-4 phrases, cite les passages pertinents du PDF avec la page si possible]\n"
        "Note: [0/1 pour QCM; 0-1 pour QRO/Épreuves selon exactitude]\n"
        "Ne révéle pas de réponses qui ne sont pas soutenues par le PDF.\n"
        "Respects strictement et formellement les formats et instructions donnés.\n"
    )
    prompt = (
        "EXERCICES GÉNÉRÉS:\n" + generated_text + "\n\n" +
        "RÉPONSES ÉTUDIANT (clé -> réponse):\n" + str(answers) + "\n\n" +
        "TEXTE PDF:\n" + texte_pdf
    )
    with st.spinner("Correction en cours..."):
        result = reponse(consignes, prompt)
    if result:
        st.subheader("📘 Correction détaillée")
        st.markdown(result)
        st.session_state["corrected_report"] = result
        st.subheader("🏁 Synthèse et Note Globale")
        synthese = compute_global_score(result)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Réponses justes", f"{synthese['num_correct']} / {synthese['num_questions']}")
        with col2:
            st.metric("% réponses justes", f"{synthese['percent_correct']}%")
        with col3:
            st.metric("Score points", f"{synthese['total_points_obtenus']} / {synthese['total_points_max']}")
        with col4:
            st.metric("Mention", synthese['appreciation'])
        with st.expander("Détails de la notation par question"):
            for item in synthese["details"]:
                st.write(f"- {item['question']} → {item['points']} pt(s)")
    else:
        st.error("La correction n'a pas pu être générée.")
        st.write("Debug - Résultat IA :", result)
        

def compute_global_score(correction_text: str) -> Dict[str, Any]:
    """Extrait des points depuis le texte de correction et calcule une note globale.
    Hypothèse: le modèle renvoie des lignes comme `Note: 0/1` ou `Note: 1/1`.
    """
    details = []
    total = 0.0
    maxi = 0.0
    current_q = None
    num_questions = 0
    num_correct = 0
    
    # st.write(correction_text.splitlines())
    for line in correction_text.splitlines():
        #st.write(line.strip().lower().startswith("question:"))
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("question:"):
            current_q = s
            num_questions += 1
        m = re.search(r"Note\s*:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
        if m:
            pts = float(m.group(1))
            mx = float(m.group(2))
            total += pts
            maxi += mx
            details.append({"question": current_q or "Question", "points": f"{pts}/{mx}"})
            #st.write(mx)
            if mx > 0 and abs(pts - mx) < 1e-6:
                num_correct += 1
    pourcentage_points = round(100.0 * total / maxi, 1) if maxi > 0 else 0.0
    percent_correct = round(100.0 * num_correct / num_questions, 1) if num_questions > 0 else 0.0
    # Mention sur la base du pourcentage de réponses justes
    appreciation = (
        "Excellent" if percent_correct >= 85 else
        "Très bien" if percent_correct >= 75 else
        "Bien" if percent_correct >= 65 else
        "Assez bien" if percent_correct >= 55 else
        "Passable" if percent_correct >= 50 else
        "Insuffisant"
    )
    return {
        "total_points_obtenus": int(total) if float(total).is_integer() else round(total, 2),
        "total_points_max": int(maxi) if float(maxi).is_integer() else round(maxi, 2),
        "pourcentage": pourcentage_points,
        "num_questions": num_questions,
        "num_correct": num_correct,
        "percent_correct": percent_correct,
        "appreciation": appreciation,
        "details": details,
    }


if __name__ == "__main__":
    main()
