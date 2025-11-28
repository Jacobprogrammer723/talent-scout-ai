import streamlit as st
import pypdf
import os
import pandas as pd
import json
import google.generativeai as genai
from dotenv import load_dotenv
import docx
import plotly.express as px

import time

# Load environment variables
# Load environment variables
load_dotenv()

def extract_text_from_file(file):
    """
    Extracts text from a PDF or DOCX file.
    
    Args:
        file: A file-like object (uploaded file).
        
    Returns:
        str: The extracted text.
    """
    try:
        if file.name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            return ""
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
        return ""

@st.cache_data
def analyze_candidates(job_description, cv_text_list):
    """
    Analyzes candidates using Google Gemini.

    Args:
        job_description: The job description text.
        cv_text_list: A list of dictionaries [{'name': filename, 'text': text}, ...].

    Returns:
        str: The raw JSON response from Gemini.
    """
    prompt = f"""
    You are an expert HR AI assistant. Your task is to screen resumes against a job description.

    **Job Description:**
    {job_description}

    **Candidates:**
    """
    
    for cv in cv_text_list:
        prompt += f"\n--- Candidate: {cv['name']} ---\n{cv['text']}\n"

    prompt += """
    \n**Instructions:**
    Analyze each candidate based on the job description.
    
    **CRITICAL: Strict Semantic Matching & Context Verification**
    1. **Strict Synonym Mapping ONLY:**
       - Treat exact synonyms as matches (e.g., "Remote" == "Distansarbete", "Javascript" == "JS").
       - Do NOT conflate distinct roles (e.g., "Account Management" != "New Business Sales") unless explicitly supported by context.
    
    2. **Context Verification:**
       - Before matching a skill using a different word, verify the *context* in the CV bullet points.
       - *Example:* If CV says "Kundansvar" but describes "Upselling and contract negotiation", THEN it matches "Sales".
       - *Example:* If CV says "Kundansvar" but describes "Technical support and ticket handling", do NOT match it to "Sales".
    
    3. **Output Transparency:**
       - If a match is made based on context rather than exact words, explicitly state this in the "summary" or "pros".
       - Example: "Matched 'Sales' requirement based on described tasks (Upselling/Negotiation), despite job title being 'Support'."

    Return the response STRICTLY as a valid JSON list of objects. 
    Do not include any markdown formatting (like ```json ... ```).
    
    The JSON structure for each candidate must be:
    {
        "name": "string (filename or extracted name)",
        "match_score": int (0-100),
        "summary": "string (brief summary of qualifications)",
        "pros": ["string", "string"],
        "cons": ["string", "string"],
        "years_experience": int (estimated),
        "radar_scores": {
            "Technical Skills": int (0-10),
            "Communication": int (0-10),
            "Experience": int (0-10),
            "Culture Fit": int (0-10),
            "Leadership": int (0-10)
        },
        "email": "string (extracted email or empty)",
        "phone": "string (extracted phone or empty)",
        "linkedin_url": "string (extracted linkedin url or empty)"
    }
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return "[]"

def generate_interview_questions(candidate_name, job_description, cons_list):
    """
    Generates interview questions based on candidate weaknesses.
    """
    cons_text = ", ".join(cons_list)
    prompt = f"""
    You are an expert interviewer. Generate 3 specific, hard interview questions for a candidate named {candidate_name}.
    
    **Job Description:**
    {job_description}
    
    **Candidate Weaknesses/Cons:**
    {cons_text}
    
    **Goal:**
    The questions should probe these weaknesses to verify if they are deal-breakers.
    Return ONLY the 3 questions as a numbered list.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Could not generate questions."

def main():
    st.set_page_config(layout="wide", page_title="TalentScout AI")

    # Initialize Session State
    if 'is_paid' not in st.session_state:
        st.session_state['is_paid'] = False
    
    # Rate Limiting State
    if 'rate_limit_count' not in st.session_state:
        st.session_state['rate_limit_count'] = 0
    if 'rate_limit_start_time' not in st.session_state:
        st.session_state['rate_limit_start_time'] = time.time()

    # Top Banner
    if not st.session_state['is_paid']:
        st.warning("🌟 **Try for free. Pay only if you like the results.** Top 3 candidates are free forever.")

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # Secure API Key Handling
        api_key = None
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
        elif os.getenv("GOOGLE_API_KEY"):
            api_key = os.getenv("GOOGLE_API_KEY")

        if api_key:
            genai.configure(api_key=api_key)
        else:
            st.error("⚠️ Missing API Key. Please configure `GOOGLE_API_KEY` in Streamlit Secrets.")

        st.markdown("---")
        st.subheader("Advanced Options")
        blind_hiring = st.checkbox("Blind Hiring Mode (Hide Names)")
        min_score = st.slider("Minimum Match Score", 0, 100, 0)
        
        st.markdown("---")
        if st.session_state['is_paid']:
            st.success("✅ Premium Active")
        else:
            st.info("🔒 Free Plan (Top 3 Only)")


    # Main Content
    st.title("TalentScout AI")
    st.markdown("### 🚀 Smart Resume Screening & Analysis Tool")
    st.markdown("""
    Upload multiple resumes (PDF or DOCX) and let AI rank them against your Job Description.
    Enable **Blind Hiring Mode** to remove bias, and use the **Dashboard** to filter top talent.
    """)

    # Job Description Input
    job_description = st.text_area("Paste the Job Description here:", height=200)

    # File Uploader
    uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    st.caption("🔒 **Privacy First:** No database. Files are processed in memory and discarded.")

    if uploaded_files:
        st.write(f"**Number of files uploaded:** {len(uploaded_files)}")
        
        if st.button("Analyze Candidates"):
            if not job_description:
                st.warning("Please enter a Job Description.")
            elif not api_key:
                st.error("System Error: API Key not configured.")
            else:
                # --- SECURITY CHECKS ---
                
                # 1. Rate Limiting (10 requests per hour)
                current_time = time.time()
                elapsed_time = current_time - st.session_state['rate_limit_start_time']
                
                if elapsed_time > 3600: # Reset after 1 hour
                    st.session_state['rate_limit_count'] = 0
                    st.session_state['rate_limit_start_time'] = current_time
                
                if st.session_state['rate_limit_count'] >= 10:
                    st.error("⚠️ För många förfrågningar. För att förhindra missbruk har vi en gräns på 10 analyser per timme. Försök igen senare eller kontakta oss för Enterprise-access.")
                    st.stop()
                
                # 2. File Validation
                if len(uploaded_files) > 50:
                    st.error("⚠️ Too many files. Maximum 50 files allowed at once.")
                    st.stop()
                
                for file in uploaded_files:
                    if file.size > 5 * 1024 * 1024: # 5MB
                        st.error(f"⚠️ File '{file.name}' is too large. Maximum 5MB per file.")
                        st.stop()
                
                # Increment Counter
                st.session_state['rate_limit_count'] += 1
                
                with st.spinner("Analyzing candidates..."):
                    cv_text_list = []
                    for file in uploaded_files:
                        text = extract_text_from_file(file)
                        cv_text_list.append({"name": file.name, "text": text})
                    
                    json_response = analyze_candidates(job_description, cv_text_list)
                    
                    # Clean up response if it contains markdown code blocks
                    clean_response = json_response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]

                    try:
                        data = json.loads(clean_response)
                        df = pd.DataFrame(data)
                        
                        # Apply Bias Blocker
                        if blind_hiring:
                            df['name'] = [f"Candidate #{i+1}" for i in range(len(df))]
                        
                        # Filter by Score
                        df_filtered = df[df['match_score'] >= min_score]
                        
                        # Sort by Score Descending
                        df_filtered = df_filtered.sort_values(by='match_score', ascending=False).reset_index(drop=True)
                        
                        if df_filtered.empty:
                            st.warning("No candidates matched the criteria.")
                        else:
                            st.success("Analysis Complete!")
                            
                            # --- FREEMIUM LOGIC ---
                            display_df = df_filtered.copy()
                            
                            if not st.session_state['is_paid']:
                                # Mask candidates ranked 4 and below (index 3+)
                                for i in range(len(display_df)):
                                    if i > 2: # 0, 1, 2 are top 3
                                        display_df.at[i, 'name'] = "🔒 Upgrade to Unlock"
                                        display_df.at[i, 'summary'] = "🔒 Upgrade to unlock details..."
                                        display_df.at[i, 'pros'] = []
                                        display_df.at[i, 'cons'] = []
                                        display_df.at[i, 'email'] = ""
                                        display_df.at[i, 'phone'] = ""
                                        display_df.at[i, 'linkedin_url'] = ""
                                        # Keep match_score visible or mask it? User said "blur names and hide reasoning". 
                                        # Usually score is a good teaser. Let's keep score but maybe blur it if requested, 
                                        # but user prompt said "show them in the table but blur the names and hide the reasoning".
                            
                            # CSV Download (Only if paid)
                            if st.session_state['is_paid']:
                                csv = df_filtered.to_csv(index=False).encode('utf-8')
                                st.sidebar.download_button(
                                    "Download Report (CSV)",
                                    csv,
                                    "talent_scout_report.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                            else:
                                st.sidebar.button("🔒 Download Report (Premium)", disabled=True)
                            
                            # Visual Dashboard
                            st.subheader("Candidate Dashboard")
                            st.data_editor(
                                display_df[['name', 'match_score', 'years_experience', 'email', 'linkedin_url']],
                                column_config={
                                    "match_score": st.column_config.ProgressColumn(
                                        "Match Score",
                                        help="The AI's estimated match score (0-100)",
                                        format="%d",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    "linkedin_url": st.column_config.LinkColumn("LinkedIn"),
                                    "email": st.column_config.TextColumn("Email"),
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Paywall Button
                            if not st.session_state['is_paid'] and len(df_filtered) > 3:
                                st.markdown("---")
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    if st.button("🔓 Unlock Full Report & Export ($49)", type="primary", use_container_width=True):
                                        with st.spinner("Processing Payment..."):
                                            time.sleep(2)
                                            st.session_state['is_paid'] = True
                                            st.rerun()
                            
                            # Detailed View
                            st.subheader("Detailed Analysis")
                            for index, row in display_df.iterrows():
                                # Check if masked
                                is_masked = not st.session_state['is_paid'] and index > 2
                                
                                expander_title = f"{row['name']} - Score: {row['match_score']}%"
                                if is_masked:
                                    expander_title = "🔒 Upgrade to Unlock Candidate"
                                
                                with st.expander(expander_title):
                                    if is_masked:
                                        st.info("💎 **Pay to see full analysis, pros, cons, and interview questions.**")
                                    else:
                                        st.write(f"**Summary:** {row['summary']}")
                                        
                                        # Contact Info
                                        contact_cols = st.columns(3)
                                        if row.get('email'):
                                            contact_cols[0].markdown(f"📧 **Email:** {row['email']}")
                                        if row.get('phone'):
                                            contact_cols[1].markdown(f"📱 **Phone:** {row['phone']}")
                                        if row.get('linkedin_url'):
                                            contact_cols[2].markdown(f"🔗 [LinkedIn Profile]({row['linkedin_url']})")
                                        
                                        st.markdown("---")
                                        
                                        # Radar Chart
                                        if 'radar_scores' in row and row['radar_scores']:
                                            scores = row['radar_scores']
                                            df_radar = pd.DataFrame(dict(
                                                r=list(scores.values()),
                                                theta=list(scores.keys())
                                            ))
                                            fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                                            fig.update_traces(fill='toself')
                                            fig.update_layout(
                                                polar=dict(
                                                    radialaxis=dict(
                                                        visible=True,
                                                        range=[0, 10]
                                                    )
                                                ),
                                                showlegend=False,
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("#### ✅ Pros")
                                            for pro in row.get('pros', []):
                                                st.markdown(f"- {pro}")
                                        
                                        with col2:
                                            st.markdown("#### ❌ Cons")
                                            for con in row.get('cons', []):
                                                st.markdown(f"- {con}")
                                        
                                        st.markdown("---")
                                        if st.button(f"Generate Interview Questions for {row['name']}", key=f"btn_{index}"):
                                            with st.spinner("Generating questions..."):
                                                questions = generate_interview_questions(row['name'], job_description, row.get('cons', []))
                                                st.markdown("### 🎤 Recommended Interview Questions")
                                                st.markdown(questions)

                    except json.JSONDecodeError:
                        st.error("Failed to parse API response. Raw response:")
                        st.text(json_response)

if __name__ == "__main__":
    main()
