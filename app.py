import streamlit as st
import pypdf
import os
import pandas as pd
import json
import google.generativeai as genai
from dotenv import load_dotenv
import docx
import plotly.express as px
import urllib.parse

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
def analyze_candidates(job_description, cv_text_list, competitors_list=""):
    """
    Analyzes candidates using Google Gemini.

    Args:
        job_description: The job description text.
        cv_text_list: A list of dictionaries [{'name': filename, 'text': text}, ...].

    Returns:
        str: The raw JSON response from Gemini.
    """
    prompt = f"""
    You are an expert HR AI. Analyze the following candidates against this Job Description:
    
    JOB DESCRIPTION:
    {job_description}
    
    COMPETITOR WATCHLIST (Target Companies):
    {competitors_list}
    
    CANDIDATES (Text extracted from CVs):
**
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

    4. **Risk Assessment:**
       - **Red Flags:** Check for employment gaps (>6 months), job hopping (>3 jobs in 2 years), or vague descriptions.
       - **Skill Gaps:** Compare extracted skills vs JD requirements. Be specific about what is missing.

    5. **Deep Profiling:**
       - **Personality Decoder:** Analyze tone/style. Extract 3-4 tags (e.g., "High Agency", "Collaborative", "Analytical", "Leader", "Doer").
       - **Competitor Check:** Check if the candidate has worked at any company in the COMPETITOR WATCHLIST. Set "is_competitor_match" to true/false.

    Return the response STRICTLY as a valid JSON list of objects. 
    Do not include any markdown formatting (like ```json ... ```).
    
    The JSON structure for each candidate must be:
    {
        "name": "string (filename or extracted name)",
        "score_skills": int (0-100),
        "score_experience": int (0-100),
        "score_soft_skills": int (0-100),
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
        "linkedin_url": "string (extracted linkedin url or empty)",
        "red_flags": ["string", "string"],
        "skill_gaps": ["string", "string"],
        "personality_tags": ["string", "string"],
        "is_competitor_match": boolean
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
        
        # Custom Scoring Weights
        with st.expander("⚙️ Anpassa Viktning (Scoring Weights)"):
            w_skills = st.slider("Vikt: Hårda Kunskaper (Skills)", 0, 100, 33)
            w_exp = st.slider("Vikt: Erfarenhet (Experience)", 0, 100, 33)
            w_soft = st.slider("Vikt: Potential & Driv (Soft Skills)", 0, 100, 34)
            
            # Normalize
            total_weight = w_skills + w_exp + w_soft
            if total_weight == 0:
                norm_skills, norm_exp, norm_soft = 0.33, 0.33, 0.34
            else:
                norm_skills = w_skills / total_weight
                norm_exp = w_exp / total_weight
                norm_soft = w_soft / total_weight

        # Competitor Watchlist
        competitors_input = st.text_input("🎯 Target Companies (Competitor Watchlist)", placeholder="e.g. Spotify, Klarna, Volvo")

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
                    
                    json_response = analyze_candidates(job_description, cv_text_list, competitors_input)
                    
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
                                        display_df.at[i, 'red_flags'] = []
                                        display_df.at[i, 'skill_gaps'] = []
                                        display_df.at[i, 'personality_tags'] = []
                                        display_df.at[i, 'is_competitor_match'] = False
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
                                        # Header with Name and Badges
                                        header_col1, header_col2 = st.columns([3, 1])
                                        with header_col1:
                                            st.write(f"**Summary:** {row['summary']}")
                                        with header_col2:
                                            if row.get('is_competitor_match'):
                                                st.error("🎯 **Competitor Match!**")
                                            
                                            # Hidden Gem Logic
                                            # Criteria: Exp < 60, Skills > 80, Soft Skills > 80
                                            score_skills = row.get('score_skills', 0)
                                            score_exp = row.get('score_experience', 0)
                                            score_soft = row.get('score_soft_skills', 0)
                                            
                                            if score_exp < 60 and score_skills > 80 and score_soft > 80:
                                                st.success("💎 **Hidden Gem!**")
                                        
                                        # Personality Tags
                                        tags = row.get('personality_tags', [])
                                        if tags:
                                            st.markdown("**Personality Vibe:** " + " ".join([f"`{tag}`" for tag in tags]))

                                        # Contact Info
                                        contact_cols = st.columns(3)
                                        candidate_email = row.get('email', '')
                                        
                                        if candidate_email:
                                            contact_cols[0].markdown(f"📧 **Email:** {candidate_email}")
                                        if row.get('phone'):
                                            contact_cols[1].markdown(f"📱 **Phone:** {row['phone']}")
                                        if row.get('linkedin_url'):
                                            contact_cols[2].markdown(f"🔗 [LinkedIn Profile]({row['linkedin_url']})")
                                        
                                        # Smart Email Actions
                                        if candidate_email:
                                            st.markdown("---")
                                            email_cols = st.columns(2)
                                            
                                            # Invite Email
                                            subject_invite = f"Interview Invitation: {row['name']}"
                                            body_invite = f"Hi {row['name']},\n\nI reviewed your application and was impressed by your profile. We would love to schedule an interview.\n\nBest,\nTalentScout Team"
                                            mailto_invite = f"mailto:{candidate_email}?subject={urllib.parse.quote(subject_invite)}&body={urllib.parse.quote(body_invite)}"
                                            email_cols[0].link_button("🟢 Draft Interview Invite", mailto_invite)
                                            
                                            # Rejection Email
                                            subject_reject = f"Update on your application: {row['name']}"
                                            body_reject = f"Hi {row['name']},\n\nThank you for your application. Unfortunately, we have decided to proceed with other candidates at this time.\n\nBest,\nTalentScout Team"
                                            mailto_reject = f"mailto:{candidate_email}?subject={urllib.parse.quote(subject_reject)}&body={urllib.parse.quote(body_reject)}"
                                            email_cols[1].link_button("🔴 Draft Rejection", mailto_reject)

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
                                        
                                        # Risk Assessment Section
                                        st.markdown("---")
                                        st.subheader("⚠️ Risk Assessment")
                                        
                                        risk_col1, risk_col2 = st.columns(2)
                                        
                                        with risk_col1:
                                            st.markdown("#### 🚩 Red Flags")
                                            red_flags = row.get('red_flags', [])
                                            if red_flags:
                                                for flag in red_flags:
                                                    st.error(f"{flag}")
                                            else:
                                                st.success("No major red flags detected.")
                                        
                                        with risk_col2:
                                            st.markdown("#### 🧩 Skill Gaps")
                                            skill_gaps = row.get('skill_gaps', [])
                                            if skill_gaps:
                                                for gap in skill_gaps:
                                                    st.warning(f"Missing: {gap}")
                                            else:
                                                st.success("No critical skill gaps detected.")

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
