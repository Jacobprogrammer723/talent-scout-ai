# TalentScout AI 🚀

TalentScout AI is a smart resume screening and analysis tool powered by Google Gemini. It helps HR professionals and recruiters efficiently screen multiple resumes against a job description, providing actionable insights, match scores, and interview questions.

## Features

-   **📄 Bulk PDF Upload**: Upload multiple resumes at once.
-   **🧠 AI Analysis**: Automatically ranks candidates based on job description fit.
-   **🙈 Bias Blocker**: "Blind Hiring Mode" to hide names and reduce unconscious bias.
-   **📊 Visual Dashboard**: Interactive data table with progress bars and score filtering.
-   **📝 Detailed Insights**: Pros, Cons, and Summaries for each candidate.
-   **🎤 Interview Generator**: Generates specific, hard interview questions based on candidate weaknesses.
-   **📥 Export**: Download the analysis report as a CSV file.

## Setup & Installation

### Prerequisites

-   Python 3.9+
-   A Google Cloud API Key (for Gemini)

### Local Development

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd talent-scout-ai
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

5.  **Enter API Key:**
    Open the app in your browser (usually `http://localhost:8501`) and enter your Google API Key in the sidebar.

### Docker

1.  **Build the image:**
    ```bash
    docker build -t talentscout-ai .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 talentscout-ai
    ```

### Streamlit Cloud Deployment

1.  Push your code to a GitHub repository.
2.  Go to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Connect your GitHub account and select your repository.
4.  Select `app.py` as the main file.
5.  In "Advanced Settings", add your `GOOGLE_API_KEY` as a secret (optional, or users can enter it in the UI).
6.  Click **Deploy**!

## Tech Stack

-   **Frontend**: Streamlit
-   **AI Model**: Google Gemini Pro
-   **PDF Processing**: pypdf
-   **Data Manipulation**: Pandas
