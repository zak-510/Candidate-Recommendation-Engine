# Candidate Recommendation Engine

Instead of manually reviewing resumes, this application analyzes and ranks candidates based on how well their qualifications align with your job description.

## What It Does

This application serves as a recruitment assistant that automates the initial candidate screening process. You provide a job description and upload candidate resumes, and the system ranks candidates based on semantic similarity rather than simple keyword matching. The AI understands the context and meaning behind the text, providing more accurate matches.

Key capabilities include:
- **Rankings**: Candidates automatically sorted by relevance to job requirements
- **Similarity Scoring**: Each candidate receives a percentage match score for easy comparison
- **AI-Generated Summaries**: Personalized explanations of why each candidate is a strong fit
- **Streamlined Interface**: Clean, web-based application

## Core Features

**Semantic Matching**: Natural language processing that goes beyond keyword matching to understand skills and experience alignment

**Similarity Scoring**: Match percentages using cosine similarity helps prioritize which candidates to review first

**AI-Powered Summaries**: Automated 2-3 sentence explanations highlighting each candidate's strengths relative to the role

**Batch Processing**: Upload and process multiple resumes simultaneously for efficient screening

**Name Detection**: Automatically extracts candidate names from resumes for easy identification  

## Deployment

This application is designed to run on Streamlit Cloud with cloud-based AI processing.

### Local Development (Optional)
If you want to run locally for development:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenRouter API key in `.streamlit/secrets.toml`
4. Run: `streamlit run app.py`

## Usage Instructions

1. **Enter Job Description**: Paste the job description in the provided text area. More detailed descriptions gives better matching results.
2. **Upload Resume Files**: Select and upload candidate resumes in PDF or plain text format.
3. **Run Analysis**: Click the analyze button to process the uploaded resumes against the job requirements.
4. **Review Results**: Candidates are displayed in ranked order with similarity scores.

## Requirements and Considerations

**Supported File Formats**: The application takes PDF and plain text files. Password-protected PDFs cannot be processed.

**Performance**: Processing time varies based on the number of resumes and document length.

**Data Quality**: Results depend on the clarity and completeness of both job descriptions and resume content. Well-formatted, detailed documents produce more accurate rankings.

## Tech Architecture

- **Streamlit**: Provides the web-based user interface and application framework
- **Sentence Transformers**: Handles semantic text embeddings using the all-MiniLM-L6-v2 model
- **Mistral Small 3.1 24B**: Advanced 24B parameter model via OpenRouter API for high-quality candidate assessments
- **PyMuPDF**: Extracts text content from PDF documents
- **Scikit-learn**: Performs cosine similarity calculations for ranking
