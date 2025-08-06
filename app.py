import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from transformers import pipeline
import torch


st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def generate_embeddings(_model, texts):
    return _model.encode(texts, show_progress_bar=False, convert_to_tensor=False)

@st.cache_resource
def load_summarizer():
    import requests
    import os
    from pathlib import Path
    
    if Path("./mistral-7b-instruct-v0.2.Q4_K_M.gguf").exists():
        from llama_cpp import Llama
        os.environ['GGML_METAL'] = '0'
        
        llm = Llama(
            model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=32768,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False
        )
        return llm
    else:
        return "huggingface_api"

def extract_name_from_text(text):
    lines = text.strip().split('\n')[:5]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        words = line.split()
        if 2 <= len(words) <= 4:
            if all(word.replace('-', '').replace("'", "").isalpha() for word in words):
                if not line.isupper() and any(c.isupper() for c in line):
                    return line
    
    return None

def extract_text_from_files(uploaded_files):
    processed_resumes = []
    
    for file in uploaded_files:
        try:
            text = ""
            
            if file.type == "application/pdf":
                with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                
            elif file.type == "text/plain":
                text = file.read().decode('utf-8')
            
            if not text.strip():
                st.warning(f"No text could be extracted from {file.name}")
                continue
            
            candidate_name = extract_name_from_text(text)
            if not candidate_name:
                candidate_name = file.name.rsplit('.', 1)[0]
            
            processed_resumes.append({
                'name': candidate_name,
                'text': text.strip(),
                'filename': file.name
            })
            
        except Exception as e:
            st.warning(f"Could not process {file.name}: {str(e)}")
            continue
    
    return processed_resumes

def generate_ai_summary(job_description: str,
                        resume_text: str,
                        similarity_score: float,
                        mistral_llm) -> str:
    try:
        match_percentage = similarity_score * 100
        
        prompt = f"""[INST] You are an expert technical recruiter reviewing a candidate's resume in relation to a specific job description. A semantic similarity score of {match_percentage:.1f}% was computed, reflecting the degree of alignment between the candidate's experience and the role's requirements. Use this score to guide your judgment: scores around 20–40% indicate limited overlap with some relevant elements; 40–60% suggests partial alignment; 60–80% signals strong fit across core criteria; and scores above 80% reflect an exceptional match.

Write a concise, specific, and insightful 2–3 sentence evaluation that accurately reflects the candidate's qualifications. Prioritize clarity over formality, and focus on areas of demonstrated alignment, notable gaps, and real potential to grow into the role. Avoid generalities or generic soft skill phrases unless grounded in observable evidence from the resume. This summary should read as if you were presenting the candidate to a hiring manager, honest, efficient, and grounded in substance.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

ASSESSMENT: [/INST]"""

        if mistral_llm == "huggingface_api":
            import requests
            import os
            
            try:
                hf_token = st.secrets.get("HF_TOKEN", None)
            except:
                hf_token = os.getenv("HF_TOKEN")
            
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            
            api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "stop": ["[INST]"]
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get("generated_text", "").replace(prompt, "").strip()
                else:
                    summary = "Could not generate summary."
            else:
                summary = "Could not generate summary."
        else:
            response = mistral_llm(
                prompt,
                max_tokens=500,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.1,
                stop=["[INST]"],
                echo=False
            )
            
            if 'choices' in response and len(response['choices']) > 0:
                summary = response['choices'][0]['text'].strip()
            else:
                summary = "Could not generate summary."
        
        if summary and len(summary) > 10:
            summary = summary.replace("[/INST]", "").replace("</s>", "").strip()
            
            if (summary.lower().startswith(job_description[:50].lower()) or 
                summary.lower().startswith(resume_text[:50].lower())):
                summary = "Could not generate summary."
            
            return summary
        else:
            return "Could not generate summary."

    except Exception as e:
        return f"Could not generate summary due to an error: {str(e)}"

st.title("Candidate Recommendation Engine")

st.subheader("Job Description")
job_description = st.text_area(
    "Paste the Job Description Here",
    height=200,
    placeholder="Enter the job description that you want to match candidates against..."
)

st.subheader("Upload Resumes")
uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Upload multiple resume files in PDF or TXT format"
)

st.subheader("Configuration")

col1, col2 = st.columns(2)

with col1:
    enable_ai_summaries = st.checkbox(
        "Enable AI-generated summaries",
        value=True,
        help="Generate AI summaries explaining the candidate's fit for the job"
    )

with col2:
    if uploaded_files:
        max_candidates = len(uploaded_files)
        default_candidates = max_candidates
        
        num_candidates = st.slider(
            "Number of top candidates to display",
            min_value=0,
            max_value=max_candidates,
            value=default_candidates,
            help="Select how many top-ranked candidates to show in results"
        )

st.subheader("Analysis")
analyze_button = st.button(
    "Analyze Candidates",
    type="primary",
    help="Start the candidate analysis process"
)

if analyze_button:
    if not job_description.strip():
        st.warning("Please enter a job description before analyzing candidates.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume file before analyzing candidates.")
    else:
        with st.spinner('Analyzing resumes...'):
            model = load_model()
            
            mistral_llm = None
            if enable_ai_summaries:
                mistral_llm = load_summarizer()
            
            processed_resumes = extract_text_from_files(uploaded_files)
            
            if not processed_resumes:
                st.error("No valid resumes to process. Please check your uploaded files.")
            else:
                st.success(f"Successfully processed {len(processed_resumes)} resume(s)")
                
                job_embedding = generate_embeddings(model, [job_description])
                
                resume_texts = [resume['text'] for resume in processed_resumes]
                resume_embeddings = generate_embeddings(model, resume_texts)
                
                similarity_scores = cosine_similarity(job_embedding, resume_embeddings)[0]
                
                results = []
                for i, resume in enumerate(processed_resumes):
                    results.append({
                        'name': resume['name'],
                        'filename': resume['filename'],
                        'text': resume['text'],
                        'similarity_score': similarity_scores[i]
                    })
                
                results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                st.subheader(f"Top {min(num_candidates, len(results))} Candidates")
                
                for i, candidate in enumerate(results[:num_candidates]):
                    with st.container():
                        st.subheader(f"{i+1}. {candidate['name']}")
                        
                        metric_col, info_col = st.columns([1, 2])
                        
                        with metric_col:
                            similarity_percentage = candidate['similarity_score'] * 100
                            st.metric(
                                label="Match Score",
                                value=f"{similarity_percentage:.1f}%"
                            )
                        
                        with info_col:
                            if candidate['name'] != candidate['filename'].rsplit('.', 1)[0]:
                                st.caption(f"File: {candidate['filename']}")
                        
                        if enable_ai_summaries and mistral_llm is not None:
                            with st.expander("Candidate Assessment"):
                                with st.spinner("Generating AI summary..."):
                                    summary = generate_ai_summary(
                                        job_description, 
                                        candidate['text'], 
                                        candidate['similarity_score'],
                                        mistral_llm
                                    )
                                    st.write(summary)
                        
                        st.divider()