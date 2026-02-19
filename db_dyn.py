from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import groq
import json
import re
from pydub import AudioSegment, silence
import whisper
import librosa
import numpy as np
import os
import subprocess
import cv2
import time
import uuid
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dotenv import load_dotenv
from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import fitz  # PyMuPDF for resume parsing
import google.generativeai as genai
from gtts import gTTS
import os
import uuid
from multiprocessing import Process, Semaphore
from voice_feedback_generator import generate_dynamic_voice_feedback
from concurrent.futures import ThreadPoolExecutor

from face import analyze_video, store_face_feedback  # at the top, after other imports

AUDIO_FOLDER = "static/audio"

def generate_tts(question_text):
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    tts = gTTS(text=question_text, lang="en")
    tts.save(filepath)

    return f"/static/audio/{filename}"

# This is the single, integrated Flask app
app = Flask(__name__)
CORS(app)

# MySQL + Bcrypt for login and question generation (flask_mysqldb)
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'interview@1234')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'mockinterview')

mysql_db= MySQL(app)
bcrypt = Bcrypt(app)

# Gemini / question generation config
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# ================================
# DATABASE CONFIGURATION
# ================================

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'interview@1234'),
    'database': os.getenv('MYSQL_DB', 'mockinterview')
}


# Questions per level
QUESTIONS_PER_LEVEL = {
    'beginner': 10,
    'intermediate': 12,
    'advanced': 15
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def extract_text_from_pdf(file_stream):
    """Extract text from PDF resume (ported from db_qgen.py)"""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def clean_json_response(response_text):
    """Clean and parse JSON response from Gemini (ported from db_qgen.py)"""
    # Remove markdown code blocks if present
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'\s*```', '', response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass
        raise Exception("Failed to parse JSON response from AI")


def generate_beginner_questions(resume_text, job_title, company_name):
    """Generate beginner level questions (ported from db_qgen.py)"""
    prompt = f"""
You are a senior HR professional and technical interviewer conducting a BEGINNER-level interview. 
Generate exactly 10 professional, polished interview questions that assess foundational skills while maintaining a respectful, corporate tone.

*Candidate Information:*
- Resume Content: {resume_text[:3000]}
- Target Job Role: {job_title}
- Target Company: {company_name}

*Professional Tone Requirements:*
- Use formal, corporate language appropriate for a professional setting
- Avoid casual or overly simplified phrasing
- Maintain clarity while using industry-standard terminology
- Frame questions in a way that shows respect for the candidate's potential
- Ensure questions sound like they come from an experienced interviewer

*Question Generation Guidelines:*
- Assess basic technical/functional knowledge relevant to {job_title}
- Evaluate professional demeanor and workplace readiness
- Understand motivation and alignment with {company_name}'s values
- Gauge ability to handle entry-level responsibilities
- Questions should be answerable in 1-2 minutes

*Question Categories (Generate EXACTLY in this order):*

*1. PROFESSIONAL INTRODUCTION (1 question):*
- A formal introductory question that allows the candidate to present themselves professionally

*2. BEHAVIORAL & PROFESSIONAL COMPETENCIES (3 questions):*
- Professional communication and teamwork abilities
- Approach to entry-level challenges and deadlines
- Receptiveness to feedback and professional development

*3. RESUME ANALYSIS (3 questions):*
- Academic preparation and relevant coursework for {job_title}
- Application of listed technical skills in practical contexts
- Learning outcomes from projects or internships

*4. ROLE-SPECIFIC UNDERSTANDING (2 questions):*
- Comprehension of {job_title} responsibilities
- Foundational knowledge in required technical/functional areas

*5. COMPANY ALIGNMENT (1 question):*
- Interest in and basic understanding of {company_name}

*Output Format:*
Return a JSON array with exactly 10 questions. Each question should have:
- question_number (1-10)
- question_category
- question_text (professionally phrased)
- purpose
- expected_answer_duration
- evaluation_criteria
- good_answer_indicators (array)
- red_flags (array)

Generate the questions now, ensuring they sound professional and polished.
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return clean_json_response(response.text)


def generate_intermediate_questions(resume_text, job_title, company_name):
    """Generate intermediate level questions (ported from db_qgen.py)"""
    prompt = f"""
You are an expert interviewer conducting an INTERMEDIATE-level interview.
You are a seasoned technical and hr manager conducting an INTERMEDIATE-level interview.
Generate exactly 12 professional interview questions that assess depth of expertise and workplace effectiveness.

*Candidate Information:*
- Resume: {resume_text[:3000]}
- Job Role: {job_title}
- Company: {company_name}

*Professional Tone Requirements:*
- Use authoritative, experienced interviewer tone
- Employ industry-standard technical and business terminology
- Frame questions to assess professional judgment and decision-making
- Maintain respectful yet challenging tone appropriate for experienced candidates

*Interview Focus:*
- Depth of technical understanding and practical application
- Problem-solving methodology in professional contexts
- Project ownership and team collaboration
- Communication effectiveness with stakeholders
- Applied knowledge from professional experience

*Question Distribution:*
1. Professional Background & Career Trajectory (1)
2. Behavioral & Professional Scenarios (3)
3. Experience Analysis & Technical Deep Dive (3)
4. Role-Specific Technical Assessment (3)
5. Company Alignment & Professional Growth (2)

*Output Format:*
Return ONLY a valid JSON array (no markdown, no explanations) like this:
[
  {{
    "question_number": 1,
    "question_category": "Professional Background",
    "question_text": "Could you walk me through your career progression and how it has prepared you for this role?",
    "purpose": "Assess career trajectory and role alignment",
    "expected_answer_duration": "2-3 minutes",
    "evaluation_criteria": "Clarity, relevance, progression logic",
    "good_answer_indicators": ["Structured narrative", "Clear role connections"],
    "red_flags": ["Disjointed explanation", "Lack of preparation"]
  }},
  ...
]

Generate exactly 12 professionally phrased questions.
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return clean_json_response(response.text)


def generate_advanced_questions(resume_text, job_title, company_name):
    """Generate advanced level questions (ported from db_qgen.py)"""
    prompt = f"""
You are a C-level executive or senior director conducting an ADVANCED-level interview.
Generate exactly 15 high-stakes, strategic interview questions that assess leadership, vision, and executive presence.

*Candidate Information:*
- Resume: {resume_text[:3000]}
- Target Role: {job_title}
- Company: {company_name}

*Professional Tone Requirements:*
- Use executive-level, strategic language
- Frame questions to assess thought leadership and strategic impact
- Employ business and technical terminology at director/VP level
- Questions should reflect high-stakes decision-making scenarios
- Maintain tone of peer-to-peer discussion among senior professionals

*Interview Focus:*
- Strategic vision and business impact assessment
- Leadership philosophy and team development
- Complex system/process design and optimization
- Cross-functional influence and organizational change management
- Alignment with {company_name}'s strategic objectives

*Question Distribution:*
1. Executive Introduction & Strategic Perspective (1)
2. Leadership & Organizational Influence (4)
3. Strategic Technical/Business Decision Making (4)
4. High-Complexity Problem Solving (4)
5. Company Vision & Long-term Alignment (2)

*Output Format:*
Return ONLY a JSON array, no extra text, formatted like this:
[
  {{
    "question_number": 1,
    "question_category": "Strategic Leadership",
    "question_text": "How would you approach developing a strategic roadmap for our {job_title} function over the next 3 years?",
    "purpose": "Evaluate strategic thinking and vision alignment",
    "expected_answer_duration": "3-4 minutes",
    "evaluation_criteria": "Strategic depth, business acumen, alignment",
    "good_answer_indicators": ["Clear strategic framework", "Stakeholder consideration"],
    "red_flags": ["Tactical focus only", "Lacks business context"]
  }},
  ...
]

Generate exactly 15 executive-level questions.
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return clean_json_response(response.text)


def save_questions_to_db(questions_data, session_id, difficulty_level='beginner'):
    """
    Save generated questions to database (ported from db_qgen.py)
    """
    try:
        connection = get_db_connection()
        if connection is None:
            print("DB connection failed in save_questions_to_db")
            return False

        cursor = connection.cursor()

        # Map difficulty level to match database enum
        level_mapping = {
            'beginner': 'beginner',
            'intermediate': 'intermedite',  # match existing table typo
            'advanced': 'advanced'
        }

        db_level = level_mapping.get(difficulty_level, 'beginner')

        # Insert each question
        for question in questions_data:
            # Extract question text (handle different response formats)
            if isinstance(question, dict):
                question_text = question.get('question_text', '')
            else:
                question_text = str(question)

            if not question_text:
                continue

            # Clean numbering at start
            question_text = re.sub(r'^\d+[\.\)\-\s]*', '', question_text.strip())

            sql = """
                INSERT INTO questions (session_id, question_text, difficulty_level, created_at)
                VALUES (%s, %s, %s, NOW())
            """
            cursor.execute(sql, (session_id, question_text, db_level))

        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"Database error (save_questions_to_db): {str(e)}")
        try:
            connection.rollback()
        except Exception:
            pass
        return False

def get_questions_for_level(level):
    """
    Retrieve questions for a specific level from the database
    Returns: List of questions in the order they should be asked
    """
    connection = get_db_connection()
    if connection is None:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get questions for this level
        cursor.execute("""
            SELECT questions_id, question_text, difficulty_level
            FROM questions 
            WHERE difficulty_level = %s 
            ORDER BY questions_id 
            LIMIT %s
        """, (level, QUESTIONS_PER_LEVEL[level]))
        
        questions = cursor.fetchall()
        
        print(f"Retrieved {len(questions)} questions for level: {level}")
        
        cursor.close()
        connection.close()
        
        return questions
        
    except Error as e:
        print(f"Error retrieving questions: {e}")
        if connection:
            connection.close()
        return []

def check_if_questions_exist_for_level(level):
    """Check if we have enough questions for a specific level"""
    connection = get_db_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM questions 
            WHERE difficulty_level = %s
        """, (level,))
        
        result = cursor.fetchone()
        count = result['count'] if result else 0
        
        cursor.close()
        connection.close()
        
        required_count = QUESTIONS_PER_LEVEL.get(level, 10)
        return count >= required_count
        
    except Error as e:
        print(f"Error checking questions: {e}")
        if connection:
            connection.close()
        return False



def store_voice_feedback(student_id, session_id, question_id, question_number, strengths, improvements):
    """Store voice feedback in database"""
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to database for voice feedback")
        return False
    
    try:
        cursor = connection.cursor()
        
        # Convert lists to JSON strings
        strengths_json = json.dumps(strengths) if strengths else "[]"
        improvements_json = json.dumps(improvements) if improvements else "[]"
        
        # First check if table exists and has correct column names
        cursor.execute("SHOW COLUMNS FROM voice_feedback LIKE 'session_id'")
        session_id_exists = cursor.fetchone()
        
        if session_id_exists:
            # Use session_id column
            try:
                cursor.execute("""
                    INSERT INTO voice_feedback 
                    (student_id, session_id, strengths, improvements, q_no)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    strengths_json,
                    improvements_json,
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error inserting into voice_feedback (session_id): {e}")
        else:
            # Use resumeid column
            try:
                cursor.execute("""
                    INSERT INTO voice_feedback 
                    (student_id, resumeid, strengths, improvements, q_no)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    strengths_json,
                    improvements_json,
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error inserting into voice_feedback (resumeid): {e}")
        
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Voice feedback stored for question {question_id}")
        return True
        
    except Error as e:
        print(f"❌ Error storing voice feedback: {e}")
        if connection:
            connection.close()
        return False

def store_content_feedback(student_id, session_id, question_id, question_number, response, content_analysis, sample_answer):
    """Store content feedback in database"""
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to database for content feedback")
        return False
    
    try:
        cursor = connection.cursor()
        
        # Extract scores from content_analysis
        if content_analysis and isinstance(content_analysis, dict):
            content_score = content_analysis.get('content_score', 0)
            overall_score = content_analysis.get('overall_score', 0)
            relevance_score = content_analysis.get('relevance_score', 0)
            structure_score = content_analysis.get('structure_score', 0)
            improvements = content_analysis.get('improvements', [])
            strengths = content_analysis.get('strengths', [])
        else:
            content_score = 0
            overall_score = 0
            relevance_score = 0
            structure_score = 0
            improvements = []
            strengths = []
        
        # Convert lists to JSON strings
        improvements_json = json.dumps(improvements) if improvements else "[]"
        strengths_json = json.dumps(strengths) if strengths else "[]"
        
        # First check if table exists and has correct column names
        cursor.execute("SHOW COLUMNS FROM content_feedback LIKE 'session_id'")
        session_id_exists = cursor.fetchone()
        
        if session_id_exists:
            # Use session_id column
            try:
                cursor.execute("""
                    INSERT INTO content_feedback 
                    (student_id, session_id, response, content_score, overall, relevance, 
                     structure, improvements, strengths, sample_answer, q_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    response,
                    str(content_score),
                    str(overall_score),
                    str(relevance_score),
                    str(structure_score),
                    improvements_json,
                    strengths_json,
                    sample_answer if sample_answer else "",
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error inserting into content_feedback (session_id): {e}")
        else:
            # Use resumeid column
            try:
                cursor.execute("""
                    INSERT INTO content_feedback 
                    (student_id, resumeid, response, content_score, overall, relevance, 
                     structure, improvements, strengths, sample_answer, q_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    response,
                    str(content_score),
                    str(overall_score),
                    str(relevance_score),
                    str(structure_score),
                    improvements_json,
                    strengths_json,
                    sample_answer if sample_answer else "",
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error inserting into content_feedback (resumeid): {e}")
        
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Content feedback stored for question {question_id}")
        return True
        
    except Error as e:
        print(f"❌ Error storing content feedback: {e}")
        if connection:
            connection.close()
        return False
    
def store_skipped_question(student_id, session_id, question_id, question_number, sample_answer=""):
    """Store skipped question feedback in database"""
    connection = get_db_connection()
    if connection is None:
        print("Failed to connect to database for skipped question")
        return False
    
    try:
        cursor = connection.cursor()
        
        # Check table structure for voice_feedback
        cursor.execute("SHOW COLUMNS FROM voice_feedback LIKE 'session_id'")
        voice_session_id_exists = cursor.fetchone()
        
        if voice_session_id_exists:
            # Use session_id column
            try:
                cursor.execute("""
                    INSERT INTO voice_feedback 
                    (student_id, session_id, strengths, improvements, q_no)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    json.dumps(["NOT_ANSWERED"]),
                    json.dumps(["Question was skipped"]),
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error storing skipped voice_feedback: {e}")
        else:
            # Use resumeid column
            try:
                cursor.execute("""
                    INSERT INTO voice_feedback 
                    (student_id, resumeid, strengths, improvements, q_no)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    json.dumps(["NOT_ANSWERED"]),
                    json.dumps(["Question was skipped"]),
                    int(question_id)
                ))
            except Exception as e:
                print(f"❌ Error storing skipped voice_feedback (resumeid): {e}")
        
        # Check table structure for content_feedback
        cursor.execute("SHOW COLUMNS FROM content_feedback LIKE 'session_id'")
        content_session_id_exists = cursor.fetchone()
        
        if content_session_id_exists:
            # Use session_id column
            try:
                cursor.execute("""
                    INSERT INTO content_feedback 
                    (student_id, session_id, response, content_score, overall, relevance, 
                     structure, improvements, strengths, sample_answer, q_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    "NOT_ANSWERED",
                    "0",
                    "0",
                    "0",
                    "0",
                    json.dumps(["Question was skipped"]),
                    json.dumps(["NOT_ANSWERED"]),
                    sample_answer,
                    int(question_id)  # Use question_id (PK), not question_number
                ))
            except Exception as e:
                print(f"❌ Error storing skipped content_feedback: {e}")
        else:
            # Use resumeid column
            try:
                cursor.execute("""
                    INSERT INTO content_feedback 
                    (student_id, resumeid, response, content_score, overall, relevance, 
                     structure, improvements, strengths, sample_answer, q_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    "NOT_ANSWERED",
                    "0",
                    "0",
                    "0",
                    "0",
                    json.dumps(["Question was skipped"]),
                    json.dumps(["NOT_ANSWERED"]),
                    sample_answer,
                    int(question_id)
                ))
            except Exception as e:
                 print(f"❌ Error storing skipped content_feedback (resumeid): {e}")
        
        # Check table structure for face_feedback (for session_id support)
        cursor.execute("SHOW COLUMNS FROM face_feedback LIKE 'session_id'")
        face_session_id_exists = cursor.fetchone()
        
        if face_session_id_exists:
            # Use session_id column
            try:
                cursor.execute("""
                    INSERT INTO face_feedback 
                    (student_id, session_id, posture_quality, posture_feedback, alignment, 
                     alignment_feedback, eye_contact, eyecontact_feedback, touch, touch_feedback, 
                     strength, improvements, tips, qno)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student_id,
                    session_id,
                    "NOT_ANSWERED",
                    "Question was skipped",
                    "NOT_ANSWERED",
                    "Question was skipped",
                    "NOT_ANSWERED",
                    "Question was skipped",
                    "NOT_ANSWERED",
                    "Question was skipped",
                    "NOT_ANSWERED",
                    "Question was skipped",
                    "Question was skipped",
                    int(question_id)  # Fix: Use question_id (PK), not question_number
                ))
            except Exception as face_err:
                print(f"⚠️ Could not store skipped face_feedback row: {face_err}")
                
        # Also store the skipped attempt in responses table
        store_response(session_id, question_id, student_id, "NOT_ANSWERED", "Question was skipped")
        
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Skipped question {question_id} stored in database")
        return True
        
    except Error as e:
        print(f"❌ Error storing skipped question: {e}")
        print(f"❌ Full error details: {e}")
        if connection:
            connection.close()
        return False               
        
        # ================================
# MULTIPROCESSING CONFIGURATION
# ================================
MAX_PROCESSES = 2
process_semaphore = Semaphore(MAX_PROCESSES)

def analyze_answer_process(video_path, question, question_id, session_id, student_id, question_number):
    """
    Background process function - contains exact same analysis logic as before.
    Creates its own DB connection inside the process.
    """
    # Store video response first - transcript will be updated if needed or left empty
    store_response(session_id, question_id, student_id, video_path)

    with process_semaphore:
        try:
            print(f"🔄 [Process] Starting analysis for Question {question_number}")
            
            # Audio processing for voice analysis
            audio_path = extract_audio(video_path)
            transcribed_text = transcribe_text(audio_path)
            
            print(f"📝 [Process] Transcribed text: {transcribed_text}")
            
            # VOICE ANALYSIS (using existing tools)
            audio_features = analyze_audio_features(audio_path)
            pause_analysis = analyze_pauses_enhanced(audio_path)
            filler_analysis = analyze_filler_patterns(transcribed_text)
            
            analysis_results = {
                'pause_analysis': pause_analysis,
                'filler_analysis': filler_analysis,
                'audio_features': audio_features
            }
            
            confidence_score = calculate_advanced_confidence(analysis_results)
            confidence_category = get_confidence_category(confidence_score)

            sample_answer = None
            content_analysis = None
            face_result = None
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Voice Groq (uses GROQ_VOICE_API_KEY via voice_feedback_generator)
                voice_future = executor.submit(
                    generate_dynamic_voice_feedback,
                    analysis_results,
                    confidence_score
                )
                
                # Face Analysis - Parallelized
                print(f"🎥 [Process] Submitting face analysis to parallel executor...")
                face_future = executor.submit(analyze_video, video_path)
                
                sample_future = None
                content_future = None
                if question and transcribed_text.strip():
                    # Content Groq (uses GROQ_API_KEY in this file)
                    sample_future = executor.submit(generate_sample_answer, question)
                    content_future = executor.submit(
                        analyze_answer_content,
                        question,
                        transcribed_text
                    )
                    
                # Wait for results
                feedback_data = voice_future.result()
                print(f"✅ [Process] Dynamic feedback generated")
                
                face_result = face_future.result()
                print(f"✅ [Process] Face analysis completed")
                if sample_future is not None:
                    sample_answer = sample_future.result()
                    print(f"📋 [Process] Sample answer generated: {sample_answer is not None}")
                if content_future is not None:
                    content_analysis = content_future.result()
                    print(f"📊 [Process] Content analysis completed: {content_analysis is not None}")

            # Calculate speaking rate for metrics
            speaking_rate = 0
            if audio_features['duration'] > 0:
                speaking_rate = (filler_analysis['word_count'] / audio_features['duration']) * 60

            # STORE FACE FEEDBACK
            if face_result and session_id and question_id:
                try:
                    face_db_success = store_face_feedback(
                        face_result, 
                        student_id=student_id,
                        session_id=session_id, 
                        qno=int(question_id)
                    )
                    if face_db_success:
                        print(f"✅ [Process] Face feedback stored in database")
                    else:
                        print(f"⚠️ [Process] Face feedback storage returned False")
                except Exception as e:
                    print(f"⚠️ [Process] Failed to store face feedback: {e}")

            # STORE FEEDBACK IN DATABASE
            if session_id and question_id:
                print(f"📊 [Process] Attempting to store feedback...")
                print(f"   Session ID: {session_id}")
                print(f"   Question ID: {question_id}")
                print(f"   Strengths count: {len(feedback_data.get('strengths', []))}")
                print(f"   Improvements count: {len(feedback_data.get('improvements', []))}")
                try:
                    # Store voice feedback
                    voice_result = store_voice_feedback(
                        student_id=student_id,
                        session_id=session_id,
                        question_id=question_id,
                        question_number=int(question_number),
                        strengths=feedback_data['strengths'],
                        improvements=feedback_data['improvements']
                    )
                    print(f"📊 [Process] store_voice_feedback returned: {voice_result}")
                    
                    # Store content feedback if analysis was performed
                    if content_analysis and transcribed_text.strip():
                        content_result = store_content_feedback(
                            student_id=student_id,
                            session_id=session_id,
                            question_id=question_id,
                            question_number=int(question_number),
                            response=transcribed_text,
                            content_analysis=content_analysis,
                            sample_answer=sample_answer
                        )
                        print(f"📊 [Process] store_content_feedback returned: {content_result}")
                    elif transcribed_text.strip():  # If we have transcript but no content analysis
                        content_result = store_content_feedback(
                            student_id=student_id,
                            session_id=session_id,
                            question_id=question_id,
                            question_number=int(question_number),
                            response=transcribed_text,
                            content_analysis=None,
                            sample_answer=sample_answer
                        )
                        print(f"📊 [Process] store_content_feedback returned: {content_result}")
                    
                    print(f"✅ [Process] Feedback stored in database for Q{question_number}")
                except Exception as e:
                    print(f"⚠️ [Process] Failed to store feedback in database: {e}")
                    import traceback
                    traceback.print_exc()

            # Cleanup
            for p in [audio_path, video_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            print(f"✅ [Process] Analysis completed for Question {question_number}")
            
        except Exception as e:
            print(f"❌ [Process] Analysis error: {str(e)}")
            
            # Cleanup on error
            for p in [video_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# ===============================
# GROQ API CONFIGURATION
# ================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=GROQ_API_KEY)


SAMPLE_ANSWER_MODEL = "llama-3.1-8b-instant"
ANALYSIS_MODEL = "llama-3.1-8b-instant"

print(f"✅ Using Groq with model: {SAMPLE_ANSWER_MODEL}")

# Load Whisper model once
whisper_model = whisper.load_model("small")

# Enhanced filler words list
FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually", "basically", "literally", "right", "okay"]

# Common words that don't matter if repeated
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
    'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'can', 'may', 'might', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these',
    'those', 'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how'
}

# ================================
# VOICE ANALYSIS FUNCTIONS
# ================================

def extract_audio(video_path):
    """Extract audio from video file"""
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    try:
        result = subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
            "-ar", "44100", "-ac", "1", audio_path, "-y"
        ], check=True, capture_output=True, text=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr}")

def analyze_audio_features(audio_path):
    """Analyze audio features including pitch, energy, and duration"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=400)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            if magnitudes[index, t] > 0.1:
                pitch_values.append(pitches[index, t])
        
        pitch_values = np.array(pitch_values)
        avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # Energy analysis
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        energy_std = float(np.std(rms))
        
        return {
            'duration': duration,
            'avg_pitch': avg_pitch,
            'pitch_std': pitch_std,
            'avg_energy': avg_energy,
            'energy_std': energy_std,
            'raw_audio': y,
            'sample_rate': sr
        }
    except Exception as e:
        raise Exception(f"Audio analysis error: {str(e)}")

def analyze_pauses_enhanced(audio_path):
    """Analyze pauses in speech"""
    try:
        audio = AudioSegment.from_wav(audio_path)
        silences = silence.detect_silence(audio, min_silence_len=300, silence_thresh=audio.dBFS - 20)
        
        total_silence = sum([end - start for start, end in silences])
        total_duration = len(audio)
        silence_ratio = total_silence / total_duration if total_duration > 0 else 0
        
        strategic_pauses = []
        hesitant_pauses = []
        
        for start, end in silences:
            pause_duration = end - start
            if 200 <= pause_duration <= 800:
                strategic_pauses.append(pause_duration)
            elif pause_duration > 800:
                hesitant_pauses.append(pause_duration)
        
        return {
            'silence_ratio': silence_ratio,
            'strategic_pauses_count': len(strategic_pauses),
            'hesitant_pauses_count': len(hesitant_pauses),
            'total_pauses': len(silences)
        }
    except Exception as e:
        raise Exception(f"Pause analysis error: {str(e)}")

def analyze_filler_patterns(transcribed_text):
    """Analyze filler words and repetition patterns"""
    try:
        words = transcribed_text.lower().split()
        word_count = len(words)
        
        filler_count = sum(words.count(filler) for filler in FILLER_WORDS)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        word_freq = {}
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word)
            if len(clean_word) > 0:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        important_repeated_words = {}
        repetition_penalty = 0
        
        for word, count in word_freq.items():
            if (word not in COMMON_WORDS and len(word) > 3 and count > 4):
                important_repeated_words[word] = count
                repetition_penalty += (count - 4) * 2
        
        repetition_penalty = min(20, repetition_penalty)
        
        return {
            'filler_count': filler_count,
            'filler_ratio': filler_ratio,
            'important_repeated_words': important_repeated_words,
            'repetition_penalty': repetition_penalty,
            'word_count': word_count
        }
    except Exception as e:
        raise Exception(f"Filler analysis error: {str(e)}")

def transcribe_text(audio_path):
    """Transcribe audio to text"""
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

def calculate_advanced_confidence(analysis_results):
    """Calculate confidence score based on multiple factors"""
    try:
        base_score = 100
        
        pause_analysis = analysis_results['pause_analysis']
        filler_analysis = analysis_results['filler_analysis']
        audio_features = analysis_results['audio_features']
        word_count = filler_analysis['word_count']
        duration = audio_features['duration']
        
        speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
        
        # Calculate penalties
        pause_penalty = pause_analysis['silence_ratio'] * 30
        filler_penalty = filler_analysis['filler_ratio'] * 40
        repetition_penalty = filler_analysis['repetition_penalty']
        
        rate_penalty = abs(speaking_rate - 150) * 0.15
        energy_penalty = max(0, (0.02 - audio_features['avg_energy']) * 500)
        pitch_penalty = max(0, (50 - audio_features['pitch_std']) * 0.2)
        
        hesitant_penalty = pause_analysis['hesitant_pauses_count'] * 2
        
        total_penalty = (pause_penalty + filler_penalty + repetition_penalty + 
                        rate_penalty + energy_penalty + pitch_penalty + hesitant_penalty)
        
        final_score = max(0, min(100, base_score - total_penalty))
        return round(final_score, 2)
    except Exception as e:
        print(f"Confidence calculation error: {e}")
        return 50.0

def get_confidence_category(score):
    """Convert score to category with updated ranges"""
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Average"
    elif score >= 30:
        return "Below Average"
    else:
        return "Poor"



# ================================
# GROQ API CALL FUNCTION
# ================================

def call_groq_api(prompt, model=SAMPLE_ANSWER_MODEL):
    """Call Groq API with retry logic"""
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {str(e)[:200]}")
        raise Exception(f"Groq API failed: {str(e)}")

def extract_json_from_text(text):
    """Extract JSON from text response"""
    try:
        # Try to find JSON pattern in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return None

# ================================
# QUESTION TYPE DETECTION SYSTEM
# ================================

def detect_question_type(question):
    """
    Automatically detect the type of interview question using keyword detection first,
    then fall back to Groq API classification if needed.
    
    Returns: 'introduction', 'technical', 'behavioral', 'hr', 'company', 'situational', 'general'
    """
    # First try keyword-based detection
    question_lower = question.lower().strip()
    
    # Introduction/Self-introduction questions
    intro_keywords = [
        'introduce yourself', 'tell me about yourself', 'describe yourself',
        'walk me through your background', 'tell us about you', 'who are you',
        'brief introduction', 'about yourself', 'your background', 'personal introduction'
    ]
    if any(keyword in question_lower for keyword in intro_keywords):
        print(f"🔍 Keyword detection: introduction")
        return 'introduction'
    
    # Technical questions - programming, CS concepts, domain-specific
    technical_keywords = [
        'explain', 'what is', 'what are', 'define', 'difference between', 'how does',
        'implement', 'algorithm', 'data structure', 'code', 'function', 'method',
        'class', 'object', 'oops', 'oop', 'inheritance', 'polymorphism', 'encapsulation',
        'database', 'sql', 'query', 'programming', 'technical', 'system design',
        'complexity', 'time complexity', 'space complexity', 'big o', 'runtime',
        'framework', 'library', 'api', 'rest', 'http', 'protocol', 'network',
        'memory', 'pointer', 'variable', 'syntax', 'compile', 'runtime', 'python',
        'java', 'javascript', 'react', 'node', 'aws', 'cloud', 'docker', 'kubernetes'
    ]
    if any(keyword in question_lower for keyword in technical_keywords):
        print(f"🔍 Keyword detection: technical")
        return 'technical'
    
    # Behavioral questions - past experiences using STAR method
    behavioral_keywords = [
        'tell me about a time', 'describe a situation', 'give me an example',
        'tell us about when', 'share an experience', 'describe when you',
        'have you ever', 'can you recall', 'think of a time',
        'describe how you handled', 'give an example of', 'talk about a time',
        'example from your past', 'experience where', 'situation you faced'
    ]
    if any(keyword in question_lower for keyword in behavioral_keywords):
        print(f"🔍 Keyword detection: behavioral")
        return 'behavioral'
    
    # Company-specific questions
    company_keywords = [
        'why this company', 'why do you want to work here', 'why our company',
        'what do you know about', 'why are you interested in',
        'what attracts you', 'why should we hire you', 'why you',
        'what interests you about', 'our company', 'our organization',
        'company culture', 'our mission', 'our values', 'our product'
    ]
    if any(keyword in question_lower for keyword in company_keywords):
        print(f"🔍 Keyword detection: company")
        return 'company'
    
    # Situational/Hypothetical questions
    situational_keywords = [
        'what would you do', 'how would you handle', 'if you were',
        'imagine', 'suppose', 'what if', 'how would you approach',
        'in a situation where', 'if faced with', 'scenario', 'hypothetical',
        'case study', 'what will you do', 'how will you respond'
    ]
    if any(keyword in question_lower for keyword in situational_keywords):
        print(f"🔍 Keyword detection: situational")
        return 'situational'
    
    # HR/General questions - strengths, weaknesses, goals
    hr_keywords = [
        'strength', 'weakness', 'goals', 'where do you see yourself',
        'career goals', 'aspirations', 'salary', 'expectations',
        'greatest achievement', 'biggest failure', 'work style',
        'management style', 'conflict resolution', 'teamwork',
        'leadership', 'motivation', 'work-life balance', 'stress management'
    ]
    if any(keyword in question_lower for keyword in hr_keywords):
        print(f"🔍 Keyword detection: hr")
        return 'hr'
    
    # If keyword detection fails, use Groq API for classification
    print(f"🔍 Keyword detection inconclusive, using Groq API for classification")
    return classify_question_with_groq(question)

def classify_question_with_groq(question):
    """Classify question type using Groq API when keyword detection fails"""
    try:
        prompt = f"""Classify this interview question into exactly one of these 6 categories:

1. introduction - Questions about self-introduction, background, experience
2. technical - Questions about technical knowledge, programming, systems, concepts
3. behavioral - Questions about past experiences using STAR method (Situation, Task, Action, Result)
4. company - Questions about the specific company, why you want to work here
5. hr - Questions about strengths, weaknesses, career goals, salary, teamwork
6. situational - Hypothetical scenarios, what would you do if situations

Question: "{question}"

Return ONLY the category name (one word: introduction, technical, behavioral, company, hr, or situational) without any additional text, explanations, or punctuation."""

        print(f"🤖 Calling Groq API for question classification: {question[:100]}...")
        
        response = call_groq_api(prompt, model=SAMPLE_ANSWER_MODEL)
        
        if response:
            category = response.strip().lower()
            valid_categories = ['introduction', 'technical', 'behavioral', 'company', 'hr', 'situational']
            
            # Clean up the response
            category = category.replace('.', '').replace('"', '').replace("'", "").strip()
            
            # Check for exact match
            if category in valid_categories:
                print(f"✅ Groq API classification: {category}")
                return category
            
            # Check for partial matches
            if 'intro' in category or 'yourself' in category:
                print(f"✅ Groq API classification (mapped): introduction")
                return 'introduction'
            elif 'tech' in category:
                print(f"✅ Groq API classification (mapped): technical")
                return 'technical'
            elif 'behavior' in category or 'star' in category or 'experience' in category:
                print(f"✅ Groq API classification (mapped): behavioral")
                return 'behavioral'
            elif 'compan' in category:
                print(f"✅ Groq API classification (mapped): company")
                return 'company'
            elif 'hr' in category or 'strength' in category or 'weakness' in category or 'goal' in category:
                print(f"✅ Groq API classification (mapped): hr")
                return 'hr'
            elif 'situation' in category or 'scenario' in category or 'hypothet' in category:
                print(f"✅ Groq API classification (mapped): situational")
                return 'situational'
            else:
                print(f"⚠️ Groq API returned unexpected category: {category}, defaulting to general")
                return 'general'
        else:
            print("⚠️ Groq API returned no response for classification, defaulting to general")
            return 'general'
            
    except Exception as e:
        print(f"❌ Groq API classification error: {str(e)}")
        # If API fails, return 'general' as fallback
        return 'general'

# ================================
# ADAPTIVE PROMPT GENERATION
# ================================

def get_sample_answer_prompt(question, question_type):
    """Generate context-appropriate prompt for sample answer based on detected question type"""
    
    if question_type == 'introduction':
        return f"""You are an expert career coach. Generate a professional sample answer for this self-introduction question.

QUESTION: "{question}"

Generate a well-structured self-introduction (150-250 words) that includes:

1. PROFESSIONAL BACKGROUND:
   - Current status (student/professional)
   - Educational background (degree, institution)
   - Relevant coursework or specialization

2. KEY SKILLS & STRENGTHS:
   - 3-4 core technical or professional skills
   - Any relevant certifications or achievements

3. EXPERIENCE:
   - Academic projects, internships, or work experience
   - Highlight measurable outcomes or learnings

4. CAREER GOALS & MOTIVATION:
   - Short-term career objectives
   - Why interested in this role/field
   - What you're looking to achieve

Requirements:
- Use first-person narrative (I am, I have, My background)
- Professional yet personable tone
- Enthusiastic and confident
- Well-structured with smooth flow
- Specific examples rather than generic statements
- Appropriate length (1-2 minutes when spoken)

Provide ONLY the sample answer text, no additional explanations."""

    elif question_type == 'technical':
        return f"""You are a technical interview expert with deep domain knowledge. Generate an accurate, comprehensive technical answer.

QUESTION: "{question}"

Generate a clear technical answer (150-250 words) that includes:

1. DEFINITION/CONCEPT:
   - Clear explanation of the core concept
   - Accurate technical terminology

2. KEY POINTS:
   - Main characteristics or principles
   - How it works or why it's important

3. EXAMPLES:
   - Practical examples or use cases
   - Code snippets if relevant (keep brief)

4. COMPARISON (if applicable):
   - Differences from related concepts
   - When to use vs alternatives

Requirements:
- Technically accurate and up-to-date
- Clear explanations accessible to interviewers
- Structured logically (definition → explanation → example)
- Avoid overly complex jargon unless necessary
- Demonstrate deep understanding
- Concise but comprehensive

Provide ONLY the sample answer text, no additional explanations."""

    elif question_type == 'behavioral':
        return f"""You are an interview coach specializing in behavioral interviews. Generate a strong STAR method answer.

QUESTION: "{question}"

Generate a compelling behavioral answer (200-300 words) using the STAR framework:

1. SITUATION (Context):
   - Set the scene with specific details
   - When and where did this happen?
   - What was the context?

2. TASK (Challenge/Responsibility):
   - What was the challenge or goal?
   - What was at stake?
   - Your specific responsibility

3. ACTION (Your Actions):
   - Specific steps YOU took
   - Skills and qualities demonstrated
   - Why you chose this approach

4. RESULT (Outcome):
   - Measurable outcomes or achievements
   - What did you learn?
   - How did this experience shape you?

Requirements:
- Specific, real example (not hypothetical)
- Focus on "I" not "we" (your individual contribution)
- Demonstrate key competencies (leadership, problem-solving, teamwork)
- Quantifiable results where possible
- Show self-awareness and learning
- Professional yet authentic tone

Provide ONLY the sample answer text, no additional explanations."""

    elif question_type == 'company':
        return f"""You are a career strategist. Generate a well-researched, enthusiastic answer about company fit.

QUESTION: "{question}"

Generate a thoughtful company-focused answer (150-250 words) that includes:

1. COMPANY KNOWLEDGE:
   - What the company does (products/services)
   - Company mission, values, or culture
   - Recent achievements or news (general examples)

2. PERSONAL ALIGNMENT:
   - Why their mission resonates with you
   - How your values align with company culture
   - What aspects of their work excite you

3. CONTRIBUTION:
   - What you can bring to the company
   - How your skills fit their needs
   - Your potential impact

4. ENTHUSIASM:
   - Genuine interest and excitement
   - Long-term vision with the company

Requirements:
- Specific to company (not generic "great company" statements)
- Show you've done research
- Connect your background to their needs
- Authentic enthusiasm
- Professional and mature
- Focus on mutual benefit

Provide ONLY the sample answer text, no additional explanations."""

    elif question_type == 'hr':
        return f"""You are an HR interview specialist. Generate a balanced, self-aware answer to this HR question.

QUESTION: "{question}"

Generate a thoughtful HR answer (150-250 words) that includes:

1. HONEST SELF-ASSESSMENT:
   - Genuine reflection on the topic
   - Self-awareness and maturity

2. SPECIFIC EXAMPLES:
   - Concrete examples from your experience
   - Real situations, not abstract statements

3. BALANCE:
   - For weaknesses: acknowledge + improvement efforts
   - For strengths: evidence + humility
   - Show growth mindset

4. PROFESSIONAL RELEVANCE:
   - Connect to work/career context
   - Show how this relates to professional development

Requirements:
- Authentic and honest (not cliché responses)
- Professional and appropriate
- Shows self-awareness
- Demonstrates continuous improvement
- Relevant to career/work context
- Mature and thoughtful perspective

Provide ONLY the sample answer text, no additional explanations."""

    elif question_type == 'situational':
        return f"""You are an interview expert. Generate a well-reasoned answer to this situational question.

QUESTION: "{question}"

Generate a structured situational answer (150-250 words) that includes:

1. APPROACH/FRAMEWORK:
   - How you would think through the problem
   - Key factors to consider

2. ANALYSIS:
   - Multiple perspectives or options
   - Trade-offs and considerations

3. DECISION-MAKING:
   - Your recommended approach
   - Rationale behind the decision

4. IMPLEMENTATION:
   - Steps you would take
   - How you'd handle potential challenges

Requirements:
- Logical and structured thinking
- Consider multiple angles
- Demonstrate sound judgment
- Professional problem-solving approach
- Communication and collaboration aspects
- Show leadership and initiative

Provide ONLY the sample answer text, no additional explanations."""

    else:  # general
        return f"""You are an interview expert. Generate a professional, comprehensive answer to this question.

QUESTION: "{question}"

Generate a clear answer (150-250 words) that:
- Directly addresses the question
- Provides specific information or examples
- Demonstrates relevant knowledge or experience
- Is well-structured and easy to follow
- Shows professionalism and competence

Provide ONLY the sample answer text, no additional explanations."""

def get_analysis_prompt(question, user_answer, question_type):
    """Generate context-appropriate analysis prompt based on detected question type"""
    
    base_json_structure = """
Return ONLY valid JSON (no markdown, no explanations) in this EXACT structure:
{
    "content_score": 7,
    "structure_score": 6,
    "relevance_score": 8,
    "overall_score": 7,
    "content_feedback": "Brief specific assessment",
    "missing_elements": ["Element 1", "Element 2"],
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "improvements": ["Improvement 1", "Improvement 2", "Improvement 3"],
    "key_points_covered": ["Point 1", "Point 2", "Point 3"]
}

All scores are 0-10. Be specific, constructive, and actionable in feedback.
"""
    
    if question_type == 'introduction':
        return f"""Analyze this self-introduction answer comprehensively.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR INTRODUCTION:

1. CONTENT SCORE (0-10) - Evaluate completeness:
   ✓ Did they mention educational background?
   ✓ Did they highlight relevant skills?
   ✓ Did they discuss experience (projects/internships/work)?
   ✓ Did they express career goals/motivation?
   ✓ Was it personalized and authentic?

2. STRUCTURE SCORE (0-10) - Evaluate organization:
   ✓ Clear beginning, middle, end?
   ✓ Logical flow (past → present → future)?
   ✓ Smooth transitions between points?
   ✓ Appropriate length (not too brief/long)?
   ✓ Well-paced delivery?

3. RELEVANCE SCORE (0-10) - Evaluate fit:
   ✓ Relevant to professional context?
   ✓ Appropriate level of detail?
   ✓ Memorable and engaging?
   ✓ Professional yet personable tone?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess completeness and authenticity
- Missing Elements: What key intro components were not mentioned?
- Strengths: What made the introduction effective? (min 2-3 points)
- Improvements: Specific suggestions to enhance the introduction (min 2-3 points)
- Key Points Covered: What they successfully included

{base_json_structure}"""

    elif question_type == 'technical':
        return f"""Analyze this technical interview answer with precision.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR TECHNICAL QUESTIONS:

1. CONTENT SCORE (Technical Accuracy) (0-10):
   ✓ Are technical concepts explained correctly?
   ✓ Is the information accurate and current?
   ✓ Are there any technical errors?
   ✓ Is the depth appropriate?
   ✓ Are definitions precise?

2. STRUCTURE SCORE (Clarity) (0-10):
   ✓ Is the explanation clear and logical?
   ✓ Are examples provided?
   ✓ Is it easy to follow?
   ✓ Good use of analogies/comparisons?
   ✓ Appropriate technical terminology?

3. RELEVANCE SCORE (Completeness) (0-10):
   ✓ Does it fully answer the question?
   ✓ Are all key concepts covered?
   ✓ Demonstrates deep understanding?
   ✓ Addresses nuances?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess technical accuracy and depth
- Missing Elements: What technical concepts/details are missing?
- Strengths: What technical aspects were explained well? (min 2-3 points)
- Improvements: How to improve technical accuracy/clarity (min 2-3 points)
- Key Points Covered: Which technical concepts were correctly addressed

{base_json_structure}"""

    elif question_type == 'behavioral':
        return f"""Analyze this behavioral interview answer using STAR method criteria.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR BEHAVIORAL QUESTIONS (STAR):

1. CONTENT SCORE (0-10):
   ✓ Specific, real example provided?
   ✓ Shows relevant skills/competencies?
   ✓ Demonstrates problem-solving?
   ✓ Measurable results mentioned?
   ✓ Learning/growth demonstrated?

2. STRUCTURE SCORE (STAR Format) (0-10):
   ✓ SITUATION: Was context clearly set?
   ✓ TASK: Was the challenge/responsibility explained?
   ✓ ACTION: Were specific actions described (focus on "I")?
   ✓ RESULT: Was outcome explained with impact?
   ✓ Logical flow through STAR?

3. RELEVANCE SCORE (0-10):
   ✓ Example directly relates to question?
   ✓ Demonstrates right competencies?
   ✓ Appropriate depth of detail?
   ✓ Shows self-awareness?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess STAR completeness and effectiveness
- Missing Elements: Which STAR components are weak/missing?
- Strengths: What made the story effective? (min 2-3 points)
- Improvements: How to better use STAR structure (min 2-3 points)
- Key Points Covered: Which STAR elements were successfully present

{base_json_structure}"""

    elif question_type == 'company':
        return f"""Analyze this company-focused answer for research and authenticity.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR COMPANY QUESTIONS:

1. CONTENT SCORE (0-10):
   ✓ Shows company knowledge?
   ✓ Mentions specific company details?
   ✓ Not generic "great company" statements?
   ✓ Demonstrates research?
   ✓ Authentic and genuine?

2. STRUCTURE SCORE (0-10):
   ✓ Clear explanation of interest?
   ✓ Personal connection to company?
   ✓ Links own goals to company mission?
   ✓ Enthusiastic tone?

3. RELEVANCE SCORE (0-10):
   ✓ Shows company/culture fit?
   ✓ Explains how they can contribute?
   ✓ Demonstrates alignment?
   ✓ Long-term commitment?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess research depth and authenticity
- Missing Elements: What company knowledge is lacking?
- Strengths: What shows good company understanding? (min 2-3 points)
- Improvements: How to demonstrate better company fit (min 2-3 points)
- Key Points Covered: What company aspects were mentioned

{base_json_structure}"""

    elif question_type == 'hr':
        return f"""Analyze this HR answer for self-awareness and professionalism.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR HR QUESTIONS:

1. CONTENT SCORE (0-10):
   ✓ Honest and thoughtful response?
   ✓ Specific examples provided?
   ✓ Shows self-awareness?
   ✓ Demonstrates growth mindset?
   ✓ Avoids clichés?

2. STRUCTURE SCORE (0-10):
   ✓ Well-balanced answer?
   ✓ Appropriate depth?
   ✓ Professional tone?
   ✓ Mature perspective?

3. RELEVANCE SCORE (0-10):
   ✓ Directly addresses question?
   ✓ Career-focused?
   ✓ Appropriate for professional context?
   ✓ Shows continuous improvement?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess authenticity and self-awareness
- Missing Elements: What aspects need more depth?
- Strengths: What shows good self-awareness? (min 2-3 points)
- Improvements: How to provide more balanced answer (min 2-3 points)
- Key Points Covered: What was successfully addressed

{base_json_structure}"""

    elif question_type == 'situational':
        return f"""Analyze this situational answer for problem-solving and judgment.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA FOR SITUATIONAL QUESTIONS:

1. CONTENT SCORE (0-10):
   ✓ Structured problem-solving approach?
   ✓ Multiple factors considered?
   ✓ Shows sound judgment?
   ✓ Practical and realistic?

2. STRUCTURE SCORE (0-10):
   ✓ Logical thinking process?
   ✓ Clear steps outlined?
   ✓ Well-organized response?
   ✓ Considers consequences?

3. RELEVANCE SCORE (0-10):
   ✓ Addresses the situation directly?
   ✓ Professional approach?
   ✓ Shows leadership/initiative?
   ✓ Communication considered?

FEEDBACK REQUIREMENTS:
- Content Feedback: Assess problem-solving approach
- Missing Elements: What considerations are missing?
- Strengths: What shows good judgment? (min 2-3 points)
- Improvements: How to demonstrate better approach (min 2-3 points)
- Key Points Covered: What was successfully addressed

{base_json_structure}"""

    else:  # general
        return f"""Analyze this interview answer comprehensively.

QUESTION: "{question}"
USER'S ANSWER: "{user_answer}"

EVALUATION CRITERIA:

1. CONTENT SCORE (0-10):
   ✓ Addresses question directly?
   ✓ Sufficient information/depth?
   ✓ Accurate and relevant?

2. STRUCTURE SCORE (0-10):
   ✓ Well-organized?
   ✓ Clear and logical?
   ✓ Appropriate length?

3. RELEVANCE SCORE (0-10):
   ✓ On-topic and focused?
   ✓ Professional?
   ✓ Demonstrates competence?

FEEDBACK REQUIREMENTS:
- Content Feedback: Overall assessment
- Missing Elements: What should be added?
- Strengths: What was done well? (min 2-3 points)
- Improvements: Specific suggestions (min 2-3 points)
- Key Points Covered: What was successfully addressed

{base_json_structure}"""

# ================================
# SAMPLE ANSWER AND ANALYSIS FUNCTIONS
# ================================

def generate_sample_answer(question):
    """Generate a sample answer using Groq/Llama with question type detection"""
    try:
        # Detect question type
        question_type = detect_question_type(question)
        print(f"🔍 Question type detected: {question_type}")
        
        # Get appropriate prompt
        prompt = get_sample_answer_prompt(question, question_type)
        
        print(f"📝 Generating {question_type} sample answer for: {question[:100]}...")
        
        # Call Groq API
        answer = call_groq_api(prompt, model=SAMPLE_ANSWER_MODEL)
        
        if answer and len(answer) > 10:
            print(f"✅ Sample answer generated ({len(answer)} chars)")
            return answer.strip()
        else:
            print("❌ No response from Groq")
            # If API fails, raise exception
            raise Exception("Groq API failed to generate sample answer")
            
    except Exception as e:
        print(f"❌ Sample answer generation error: {str(e)}")
        raise Exception(f"Failed to generate sample answer: {str(e)}")

def analyze_answer_content(question, user_answer):
    """Analyze user's answer using Groq/Llama with question type detection"""
    try:
        # Detect question type
        question_type = detect_question_type(question)
        print(f"🔍 Analyzing {question_type} answer...")
        
        # Get appropriate analysis prompt
        prompt = get_analysis_prompt(question, user_answer, question_type)
        
        print(f"📊 Analyzing answer content with Groq...")
        print(f"Question: {question[:100]}...")
        print(f"User answer length: {len(user_answer)} characters")
        
        # Call Groq API
        response = call_groq_api(prompt, model=ANALYSIS_MODEL)
        
        if response:
            print("✅ Groq response received")
            
            # Try multiple methods to extract JSON
            content_analysis = None
            
            # Method 1: Direct JSON parse
            try:
                content_analysis = json.loads(response.strip())
                print("✅ JSON parsed directly")
                # Add question_type to the analysis result
                if content_analysis and isinstance(content_analysis, dict):
                    content_analysis['question_type'] = question_type
                return content_analysis
            except json.JSONDecodeError:
                # Method 2: Extract JSON from text
                content_analysis = extract_json_from_text(response)
                if content_analysis:
                    print("✅ JSON extracted from text")
                    # Add question_type to the analysis result
                    if isinstance(content_analysis, dict):
                        content_analysis['question_type'] = question_type
                    return content_analysis
                else:
                    print("❌ Could not parse JSON")
                    raise Exception("Failed to parse JSON from Groq response")
        
        else:
            raise Exception("No response from Groq API for analysis")
            
    except Exception as e:
        print(f"❌ Answer analysis error: {str(e)}")
        raise Exception(f"Failed to analyze answer: {str(e)}")

# ================================
# LOGIN + FRONT PAGES
# ================================

@app.route('/')
def index():
    """Login page"""
    return render_template('login.html')


@app.route('/signup', methods=['POST'])
def signup():
    """User signup"""
    data = request.json
    name = data['name']
    email = data['email']
    password = data['password']

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')

    cur = mysql_db.connection.cursor()
    cur.execute("SELECT * FROM login WHERE email=%s", (email,))
    if cur.fetchone():
        return jsonify({"error": "User already exists"}), 409

    cur.execute(
        "INSERT INTO login (name, email, password) VALUES (%s, %s, %s)",
        (name, email, hashed_pw)
    )
    mysql_db.connection.commit()
    cur.close()

    return jsonify({"message": "Signup successful"}), 201


@app.route('/login', methods=['POST'])
def login():
    """User login"""
    data = request.json
    email = data['email']
    password = data['password']

    cur = mysql_db.connection.cursor()
    cur.execute(
        "SELECT student_id, name, email, password FROM login WHERE email=%s",
        (email,)
    )
    user = cur.fetchone()
    cur.close()

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    student_id, name, email, hashed_pw = user

    if not bcrypt.check_password_hash(hashed_pw, password):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "student_id": student_id,
            "name": name,
            "email": email
        }
    }), 200


@app.route('/qgen')
def qgen_home():
    """Serve the resume/question generation page"""
    return render_template('db_qgen.html')


@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Main endpoint for generating questions based on level (ported from db_qgen.py)"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume uploaded'}), 400

        level = request.form.get('level')
        job_title = request.form.get('job_title', '').strip()
        company_name = request.form.get('company_name', '').strip()

        if not level or level not in ['beginner', 'intermediate', 'advanced']:
            return jsonify({'error': 'Invalid level specified'}), 400

        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400

        # Extract resume text
        resume_file = request.files['resume']
        resume_text = extract_text_from_pdf(resume_file)

        if not resume_text:
            return jsonify({'error': 'Could not extract text from resume'}), 400

        # Get session_id from form
        session_id = request.form.get('session_id')
        if not session_id:
            session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Ensure session exists as 'ongoing'
        ensure_session_exists(session_id, level)

        # Generate questions based on level
        if level == 'beginner':
            questions = generate_beginner_questions(resume_text, job_title, company_name)
        elif level == 'intermediate':
            questions = generate_intermediate_questions(resume_text, job_title, company_name)
        else:
            questions = generate_advanced_questions(resume_text, job_title, company_name)

        # Save questions to database
        db_success = save_questions_to_db(questions, session_id=session_id, difficulty_level=level)

        # Prepare simplified response for frontend
        simplified_questions = []
        for q in questions:
            if isinstance(q, dict):
                simplified_questions.append({
                    'question_text': q.get('question_text', ''),
                    'category': q.get('question_category', ''),
                    'purpose': q.get('purpose', '')
                })
            else:
                simplified_questions.append({
                    'question_text': str(q),
                    'category': 'General',
                    'purpose': 'Assessment'
                })

        return jsonify({
            'success': True,
            'level': level,
            'job_title': job_title,
            'company_name': company_name,
            'questions': simplified_questions,
            'total_questions': len(questions),
            'database_saved': db_success,
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/interview')
def interview_home():
    """Serve the main interview practice page"""
    return render_template('db_vans_ch.html')

# ================================
# DATABASE ROUTES
# ================================

@app.route('/get_questions/<level>', methods=['GET'])
def get_questions(level):
    """Get questions for a specific level"""
    try:
        if level not in ['beginner', 'intermediate', 'advanced']:
            return jsonify({'error': 'Invalid level. Use beginner, intermediate, or advanced'}), 400
        
        # Check if we have enough questions for this level
        has_enough_questions = check_if_questions_exist_for_level(level)
        
        if not has_enough_questions:
            return jsonify({
                'error': f'Not enough questions found for {level} level',
                'suggestion': f'Please generate at least {QUESTIONS_PER_LEVEL[level]} questions for {level} level first'
            }), 404
        
        # Get session_id from query params (Optional now due to fallback)
        session_id = request.args.get('session_id')
        
        questions = []
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                # Map level to DB stored value (handle 'intermedite' typo)
                level_mapping = {'beginner': 'beginner', 'intermediate': 'intermedite', 'advanced': 'advanced'}
                db_level = level_mapping.get(level, level)

                # 1. Try to get questions for the specific session
                if session_id:
                    cursor.execute("""
                        SELECT questions_id, question_text, difficulty_level, session_id
                        FROM questions
                        WHERE difficulty_level = %s AND session_id = %s
                        ORDER BY questions_id ASC
                    """, (db_level, session_id))
                    questions = cursor.fetchall()
                
                # 2. Fallback: If no session_id or no questions found for session, get latest questions
                if not questions:
                    limit = QUESTIONS_PER_LEVEL.get(level, 10)
                    print(f"⚠️ No questions found for session '{session_id}'. Falling back to latest {limit} questions for level {level}.")
                    
                    cursor.execute("""
                        SELECT questions_id, question_text, difficulty_level, session_id
                        FROM questions
                        WHERE difficulty_level = %s
                        ORDER BY questions_id DESC
                        LIMIT %s
                    """, (db_level, limit))
                    questions = cursor.fetchall()
                    
                    # Reverse to show in correct order (Q1 -> Q10)
                    questions.reverse()

                cursor.close()
                connection.close()
            except Exception as e:
                print(f"Error fetching questions for level/session: {e}")
                try:
                    connection.close()
                except Exception:
                    pass
                return jsonify({'error': f'Database error: {str(e)}'}), 500
        
        # Verify we have questions after fallback
        if not questions:
            return jsonify({
                'error': f'No questions found in database for {level} level',
                'suggestion': 'Please generate questions first'
            }), 404
        
        # Format questions for response
        formatted_questions = []
        for i, q in enumerate(questions):
            formatted_questions.append({
                'question_id': q['questions_id'],
                'question_number': i + 1,
                'question_text': q['question_text'],
                'difficulty_level': q['difficulty_level'],
                'total_questions': len(questions)
            })
        
        return jsonify({
            'success': True,
            'level': level,
            'total_questions': len(questions),
            'questions_per_level': QUESTIONS_PER_LEVEL[level],
            'questions': formatted_questions
        })
        
    except Exception as e:
        print(f"Error in get_questions: {str(e)}")
        return jsonify({'error': f'Failed to get questions: {str(e)}'}), 500

def store_response(session_id, question_id, student_id, video_path, transcript=None):
    """Store video response in the responses table"""
    connection = get_db_connection()
    if connection is None:
        return False
    try:
        cursor = connection.cursor()
        sql = """
            INSERT INTO responses (session_id, question_id, student_id, video_path, transcript)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (session_id, int(question_id), student_id, video_path, transcript))
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"❌ Error storing response: {e}")
        if connection:
            connection.close()
        return False

def ensure_session_exists(session_id, student_id=1, level='beginner'):
    """Ensure a session exists in the interview_session table with 'ongoing' status"""
    if not session_id:
        return
    
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            # Check if session exists
            cursor.execute("SELECT session_id FROM interview_session WHERE session_id = %s", (session_id,))
            if not cursor.fetchone():
                # Create session with 'ongoing' status
                cursor.execute("""
                    INSERT INTO interview_session (session_id, student_id, level, total_questions, status)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, student_id, level, 10, 'ongoing'))
                connection.commit()
                print(f"🆕 Created ongoing session: {session_id}")
            cursor.close()
            connection.close()
        except Exception as e:
            print(f"⚠️ Error ensuring session exists: {e}")
            if connection:
                connection.close()


@app.route('/generate_question_audio', methods=['POST'])
def generate_question_audio():
    """Generate audio for a question text"""
    try:
        data = request.get_json()
        question_text = data.get('question_text', '')
        
        if not question_text:
            return jsonify({'error': 'Question text is required'}), 400
        
        # Ensure audio folder exists
        os.makedirs(AUDIO_FOLDER, exist_ok=True)
        
        # Generate TTS audio
        audio_url = generate_tts(question_text)
        
        return jsonify({
            'success': True,
            'audio_url': audio_url
        })
        
    except Exception as e:
        print(f"Error generating question audio: {str(e)}")
        return jsonify({'error': f'Failed to generate audio: {str(e)}'}), 500

@app.route('/check_database', methods=['GET'])
def check_database():
    """Check database connection and table status"""
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # Check if questions table exists and has data
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM questions WHERE difficulty_level = 'beginner') as beginner_count,
                (SELECT COUNT(*) FROM questions WHERE difficulty_level = 'intermediate') as intermediate_count,
                (SELECT COUNT(*) FROM questions WHERE difficulty_level = 'advanced') as advanced_count
        """)
        
        counts = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'database_status': 'connected',
            'question_counts': {
                'beginner': counts['beginner_count'] if counts else 0,
                'intermediate': counts['intermediate_count'] if counts else 0,
                'advanced': counts['advanced_count'] if counts else 0
            },
            'required_counts': QUESTIONS_PER_LEVEL
        })
        
    except Exception as e:
        print(f"Database check error: {str(e)}")
        return jsonify({'error': f'Database check failed: {str(e)}'}), 500

# ================================
# ANALYZE VOICE ROUTE
# ================================

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    """Analyze voice recording and answer content"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    os.makedirs('uploads', exist_ok=True)
    video_path = os.path.join('uploads', f"{uuid.uuid4().hex}_{video.filename}")
    video.save(video_path)

    # Get form data
    question = request.form.get('question', '').strip()
    question_id = request.form.get('question_id', '0')
    session_id = request.form.get('session_id', '')
    student_id = request.form.get('student_id', '1')
    question_number = request.form.get('question_number', '1')
    is_last_question = request.form.get('is_last_question', 'false').lower() == 'true'
    
    if not question:
        question = "What are the four pillars of OOPS (Object-Oriented Programming)? Explain each."

    print(f"🎯 Received analysis request for question: {question}")
    print(f"📝 Question ID: {question_id}, Session ID: {session_id}, Student ID: {student_id}, Question #: {question_number}")

    # Ensure session exists as 'ongoing'
    ensure_session_exists(session_id, student_id=student_id)

    # If it's the last question, process synchronously (user will wait)
    if is_last_question:
        print(f"⏳ Last question - processing synchronously...")
        try:
            # Run analysis in current process for last question
            analyze_answer_process(video_path, question, question_id, session_id, student_id, question_number)
            
            return jsonify({
                'success': True,
                'message': 'Analysis completed',
                'processing_mode': 'synchronous',
                'question_number': question_number
            })
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    else:
        # For non-last questions, process in background
        try:
            # Start background process
            process = Process(
                target=analyze_answer_process,
                args=(video_path, question, question_id, session_id, student_id, question_number)
            )
            process.daemon = True
            process.start()
            
            print(f"✅ Background process started for Question {question_number}")
            
            # Return immediately
            return jsonify({
                'success': True,
                'message': 'Analysis started in background',
                'processing_mode': 'asynchronous',
                'question_number': question_number
            })
        except Exception as e:
            print(f"❌ Failed to start background process: {str(e)}")
            return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

# ================================
# ROUTE FOR SKIPPED QUESTIONS
# ================================

@app.route('/skip_question', methods=['POST'])
def skip_question():
    """Handle skipped questions"""
    try:
        data = request.get_json()
        question_id = data.get('question_id', '0')
        question_text = data.get('question_text', '')
        session_id = data.get('session_id', '')
        student_id = data.get('student_id', '1')
        level = data.get('level', 'beginner')
        question_number = data.get('question_number', '1')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Ensure session exists as 'ongoing'
        ensure_session_exists(session_id, student_id=student_id, level=level)
        
        print(f"⏭️ Skipping question {question_number} (ID: {question_id}) for session {session_id}")
        
        # Generate sample answer if question text is available
        sample_answer = ""
        if question_text:
            try:
                sample_answer = generate_sample_answer(question_text)
                print(f"📋 Generated sample answer for skipped question: {len(sample_answer)} chars")
            except Exception as e:
                print(f"⚠️ Failed to generate sample answer for skip: {e}")
        
        # Store skipped question in database with sample answer
        if store_skipped_question(student_id, session_id, question_id, int(question_number), sample_answer=sample_answer):
            return jsonify({
                'success': True,
                'message': f'Question {question_number} marked as skipped',
                'question_id': question_id,
                'question_number': question_number,
                'sample_answer_generated': bool(sample_answer)
            })
        else:
            return jsonify({'error': 'Failed to store skipped question in database'}), 500
            
    except Exception as e:
        print(f"❌ Skip question error: {str(e)}")
        return jsonify({'error': f'Failed to skip question: {str(e)}'}), 500

@app.route('/start-interview')
def interview():
    return render_template('db_vans_ch.html')

@app.route('/complete_session', methods=['POST'])
def complete_session():
    """Mark a session as completed and track progress"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        student_id = data.get('student_id')
        level = data.get('level')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
            
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # 1. Update session status
            cursor.execute("""
                UPDATE interview_session 
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE session_id = %s
            """, (session_id,))
            
            # 2. Add to progress_tracking
            if student_id and level:
                try:
                    cursor.execute("""
                        INSERT INTO progress_tracking (student_id, session_id, level, completed_at)
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    """, (student_id, session_id, level))
                    print(f"📈 Progress tracked for Student {student_id} in session {session_id}")
                except Exception as track_err:
                    print(f"⚠️ Failed to log progress tracking: {track_err}")
            
            connection.commit()
            cursor.close()
            connection.close()
            print(f"🏁 Session {session_id} marked as completed")
            return jsonify({'success': True, 'message': 'Session completed and progress tracked'})
        return jsonify({'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error completing session: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/get_latest_ongoing_session', methods=['GET'])
def get_latest_ongoing_session():
    """Get the session_id of the most recent 'ongoing' session"""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT session_id FROM interview_session 
                WHERE status = 'ongoing' 
                ORDER BY started_at DESC LIMIT 1
            """)
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result:
                return jsonify({'success': True, 'session_id': result['session_id']})
            return jsonify({'success': False, 'message': 'No ongoing session found'}), 404
        return jsonify({'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error getting latest session: {e}")
        return jsonify({'error': str(e)}), 500

# ================================
# DEBUG ROUTE TO CHECK DATABASE
# ================================
@app.route('/analyze_interview', methods=['POST'])
def analyze_interview():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    filename = f"interview_{int(time.time())}.webm"
    filepath = os.path.join("uploads", filename)
    video.save(filepath)

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            mp4_path = filepath.replace(".webm", ".mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", filepath, "-vcodec", "libx264", mp4_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                return jsonify({'error': 'Cannot open video file even after conversion.'}), 500
            filepath = mp4_path

        cap.release()
        print(f"🎥 Analyzing video: {filepath}")
        result = analyze_video(filepath)
        print(f"✅ Video analysis complete. Result structure: {result.keys() if isinstance(result, dict) else type(result)}")

        # Question number → qno in face_feedback
        qno_raw = request.form.get("question_number") or request.form.get("qno")
        try:
            qno_val = int(qno_raw) if qno_raw is not None else None
        except (ValueError, TypeError):
            qno_val = None


        # Get session_id
        session_id = request.form.get("session_id")

        print(f"📊 Parameters: qno={qno_val}, session_id={session_id}")
        
        # Store face feedback with error handling
        if isinstance(result, dict):
            student_id = request.form.get("student_id") or 1
            db_success = store_face_feedback(result, student_id=student_id, session_id=session_id, qno=qno_val)
            if db_success:
                print(f"✅ Face feedback successfully stored in database")
            else:
                print(f"⚠️ Face feedback storage returned False")
        else:
            print(f"❌ Unexpected result type from analyze_video: {type(result)}")

        return jsonify(result)

    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/debug_database', methods=['GET'])
def debug_database():
    """Debug endpoint to check database tables"""
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        
        # Check all tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        table_details = {}
        for table in tables:
            table_name = list(table.values())[0]
            cursor.execute(f"DESCRIBE {table_name}")
            table_details[table_name] = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'tables': tables,
            'table_details': table_details
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"🚀 Interview Answer Analyzer")
    print(f"{'='*50}")
    print(f"✓ Groq API: Configured")
    print(f"✓ Model: {SAMPLE_ANSWER_MODEL}")
    print(f"✓ Whisper: Loaded")
    print(f"✓ Database: MySQL connector ready")
    print(f"✓ Questions per level: {QUESTIONS_PER_LEVEL}")
    print(f"✓ Voice & Content Analysis: READY")
    print(f"✓ Feedback Storage: ENABLED")
    print(f"✓ Skip Question: ENABLED for all questions except last")
    print(f"✓ Port: 5000")
    print(f"{'='*50}\n")
    
    # Test database connection
    connection = get_db_connection()
    if connection:
        print("✅ Database connection successful")
        cursor = connection.cursor()
        
        # Check if voice_feedback and content_feedback tables exist
        cursor.execute("SHOW TABLES LIKE 'voice_feedback'")
        voice_feedback_exists = cursor.fetchone()
        
        cursor.execute("SHOW TABLES LIKE 'content_feedback'")
        content_feedback_exists = cursor.fetchone()
        
        if voice_feedback_exists and content_feedback_exists:
            print("✅ Required tables exist: voice_feedback, content_feedback")
        else:
            print("⚠️ Warning: Some tables might be missing")
            print("   If you get database errors, please create these tables:")
            print("   CREATE TABLE voice_feedback (id INT AUTO_INCREMENT PRIMARY KEY, studentid INT, session_id VARCHAR(100), strengths JSON, improvements JSON, q_no INT);")
            print("   CREATE TABLE content_feedback (id INT AUTO_INCREMENT PRIMARY KEY, studentid INT, session_id VARCHAR(100), response TEXT, content_score VARCHAR(10), overall VARCHAR(10), relevance VARCHAR(10), structure VARCHAR(10), improvements JSON, strengths JSON, sample_answer TEXT, q_no INT);")
        
        cursor.close()
        connection.close()
    else:
        print("⚠️ Database connection failed - check DB_CONFIG settings")
    
    app.run(debug=True, port=5000)
    
    