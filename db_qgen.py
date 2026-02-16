from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_mysqldb import MySQL
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import re
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)



# MySQL Configuration
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'interview@1234')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'mockinterview')
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Configure Gemini - in production, use environment variables
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Database connection helper
def get_db_connection():
    """Get MySQL database connection"""
    return mysql.connection

def save_questions_to_db(questions_data, session_id=100, difficulty_level='beginner'):
    """
    Save generated questions to database
    
    Args:
        questions_data: List of question dictionaries
        session_id: Foreign key to interview_session table
        difficulty_level: beginner/intermediate/advanced
    """
    try:
        cursor = mysql.connection.cursor()
        
        # Map difficulty level to match database enum
        level_mapping = {
            'beginner': 'beginner',
            'intermediate': 'intermedite',  # Note: Match the typo in your table
            'advanced': 'advanced'
        }
        
        db_level = level_mapping.get(difficulty_level, 'beginner')
        
        # Insert into interview_session first to satisfy FK constraint
        # Use difficulty_level (without mapping) for level column in interview_session
        # Use student_id=1, total_questions=len(questions_data)
        
        # Check if session exists
        check_sql = "SELECT session_id FROM interview_session WHERE session_id = %s"
        cursor.execute(check_sql, (session_id,))
        if not cursor.fetchone():
            session_sql = """
                INSERT INTO interview_session (session_id, student_id, level, total_questions, status) 
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(session_sql, (session_id, 1, difficulty_level, len(questions_data), 'ongoing'))

        # Insert each question using mapped db_level (with potential typo)
        for question in questions_data:
            # Extract question text (handle different response formats)
            if isinstance(question, dict):
                question_text = question.get('question_text', '')
            else:
                question_text = str(question)
            
            # Clean up question text if needed
            if not question_text:
                continue
                
            # Remove any numbering or special characters at the beginning
            question_text = re.sub(r'^\d+[\.\)\-\s]*', '', question_text.strip())
            
            # Insert into database
            sql = """
                INSERT INTO questions (session_id, question_text, difficulty_level, created_at)
                VALUES (%s, %s, %s, NOW())
            """
            cursor.execute(sql, (session_id, question_text, db_level))
        
        mysql.connection.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Database error: {str(e)}")
        mysql.connection.rollback()
        return False

def extract_text_from_pdf(file_stream):
    """Extract text from PDF resume"""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def clean_json_response(response_text):
    """Clean and parse JSON response from Gemini"""
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
            except:
                pass
        raise Exception("Failed to parse JSON response from AI")

def generate_beginner_questions(resume_text, job_title, company_name):
    """Generate beginner level questions"""
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
    """Generate intermediate level questions"""
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
    """Generate professional advanced level questions"""
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

@app.route('/')
def home():
    """Serve the main frontend page"""
    return render_template('db_qgen.html')

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Main endpoint for generating questions based on level"""
    try:
        # Validate request
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
        
        # Generate questions based on level
        if level == 'beginner':
            questions = generate_beginner_questions(resume_text, job_title, company_name)
        elif level == 'intermediate':
            questions = generate_intermediate_questions(resume_text, job_title, company_name)
        else:  # advanced
            questions = generate_advanced_questions(resume_text, job_title, company_name)
        
        # Generate a random integer session_id (100000 - 2000000000)
        import random
        session_id = random.randint(100000, 2000000000)

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
            'questions': simplified_questions,
            'total_questions': len(questions),
            'database_saved': db_success,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-questions', methods=['GET'])
def test_questions():
    """Test endpoint to check question generation without file upload"""
    try:
        # Sample data for testing
        sample_resume = """
        John Doe - Computer Science Graduate
        EDUCATION: Bachelor of Computer Science, University of Tech (2020-2024)
        SKILLS: Python, Java, JavaScript, SQL, React, Node.js
        PROJECTS: E-commerce website, Task management app, Data analysis tool
        EXPERIENCE: Software Developer Intern at TechCorp (Summer 2023)
        """
        
        level = request.args.get('level', 'beginner')
        job_title = request.args.get('job_title', 'Software Developer')
        company_name = request.args.get('company_name', 'Google')
        
        if level == 'beginner':
            questions = generate_beginner_questions(sample_resume, job_title, company_name)
        elif level == 'intermediate':
            questions = generate_intermediate_questions(sample_resume, job_title, company_name)
        else:
            questions = generate_advanced_questions(sample_resume, job_title, company_name)
        
        # Save test questions to database
        db_success = save_questions_to_db(questions, session_id=100, difficulty_level=level)
        
        # Simplify for response
        simplified_questions = []
        for q in questions:
            if isinstance(q, dict):
                simplified_questions.append({
                    'question_text': q.get('question_text', ''),
                    'category': q.get('question_category', '')
                })
        
        return jsonify({
            'success': True,
            'level': level,
            'questions': simplified_questions,
            'total_questions': len(questions),
            'database_saved': db_success
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view-questions', methods=['GET'])
def view_questions():
    """Endpoint to view all questions in database"""
    try:
        cursor = mysql.connection.cursor()
        
        # Get all questions
        cursor.execute("""
            SELECT questions_id, session_id, question_text, 
                   difficulty_level, created_at 
            FROM questions 
            ORDER BY created_at DESC, questions_id
        """)
        
        questions = cursor.fetchall()
        cursor.close()
        
        return jsonify({
            'success': True,
            'total_questions': len(questions),
            'questions': questions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    #3lines
    
#@app.route('/start-interview')
#def start_interview():
 #   return render_template('db_vans_ch.html')


@app.route('/clear-questions', methods=['DELETE'])
def clear_questions():
    """Endpoint to clear all questions from database"""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM questions")
        mysql.connection.commit()
        affected_rows = cursor.rowcount
        cursor.close()
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {affected_rows} questions'
        })
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'error': str(e)}), 500
    
# Add this ONE route to your existing db_qgen.py file
@app.route('/start-interview')
def start_interview():
    """Redirect to the integrated db_dyn interview application."""
    # db_dyn.py is expected to be running on port 5000 and serving db_vans_ch.html at "/"
    return '''
    <html>
        <head>
            <title>Launching Interview...</title>
            <meta http-equiv="refresh" content="2;url=http://localhost:5000/?session_id={{ request.args.get('session_id', '') }}" />
            <style>
                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .message {
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                }
                .spinner {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #667eea;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 2s linear infinite;
                    margin: 0 auto 20px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="message">
                <div class="spinner"></div>
                <h2>🚀 Launching Interview App...</h2>
                <p>You will be redirected in 2 seconds.</p>
                <p>If not redirected, <a href="http://localhost:5000/?session_id={{ request.args.get('session_id', '') }}">click here</a></p>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Create .env file if it doesn't exist
    env_file = '.env'
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("""# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=interview@1234
MYSQL_DB=mockinterview

# Gemini API Key
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY'))
""")
    
    # Check database connection
    with app.app_context():
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            print("✓ Database connection successful")
            cursor.close()
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            print("Please ensure MySQL is running and database exists")
    
    app.run(debug=True, port=5001)    