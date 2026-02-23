from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import json
import re
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = "interview_feedback_secret_2025"

# Add route to serve video responses from uploads folder
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'interview@1234'),
    'database': os.getenv('MYSQL_DB', 'mockinterview')
}

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

@app.route('/')
def index():
    return redirect(url_for('feedback'))

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/get_feedback')
def get_feedback():
    session_id = request.args.get('session_id')
    
    # Fallback to default session if not provided
    if not session_id:
        session_id = 'session_1771766105079_sy41hzikg'
        print(f"⚠️ No session_id provided, using default: {session_id}")
    
    print(f"🔍 Fetching feedback for session: {session_id}")

    connection = get_db_connection()
    if not connection:
        print("❌ Database connection failed")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = connection.cursor(dictionary=True, buffered=True)
        
        # 1. Fetch all questions for this session
        cursor.execute("""
            SELECT questions_id, question_text 
            FROM questions 
            WHERE session_id = %s 
            ORDER BY questions_id ASC
        """, (session_id,))
        questions_list = cursor.fetchall()
        print(f"📊 Found {len(questions_list)} questions for session {session_id}")
        
        if not questions_list:
            return jsonify({}), 200

        # Construct organized feedback dictionary
        feedback_data = {}
        
        for idx, q_item in enumerate(questions_list):
            q_id = q_item['questions_id']
            q_text = q_item['question_text']
            q_num_label = str(idx + 1) # Frontend uses "1", "2", etc.
            
            print(f"  ➡️ Processing Q{q_num_label} (ID: {q_id})")

            # Fetch Response (Video & Transcript)
            cursor.execute("SELECT video_path, transcript FROM responses WHERE session_id = %s AND question_id = %s", (session_id, q_id))
            resp_item = cursor.fetchone() or {}
            
            # Fetch Content Feedback
            cursor.execute("SELECT * FROM content_feedback WHERE session_id = %s AND q_no = %s", (session_id, q_id))
            cf = cursor.fetchone() or {}
            
            # Fetch Voice Feedback
            cursor.execute("SELECT * FROM voice_feedback WHERE session_id = %s AND q_no = %s", (session_id, q_id))
            vf = cursor.fetchone() or {}
            
            # Fetch Face Feedback
            cursor.execute("SELECT * FROM face_feedback WHERE session_id = %s AND qno = %s", (session_id, q_id))
            ff = cursor.fetchone() or {}
            
            print(f"    - Response found: {bool(resp_item)}")
            print(f"    - Content feedback found: {bool(cf)}")
            print(f"    - Voice feedback found: {bool(vf)}")
            print(f"    - Face feedback found: {bool(ff)}")

            # Helper to parse lists from various string formats (JSON, comma-separated, pipe-separated)
            def parse_list(val, default=[]):
                if not val: return default
                if isinstance(val, (list, tuple)): return list(val)
                
                # CLEAN string - remove outer quotes if double quoted
                if isinstance(val, str):
                    val = val.strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1].strip()

                # CRITICAL: Prioritize JSON format detection before string splitting
                if isinstance(val, str) and ((val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}'))):
                    try:
                        res = json.loads(val)
                        if isinstance(res, list): return res
                        if isinstance(res, dict): return [str(v) for v in res.values()]
                        return [str(res)]
                    except:
                        pass
                
                # Fallback to string splitting
                if isinstance(val, str):
                    # Check for separators
                    if '|' in val:
                        items = [i.strip() for i in val.split('|') if i.strip()]
                    elif ',' in val:
                        items = [i.strip() for i in val.split(',') if i.strip()]
                    else:
                        items = [val.strip()]
                else:
                    items = [str(val)]
                
                # Clean up items (remove leading numbers, bullets, etc.)
                cleaned = []
                for item in items:
                  c = re.sub(r'^[•\-\*✓]\s*', '', item)
                  c = re.sub(r'^\d+[\.\)\-\s]*', '', c)
                  if c.strip():
                    cleaned.append(c.strip())
                
                return cleaned if cleaned else default

            # Helper for metrics
            def parse_metric(val, default=60):
                try: return int(float(val)) if val is not None else default
                except: return default

            # Map the database data to the frontend structure
            db_video_path = resp_item.get('video_path', '')
            is_skipped = (db_video_path == 'NOT_ANSWERED')
            
            video_url = ""
            if db_video_path and not is_skipped:
                video_url = db_video_path.replace('\\', '/')
            
            # Prioritize response column from content_feedback, then responses table
            transcript = cf.get('response', resp_item.get('transcript', ''))
            if not transcript and is_skipped:
                transcript = "question was skipped"

            feedback_data[q_num_label] = {
                "q": q_text,
                "transcript": transcript,
                "video_path": video_url,
                "is_skipped": is_skipped,
                "content": {
                    "status": "Excellent" if parse_metric(cf.get('overall'), 0) >= 8 else "Good" if parse_metric(cf.get('overall'), 0) >= 6 else "Needs Improvement" if cf.get('overall') else "Not Evaluated",
                    # User: strengths column in content_feedback
                    "strengths": parse_list(cf.get('strengths'), []),
                    # User: improvements column in content_feedback
                    "improvements": parse_list(cf.get('improvements'), []),
                    "sample": cf.get('sample_answer', "")
                },
                "facial": {
                    "status": "Excellent" if ff.get('strength') else "Good" if ff.get('eye_contact') else "Not Evaluated",
                    "posture_feedback": ff.get('posture_feedback', ''),
                    "alignment_feedback": ff.get('alignment_feedback', ''),
                    "eyecontact_feedback": ff.get('eyecontact_feedback', ''),
                    "touch_feedback": ff.get('touch_feedback', ''),
                    "strengths": parse_list(ff.get('strength'), []),
                    "improvements": parse_list(ff.get('improvements'), [])
                },
                "voice": {
                    "status": "Excellent" if vf.get('overall_score', 0) >= 80 else "Good" if vf.get('overall_score') else "Not Evaluated",
                    "pace": parse_metric(vf.get('pace'), 0),
                    "pitch": parse_metric(vf.get('pitch'), 0),
                    "energy": parse_metric(vf.get('energy'), 0),
                    "clarity": parse_metric(vf.get('clarity'), 0),
                    # Data correctly fetched from voice_feedback table (vf)
                    "strengths": parse_list(vf.get('strengths'), []),
                    "improvements": parse_list(vf.get('improvements'), [])
                }
            }

        cursor.close()
        connection.close()
        return jsonify(feedback_data)

    except Error as e:
        print(f"Error fetching feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_feedback', methods=['POST'])
def clear_feedback():
    # Since we are db linked, clear feedback logic might change or remain for session
    return jsonify({"success": True})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀  Interview Feedback Server (DB Linked)")
    print("="*50)
    print("Feedback Page : http://localhost:5001/feedback")
    print("="*50 + "\n")
    app.run(debug=True, port=5001)
