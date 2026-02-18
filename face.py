from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from mediapipe import solutions

mp_face_mesh = solutions.face_mesh
mp_drawing = solutions.drawing_utils
mp_pose = solutions.pose

# Load database configuration from .env
load_dotenv()
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "")

DEFAULT_RESUME_ID = "FACE_DEFAULT"


def get_db_connection():
    """Create database connection using .env configuration"""
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL in face.py: {e}")
        return None


def store_face_feedback(feedback, student_id=None, session_id=None, qno=None):
    """
    Store face analysis feedback into face_feedback table.
    Expects feedback in the structure returned by generate_comprehensive_feedback.
    
    Args:
        feedback: Dictionary containing face analysis feedback
        student_id: ID of the student
        session_id: Interview session ID
        qno: Question number (must be a valid questions_id from the questions table)
    """
    connection = get_db_connection()
    if connection is None:
        print("❌ Failed to connect to database for face feedback")
        return False

    try:
        cursor = connection.cursor()

        # Extract feedback data from the feedback dictionary
        posture = feedback.get("posture", {})
        alignment = feedback.get("face_alignment", {})
        eye_contact = feedback.get("eye_contact", {})
        body_touch = feedback.get("body_touch", {})
        recommendations = feedback.get("recommendations", {})

        # Extract list data from recommendations and join with separator
        strengths_list = recommendations.get("strengths", []) or []
        improvements_list = recommendations.get("improvements", []) or []
        tips_list = recommendations.get("tips", []) or []

        strengths_str = " | ".join(str(s) for s in strengths_list)[:1000] if strengths_list else ""
        improvements_str = " | ".join(str(i) for i in improvements_list)[:1000] if improvements_list else ""
        tips_str = " | ".join(str(t) for t in tips_list)[:1000] if tips_list else ""

        # Prepare values - convert None to None (not empty string) for proper NULL handling
        posture_quality = posture.get("quality") or None
        posture_feedback_text = posture.get("feedback") or None
        alignment_quality = alignment.get("quality") or None
        alignment_feedback_text = alignment.get("feedback") or None
        eye_contact_quality = eye_contact.get("quality") or None
        eye_contact_feedback_text = eye_contact.get("feedback") or None
        touch_quality = body_touch.get("quality") or None
        touch_feedback_text = body_touch.get("feedback") or None

        insert_query = """
            INSERT INTO face_feedback (
                `student_id`,
                `posture_quality`,
                `posture_feedback`,
                `alignment`,
                `alignment_feedback`,
                `eye_contact`,
                `eyecontact_feedback`,
                `touch`,
                `touch_feedback`,
                `strength`,
                `improvements`,
                `tips`,
                `qno`,
                `session_id`
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            student_id,
            posture_quality,
            posture_feedback_text,
            alignment_quality,
            alignment_feedback_text,
            eye_contact_quality,
            eye_contact_feedback_text,
            touch_quality,
            touch_feedback_text,
            strengths_str,
            improvements_str,
            tips_str,
            int(qno) if qno is not None else None,
            session_id
        )

        print(f"📝 Inserting face feedback: student_id={student_id}, qno={qno}, session_id={session_id}")
        
        cursor.execute(insert_query, values)
        connection.commit()
        affected_rows = cursor.rowcount
        cursor.close()
        connection.close()
        
        if affected_rows > 0:
            print(f"✅ Face feedback stored in face_feedback table (rows: {affected_rows})")
            return True
        else:
            print("⚠️ No rows were inserted")
            return False
            
    except Error as e:
        print(f"❌ Error storing face feedback: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        try:
            if connection:
                connection.close()
        except Exception:
            pass
        return False

# ---------------- Video Analysis ----------------
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Track metrics over frames
    posture_metrics = []
    alignment_metrics = []
    eye_contact_metrics = []
    body_touch_metrics = []

    frame_count = 0
    while cap.isOpened() and frame_count < 150:  # Analyze more frames for better accuracy
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # Analyze posture
        if pose_results.pose_landmarks:
            posture_feedback = analyze_posture_complete(pose_results.pose_landmarks.landmark)
            posture_metrics.append(posture_feedback)

        # Analyze face alignment and eye contact
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                alignment_feedback = analyze_head_alignment(face_landmarks.landmark)
                alignment_metrics.append(alignment_feedback)
                
                eye_contact_feedback = analyze_eye_contact_improved(face_landmarks.landmark, 
                                                                      pose_results.pose_landmarks if pose_results.pose_landmarks else None)
                eye_contact_metrics.append(eye_contact_feedback)
                
                # Analyze body touch
                if pose_results.pose_landmarks:
                    body_touch_feedback = analyze_body_touch(face_landmarks.landmark, pose_results.pose_landmarks)
                    body_touch_metrics.append(body_touch_feedback)

        frame_count += 1

    cap.release()
    face_mesh.close()
    pose.close()

    # Generate final feedback
    final_feedback = generate_comprehensive_feedback(
        posture_metrics,
        alignment_metrics, 
        eye_contact_metrics,
        body_touch_metrics
    )

    return final_feedback

# ---------------- Posture Analysis ----------------
def analyze_posture_complete(landmarks):
    """Detect posture issues including slouching and leaning"""
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate metrics
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        ear_center_y = (left_ear.y + right_ear.y) / 2
        forward_slouch = ear_center_y - shoulder_center_y

        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        head_center_x = (left_ear.x + right_ear.x) / 2
        lateral_lean = abs(head_center_x - shoulder_center_x)

        # Count issues
        issues = []
        
        if forward_slouch > 0.12:
            issues.append("slouching forward")
        elif forward_slouch > 0.07:
            issues.append("slight forward lean")

        if shoulder_tilt > 0.06:
            issues.append("uneven shoulders")
        elif shoulder_tilt > 0.03:
            issues.append("slightly tilted shoulders")

        if lateral_lean > 0.08:
            issues.append("leaning to one side")
        elif lateral_lean > 0.04:
            issues.append("slight side lean")

        # Determine quality
        if len(issues) == 0:
            return {'quality': 'good', 'issues': []}
        elif len(issues) <= 1 and all('slight' in i for i in issues):
            return {'quality': 'average', 'issues': issues}
        else:
            return {'quality': 'needs_improvement', 'issues': issues}

    except Exception as e:
        return {'quality': 'average', 'issues': ['posture could not be analyzed']}

# ---------------- Head Alignment Analysis ----------------
def analyze_head_alignment(face_landmarks):
    """Detect if person is looking down, up, or to the sides"""
    try:
        # Get key face points
        nose_tip = face_landmarks[1]
        chin = face_landmarks[152]
        forehead = face_landmarks[10]
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        left_cheek = face_landmarks[234]
        right_cheek = face_landmarks[454]

        # Calculate face orientation
        # Vertical alignment (looking up/down)
        face_height = abs(forehead.y - chin.y)
        nose_vertical_pos = (nose_tip.y - forehead.y) / face_height if face_height > 0 else 0.5
        
        # Horizontal alignment (looking left/right)
        face_width = abs(left_cheek.x - right_cheek.x)
        nose_horizontal_pos = abs(nose_tip.x - (left_cheek.x + right_cheek.x) / 2) / face_width if face_width > 0 else 0
        
        # Eye level check
        eye_level_diff = abs(left_eye.y - right_eye.y)
        
        # Face centering in frame
        face_center_x = (left_eye.x + right_eye.x) / 2
        center_offset = abs(face_center_x - 0.5)

        issues = []
        
        # Check if looking down
        if nose_vertical_pos > 0.65:
            issues.append("looking down")
        # Check if looking up
        elif nose_vertical_pos < 0.35:
            issues.append("looking up")
        
        # Check if looking to the side
        if nose_horizontal_pos > 0.15:
            issues.append("head turned to side")
        
        # Check head tilt
        if eye_level_diff > 0.025:
            issues.append("head tilted")
        
        # Check centering
        if center_offset > 0.2:
            issues.append("not centered in frame")

        # Determine quality
        if len(issues) == 0:
            return {'quality': 'good', 'issues': []}
        elif len(issues) == 1:
            return {'quality': 'average', 'issues': issues}
        else:
            return {'quality': 'needs_improvement', 'issues': issues}

    except Exception as e:
        return {'quality': 'average', 'issues': ['alignment could not be analyzed']}

# ---------------- Eye Contact Analysis ----------------
def analyze_eye_contact_improved(face_landmarks, pose_landmarks):
    """Improved eye contact detection"""
    try:
        nose = face_landmarks[1]
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        
        # Calculate gaze direction based on eye and nose position
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Check horizontal gaze (left/right)
        horizontal_deviation = abs(eye_center_x - 0.5)
        
        # Check if looking away based on nose position relative to eyes
        nose_eye_offset_x = abs(nose.x - eye_center_x)
        
        # Use pose landmarks if available
        if pose_landmarks:
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            head_body_alignment = abs(eye_center_x - shoulder_center_x)
        else:
            head_body_alignment = horizontal_deviation

        issues = []
        
        # Strict thresholds for eye contact
        if horizontal_deviation > 0.15 or head_body_alignment > 0.12:
            issues.append("looking away from camera")
        elif horizontal_deviation > 0.08 or head_body_alignment > 0.06:
            issues.append("occasionally glancing away")
        
        if nose_eye_offset_x > 0.08:
            issues.append("head turned away")

        # Determine quality
        if len(issues) == 0:
            return {'quality': 'good', 'issues': []}
        elif len(issues) == 1 and 'occasionally' in issues[0]:
            return {'quality': 'average', 'issues': issues}
        else:
            return {'quality': 'needs_improvement', 'issues': issues}

    except Exception as e:
        return {'quality': 'average', 'issues': ['eye contact could not be analyzed']}

# ---------------- Body Touch Analysis ----------------
def analyze_body_touch(face_landmarks, pose_landmarks):
    """Detect hand touching face/hair"""
    try:
        # Face boundary points
        forehead = np.array([face_landmarks[10].x, face_landmarks[10].y])
        chin = np.array([face_landmarks[152].x, face_landmarks[152].y])
        left_cheek = np.array([face_landmarks[234].x, face_landmarks[234].y])
        right_cheek = np.array([face_landmarks[454].x, face_landmarks[454].y])
        nose = np.array([face_landmarks[1].x, face_landmarks[1].y])
        
        # Hand positions
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_index = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value]
        right_index = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value]
        
        left_hand = np.array([left_wrist.x, left_wrist.y])
        right_hand = np.array([right_wrist.x, right_wrist.y])
        left_finger = np.array([left_index.x, left_index.y])
        right_finger = np.array([right_index.x, right_index.y])
        
        # Calculate distances to face regions
        face_points = [forehead, chin, left_cheek, right_cheek, nose]
        
        min_left_hand_dist = min([np.linalg.norm(left_hand - point) for point in face_points])
        min_right_hand_dist = min([np.linalg.norm(right_hand - point) for point in face_points])
        min_left_finger_dist = min([np.linalg.norm(left_finger - point) for point in face_points])
        min_right_finger_dist = min([np.linalg.norm(right_finger - point) for point in face_points])
        
        # Check if hands are near face
        touching = False
        near_face = False
        
        # Stricter thresholds
        if (min_left_hand_dist < 0.12 or min_right_hand_dist < 0.12 or 
            min_left_finger_dist < 0.12 or min_right_finger_dist < 0.12):
            touching = True
        elif (min_left_hand_dist < 0.20 or min_right_hand_dist < 0.20 or
              min_left_finger_dist < 0.20 or min_right_finger_dist < 0.20):
            near_face = True
        
        # Check if hands are high (touching hair)
        left_hand_high = left_hand[1] < forehead[1] + 0.05
        right_hand_high = right_hand[1] < forehead[1] + 0.05
        
        issues = []
        quality = 'good'
        
        if touching:
            if left_hand_high or right_hand_high:
                issues.append("touching hair")
            else:
                issues.append("touching face")
            quality = 'needs_improvement'
        elif near_face:
            issues.append("hands near face")
            quality = 'average'
        
        return {'quality': quality, 'issues': issues, 'detected': touching or near_face}

    except Exception as e:
        return {'quality': 'good', 'issues': [], 'detected': False}

# ---------------- Generate Feedback ----------------
def generate_comprehensive_feedback(posture_metrics, alignment_metrics, eye_contact_metrics, body_touch_metrics):
    """Generate comprehensive feedback based on all metrics"""
    
    def analyze_metric_quality(metrics):
        if not metrics:
            return {'quality': 'average', 'issues': ['not enough data']}
        
        # Count quality occurrences
        good_count = sum(1 for m in metrics if m['quality'] == 'good')
        avg_count = sum(1 for m in metrics if m['quality'] == 'average')
        bad_count = sum(1 for m in metrics if m['quality'] == 'needs_improvement')
        
        total = len(metrics)
        good_pct = (good_count / total) * 100
        bad_pct = (bad_count / total) * 100
        
        # Collect all unique issues
        all_issues = []
        for m in metrics:
            all_issues.extend(m.get('issues', []))
        unique_issues = list(set(all_issues))
        
        # Determine overall quality (stricter thresholds)
        if bad_pct > 25:  # More than 25% bad = needs improvement
            quality = 'needs_improvement'
        elif good_pct < 60:  # Less than 60% good = average
            quality = 'average'
        else:
            quality = 'good'
        
        return {'quality': quality, 'issues': unique_issues, 'good_pct': good_pct, 'bad_pct': bad_pct}
    
    # Analyze each metric
    posture = analyze_metric_quality(posture_metrics)
    alignment = analyze_metric_quality(alignment_metrics)
    eye_contact = analyze_metric_quality(eye_contact_metrics)
    
    # Body touch analysis
    if body_touch_metrics:
        touch_detected = sum(1 for m in body_touch_metrics if m.get('detected', False))
        touch_pct = (touch_detected / len(body_touch_metrics)) * 100
        
        if touch_pct > 20:
            body_touch = {'quality': 'needs_improvement', 'issues': ['frequent hand-to-face touching'], 'touch_pct': touch_pct}
        elif touch_pct > 5:
            body_touch = {'quality': 'average', 'issues': ['occasional hand-to-face touching'], 'touch_pct': touch_pct}
        else:
            body_touch = {'quality': 'good', 'issues': [], 'touch_pct': touch_pct}
    else:
        body_touch = {'quality': 'good', 'issues': [], 'touch_pct': 0}
    
    # Generate detailed feedback messages
    posture_feedback = generate_posture_feedback(posture)
    alignment_feedback = generate_alignment_feedback(alignment)
    eye_contact_feedback = generate_eye_contact_feedback(eye_contact)
    body_touch_feedback = generate_body_touch_feedback(body_touch)
    
    # Overall assessment
    all_qualities = [posture['quality'], alignment['quality'], eye_contact['quality'], body_touch['quality']]
    needs_improvement_count = all_qualities.count('needs_improvement')
    good_count = all_qualities.count('good')
    
    if needs_improvement_count >= 2:
        overall = "Significant improvements needed - practice and refine your interview presence"
    elif needs_improvement_count == 1:
        overall = "Good foundation with areas needing improvement"
    elif good_count >= 3:
        overall = "Excellent interview presence - well done!"
    else:
        overall = "Good overall performance - minor refinements will make it excellent"
    
    # Generate recommendations
    recommendations = generate_detailed_recommendations(posture, alignment, eye_contact, body_touch)
    
    return {
        'posture': {
            'quality': posture['quality'],
            'feedback': posture_feedback
        },
        'face_alignment': {
            'quality': alignment['quality'],
            'feedback': alignment_feedback
        },
        'eye_contact': {
            'quality': eye_contact['quality'],
            'feedback': eye_contact_feedback
        },
        'body_touch': {
            'quality': body_touch['quality'],
            'feedback': body_touch_feedback
        },
        'overall_assessment': overall,
        'recommendations': recommendations
    }

def generate_posture_feedback(posture):
    """Generate posture feedback message"""
    quality = posture['quality']
    issues = posture.get('issues', [])
    
    if quality == 'good':
        return "Excellent posture! You maintained an upright, well-aligned position throughout."
    elif quality == 'average':
        if issues:
            return f"Generally good posture with minor issues: {', '.join(issues)}. Try to maintain better alignment."
        return "Your posture was generally acceptable with some minor fluctuations."
    else:
        if issues:
            return f"Posture needs improvement. Detected: {', '.join(issues)}. Sit upright with shoulders back and level."
        return "Posture needs work. Focus on sitting straight with proper alignment."

def generate_alignment_feedback(alignment):
    """Generate alignment feedback message"""
    quality = alignment['quality']
    issues = alignment.get('issues', [])
    
    if quality == 'good':
        return "Perfect camera alignment! Your face remained centered and properly positioned."
    elif quality == 'average':
        if issues:
            return f"Camera alignment was mostly good but noted: {', '.join(issues)}. Keep your head level and face the camera directly."
        return "Your head alignment was acceptable with occasional adjustments needed."
    else:
        if issues:
            return f"Camera alignment needs attention. Issues detected: {', '.join(issues)}. Face the camera directly and keep your head level."
        return "Head alignment needs improvement. Ensure you're looking straight at the camera."

def generate_eye_contact_feedback(eye_contact):
    """Generate eye contact feedback message"""
    quality = eye_contact['quality']
    issues = eye_contact.get('issues', [])
    
    if quality == 'good':
        return "Outstanding eye contact! You maintained consistent focus on the camera."
    elif quality == 'average':
        if issues:
            return f"Eye contact was decent but noticed: {', '.join(issues)}. Try to maintain more consistent focus on the camera lens."
        return "Your eye contact was acceptable but could be more consistent."
    else:
        if issues:
            return f"Eye contact needs significant improvement. Detected: {', '.join(issues)}. Look directly at the camera lens, not at the screen or elsewhere."
        return "Eye contact needs work. Practice looking directly at the camera lens throughout."

def generate_body_touch_feedback(body_touch):
    """Generate body touch feedback message"""
    quality = body_touch['quality']
    touch_pct = body_touch.get('touch_pct', 0)
    
    if quality == 'good':
        return "Excellent! You kept your hands away from your face and maintained professional composure."
    elif quality == 'average':
        return f"Occasional hand-to-face touching detected ({int(touch_pct)}% of time). Be mindful of keeping hands away from your face and hair."
    else:
        return f"Frequent hand-to-face touching detected ({int(touch_pct)}% of time). This is distracting - keep hands resting on the desk or in your lap."

def generate_detailed_recommendations(posture, alignment, eye_contact, body_touch):
    """Generate detailed recommendations with what's good and what needs work"""
    recommendations = {
        'strengths': [],
        'improvements': [],
        'tips': []
    }
    
    # Identify strengths
    if posture['quality'] == 'good':
        recommendations['strengths'].append("✓ Excellent posture maintained")
    if alignment['quality'] == 'good':
        recommendations['strengths'].append("✓ Perfect head alignment and camera positioning")
    if eye_contact['quality'] == 'good':
        recommendations['strengths'].append("✓ Strong, consistent eye contact")
    if body_touch['quality'] == 'good':
        recommendations['strengths'].append("✓ Professional hand positioning - no fidgeting")
    
    # Identify areas for improvement
    if posture['quality'] == 'needs_improvement':
        recommendations['improvements'].append("⚠ Posture: Sit up straight, keep shoulders level and relaxed")
        recommendations['tips'].append("• Place a cushion behind your lower back for support")
        recommendations['tips'].append("• Adjust chair height so feet are flat on the floor")
    elif posture['quality'] == 'average':
        recommendations['improvements'].append("⚠ Posture: Minor adjustments needed for better alignment")
        recommendations['tips'].append("• Check your sitting position before starting")
    
    if alignment['quality'] == 'needs_improvement':
        recommendations['improvements'].append("⚠ Head Alignment: Keep face centered and level with camera")
        recommendations['tips'].append("• Position camera at eye level")
        recommendations['tips'].append("• Sit directly in front of the camera, not at an angle")
    elif alignment['quality'] == 'average':
        recommendations['improvements'].append("⚠ Head Alignment: Maintain more consistent positioning")
    
    if eye_contact['quality'] == 'needs_improvement':
        recommendations['improvements'].append("⚠ Eye Contact: Look directly at the camera lens, not the screen")
        recommendations['tips'].append("• Place a small sticky note near the camera as a reminder")
        recommendations['tips'].append("• Practice speaking to the camera lens in mock interviews")
    elif eye_contact['quality'] == 'average':
        recommendations['improvements'].append("⚠ Eye Contact: Reduce glancing away from camera")
        recommendations['tips'].append("• Focus on maintaining camera gaze for longer periods")
    
    if body_touch['quality'] == 'needs_improvement':
        recommendations['improvements'].append("⚠ Hand Movements: Avoid touching face, hair, or eyes")
        recommendations['tips'].append("• Keep hands folded on desk or resting in lap")
        recommendations['tips'].append("• Be conscious of nervous habits - practice reducing fidgeting")
    elif body_touch['quality'] == 'average':
        recommendations['improvements'].append("⚠ Hand Movements: Minimize face touching")
        recommendations['tips'].append("• Rest hands in a comfortable, visible position")
    
    # If everything is good
    if not recommendations['improvements']:
        recommendations['improvements'].append("✓ All aspects look great!")
        recommendations['tips'].append("• Continue practicing to maintain consistency")
        recommendations['tips'].append("• Try varying question types to test adaptability")
    
    return recommendations
