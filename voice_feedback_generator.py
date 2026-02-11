"""
Dynamic Voice Feedback Generation Module
Handles all voice feedback generation using Groq API with precise metric-based analysis
"""

import json
import re
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
ANALYSIS_MODEL = "llama-3.1-8b-instant"


def call_groq_api(prompt, model=ANALYSIS_MODEL):
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
            temperature=0.7,  # Higher temperature for more dynamic/varied responses
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {str(e)[:200]}")
        raise Exception(f"Groq API failed: {str(e)}")


def analyze_metrics_and_generate_conditions(analysis_results):
    """
    Analyze all voice metrics against precise thresholds.
    Returns structured conditions with severity levels for each metric.
    
    METRICS ANALYZED:
    1. Speaking Rate (words per minute)
    2. Pause Patterns (silence ratio, strategic vs hesitant pauses)
    3. Filler Words (um, uh, like, etc.)
    4. Word Repetition (repeated important words)
    5. Voice Energy/Projection (volume/loudness)
    6. Pitch Variation (tone expressiveness)
    """
    pause_analysis = analysis_results['pause_analysis']
    filler_analysis = analysis_results['filler_analysis']
    audio_features = analysis_results['audio_features']
    
    word_count = filler_analysis['word_count']
    duration = audio_features['duration']
    speaking_rate = (word_count / duration * 60) if duration > 0 else 0
    
    silence_ratio = pause_analysis['silence_ratio']
    strategic_pauses = pause_analysis['strategic_pauses_count']
    hesitant_pauses = pause_analysis['hesitant_pauses_count']
    total_pauses = pause_analysis['total_pauses']
    
    filler_count = filler_analysis['filler_count']
    filler_ratio = filler_analysis['filler_ratio']
    repeated_words = filler_analysis['important_repeated_words']
    repetition_penalty = filler_analysis['repetition_penalty']
    
    avg_energy = audio_features['avg_energy']
    energy_std = audio_features['energy_std']
    avg_pitch = audio_features['avg_pitch']
    pitch_std = audio_features['pitch_std']
    
    conditions = {}
    
    # ============================================================
    # CRITICAL CHECK: SILENT OR NO RESPONSE
    # ============================================================
    if word_count == 0 or duration < 2 or avg_energy < 0.005:
        conditions['is_silent'] = True
        conditions['silent_reason'] = "No speech detected" if word_count == 0 else "Audio too quiet or too brief"
        return conditions  # Return immediately for silent responses
    
    conditions['is_silent'] = False
    
    # ============================================================
    # SECTION 1: SPEAKING PACE (Words Per Minute)
    # ============================================================
    # METRIC SCALE: 80-220 wpm (typical human speech range)
    # IDEAL RANGE: 140-160 wpm (clear, engaging interview pace)
    # TOOL: Calculated from word_count / duration
    
    if speaking_rate < 100:
        conditions['pace_status'] = "TOO_SLOW"
        conditions['pace_severity'] = "HIGH"
        conditions['pace_context'] = "Speaking very slowly, may sound uncertain or unprepared"
    elif speaking_rate < 130:
        conditions['pace_status'] = "SLOW"
        conditions['pace_severity'] = "MODERATE"
        conditions['pace_context'] = "Speaking slower than conversational pace"
    elif 140 <= speaking_rate <= 160:
        conditions['pace_status'] = "OPTIMAL"
        conditions['pace_severity'] = "EXCELLENT"
        conditions['pace_context'] = "Perfect conversational pace for interviews"
    elif 160 < speaking_rate <= 180:
        conditions['pace_status'] = "SLIGHTLY_FAST"
        conditions['pace_severity'] = "MINOR"
        conditions['pace_context'] = "Speaking a bit quickly but still comprehensible"
    elif speaking_rate > 180:
        conditions['pace_status'] = "TOO_FAST"
        conditions['pace_severity'] = "HIGH"
        conditions['pace_context'] = "Speaking too rapidly, may lose clarity"
    else:
        conditions['pace_status'] = "ACCEPTABLE"
        conditions['pace_severity'] = "MINOR"
        conditions['pace_context'] = "Speaking pace is workable but not ideal"
    
    # ============================================================
    # SECTION 2: PAUSE PATTERNS & FLOW
    # ============================================================
    # METRIC SCALE: silence_ratio (0.0 to 1.0 = 0% to 100%)
    # GOOD: < 0.15 (less than 15% silence)
    # MODERATE: 0.15-0.25 (15-25% silence)
    # POOR: > 0.25 (more than 25% silence)
    # TOOLS: pydub.silence.detect_silence() with 300ms min silence
    #        Strategic pauses: 200-800ms (good for emphasis)
    #        Hesitant pauses: >800ms (indicates uncertainty)
    
    if silence_ratio > 0.40:
        conditions['pause_status'] = "EXCESSIVE_PAUSES"
        conditions['pause_severity'] = "HIGH"
        conditions['pause_context'] = f"More than 40% of response is silence - severely disrupts flow"
    elif silence_ratio > 0.25:
        conditions['pause_status'] = "MANY_PAUSES"
        conditions['pause_severity'] = "MODERATE"
        conditions['pause_context'] = f"25-40% silence - noticeable flow interruption"
    elif silence_ratio < 0.10 and strategic_pauses >= 3:
        conditions['pause_status'] = "EXCELLENT_FLOW"
        conditions['pause_severity'] = "EXCELLENT"
        conditions['pause_context'] = f"Very smooth delivery with {strategic_pauses} strategic pauses for emphasis"
    elif silence_ratio < 0.15:
        conditions['pause_status'] = "GOOD_FLOW"
        conditions['pause_severity'] = "GOOD"
        conditions['pause_context'] = f"Less than 15% silence - natural, confident flow"
    elif hesitant_pauses > strategic_pauses * 2:
        conditions['pause_status'] = "HESITANT"
        conditions['pause_severity'] = "MODERATE"
        conditions['pause_context'] = f"{hesitant_pauses} long hesitation pauses vs {strategic_pauses} strategic pauses"
    else:
        conditions['pause_status'] = "MODERATE_FLOW"
        conditions['pause_severity'] = "MINOR"
        conditions['pause_context'] = f"15-25% silence - acceptable but could be smoother"
    
    # ============================================================
    # SECTION 3: FILLER WORDS
    # ============================================================
    # METRIC SCALE: filler_ratio (0.0 to 1.0 = 0% to 100%)
    # EXCELLENT: < 0.03 (less than 3% fillers)
    # ACCEPTABLE: 0.03-0.06 (3-6% fillers)
    # POOR: > 0.06 (more than 6% fillers)
    # TOOL: Text analysis counting "um", "uh", "like", "you know", etc.
    # Analyzed from Whisper transcription
    
    if filler_ratio > 0.10:
        conditions['filler_status'] = "EXCESSIVE_FILLERS"
        conditions['filler_severity'] = "HIGH"
        conditions['filler_context'] = f"{filler_count} filler words detected - more than 10% of speech"
    elif filler_ratio > 0.06:
        conditions['filler_status'] = "HIGH_FILLERS"
        conditions['filler_severity'] = "MODERATE"
        conditions['filler_context'] = f"{filler_count} filler words - noticeable usage affecting clarity"
    elif filler_ratio > 0.03:
        conditions['filler_status'] = "SOME_FILLERS"
        conditions['filler_severity'] = "MINOR"
        conditions['filler_context'] = f"{filler_count} filler words - minor usage, mostly clear"
    elif filler_ratio > 0:
        conditions['filler_status'] = "MINIMAL_FILLERS"
        conditions['filler_severity'] = "GOOD"
        conditions['filler_context'] = f"Only {filler_count} filler words - very clear speech"
    else:
        conditions['filler_status'] = "NO_FILLERS"
        conditions['filler_severity'] = "EXCELLENT"
        conditions['filler_context'] = "Zero filler words detected - exceptional clarity"
    
    # ============================================================
    # SECTION 4: WORD REPETITION
    # ============================================================
    # METRIC: Count of important words (>3 letters, not common words) repeated >2 times
    # GOOD: No words repeated more than 2-3 times
    # MODERATE: Some words repeated 4-5 times
    # POOR: Words repeated 6+ times
    # TOOL: Text frequency analysis excluding common words
    # Repetition penalty: (count - 4) * 2, max 20 points
    
    if repeated_words:
        top_repeated = sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)
        most_repeated_word = top_repeated[0][0]
        repetition_count = top_repeated[0][1]
        
        # Get top 2-3 repeated words
        repeated_words_list = [f"'{word}' ({count}x)" for word, count in top_repeated[:3]]
        repeated_words_str = ", ".join(repeated_words_list)
        
        if repetition_count >= 6:
            conditions['repetition_status'] = "HIGH_REPETITION"
            conditions['repetition_severity'] = "HIGH"
            conditions['repetition_context'] = f"Word '{most_repeated_word}' repeated {repetition_count} times - significantly affects variety"
            conditions['repeated_words'] = repeated_words_str
        elif repetition_count >= 4:
            conditions['repetition_status'] = "SOME_REPETITION"
            conditions['repetition_severity'] = "MODERATE"
            conditions['repetition_context'] = f"Word '{most_repeated_word}' repeated {repetition_count} times - noticeable"
            conditions['repeated_words'] = repeated_words_str
        else:
            conditions['repetition_status'] = "MINOR_REPETITION"
            conditions['repetition_severity'] = "MINOR"
            conditions['repetition_context'] = f"Word '{most_repeated_word}' repeated {repetition_count} times - acceptable"
            conditions['repeated_words'] = repeated_words_str
    else:
        conditions['repetition_status'] = "EXCELLENT_VARIETY"
        conditions['repetition_severity'] = "EXCELLENT"
        conditions['repetition_context'] = "No excessive word repetition - good vocabulary variety"
        conditions['repeated_words'] = "None"
    
    # ============================================================
    # SECTION 5: VOICE ENERGY / PROJECTION (Volume/Loudness)
    # ============================================================
    # METRIC SCALE: avg_energy typically 0.005 to 0.060 (RMS amplitude)
    # TOOL: librosa.feature.rms() - Root Mean Square energy
    # QUIET: < 0.015 (too soft, hard to hear)
    # MODERATE: 0.015-0.025 (acceptable but could be stronger)
    # GOOD: 0.025-0.035 (confident, clear projection)
    # STRONG: > 0.035 (excellent projection)
    
    if avg_energy < 0.010:
        conditions['energy_status'] = "VERY_QUIET"
        conditions['energy_severity'] = "HIGH"
        conditions['energy_context'] = "Voice is too soft - difficult to hear, lacks confidence"
    elif avg_energy < 0.018:
        conditions['energy_status'] = "QUIET"
        conditions['energy_severity'] = "MODERATE"
        conditions['energy_context'] = "Voice projection is low - needs more volume"
    elif avg_energy > 0.035:
        conditions['energy_status'] = "STRONG_PROJECTION"
        conditions['energy_severity'] = "EXCELLENT"
        conditions['energy_context'] = "Excellent voice projection - commands attention"
    elif avg_energy > 0.025:
        conditions['energy_status'] = "GOOD_PROJECTION"
        conditions['energy_severity'] = "GOOD"
        conditions['energy_context'] = "Good confident voice projection"
    else:
        conditions['energy_status'] = "MODERATE_PROJECTION"
        conditions['energy_severity'] = "MINOR"
        conditions['energy_context'] = "Voice projection is acceptable but could be stronger"
    
    # ============================================================
    # SECTION 6: PITCH VARIATION (Tone Expressiveness)
    # ============================================================
    # METRIC SCALE: pitch_std typically 10-80 Hz (standard deviation of pitch)
    # TOOL: librosa.piptrack() - Pitch tracking algorithm
    #       Extracts fundamental frequency (F0) over time
    # MONOTONE: < 25 Hz (very flat, boring)
    # SOMEWHAT FLAT: 25-35 Hz (needs more expression)
    # MODERATE: 35-45 Hz (acceptable variety)
    # GOOD: 45-55 Hz (engaging, expressive)
    # EXCELLENT: > 55 Hz (very dynamic and engaging)
    
    if pitch_std < 20:
        conditions['tone_status'] = "VERY_MONOTONE"
        conditions['tone_severity'] = "HIGH"
        conditions['tone_context'] = "Voice is very flat - lacks expressiveness and engagement"
    elif pitch_std < 30:
        conditions['tone_status'] = "SOMEWHAT_FLAT"
        conditions['tone_severity'] = "MODERATE"
        conditions['tone_context'] = "Voice needs more tonal variation to sound engaging"
    elif pitch_std > 55:
        conditions['tone_status'] = "VERY_EXPRESSIVE"
        conditions['tone_severity'] = "EXCELLENT"
        conditions['tone_context'] = "Excellent tonal variation - very engaging and dynamic"
    elif pitch_std > 40:
        conditions['tone_status'] = "GOOD_EXPRESSION"
        conditions['tone_severity'] = "GOOD"
        conditions['tone_context'] = "Good variety in tone - keeps listener engaged"
    else:
        conditions['tone_status'] = "MODERATE_EXPRESSION"
        conditions['tone_severity'] = "MINOR"
        conditions['tone_context'] = "Some tonal variation present - could add more expression"
    
    return conditions


def generate_dynamic_voice_feedback(analysis_results, confidence_score):
    """
    Generate dynamic, personalized voice feedback using Groq API.
    Analyzes precise thresholds and generates varied, natural feedback.
    
    The word "DYNAMIC" means: Different sentences for the same issue each time.
    Example: "Try speaking louder" vs "Project your voice more" vs "Increase your volume"
    
    Args:
        analysis_results: Dictionary containing pause_analysis, filler_analysis, audio_features
        confidence_score: Calculated confidence score (0-100) - used internally only
    
    Returns:
        Dictionary with 'strengths' and 'improvements' arrays containing natural feedback
    """
    try:
        # Analyze all metrics and get conditions
        conditions = analyze_metrics_and_generate_conditions(analysis_results)
        
        # ============================================================
        # HANDLE SILENT / NO RESPONSE CASE
        # ============================================================
        if conditions.get('is_silent', False):
            print("🔇 [Dynamic Feedback] Silent response detected - no speech or too quiet")
            return {
                'strengths': [],
                'improvements': [
                    "No response was detected - please ensure your microphone is working and speak clearly when recording",
                    "Make sure to unmute your microphone and speak at a normal volume"
                ]
            }
        
        # Extract metric details for repeated words
        repeated_words_detail = conditions.get('repeated_words', 'None')
        
        # ============================================================
        # BUILD COMPREHENSIVE PROMPT FOR GROQ API
        # ============================================================
        prompt = f"""You are a supportive interview coach providing warm, encouraging feedback on voice delivery.

**IMPORTANT: METRIC SCALES (for your understanding - NEVER mention numbers/scales in feedback):**

Speaking Rate Scale: 80-220 wpm (ideal: 140-160 wpm) - Calculated from word count / duration
Silence Ratio Scale: 0-100% (good: <15%, poor: >25%) - Detected by pydub silence analysis
Filler Ratio Scale: 0-100% (excellent: <3%, poor: >6%) - Counted from transcription
Voice Energy Scale: 0.005-0.060 RMS (good: >0.025, quiet: <0.018) - Measured by librosa RMS
Pitch Variation Scale: 10-80 Hz std dev (expressive: >40, flat: <30) - Tracked by librosa piptrack

**ANALYSIS RESULTS - WHAT YOU OBSERVED:**

1. SPEAKING PACE:
   Status: {conditions['pace_status']}
   Severity: {conditions['pace_severity']}
   Context: {conditions['pace_context']}

2. PAUSE PATTERNS & FLOW:
   Status: {conditions['pause_status']}
   Severity: {conditions['pause_severity']}
   Context: {conditions['pause_context']}

3. FILLER WORDS:
   Status: {conditions['filler_status']}
   Severity: {conditions['filler_severity']}
   Context: {conditions['filler_context']}

4. WORD REPETITION:
   Status: {conditions['repetition_status']}
   Severity: {conditions['repetition_severity']}
   Context: {conditions['repetition_context']}
   Repeated Words: {repeated_words_detail}

5. VOICE PROJECTION / ENERGY:
   Status: {conditions['energy_status']}
   Severity: {conditions['energy_severity']}
   Context: {conditions['energy_context']}

6. PITCH VARIATION / TONE:
   Status: {conditions['tone_status']}
   Severity: {conditions['tone_severity']}
   Context: {conditions['tone_context']}

**YOUR TASK:**
Generate natural, friendly feedback based on these conditions. Create VARIED, DIFFERENT sentences each time even for the same issue type.

**DYNAMIC SENTENCE GENERATION RULES:**
- For the SAME issue, use DIFFERENT wordings/phrasings each time
- Example for "speak louder": 
  * "Try speaking up a bit more to project confidence"
  * "Increase your volume so your voice carries better"
  * "Make sure to speak louder so you can be clearly heard"
  * "Project your voice more confidently"
- Vary sentence structure, word choice, and coaching style
- Keep it natural - like a real human coach would say it

**SEVERITY-TO-TONE MAPPING:**

HIGH Severity (serious issues):
- Direct but supportive: "You're speaking too [issue]..."
- Emphasize importance: "needs significant improvement", "requires attention"
- Examples: "You're using too many filler words which really disrupts your clarity"

MODERATE Severity (noticeable issues):
- Friendly coaching: "Try to...", "Work on...", "Focus on..."
- Encouraging: "would help", "would make a difference"
- Examples: "Try to reduce those long pauses by preparing your points beforehand"

MINOR Severity (small improvements):
- Gentle suggestions: "Consider...", "You might...", "Could..."
- Optional tone: "could help", "would polish", "might enhance"
- Examples: "You might want to add a bit more expression to your tone"

GOOD Severity (strengths):
- Positive reinforcement: "Your [aspect] is/was..."
- Specific praise: "well", "nicely", "effectively", "naturally"
- Examples: "Your speaking pace flows naturally and is easy to follow"

EXCELLENT Severity (outstanding):
- Strong positive: "Excellent...", "Outstanding...", "Great..."
- Enthusiastic: "very", "really", "particularly"
- Examples: "Your voice projection is excellent and commands attention"

**PLACEMENT RULES:**
- EXCELLENT/GOOD severity → STRENGTHS only
- HIGH/MODERATE/MINOR severity → IMPROVEMENTS only
- Each section analyzed (pace, pauses, fillers, repetition, energy, tone) should appear in either strengths OR improvements
- Provide 2-5 points total for strengths (only include actual good things)
- Provide 2-6 points total for improvements (only include actual issues)

**CRITICAL FORMATTING RULES:**
✓ Each point = ONE clear sentence (max 2 short sentences if absolutely needed)
✓ NO numbers, percentages, scores, metrics, or technical terms
✓ NO phrases like "Hz", "wpm", "ratio", "percentage", "energy level"
✓ Use natural language: "speak louder" not "increase energy levels"
✓ Be specific about WHAT and HOW: don't just say "improve your tone"
✓ Match severity to urgency: HIGH = direct, MINOR = gentle
✓ Create VARIED sentences - never repeat the same phrasing patterns

**OUTPUT FORMAT - Return ONLY valid JSON:**
{{
    "strengths": [
        "First strength based on GOOD/EXCELLENT items",
        "Second strength if applicable",
        "Third strength if applicable"
    ],
    "improvements": [
        "First improvement based on HIGH severity (urgent tone)",
        "Second improvement based on MODERATE severity (coaching tone)",
        "Third improvement based on MINOR severity (gentle suggestion)",
        "Additional improvements as needed"
    ]
}}

**EXAMPLE OUTPUTS SHOWING DYNAMIC VARIETY:**

For EXCESSIVE_FILLERS (HIGH):
- Option A: "You're using too many filler words like 'um' and 'like' which disrupts your message - practice pausing silently instead"
- Option B: "Work on cutting down the filler words by taking brief pauses when you need to think"
- Option C: "Reduce those 'ums' and 'likes' by staying focused on your prepared talking points"

For SLOW pace (MODERATE):
- Option A: "Try speaking a bit faster to maintain energy and keep the interviewer engaged"
- Option B: "Pick up your pace slightly so your delivery feels more confident and dynamic"
- Option C: "Speed up your speaking rate to sound more enthusiastic about your points"

For QUIET voice (MODERATE):
- Option A: "Speak louder and project your voice more to convey greater confidence"
- Option B: "Try increasing your volume so your voice carries better across the room"
- Option C: "Make sure to speak up more confidently so you can be clearly heard"

For GOOD_PROJECTION (GOOD):
- Option A: "Your voice projects well with good confidence throughout"
- Option B: "You maintained strong voice projection which sounds professional"
- Option C: "Your volume and projection came across clearly and confidently"

Now generate UNIQUE, VARIED feedback for THIS candidate based on their specific conditions above. Remember: create DIFFERENT sentences than the examples - be creative!"""

        print(f"🤖 [Dynamic Feedback] Calling Groq API for voice feedback generation...")
        
        # Call Groq API with higher temperature for variety
        response = call_groq_api(prompt, model=ANALYSIS_MODEL)
        
        if response:
            print(f"✅ [Dynamic Feedback] Groq API response received: {len(response)} chars")
            
            try:
                # Clean the response - remove markdown code blocks if present
                cleaned_response = response.strip()
                
                # Remove markdown code blocks
                if '```json' in cleaned_response or '```' in cleaned_response:
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_response, re.DOTALL)
                    if json_match:
                        cleaned_response = json_match.group(1)
                    else:
                        json_match = re.search(r'\{.*?\}', cleaned_response, re.DOTALL)
                        if json_match:
                            cleaned_response = json_match.group(0)
                
                feedback_data = json.loads(cleaned_response)
                
                # Validate structure
                if 'strengths' in feedback_data and 'improvements' in feedback_data:
                    if isinstance(feedback_data['strengths'], list) and isinstance(feedback_data['improvements'], list):
                        # Validate we have actual content
                        if len(feedback_data['strengths']) > 0 or len(feedback_data['improvements']) > 0:
                            print(f"✅ [Dynamic Feedback] Generated {len(feedback_data['strengths'])} strengths and {len(feedback_data['improvements'])} improvements")
                            return feedback_data
                        else:
                            raise ValueError("Empty feedback arrays")
                    else:
                        raise ValueError("Strengths and improvements must be arrays")
                else:
                    raise ValueError("Missing strengths or improvements in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ [Dynamic Feedback] Failed to parse JSON: {e}")
                print(f"Response preview: {response[:300]}...")
                
                # Fallback: try manual extraction
                try:
                    strengths_match = re.search(r'"strengths"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    improvements_match = re.search(r'"improvements"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    
                    if strengths_match and improvements_match:
                        strengths = re.findall(r'"([^"]+)"', strengths_match.group(1))
                        improvements = re.findall(r'"([^"]+)"', improvements_match.group(1))
                        
                        if strengths or improvements:
                            print(f"✅ [Dynamic Feedback] Manually extracted feedback")
                            return {
                                'strengths': strengths if strengths else ["You completed your response"],
                                'improvements': improvements if improvements else ["Keep practicing your delivery"]
                            }
                except Exception as parse_error:
                    print(f"⚠️ [Dynamic Feedback] Manual extraction failed: {parse_error}")
                
                print("⚠️ [Dynamic Feedback] Falling back to static feedback")
                return generate_static_fallback_feedback(analysis_results, confidence_score)
        else:
            print("⚠️ [Dynamic Feedback] No response from Groq API")
            return generate_static_fallback_feedback(analysis_results, confidence_score)
            
    except Exception as e:
        print(f"❌ [Dynamic Feedback] Generation error: {str(e)}")
        print("⚠️ Using static fallback feedback")
        return generate_static_fallback_feedback(analysis_results, confidence_score)


def generate_static_fallback_feedback(analysis_results, confidence_score):
    """
    Rule-based fallback feedback with precise threshold checking.
    Mirrors the dynamic feedback logic but with pre-written messages.
    Used when Groq API is unavailable.
    """
    try:
        # Get conditions using same analysis logic
        conditions = analyze_metrics_and_generate_conditions(analysis_results)
        
        # Handle silent response
        if conditions.get('is_silent', False):
            return {
                'strengths': [],
                'improvements': [
                    "No response was detected - please ensure your microphone is working and speak clearly when recording",
                    "Make sure to unmute your microphone and speak at a normal volume"
                ]
            }
        
        strengths = []
        improvements = []
        
        # === SPEAKING PACE ===
        pace_severity = conditions['pace_severity']
        if pace_severity == "EXCELLENT":
            strengths.append("Your speaking pace feels natural and comfortable to follow")
        elif pace_severity == "HIGH":
            if conditions['pace_status'] == "TOO_SLOW":
                improvements.append("You're speaking very slowly which may make you seem uncertain - try to pick up your pace to sound more confident")
            else:  # TOO_FAST
                improvements.append("You're speaking too quickly which affects clarity - slow down and pause briefly after key points")
        elif pace_severity == "MODERATE":
            if "SLOW" in conditions['pace_status']:
                improvements.append("Try speaking a bit faster to maintain energy and keep the interviewer engaged")
        elif pace_severity == "MINOR":
            if "FAST" in conditions['pace_status']:
                improvements.append("You're speaking slightly fast - consider slowing down just a bit for better clarity")
        
        # === PAUSE PATTERNS ===
        pause_severity = conditions['pause_severity']
        if pause_severity == "EXCELLENT":
            strengths.append("Your speech flows very smoothly with well-timed strategic pauses")
        elif pause_severity == "GOOD":
            strengths.append("You maintained good flow with minimal unnecessary pauses")
        elif pause_severity == "HIGH":
            improvements.append("You have too many long pauses which breaks your flow - prepare your key points beforehand to speak more smoothly")
        elif pause_severity == "MODERATE":
            if "MANY_PAUSES" in conditions['pause_status']:
                improvements.append("Try to reduce long pauses by staying focused on your main talking points for better flow")
            elif "HESITANT" in conditions['pause_status']:
                improvements.append("Use shorter intentional pauses for emphasis rather than hesitating while you think")
        
        # === FILLER WORDS ===
        filler_severity = conditions['filler_severity']
        if filler_severity == "EXCELLENT":
            strengths.append("You speak with exceptional clarity and no filler words")
        elif filler_severity == "GOOD":
            strengths.append("You spoke clearly with very few filler words")
        elif filler_severity == "HIGH":
            improvements.append("You're using too many filler words which disrupts your clarity - practice taking brief silent pauses instead of saying 'um' or 'like'")
        elif filler_severity == "MODERATE":
            improvements.append("You have noticeable filler word usage - try pausing briefly instead when you need time to think")
        elif filler_severity == "MINOR":
            improvements.append("You use some filler words - consider reducing them by staying more focused on your prepared points")
        
        # === WORD REPETITION ===
        repetition_severity = conditions['repetition_severity']
        if repetition_severity == "EXCELLENT":
            strengths.append("You used excellent variety in your vocabulary throughout the response")
        elif repetition_severity == "GOOD":
            strengths.append("Your vocabulary variety was good with minimal repetition")
        elif repetition_severity == "HIGH":
            # Extract the repeated word from context
            context = conditions['repetition_context']
            word_match = re.search(r"'(\w+)'", context)
            if word_match:
                word = word_match.group(1)
                improvements.append(f"You repeated the word '{word}' too often - practice using synonyms or different phrasing")
        elif repetition_severity == "MODERATE":
            context = conditions['repetition_context']
            word_match = re.search(r"'(\w+)'", context)
            if word_match:
                word = word_match.group(1)
                improvements.append(f"Try varying your word choice to avoid repeating '{word}' multiple times")
        
        # === VOICE PROJECTION ===
        energy_severity = conditions['energy_severity']
        if energy_severity == "EXCELLENT":
            strengths.append("Your voice projection is excellent and commands attention")
        elif energy_severity == "GOOD":
            strengths.append("You projected your voice well with good confidence")
        elif energy_severity == "HIGH":
            improvements.append("Your voice is too soft and needs much more volume - speak up confidently so you can be heard clearly")
        elif energy_severity == "MODERATE":
            improvements.append("Try speaking louder and projecting your voice more to convey greater confidence")
        elif energy_severity == "MINOR":
            improvements.append("Consider speaking a bit louder to strengthen your projection")
        
        # === TONE VARIATION ===
        tone_severity = conditions['tone_severity']
        if tone_severity == "EXCELLENT":
            strengths.append("Your tone variation is excellent and keeps your response very engaging")
        elif tone_severity == "GOOD":
            strengths.append("You varied your tone nicely which kept your response interesting")
        elif tone_severity == "HIGH":
            improvements.append("Your voice lacks expression significantly - add much more variety by emphasizing important words and varying your tone")
        elif tone_severity == "MODERATE":
            improvements.append("Add more expression to your voice by varying your tone and emphasizing key points")
        elif tone_severity == "MINOR":
            improvements.append("Consider adding a bit more variety to your tone to make your delivery more engaging")
        
        # Ensure we have at least something
        if not strengths and not improvements:
            strengths.append("You completed your response")
            improvements.append("Keep practicing to strengthen your delivery")
        
        if not strengths:
            strengths.append("You communicated your ideas with effort")
        
        if not improvements:
            improvements.append("Continue practicing to build even more confidence")
        
        return {
            'strengths': strengths[:5],
            'improvements': improvements[:6]
        }
        
    except Exception as e:
        print(f"❌ Fallback feedback error: {e}")
        return {
            'strengths': ["You completed your response"],
            'improvements': ["Continue practicing to improve your interview delivery"]
        }