"""
Dynamic Voice Feedback Generation Module
Handles all voice feedback generation using Groq API
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
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {str(e)[:200]}")
        raise Exception(f"Groq API failed: {str(e)}")


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


def generate_dynamic_voice_feedback(analysis_results, confidence_score):
    """
    Generate dynamic, personalized voice feedback using Groq API.
    Each feedback point is unique and tailored to specific metrics.
    
    Args:
        analysis_results: Dictionary containing pause_analysis, filler_analysis, audio_features
        confidence_score: Calculated confidence score (0-100)
    
    Returns:
        Dictionary with 'strengths' and 'improvements' arrays containing dynamic feedback sentences
    """
    try:
        # Extract all metrics from analysis
        pause_analysis = analysis_results['pause_analysis']
        filler_analysis = analysis_results['filler_analysis']
        audio_features = analysis_results['audio_features']
        
        # Calculate derived metrics
        word_count = filler_analysis['word_count']
        duration = audio_features['duration']
        speaking_rate = (word_count / duration * 60) if duration > 0 else 0
        
        # Pause metrics
        silence_ratio = pause_analysis['silence_ratio']
        strategic_pauses = pause_analysis['strategic_pauses_count']
        hesitant_pauses = pause_analysis['hesitant_pauses_count']
        total_pauses = pause_analysis['total_pauses']
        
        # Filler word metrics
        filler_count = filler_analysis['filler_count']
        filler_ratio = filler_analysis['filler_ratio']
        repeated_words = filler_analysis['important_repeated_words']
        repetition_penalty = filler_analysis['repetition_penalty']
        
        # Voice quality metrics
        avg_pitch = audio_features['avg_pitch']
        pitch_std = audio_features['pitch_std']
        avg_energy = audio_features['avg_energy']
        energy_std = audio_features['energy_std']
        
        # Format repeated words for prompt
        repeated_words_str = "None"
        if repeated_words:
            top_repeated = sorted(repeated_words.items(), key=lambda x: x[1], reverse=True)[:3]
            repeated_words_str = ", ".join([f"'{word}' ({count} times)" for word, count in top_repeated])
        
        # Build comprehensive prompt for Groq API
        prompt = f"""You are an expert speech and communication coach analyzing an interview candidate's voice delivery performance. Generate specific, actionable, and personalized feedback based on the detailed metrics provided below.

**PERFORMANCE OVERVIEW:**
Overall Confidence Score: {confidence_score}/100
Confidence Category: {get_confidence_category(confidence_score)}

**DETAILED METRICS ANALYSIS:**

1. SPEAKING PACE & FLOW:
   - Speaking Rate: {speaking_rate:.1f} words per minute
   - Duration: {duration:.1f} seconds
   - Total Words: {word_count} words
   - Ideal Range: 140-160 words per minute
   - Status: {"✓ Optimal" if 140 <= speaking_rate <= 160 else "⚠ Needs adjustment" if speaking_rate < 120 or speaking_rate > 180 else "~ Acceptable"}

2. PAUSE PATTERNS:
   - Total Silence Ratio: {silence_ratio:.1%} of speech duration
   - Strategic Pauses (200-800ms): {strategic_pauses} pauses
   - Hesitant Pauses (>800ms): {hesitant_pauses} pauses
   - Total Pauses Detected: {total_pauses}
   - Status: {"✓ Good flow" if silence_ratio < 0.15 else "⚠ Too many pauses" if silence_ratio > 0.25 else "~ Moderate"}

3. FILLER WORDS & REPETITION:
   - Total Filler Words: {filler_count} instances
   - Filler Ratio: {filler_ratio:.1%} of total words
   - Most Repeated Words: {repeated_words_str}
   - Repetition Impact Score: {repetition_penalty} penalty points
   - Status: {"✓ Excellent" if filler_ratio < 0.03 else "⚠ High usage" if filler_ratio > 0.06 else "~ Acceptable"}

4. VOICE QUALITY & PROJECTION:
   - Average Pitch: {avg_pitch:.1f} Hz
   - Pitch Variation (Expressiveness): {pitch_std:.1f} Hz
   - Voice Energy (Volume/Projection): {avg_energy:.4f}
   - Energy Variation (Dynamic Range): {energy_std:.4f}
   - Status: {"✓ Strong projection" if avg_energy > 0.025 else "⚠ Low energy" if avg_energy < 0.015 else "~ Moderate"}

**YOUR TASK:**
Generate personalized feedback in TWO categories. Each category should have 2-4 specific points. Each point must be ONE clear, complete sentence (maximum 2 short sentences if needed for clarity).

**CATEGORY 1: STRENGTHS (2-4 points)**
Identify what the candidate did WELL based on the metrics:
- Reference specific numbers when impressive (e.g., "Your speaking rate of 152 wpm is ideal")
- Highlight genuine strengths only - don't fabricate positives if metrics are poor
- Be encouraging and specific
- Use varied sentence structures - avoid repetitive phrasing
- Focus on the most impactful positive aspects

**CATEGORY 2: IMPROVEMENTS (2-4 points)**
Provide actionable, constructive advice:
- Identify specific areas needing work based on metrics
- Tell them HOW to improve, not just what's wrong
- Prioritize the most impactful improvements first
- Reference actual numbers when relevant (e.g., "Reduce your 12% filler word usage to under 5%")
- Use varied coaching language for variety
- Be specific about techniques or strategies

**CRITICAL GUIDELINES:**
✓ Each point = ONE complete sentence (max 2 short sentences only if necessary)
✓ Be conversational, natural, and professional
✓ Use varied language - avoid repetitive phrases
✓ Reference actual numbers/metrics when relevant
✓ Make each sentence unique and specific to THESE exact metrics
✓ Be honest - if performance is poor overall, acknowledge it constructively
✓ Avoid generic statements like "Keep practicing" or "Good job"
✓ Focus on actionable advice, not just observations

**OUTPUT FORMAT:**
Return ONLY a valid JSON object with this exact structure. No markdown, no code blocks, no explanations:

{{
    "strengths": [
        "First strength sentence referencing specific metrics",
        "Second strength sentence with varied phrasing",
        "Third strength sentence if applicable"
    ],
    "improvements": [
        "First improvement with actionable advice",
        "Second improvement with specific techniques",
        "Third improvement if applicable"
    ]
}}

**EXAMPLES OF GOOD FEEDBACK:**

STRENGTHS:
- "Your speaking pace of 152 words per minute hits the ideal range for clear, engaging communication."
- "With only 2% filler words in your response, you demonstrate strong articulation and preparation."
- "Your voice energy level of 0.028 shows confident projection that commands attention."

IMPROVEMENTS:
- "Reduce your hesitation pauses from 8 instances to under 3 by rehearsing key talking points beforehand."
- "The word 'actually' appeared 6 times; practice using alternatives like 'specifically' or 'in fact' to add variety."
- "Slow your speaking rate from 185 wpm to 150-160 wpm by pausing briefly after key points for better impact."

Now generate unique, personalized feedback for THIS candidate based on THEIR specific metrics above."""

        print(f"🤖 [Dynamic Feedback] Calling Groq API for voice feedback generation...")
        
        # Call Groq API with the prompt
        response = call_groq_api(prompt, model=ANALYSIS_MODEL)
        
        if response:
            print(f"✅ [Dynamic Feedback] Groq API response received: {len(response)} chars")
            
            try:
                # Clean the response - remove markdown code blocks if present
                cleaned_response = response.strip()
                
                # Remove markdown code blocks
                if '```json' in cleaned_response or '```' in cleaned_response:
                    # Extract JSON from markdown
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_response, re.DOTALL)
                    if json_match:
                        cleaned_response = json_match.group(1)
                    else:
                        # Try to find JSON object directly
                        json_match = re.search(r'\{.*?\}', cleaned_response, re.DOTALL)
                        if json_match:
                            cleaned_response = json_match.group(0)
                
                feedback_data = json.loads(cleaned_response)
                
                # Validate structure
                if 'strengths' in feedback_data and 'improvements' in feedback_data:
                    if isinstance(feedback_data['strengths'], list) and isinstance(feedback_data['improvements'], list):
                        # Ensure we have at least some feedback
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
                print(f"⚠️ [Dynamic Feedback] Failed to parse JSON from Groq response: {e}")
                print(f"Response preview: {response[:300]}...")
                
                # Fallback: try to extract arrays manually using regex
                try:
                    strengths_match = re.search(r'"strengths"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    improvements_match = re.search(r'"improvements"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    
                    if strengths_match and improvements_match:
                        # Extract strings from arrays
                        strengths_text = strengths_match.group(1)
                        improvements_text = improvements_match.group(1)
                        
                        # Parse individual strings
                        strengths = re.findall(r'"([^"]+)"', strengths_text)
                        improvements = re.findall(r'"([^"]+)"', improvements_text)
                        
                        if strengths or improvements:
                            print(f"✅ [Dynamic Feedback] Manually extracted feedback arrays")
                            return {
                                'strengths': strengths if strengths else ["Good effort on completing the response"],
                                'improvements': improvements if improvements else ["Continue practicing to improve"]
                            }
                except Exception as parse_error:
                    print(f"⚠️ [Dynamic Feedback] Manual extraction also failed: {parse_error}")
                
                # If all parsing fails, fall back to static feedback
                print("⚠️ [Dynamic Feedback] Falling back to static feedback")
                return generate_static_fallback_feedback(analysis_results, confidence_score)
        else:
            print("⚠️ [Dynamic Feedback] No response from Groq API, using fallback")
            return generate_static_fallback_feedback(analysis_results, confidence_score)
            
    except Exception as e:
        print(f"❌ [Dynamic Feedback] Generation error: {str(e)}")
        print("⚠️ Using static fallback feedback")
        return generate_static_fallback_feedback(analysis_results, confidence_score)


def generate_static_fallback_feedback(analysis_results, confidence_score):
    """
    Fallback function that provides basic static feedback if API fails.
    This ensures the system always returns feedback even if Groq API is unavailable.
    """
    try:
        pause_analysis = analysis_results['pause_analysis']
        filler_analysis = analysis_results['filler_analysis']
        audio_features = analysis_results['audio_features']
        
        speaking_rate = (filler_analysis['word_count'] / audio_features['duration']) * 60 if audio_features['duration'] > 0 else 0
        
        strengths = []
        improvements = []
        
        # Basic strengths based on thresholds
        if pause_analysis['silence_ratio'] < 0.15:
            strengths.append("Good speech flow with minimal pauses")
        
        if filler_analysis['filler_ratio'] < 0.03:
            strengths.append("Clear articulation with few filler words")
        
        if 140 <= speaking_rate <= 160:
            strengths.append("Appropriate speaking pace maintained")
        
        if audio_features['avg_energy'] > 0.025:
            strengths.append("Confident voice projection")
        
        if audio_features['pitch_std'] > 40:
            strengths.append("Expressive and engaging tone")
        
        if len(filler_analysis['important_repeated_words']) == 0:
            strengths.append("Good vocabulary diversity")
        
        # Ensure at least one strength
        if not strengths:
            strengths.append("Completed the response successfully")
        
        # Basic improvements based on thresholds
        if pause_analysis['silence_ratio'] > 0.25:
            improvements.append("Reduce long pauses between thoughts for better flow")
        
        if pause_analysis['hesitant_pauses_count'] > pause_analysis['strategic_pauses_count']:
            improvements.append("Use brief strategic pauses instead of long hesitations")
        
        if filler_analysis['filler_ratio'] > 0.06:
            improvements.append("Practice reducing filler words like 'um' and 'like' for clearer speech")
        
        if filler_analysis['important_repeated_words']:
            top_repeated = sorted(filler_analysis['important_repeated_words'].items(), key=lambda x: x[1], reverse=True)[:3]
            words_str = ", ".join([f"'{word}'" for word, count in top_repeated])
            improvements.append(f"Try using synonyms for frequently repeated words: {words_str}")
        
        if speaking_rate < 120:
            improvements.append("Try speaking slightly faster to maintain engagement")
        elif speaking_rate > 180:
            improvements.append("Slow down slightly for better clarity and comprehension")
        
        if audio_features['avg_energy'] < 0.015:
            improvements.append("Speak with more energy and confidence to project better")
        
        if audio_features['pitch_std'] < 30:
            improvements.append("Add more expression and variation to your voice tone")
        
        # Ensure at least one improvement
        if not improvements:
            improvements.append("Continue practicing to build confidence and improve delivery")
        
        return {
            'strengths': strengths[:4],  # Limit to 4 max
            'improvements': improvements[:4]  # Limit to 4 max
        }
        
    except Exception as e:
        print(f"❌ Fallback feedback error: {e}")
        return {
            'strengths': ["Response recorded successfully"],
            'improvements': ["Continue practicing to improve your interview skills"]
        }