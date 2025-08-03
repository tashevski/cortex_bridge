#!/usr/bin/env python3
"""Session analyzer using Gemma 3n to generate contextual prompts for future LLM interactions"""

import json
import requests
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from conversation_logger import get_conversation_history, list_conversation_sessions
import argparse

class SessionAnalyzer:
    def __init__(self, model: str = "gemma3n:e2b", ollama_url: str = "http://localhost:11434"):
        """Initialize session analyzer with Gemma model"""
        self.model = model
        self.ollama_url = ollama_url
        self.api_base = f"{ollama_url}/api"
        
        # Ensure Ollama is running and model is available
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                raise Exception("Ollama is not running")
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                print(f"Model {self.model} not found. Available models: {model_names}")
                print(f"Pulling model {self.model}...")
                self._pull_model()
            else:
                print(f"✅ Model {self.model} is available")
                
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            raise
    
    def _pull_model(self):
        """Pull the specified model"""
        try:
            response = requests.post(f"{self.api_base}/pull", json={"name": self.model})
            if response.status_code == 200:
                print(f"✅ Successfully pulled model {self.model}")
            else:
                raise Exception(f"Failed to pull model: {response.text}")
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            raise
    
    def _call_gemma(self, prompt: str, system_prompt: str = None) -> str:
        """Call Gemma model with prompt"""
        try:
            # Prepare the request
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Make the request
            response = requests.post(f"{self.api_base}/generate", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"API request failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error calling Gemma: {e}")
            return ""
    
    def format_session_for_analysis(self, session_id: str) -> str:
        """Format session data for analysis"""
        history = get_conversation_history(session_id)
        
        if not history['session'] or not history['utterances']:
            return ""
        
        session = history['session']
        utterances = history['utterances']
        
        # Format session metadata
        session_info = f"""
            CONVERSATION SESSION ANALYSIS
            Session ID: {session_id}
            Start Time: {session[2]}
            End Time: {session[3]}
            Total Utterances: {session[4]}
            Speaker Count: {session[5]}
            Emotion Summary: {session[6]}

            CONVERSATION TRANSCRIPT:
            """
        
        # Format utterances
        for i, utterance in enumerate(utterances, 1):
            timestamp = utterance[2]
            speaker = utterance[3]
            text = utterance[4]
            emotion = utterance[5]
            confidence = utterance[6]
            is_question = utterance[7]
            
            question_mark = " [QUESTION]" if is_question else ""
            session_info += f"{i}. [{timestamp}] {speaker}: {text} (Emotion: {emotion}, Confidence: {confidence:.2f}){question_mark}\n"
        
        return session_info
    
    def analyze_session_emotions(self, session_id: str) -> str:
        """Analyze emotional patterns in a session"""
        history = get_conversation_history(session_id)
        
        if not history['utterances']:
            return ""
        
        # Count emotions
        emotion_counts = {}
        speaker_emotions = {}
        question_emotions = {}
        
        for utterance in history['utterances']:
            emotion = utterance[5]
            speaker = utterance[3]
            is_question = utterance[7]
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if speaker not in speaker_emotions:
                speaker_emotions[speaker] = {}
            speaker_emotions[speaker][emotion] = speaker_emotions[speaker].get(emotion, 0) + 1
            
            if is_question:
                question_emotions[emotion] = question_emotions.get(emotion, 0) + 1
        
        # Format emotion analysis
        analysis = f"""
            EMOTION ANALYSIS FOR SESSION: {session_id}

            Overall Emotion Distribution:
            """

        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(history['utterances']) * 100
            analysis += f"- {emotion}: {count} ({percentage:.1f}%)\n"
        
        analysis += "\nEmotions by Speaker:\n"
        for speaker, emotions in speaker_emotions.items():
            analysis += f"- {speaker}:\n"
            for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                analysis += f"  * {emotion}: {count}\n"
        
        if question_emotions:
            analysis += "\nEmotions in Questions:\n"
            for emotion, count in sorted(question_emotions.items(), key=lambda x: x[1], reverse=True):
                analysis += f"- {emotion}: {count}\n"
        
        return analysis
    
    def generate_contextual_prompt(self, session_id: str, analysis_type: str = "comprehensive") -> str:
        """Generate contextual prompt based on session analysis"""
        
        # Get session data
        session_data = self.format_session_for_analysis(session_id)
        emotion_analysis = self.analyze_session_emotions(session_id)
        
        if not session_data:
            return "No session data available for analysis."
        
        # Create analysis prompt based on type
        if analysis_type == "comprehensive":
            prompt = f"""
                Please analyze this conversation session and create a comprehensive contextual prompt that captures the key aspects of this interaction. This prompt will be used to provide context for future conversations with the same person or in similar situations.

                {session_data}

                {emotion_analysis}

                Please create a contextual prompt that includes:
                1. Key topics and themes discussed
                2. Emotional patterns and dynamics
                3. Communication style and preferences
                4. Important context or background information
                5. Relationship dynamics between speakers
                6. Any recurring patterns or concerns
                7. Questions or topics that seemed important to the speakers

                Format the response as a clear, concise contextual prompt that could be used to inform future conversations.
                """
                        
        elif analysis_type == "emotional":
            prompt = f"""
                Please analyze the emotional patterns in this conversation and create a contextual prompt focused on emotional intelligence and empathy.

                {session_data}

                {emotion_analysis}

                Create a contextual prompt that captures:
                1. Primary emotional themes
                2. Emotional triggers or sensitive topics
                3. How each speaker expresses emotions
                4. Emotional support needs
                5. Mood patterns and changes
                6. Emotional communication preferences

                Format as a brief, empathetic contextual prompt for future interactions.
                """
                        
        elif analysis_type == "topical":
            prompt = f"""
                Please analyze the topics and themes in this conversation and create a contextual prompt focused on subject matter and interests.

                {session_data}

                Create a contextual prompt that captures:
                1. Main topics discussed
                2. Areas of interest or expertise
                3. Questions asked and concerns raised
                4. Knowledge gaps or learning needs
                5. Professional or personal context
                6. Future topics to explore

                Format as a concise contextual prompt for topic-focused future conversations.
                """
                        
        else:
            prompt = f"""
                Please analyze this conversation session and create a contextual prompt.

                {session_data}

                {emotion_analysis}

                Create a contextual prompt that captures the key aspects of this interaction for future reference.
                """
        
        # Call Gemma for analysis
        system_prompt = """You are an expert conversation analyst. Your task is to analyze conversation sessions and create contextual prompts that capture the essence of the interaction for future reference. Be concise, insightful, and practical in your analysis."""
        
        return self._call_gemma(prompt, system_prompt)
    
    def generate_speaker_profile(self, session_id: str, speaker_name: str) -> str:
        """Generate a speaker profile based on their conversation patterns"""
        history = get_conversation_history(session_id)
        
        if not history['utterances']:
            return ""
        
        # Filter utterances for the specific speaker
        speaker_utterances = []
        for utterance in history['utterances']:
            if utterance[3] == speaker_name:
                speaker_utterances.append(utterance)
        
        if not speaker_utterances:
            return f"No utterances found for speaker: {speaker_name}"
        
        # Analyze speaker patterns
        emotions = {}
        questions = 0
        total_utterances = len(speaker_utterances)
        
        for utterance in speaker_utterances:
            emotion = utterance[5]
            is_question = utterance[7]
            
            emotions[emotion] = emotions.get(emotion, 0) + 1
            if is_question:
                questions += 1
        
        # Create speaker analysis
        analysis = f"""
            SPEAKER PROFILE: {speaker_name}
            Session: {session_id}
            Total Utterances: {total_utterances}
            Questions Asked: {questions} ({questions/total_utterances*100:.1f}%)

            Emotional Profile:
            """
        
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_utterances * 100
            analysis += f"- {emotion}: {count} ({percentage:.1f}%)\n"
        
        analysis += "\nSample Utterances:\n"
        for i, utterance in enumerate(speaker_utterances[:5], 1):
            text = utterance[4]
            emotion = utterance[5]
            analysis += f"{i}. \"{text}\" (Emotion: {emotion})\n"
        
        # Generate profile prompt
        prompt = f"""
            Based on this speaker's conversation patterns, create a detailed speaker profile that could be used to personalize future interactions.

            {analysis}

            Please create a speaker profile that includes:
            1. Communication style and preferences
            2. Emotional patterns and triggers
            3. Topics of interest and expertise
            4. Question-asking patterns
            5. Personality insights
            6. Recommendations for future interactions

            Format as a concise, actionable speaker profile.
            """
        
        system_prompt = """You are an expert in communication analysis and personality profiling. Create insightful, practical speaker profiles that can inform future interactions."""
        
        return self._call_gemma(prompt, system_prompt)
    
    def generate_conversation_summary(self, session_id: str) -> str:
        """Generate a conversation summary"""
        session_data = self.format_session_for_analysis(session_id)
        
        if not session_data:
            return "No session data available for summary."
        
        prompt = f"""
            Please provide a comprehensive summary of this conversation session.

            {session_data}

            Create a summary that includes:
            1. Main topics and themes discussed
            2. Key insights and takeaways
            3. Important decisions or agreements made
            4. Emotional highlights and dynamics
            5. Questions raised and answered
            6. Action items or follow-up needs
            7. Overall conversation quality and engagement

            Format as a clear, structured summary suitable for future reference.
            """
        
        system_prompt = """You are an expert conversation summarizer. Create clear, comprehensive summaries that capture the essence and key points of conversations."""
        
        return self._call_gemma(prompt, system_prompt)
    
    def save_contextual_prompt(self, session_id: str, prompt: str, prompt_type: str = "contextual") -> str:
        """Save contextual prompt to file"""
        prompts_dir = Path("contextual_prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{prompt_type}_{timestamp}.txt"
        filepath = prompts_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Contextual Prompt for Session: {session_id}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Type: {prompt_type}\n")
            f.write("=" * 60 + "\n\n")
            f.write(prompt)
        
        return str(filepath)

def main():
    parser = argparse.ArgumentParser(description="Session analyzer using Gemma 3n")
    parser.add_argument("command", choices=["analyze", "profile", "summary", "save"], 
                       help="Command to execute")
    parser.add_argument("--session", "-s", required=True, help="Session ID to analyze")
    parser.add_argument("--speaker", "-p", help="Speaker name for profile generation")
    parser.add_argument("--type", "-t", choices=["comprehensive", "emotional", "topical"], 
                       default="comprehensive", help="Analysis type")
    parser.add_argument("--model", "-m", default="gemma3n:e2b", help="Ollama model to use")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    analyzer = SessionAnalyzer(model=args.model)
    
    if args.command == "analyze":
        print(f"🔍 Analyzing session: {args.session}")
        print(f"📊 Analysis type: {args.type}")
        print("=" * 60)
        
        prompt = analyzer.generate_contextual_prompt(args.session, args.type)
        print(prompt)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"\n💾 Saved to: {args.output}")
    
    elif args.command == "profile":
        if not args.speaker:
            print("❌ Please specify a speaker with --speaker")
            return
        
        print(f"👤 Generating profile for: {args.speaker}")
        print(f"📝 Session: {args.session}")
        print("=" * 60)
        
        profile = analyzer.generate_speaker_profile(args.session, args.speaker)
        print(profile)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(profile)
            print(f"\n💾 Saved to: {args.output}")
    
    elif args.command == "summary":
        print(f"📋 Generating summary for session: {args.session}")
        print("=" * 60)
        
        summary = analyzer.generate_conversation_summary(args.session)
        print(summary)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\n💾 Saved to: {args.output}")
    
    elif args.command == "save":
        print(f"💾 Generating and saving contextual prompt for session: {args.session}")
        
        prompt = analyzer.generate_contextual_prompt(args.session, args.type)
        filepath = analyzer.save_contextual_prompt(args.session, prompt, args.type)
        
        print(f"✅ Contextual prompt saved to: {filepath}")
        print("\n📄 Prompt content:")
        print("=" * 60)
        print(prompt)

if __name__ == "__main__":
    main() 