#!/usr/bin/env python3
"""Script to list available speakers in the VCTK TTS model"""

from TTS.api import TTS

def list_vctk_speakers():
    """List all available speakers in the VCTK model"""
    try:
        # Initialize the VCTK model
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
        
        # Get available speakers
        speakers = tts.speakers
        
        print("🎤 Available VCTK Speakers:")
        print("=" * 50)
        
        # Group speakers by gender (VCTK speakers are typically p225-p376)
        male_speakers = []
        female_speakers = []
        
        for speaker in speakers:
            if speaker.startswith('p'):
                # VCTK speakers are typically p225-p376
                # Even numbers are usually female, odd numbers are male
                try:
                    speaker_num = int(speaker[1:])
                    if speaker_num % 2 == 0:
                        female_speakers.append(speaker)
                    else:
                        male_speakers.append(speaker)
                except ValueError:
                    # If not a number, just add to a general list
                    pass
        
        print(f"👨 Male Speakers ({len(male_speakers)}):")
        for i, speaker in enumerate(sorted(male_speakers)):
            print(f"   {speaker}", end="  ")
            if (i + 1) % 8 == 0:  # 8 speakers per line
                print()
        print("\n")
        
        print(f"👩 Female Speakers ({len(female_speakers)}):")
        for i, speaker in enumerate(sorted(female_speakers)):
            print(f"   {speaker}", end="  ")
            if (i + 1) % 8 == 0:  # 8 speakers per line
                print()
        print("\n")
        
        print("💡 Popular speaker choices:")
        print("   p225 (male) - Clear, professional")
        print("   p226 (female) - Warm, friendly")
        print("   p227 (male) - Deep, authoritative")
        print("   p228 (female) - Bright, energetic")
        print("   p229 (male) - Calm, measured")
        print("   p230 (female) - Soft, gentle")
        
        return speakers
        
    except Exception as e:
        print(f"❌ Error loading VCTK model: {e}")
        return []

if __name__ == "__main__":
    list_vctk_speakers() 