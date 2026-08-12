import sys
import json
from whistress_service import service

def test_file(filepath):
    print(f"==================================================")
    print(f"TESTING: {filepath}")
    print(f"==================================================")
    
    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    # predict_with_intonation now returns 10 values
    corrected_words, pauses, phrases, pitch_times, pitch_f0, pitch_stylized, k_array, P_val, audio_array, sr = service.predict_with_intonation(audio_bytes)
    
    # We need the original words to compare against. The easiest way is to run the whisper inference directly,
    # or just look at `service.client.predict(..., return_pairs=True)` since the audio is processed.
    audio_dict = service.process_audio(audio_bytes)
    original_words = service.client.predict(audio=audio_dict, transcription=None, return_pairs=True)

    print("\n--- WORD BOUNDARY COMPARISON ---")
    print(f"{'Word':<15} | {'Orig Start':<10} | {'Orig End':<10} | {'Corr Start':<10} | {'Corr End':<10} | {'Change Reason'}")
    print("-" * 80)
    for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
        word, stress, o_start, o_end = orig
        c_word, c_stress, c_start, c_end = corr
        
        reason = []
        if round(float(o_start), 3) != c_start:
            reason.append("Start modified")
        if round(float(o_end), 3) != c_end:
            reason.append("End modified")
        reason_str = ", ".join(reason) if reason else "None"
        
        print(f"{word:<15} | {float(o_start):<10.3f} | {float(o_end):<10.3f} | {c_start:<10.3f} | {c_end:<10.3f} | {reason_str}")

    print("\n--- ACOUSTIC PAUSES ---")
    print(f"{'Pause Start':<12} | {'Pause End':<10} | {'Duration':<10} | {'Pause Type':<20} | {'Used For Realignment'}")
    print("-" * 80)
    for i, pause in enumerate(pauses):
        if pause is None:
            continue
        
        duration = pause["duration"]
        ptype = pause.get("type", "unknown")
        
        used = []
        # The pause array starts from word index 1.
        # pause[i] corresponds to the pause before word (i+1).
        word_idx = i + 1
        
        used = []
        # Let's check if the pause caused a change in the neighboring words
        if word_idx < len(original_words):
            prev_orig_end = round(float(original_words[word_idx-1][3]), 3)
            prev_corr_end = corrected_words[word_idx-1][3]
            curr_orig_start = round(float(original_words[word_idx][2]), 3)
            curr_corr_start = corrected_words[word_idx][2]
            
            if prev_orig_end != prev_corr_end:
                used.append("prev.end")
            if curr_orig_start != curr_corr_start:
                used.append("curr.start")
                
        if duration < 0.45:
            reason = "Ignored (duration < 0.45s)"
        elif not used:
            reason = "Ignored (discrepancy too small or not absorbed)"
        else:
            reason = "REALIGNED: " + ", ".join(used)
            
        print(f"{pause['start']:<12.3f} | {pause['end']:<10.3f} | {duration:<10.3f} | {ptype:<20} | {reason}")
    print("\n")

if __name__ == "__main__":
    service.load_model()
    test_file("example1_football.wav")
    test_file("example4.wav")
    test_file("test_two_phrases.wav")
    test_file("test_pause.mp3")
