def grade(score):
    # Normalize score to 0–1 range
    return max(0.0, min(1.0, (score + 200) / 400))