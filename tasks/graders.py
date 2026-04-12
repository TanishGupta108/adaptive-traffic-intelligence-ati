def grade(score):
    # Normalize score to (0, 1) range — strictly exclusive of 0 and 1
    normalized = (score + 200) / 400
    return max(1e-6, min(1 - 1e-6, normalized))