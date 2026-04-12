import math

def grade(score):
    # Sigmoid: mathematically impossible to return exactly 0.0 or 1.0
    return 1.0 / (1.0 + math.exp(-score / 10.0))