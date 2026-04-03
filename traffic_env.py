import random

class TrafficEnv:
    def __init__(self):
        self.max_steps = 50
        self.reset()

    def reset(self):
        # Cars in lanes: [North, South, East, West]
        self.cars = [random.randint(0, 10) for _ in range(4)]
        
        # 0 = NS green, 1 = EW green
        self.signal = 0  

        # Emergency setup
        self.emergency_lane = random.choice([0, 1, 2, 3])
        self.emergency_active = True

        self.steps = 0
        self.time = 0

        return self._get_state()

    def _get_state(self):
        return tuple(self.cars + [self.signal, self.emergency_lane, int(self.emergency_active)])

    def step(self, action):
        reward = 0
        self.steps += 1
        self.time += 1

        # Action: 0 = keep NS, 1 = switch to EW
        if action == 1:
            self.signal = 1 - self.signal

        passed = 0

        # Traffic movement
        if self.signal == 0:  # NS green
            for i in [0, 1]:
                moved = min(self.cars[i], 2)
                self.cars[i] -= moved
                passed += moved
        else:  # EW green
            for i in [2, 3]:
                moved = min(self.cars[i], 2)
                self.cars[i] -= moved
                passed += moved

        # 🚦 Rush hour logic
        if 10 <= self.time % 50 <= 30:
            arrival_rate = 3
        else:
            arrival_rate = 1

        # Direction bias (more NS traffic)
        self.cars[0] += random.randint(0, arrival_rate)
        self.cars[1] += random.randint(0, arrival_rate)
        self.cars[2] += random.randint(0, max(1, arrival_rate - 1))
        self.cars[3] += random.randint(0, max(1, arrival_rate - 1))

        # 🚑 Emergency handling
        if self.emergency_active:
            if self.cars[self.emergency_lane] == 0:
                self.emergency_active = False

        if not self.emergency_active and random.random() < 0.1:
            self.emergency_lane = random.choice([0, 1, 2, 3])
            self.emergency_active = True

        # 🎯 Reward system
        reward -= sum(self.cars)              # congestion penalty
        reward += passed * 2                 # smooth flow reward

        # Emergency priority reward
        if self.emergency_active:
            if (self.signal == 0 and self.emergency_lane in [0, 1]) or \
               (self.signal == 1 and self.emergency_lane in [2, 3]):
                reward += 15
            else:
                reward -= 25

        done = self.steps >= self.max_steps

        return self._get_state(), reward, done
    