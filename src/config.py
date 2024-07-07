
class CALCULATE_CONFIG:
    def __init__(self):
        self.OUTSIDE_CONCENTRATION = 400
        self.OUTSIDE_PRESSURE_RANGE = (1, 10)
        self.OUTSIDE_NODE_NUM = (2, 5)

        self.NODE_NUM_RANGE = (4, 10)
        self.NODE_SIZE_RANGE = (25, 60)
        self.NODE_PEOPLE_RANGE = (0, 10)

        self.CONNECTION_NUM_TIMES = (1.5, 2)
        self.COEFFICIENT_RANGE = (0.2, 0.8)

        self.PEOPLE_CHANGE_TIMES = 3

        self.TIME_OF_EACH = 200
        self.TIME_STEP = 5

        self.HUMAN_EXHALATION = 0.0001
        self.HUMAN_EXHALATION_CONCENTRATION = 40000
        self.HUMAN_EXHALATION_FLOW = self.HUMAN_EXHALATION * self.HUMAN_EXHALATION_CONCENTRATION