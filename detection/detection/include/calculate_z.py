import math

class calculate_z:

    @staticmethod
    def calculate_h(Y, Z):
        return math.acos(abs(Y)/Z)

    @staticmethod
    def calculate_true_z(Y, Z):
        temp = math.asin(abs(Y)/Z)
        r20 = math.radians(20)

        if Y < 0.0 and temp == r20:
            return Z
        elif Y < 0.0 and temp > r20:
            h = calculate_z.calculate_h(Y, Z) + r20
        elif Y < 0.0:
            h = math.radians(180) - r20 - calculate_z.calculate_h(Y, Z)
        elif Y > 0.0:
            h = calculate_z.calculate_h(Y, Z) - r20
        elif Y == 0.0:
            h = math.radians(70)
        
        return math.sin(h) * Z



