from sympy import Matrix, atan2, acos, cos, sin, pi, log, sqrt, deg

d1 = 70
a2 = 130
a3 = 130
a4 = 85
d5 = 135


def inverse_kinematics(position, z=None):
    if z is None:
        if len(position) != 3:
            raise ValueError("Position must contain 3 elements [x, y, z]")
        x, y, z = position
    else:
        if len(position) != 2:
            raise ValueError("If z is provided separately, position must contain only [x, y]")
        x, y = position

    w4, w5, w6 = 0, 0, -1

    q = Matrix([0, 0, 0, 0, 0, 0])

    q[1] = atan2(y, x)

    q234 = atan2(-w4 * cos(q[1]) - w5 * sin(q[1]), -w6)

    b1 = x * cos(q[1]) + y * sin(q[1]) - a4 * cos(q234) + d5 * sin(q234)
    b2 = d1 - a4 * sin(q234) - d5 * cos(q234) - z
    bb = b1 ** 2 + b2 ** 2

    q[3] = acos((bb - a2 ** 2 - a3 ** 2) / (2 * a2 * a3))

    q[2] = atan2((a2 + a3 * cos(q[3])) * b2 - a3 * sin(q[3]) * b1,
                 (a2 + a3 * cos(q[3])) * b1 + a3 * sin(q[3]) * b2)

    q[4] = q234 - q[2] - q[3]

    q[5] = q[1] + pi * log(sqrt(w4 ** 2 + w5 ** 2 + w6 ** 2))
    q[5] = q[5] % pi

    # Apply offset corrections
    q[1] = q[1]
    q[2] = q[2] + 2 * pi / 2
    q[3] = q[3] + 1 * pi / 2
    q[4] = q[4] + 1 * pi / 2
    q[5] = q[5]


    q_deg = []
    for i in range(1, 6):
        try:
            angle_rad = q[i]
            angle_deg = float(deg(angle_rad))
            q_deg.append(angle_deg)
        except (TypeError, ValueError) as e:
            print(f"Warning: Cannot convert joint {i} angle to float: {e}")
            print(f"Position may be unreachable or invalid")
            return None
    
    for idx, angle in enumerate(q_deg):
        if not (0 <= angle <= 180):
            print(f"Warning: Joint {idx + 1} angle {angle:.2f}° is out of range (0-180°)")
            return None

    return q_deg
