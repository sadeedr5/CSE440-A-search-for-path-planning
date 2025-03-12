def calculate_control_inputs(current_pos, next_pos, current_theta):
    """Calculate control inputs (v, omega) to reach next position"""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    # Calculate desired heading
    desired_theta = math.atan2(dy, dx)
    
    # Calculate angle difference
    angle_diff = desired_theta - current_theta
    # Normalize angle to [-pi, pi]
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    
    # Simple proportional control
    Kp_omega = 1.0  # Angular velocity gain
    Kp_v = 2    # Linear velocity gain
    
    # Calculate control inputs
    omega = Kp_omega * angle_diff
    
    # Only move forward if roughly pointing in right direction
    if abs(angle_diff) < 0.5:
        v = Kp_v * math.sqrt(dx**2 + dy**2)
        v = min(v, 0.5)  # Limit maximum velocity
    else:
        v = 0.0
    
    return v, omega
