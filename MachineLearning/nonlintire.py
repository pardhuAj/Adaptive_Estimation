import numpy as np

def nonlintire(alpha, Fz, vwx):
    """
    Nonlinear tire model based on Pacejka's Advanced Magic Formula.
    
    Parameters:
    alpha : float
        Slip angle in radians.
    Fz : float
        Vertical force in Newton.
    vwx : float
        Longitudinal velocity of the wheel center in m/sec.
    
    Returns:
    Fy : float
        Lateral force in Newton.
    """
    # Magic Formula Coefficients
    pcy1 = 1.35
    pdy1 = -0.990
    pdy2 = 0.25
    pey1 = -1.003
    pey2 = -0.537
    pey3 = -0.083
    pky1 = -14.95
    pky2 = 2.130
    pky4 = 2
    phy1 = 0.003
    phy2 = -0.001
    pvy1 = 0.045
    pvy2 = -0.024
    
    # Vertical Load Deviation
    Fz0 = 4000
    dfz = (Fz - Fz0) / Fz0
    
    # Adapted Slip for Large Angles
    alpha_star = np.tan(alpha) * np.sign(vwx)
    
    # Vertical and Horizontal Shifts
    Svy = Fz * (pvy1 + pvy2 * dfz)
    Shy = phy1 + phy2 * dfz
    
    # Magic Formula Parameter
    x = alpha_star + Shy
    mu = (pdy1 + pdy2 * dfz)
    
    C = pcy1
    D = mu * Fz
    E = (pey1 + pey2 * dfz) * (1 - pey3 * np.sign(x))
    BCD = pky1 * Fz0 * np.sin(pky4 * np.arctan(Fz / (pky2 * Fz0)))
    B = BCD / (C * D)
    
    # Lateral Force
    y = D * np.sin(C * np.arctan(B * x - E * (B * x - np.arctan(B * x))))
    Fy = y + Svy
    
    return Fy
