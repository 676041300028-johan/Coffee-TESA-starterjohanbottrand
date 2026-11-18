from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CoffeeLot:
    lot_id: str
    bags: int
    s: float  # Screen size (13 - 18)
    m: float  # Moisture (%)
    g: float  # Defect (%)
    tv: float = 0.0

    def __post_init__(self):
        if self.tv <= 0.0:
            self.tv = calculate_tv(self.s, self.m, self.g)

@dataclass
class BuyerRoute:
    route_id: str
    company_name: str
    buyer_type: str          # "SPECIALTY", "COMMERCIAL", "INSTANT"
    quota: int               # จำนวนกระสอบสูงสุดที่รับซื้อ
    max_defect: float        # g <= max_defect
    max_moisture: float      # m <= max_moisture
    min_screen: float = 0.0  # เพิ่มเงื่อนไขขนาดเมล็ดขั้นต่ำ (Specialty ใช้ 16 หรือ 17)
    base_ceiling_mult: float = 1.00
    min_bags_per_bid: int = 1

def calculate_tv(s: float, m: float, g: float) -> float:
    """คำนวณ True Value (TV) = 120 * G * M * S"""
    # Defect factor G
    if g <= 1.0:
        g_fac = 1.05
    elif g <= 3.0:
        g_fac = 1.00
    elif g <= 5.0:
        g_fac = 0.95
    elif g <= 8.0:
        g_fac = 0.88
    else:
        g_fac = 0.80

    # Moisture factor M
    if m <= 11.5:
        m_fac = 1.03
    elif m <= 12.5:
        m_fac = 1.00
    elif m <= 13.5:
        m_fac = 0.94
    else:
        m_fac = 0.85

    # Screen size factor S
    if s >= 18:
        s_fac = 1.08
    elif s >= 17:
        s_fac = 1.04
    elif s >= 16:
        s_fac = 1.00
    elif s >= 15:
        s_fac = 0.96
    else:
        s_fac = 0.90

    return round(120.0 * g_fac * m_fac * s_fac, 2)