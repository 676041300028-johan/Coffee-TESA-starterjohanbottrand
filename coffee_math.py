from dataclasses import dataclass

GRADE_TABLE = [
    (1.0, "PREMIUM", 1.08),
    (3.0, "GRADE_1", 1.04),
    (5.0, "GRADE_2", 1.00),
    (7.0, "GRADE_3", 0.96),
    (15.0, "BELOW_STANDARD", 0.85),
    (float("inf"), "REJECT", 0.60),
]

@dataclass
class CoffeeLot:
    lot_id: str
    bags: int
    s: float  # Screen size (12 - 18)
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
    min_screen: float = 0.0  # เงื่อนไขขนาดเมล็ดขั้นต่ำ (Specialty ใช้ 16 หรือ 17)
    base_ceiling_mult: float = 1.00
    min_bags_per_bid: int = 1

def calculate_tv(s: float, m: float, g: float, p0: float = 120.0) -> float:
    """คำนวณ True Value (TV) = p0 * G * M * S ตามกติกาใหม่ล่าสุด"""
    grade_mult = 0.60
    for max_d, _, g_mult in GRADE_TABLE:
        if g <= max_d:
            grade_mult = g_mult
            break
    G = round((1.0 - g / 100.0) * grade_mult, 4)

    if m <= 12.5:
        M = 1.0000
    elif m <= 15.0:
        M = round((1.0 - m / 100.0) / 0.875 - 0.02 * (m - 12.5), 4)
    else:
        M = 0.6000

    s_int = int(round(s))
    if s_int <= 12:
        S = 0.92
    elif s_int == 13:
        S = 0.97
    elif s_int in (14, 15):
        S = 1.00
    elif s_int in (16, 17):
        S = 1.03
    else:
        S = 1.06

    return round(p0 * G * M * S, 2)
