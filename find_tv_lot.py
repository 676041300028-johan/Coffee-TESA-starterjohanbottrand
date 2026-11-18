from dataclasses import dataclass, field
from typing import Dict, List, Optional

GRADE_TABLE = [
    (1.0, "PREMIUM", 1.08),
    (3.0, "GRADE_1", 1.04),
    (5.0, "GRADE_2", 1.00),
    (7.0, "GRADE_3", 0.96),
    (15.0, "BELOW_STANDARD", 0.85),
    (float("inf"), "REJECT", 0.60),
]

def evaluate_coffee_lot(
    moisture: float, defect: float, screen: int, p0: float = 120.0
) -> Dict:
    """คำนวณ Grade, G, M, S, TV และช่วงราคาแนะนำสำหรับยื่นประมูล"""
    grade, grade_mult = "REJECT", 0.60
    for max_d, g_name, g_mult in GRADE_TABLE:
        if defect <= max_d:
            grade = g_name
            grade_mult = g_mult
            break
    G = round((1.0 - defect / 100.0) * grade_mult, 4)

    if moisture <= 12.5:
        M = 1.0000
    elif moisture <= 15.0:
        M = round((1.0 - moisture / 100.0) / 0.875 - 0.02 * (moisture - 12.5), 4)
    else:
        M = 0.6000

    if screen <= 12:
        S = 0.92
    elif screen == 13:
        S = 0.97
    elif screen in (14, 15):
        S = 1.00
    elif screen in (16, 17):
        S = 1.03
    else:
        S = 1.06

    TV = round(p0 * G * M * S, 2)
    
    return {
        "grade": grade,
        "G": G,
        "M": M,
        "S": S,
        "TV": TV,
        "floor_0.80": round(0.80 * TV, 2),      # ราคาขั้นต่ำ (Ratio = 0.80)
        "instant_max_0.98": round(0.98 * TV, 2), # เพดานกลุ่ม Instant (Ratio ~ 0.98)
        "comm_max_1.03": round(1.03 * TV, 2),    # เพดานกลุ่ม Commercial (Ratio ~ 1.03)
    }

@dataclass
class CoffeeLot:
    lot_id: str
    screen: int     # s (Size)
    moisture: float # m (Moisture)
    defect: float   # g (Defect)
    bags: int
    p0: float = 120.0
    
    grade: str = field(init=False)
    G: float = field(init=False)
    M: float = field(init=False)
    S: float = field(init=False)
    TV: float = field(init=False)
    floor_price: float = field(init=False)
    instant_ceiling: float = field(init=False)
    comm_ceiling: float = field(init=False)

    def __post_init__(self):
        res = evaluate_coffee_lot(self.moisture, self.defect, self.screen, self.p0)
        self.grade = res["grade"]
        self.G = res["G"]
        self.M = res["M"]
        self.S = res["S"]
        self.TV = res["TV"]
        self.floor_price = res["floor_0.80"]
        self.instant_ceiling = res["instant_max_0.98"]
        self.comm_ceiling = res["comm_max_1.03"]

def analyze_4_lots(lots: List[CoffeeLot], target_ratio: float = 0.95):
    """ประเมินทั้ง 4 Lots และคำนวณราคาเสนอขาย (Bid) ที่ Ratio เป้าหมายเดียวกัน"""
    print(f"{'Lot ID':<10} | {'เกรด':<14} | {'S / M / D':<15} | {'TV (บ./กก.)':<11} | {'ขั้นต่ำ (0.8)':<12} | {'เพดาน Inst':<10} | {'เพดาน Comm':<10} | {'Bid @ ' + str(target_ratio):<10}")
    print("-" * 112)
    
    total_bags = 0
    weighted_tv = 0.0
    
    for lot in lots:
        suggested_bid = round(lot.TV * target_ratio, 2)
        spec_str = f"{lot.screen}/{lot.moisture:.1f}/{lot.defect:.1f}"
        print(
            f"{lot.lot_id:<10} | {lot.grade:<14} | {spec_str:<15} | "
            f"{lot.TV:<11.2f} | {lot.floor_price:<12.2f} | "
            f"{lot.instant_ceiling:<10.2f} | {lot.comm_ceiling:<10.2f} | "
            f"{suggested_bid:<10.2f}"
        )
        total_bags += lot.bags
        weighted_tv += lot.TV * lot.bags

    print("-" * 112)
    print(f"รวมทั้งหมด: {total_bags} กระสอบ | TV เฉลี่ยถ่วงน้ำหนัก: {weighted_tv / total_bags:.2f} บาท/กก.")

# ==========================================
# ทดสอบคำนวณทั้ง 4 Lots พร้อมกัน
# ==========================================
if __name__ == "__main__":
    my_4_lots = [
        CoffeeLot("LOT-1", screen=16, moisture=11.5, defect=0.8, bags=10), # เกรด Premium
        CoffeeLot("LOT-2", screen=14, moisture=12.4, defect=2.5, bags=15), # เกรด 1
        CoffeeLot("LOT-3", screen=14, moisture=13.2, defect=4.2, bags=20), # เกรด 2 (ชื้นเกิน 12.5)
        CoffeeLot("LOT-4", screen=13, moisture=14.0, defect=6.5, bags=15), # เกรด 3
    ]
    
    analyze_4_lots(my_4_lots, target_ratio=0.94)