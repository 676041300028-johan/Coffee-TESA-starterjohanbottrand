from dataclasses import dataclass, field
from typing import Dict, List

# ตารางเกรดตามกติกา
GRADE_TABLE = [
    (1.0, "PREMIUM", 1.08),
    (3.0, "GRADE_1", 1.04),
    (5.0, "GRADE_2", 1.00),
    (7.0, "GRADE_3", 0.96),
    (15.0, "BELOW_STANDARD", 0.85),
    (float("inf"), "REJECT", 0.60),
]

# เกณฑ์ผู้ซื้อ (Gate Check + เพดาน Ratio)
BUYER_SPECS = {
    "COMMERCIAL": {
        "max_defect": 7.0,
        "max_moisture": 12.5,
        "ratio_min": 1.025,
        "ratio_mid": 1.030,
        "ratio_max": 1.035,
    },
    "INSTANT": {
        "max_defect": 15.0,
        "max_moisture": 14.0,
        "ratio_min": 0.975,
        "ratio_mid": 0.980,
        "ratio_max": 0.985,
    },
}


def evaluate_coffee_lot(
    s: int, m: float, g: float, p0: float = 120.0
) -> Dict:
    """คำนวณ Grade, ตัวคูณ G/M/S, TV, ราคาขั้นต่ำ 0.8*TV และสิทธิ์การขาย"""
    # 1. ค่า G (Defect / Grade)
    grade, grade_mult = "REJECT", 0.60
    for max_d, g_name, g_mult in GRADE_TABLE:
        if g <= max_d:
            grade = g_name
            grade_mult = g_mult
            break
    G = round((1.0 - g / 100.0) * grade_mult, 4)

    # 2. ค่า M (Moisture)
    if m <= 12.5:
        M = 1.0000
    elif m <= 15.0:
        M = round((1.0 - m / 100.0) / 0.875 - 0.02 * (m - 12.5), 4)
    else:
        M = 0.6000

    # 3. ค่า S (Screen Size)
    if s <= 12:
        S = 0.92
    elif s == 13:
        S = 0.97
    elif s in (14, 15):
        S = 1.00
    elif s in (16, 17):
        S = 1.03
    else:
        S = 1.06

    # 4. คำนวณ TV และราคาขั้นต่ำรอบ 7 (0.8 * TV)
    TV = round(p0 * G * M * S, 2)
    floor_price = round(0.80 * TV, 2)

    # 5. ตรวจสอบ Gate Check ของ Commercial และ Instant
    can_sell_comm = (
        g <= BUYER_SPECS["COMMERCIAL"]["max_defect"]
        and m <= BUYER_SPECS["COMMERCIAL"]["max_moisture"]
    )
    can_sell_inst = (
        g <= BUYER_SPECS["INSTANT"]["max_defect"]
        and m <= BUYER_SPECS["INSTANT"]["max_moisture"]
    )

    return {
        "grade": grade,
        "G": G,
        "M": M,
        "S": S,
        "TV": TV,
        "floor_0.8TV": floor_price,
        "can_sell_comm": can_sell_comm,
        "can_sell_inst": can_sell_inst,
        "comm_safe_max": round(TV * BUYER_SPECS["COMMERCIAL"]["ratio_min"], 2) if can_sell_comm else None,
        "comm_mid_max": round(TV * BUYER_SPECS["COMMERCIAL"]["ratio_mid"], 2) if can_sell_comm else None,
        "inst_safe_max": round(TV * BUYER_SPECS["INSTANT"]["ratio_min"], 2) if can_sell_inst else None,
        "inst_mid_max": round(TV * BUYER_SPECS["INSTANT"]["ratio_mid"], 2) if can_sell_inst else None,
    }


@dataclass
class CoffeeLot:
    lot_id: str
    s: int        # Size (Screen)
    m: float      # Moisture (%)
    g: float      # Defect (%)
    bags: int
    p0: float = 120.0

    grade: str = field(init=False)
    G: float = field(init=False)
    M: float = field(init=False)
    S: float = field(init=False)
    TV: float = field(init=False)
    floor_price: float = field(init=False)
    can_sell_comm: bool = field(init=False)
    can_sell_inst: bool = field(init=False)
    comm_safe_max: float = field(init=False)
    inst_safe_max: float = field(init=False)

    def __post_init__(self):
        res = evaluate_coffee_lot(self.s, self.m, self.g, self.p0)
        self.grade = res["grade"]
        self.G = res["G"]
        self.M = res["M"]
        self.S = res["S"]
        self.TV = res["TV"]
        self.floor_price = res["floor_0.8TV"]
        self.can_sell_comm = res["can_sell_comm"]
        self.can_sell_inst = res["can_sell_inst"]
        self.comm_safe_max = res["comm_safe_max"]
        self.inst_safe_max = res["inst_safe_max"]


def run_4_lot_calculator(lots: List[CoffeeLot], comm_demand: int, inst_demand: int):
    """แสดงตารางสรุปทั้ง 4 Lots พร้อมวิเคราะห์ Supply เทียบ Demand ของตลาด"""
    print("=" * 118)
    print(f"{'LOT':<8} | {'เกรด':<14} | {'s / m / g':<14} | {'กระสอบ':<6} | {'TV':<8} | {'0.8xTV':<8} | {'Commercial (≤1.025)':<20} | {'Instant (≤0.975)':<18}")
    print("-" * 118)

    comm_supply = 0
    inst_only_supply = 0
    unqualified_supply = 0

    for lot in lots:
        spec = f"{lot.s} / {lot.m:.1f}% / {lot.g:.1f}%"
        comm_str = f"ผ่าน (เพดาน {lot.comm_safe_max:.2f})" if lot.can_sell_comm else "❌ ไม่ผ่านเกณฑ์"
        inst_str = f"ผ่าน (เพดาน {lot.inst_safe_max:.2f})" if lot.can_sell_inst else "❌ ไม่ผ่านเกณฑ์"

        print(
            f"{lot.lot_id:<8} | {lot.grade:<14} | {spec:<14} | {lot.bags:<6} | "
            f"{lot.TV:<8.2f} | {lot.floor_price:<8.2f} | {comm_str:<20} | {inst_str:<18}"
        )

        if lot.can_sell_comm:
            comm_supply += lot.bags
        elif lot.can_sell_inst:
            inst_only_supply += lot.bags
        else:
            unqualified_supply += lot.bags

    print("=" * 118)
    print(f"สรุปโควตากระสอบของทีมเรา (รวม {sum(l.bags for l in lots)} กระสอบ):")
    print(f"  • เกรดสูง (ขายได้ทั้ง Commercial และ Instant): {comm_supply} กระสอบ  [ตลาด Commercial ต้องการ {comm_demand} กระสอบ]")
    print(f"  • เกรดกลาง (ขายได้เฉพาะ Instant):            {inst_only_supply} กระสอบ  [ตลาด Instant ต้องการ {inst_demand} กระสอบ]")
    print(f"  • ตกเกณฑ์ (ขายไม่ได้ในตลาดปกติ):           {unqualified_supply} กระสอบ")
    print("=" * 118)


# ==========================================
# ตัวอย่างการรันจริง (รับค่า s, m, g ของทั้ง 4 Lots)
# ==========================================
if __name__ == "__main__":
    lots_input = [
        CoffeeLot("LOT-1", s=16, m=11.8, g=0.9, bags=6),   # ผ่านทั้ง Comm และ Inst
        CoffeeLot("LOT-2", s=14, m=12.5, g=4.5, bags=8),   # ผ่านทั้ง Comm และ Inst (ชื้น 12.5 พอดี)
        CoffeeLot("LOT-3", s=15, m=13.2, g=3.2, bags=7),   # ตก Comm เพราะชื้น > 12.5% -> ขายได้แค่ Inst
        CoffeeLot("LOT-4", s=13, m=13.9, g=11.5, bags=6),  # ตก Comm เพราะชื้นและเสีย -> ขายได้แค่ Inst
    ]

    # สมมติรอบนี้ตลาดมีความต้องการ Commercial (x) = 35 กระสอบ, Instant (n) = 45 กระสอบ
    run_4_lot_calculator(lots_input, comm_demand=35, inst_demand=45)