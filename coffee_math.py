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
    s: float, m: float, g: float, p0: float = 120.0, error_buffer: float = 0.005
) -> Dict:
    """
    คำนวณ Grade, ตัวคูณ G/M/S, ราคาประเมิน (TV)
    และตรวจสอบ Gate Check โดยเผื่อ error_buffer (เช่น 0.005 = 0.5%) ป้องกันการโดนตีตก
    """
    grade, grade_mult = "REJECT", 0.60
    for max_d, g_name, g_mult in GRADE_TABLE:
        if g <= max_d:
            grade = g_name
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

    TV = round(p0 * G * M * S, 2)
    floor_price = round(0.80 * TV, 2)

    # เผื่อค่าคลาดเคลื่อน +0.5% (Worst-case check) เพื่อป้องกันส่งผิดเกรด
    worst_g = g * (1.0 + error_buffer)
    worst_m = m * (1.0 + error_buffer)

    can_sell_comm = (
        worst_g <= BUYER_SPECS["COMMERCIAL"]["max_defect"]
        and worst_m <= BUYER_SPECS["COMMERCIAL"]["max_moisture"]
    )
    can_sell_inst = (
        worst_g <= BUYER_SPECS["INSTANT"]["max_defect"]
        and worst_m <= BUYER_SPECS["INSTANT"]["max_moisture"]
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
        "comm_max": round(TV * BUYER_SPECS["COMMERCIAL"]["ratio_max"], 2) if can_sell_comm else None,
        "inst_safe_max": round(TV * BUYER_SPECS["INSTANT"]["ratio_min"], 2) if can_sell_inst else None,
        "inst_mid_max": round(TV * BUYER_SPECS["INSTANT"]["ratio_mid"], 2) if can_sell_inst else None,
        "inst_max": round(TV * BUYER_SPECS["INSTANT"]["ratio_max"], 2) if can_sell_inst else None,
    }


@dataclass
class CoffeeLot:
    lot_id: str
    s: float        # ค่า Screen Size ที่บอทมองเห็น (อาจมี error ±0.5%)
    m: float        # ค่า Moisture ที่บอทมองเห็น (อาจมี error ±0.5%)
    g: float        # ค่า Defect ที่บอทมองเห็น (อาจมี error ±0.5%)
    bags: int
    p0: float = 120.0
    # ค่าจริงฝั่งเซิร์ฟเวอร์ (หากไม่ระบุ จะใช้ค่าเดียวกับที่มองเห็น)
    s_true: Optional[float] = None
    m_true: Optional[float] = None
    g_true: Optional[float] = None

    grade: str = field(init=False)
    G: float = field(init=False)
    M: float = field(init=False)
    S: float = field(init=False)
    TV: float = field(init=False)
    TV_true: float = field(init=False)
    grade_true: str = field(init=False)
    floor_price: float = field(init=False)
    can_sell_comm: bool = field(init=False)
    can_sell_inst: bool = field(init=False)
    comm_safe_max: Optional[float] = field(init=False)
    comm_mid_max: Optional[float] = field(init=False)
    inst_safe_max: Optional[float] = field(init=False)
    inst_mid_max: Optional[float] = field(init=False)

    def __post_init__(self):
        if self.s_true is None:
            self.s_true = self.s
        if self.m_true is None:
            self.m_true = self.m
        if self.g_true is None:
            self.g_true = self.g

        # คำนวณจากค่าที่บอทมองเห็น (เผื่อ buffer 0.5%)
        res_obs = evaluate_coffee_lot(self.s, self.m, self.g, self.p0, error_buffer=0.005)
        self.grade = res_obs["grade"]
        self.G = res_obs["G"]
        self.M = res_obs["M"]
        self.S = res_obs["S"]
        self.TV = res_obs["TV"]
        self.floor_price = res_obs["floor_0.8TV"]
        self.can_sell_comm = res_obs["can_sell_comm"]
        self.can_sell_inst = res_obs["can_sell_inst"]
        self.comm_safe_max = res_obs["comm_safe_max"]
        self.comm_mid_max = res_obs["comm_mid_max"]
        self.inst_safe_max = res_obs["inst_safe_max"]
        self.inst_mid_max = res_obs["inst_mid_max"]

        # คำนวณค่า TV จริงฝั่งเซิร์ฟเวอร์ (ไม่ใส่ buffer)
        res_true = evaluate_coffee_lot(self.s_true, self.m_true, self.g_true, self.p0, error_buffer=0.0)
        self.TV_true = res_true["TV"]
        self.grade_true = res_true["grade"]

    @property
    def screen(self) -> float:
        return self.s

    @property
    def moisture(self) -> float:
        return self.m

    @property
    def defect(self) -> float:
        return self.g