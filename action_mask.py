from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from coffee_math import CoffeeLot, BUYER_SPECS


@dataclass
class BuyerOrder:
    order_id: str
    buyer_type: str
    quota_bags: int
    company_name: str = ""
    max_moisture: Optional[float] = None
    max_defect: Optional[float] = None
    max_ratio: Optional[float] = None
    hidden_ceiling_ratio: Optional[float] = None

    def __post_init__(self):
        b_type = self.buyer_type.strip().upper()
        self.buyer_type = "COMMERCIAL" if "COMM" in b_type else "INSTANT"
        if not self.company_name:
            self.company_name = self.order_id

        spec = BUYER_SPECS[self.buyer_type]
        if self.max_moisture is None:
            self.max_moisture = spec["max_moisture"]
        if self.max_defect is None:
            self.max_defect = spec["max_defect"]
        if self.max_ratio is None:
            self.max_ratio = spec["ratio_mid"]


def build_action_mask(
    lots: List[CoffeeLot],
    routes: List[BuyerOrder],
    error_buffer: float = 0.005,
) -> np.ndarray:
    """
    สร้าง Boolean Matrix (4 Lots x 8 Routes)
    เผื่อ error_buffer = 0.005 (0.5%) ป้องกันกรณีค่าจริงของ s, m, g คลาดเคลื่อนจนข้ามเกรดและโดนตีตก
    """
    num_lots = len(lots)
    num_routes = len(routes)
    mask = np.zeros((num_lots, num_routes), dtype=bool)

    for i, lot in enumerate(lots):
        if lot.bags <= 0:
            continue

        # คำนวณค่ากรณีเลวร้ายที่สุด (+0.5% error)
        worst_m = lot.m * (1.0 + error_buffer)
        worst_g = lot.g * (1.0 + error_buffer)

        for j, order in enumerate(routes):
            if order.quota_bags <= 0:
                continue

            if worst_m > order.max_moisture or worst_g > order.max_defect:
                continue

            mask[i, j] = True

    return mask