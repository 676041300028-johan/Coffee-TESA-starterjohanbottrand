import numpy as np
from typing import List
from coffee_math import CoffeeLot, BuyerRoute

def is_lot_eligible_for_route(
    lot: CoffeeLot,
    route: BuyerRoute,
    error_buffer: float = 0.005
) -> bool:
    """
    ตรวจสอบว่า Lot ผ่านสเปกของ Route หรือไม่ โดยเผื่อค่าคลาดเคลื่อน +/- 0.5% ของเซิร์ฟเวอร์
    - g_true อาจสูงกว่าที่เห็น -> เช็ก lot.g * (1 + buffer) <= max_defect
    - m_true อาจสูงกว่าที่เห็น -> เช็ก lot.m * (1 + buffer) <= max_moisture
    - s_true อาจต่ำกว่าที่เห็น -> เช็ก lot.s * (1 - buffer) >= min_screen
    """
    if lot.bags <= 0 or route.quota <= 0:
        return False

    worst_g = lot.g * (1.0 + error_buffer)
    worst_m = lot.m * (1.0 + error_buffer)
    worst_s = lot.s * (1.0 - error_buffer)

    if worst_g > route.max_defect:
        return False
    if worst_m > route.max_moisture:
        return False
    if route.min_screen > 0.0 and worst_s < route.min_screen:
        return False

    return True

def get_valid_route_mask(
    lots: List[CoffeeLot],
    routes: List[BuyerRoute],
    error_buffer: float = 0.005
) -> np.ndarray:
    """สร้าง Action Mask ขนาด (len(lots), len(routes))"""
    mask = np.zeros((len(lots), len(routes)), dtype=bool)
    for i, lot in enumerate(lots):
        for j, route in enumerate(routes):
            mask[i, j] = is_lot_eligible_for_route(lot, route, error_buffer)
    return mask