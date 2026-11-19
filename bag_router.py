from typing import List, Dict
import numpy as np
from coffee_math import CoffeeLot, BuyerRoute
from action_mask import get_valid_route_mask

class BagCombinatorialRouter:
    def __init__(self, error_buffer: float = 0.005):
        self.error_buffer = error_buffer

    def route_bags(
        self,
        lots: List[CoffeeLot],
        routes: List[BuyerRoute],
        target_sell_bags: int,
        price_multiplier: float,
        estimated_ceilings: Dict[str, float],
        inst_price_multiplier: float = None
    ) -> List[Dict]:
        """
        จัดสรรกระสอบลงแต่ละ Route อย่างชาญฉลาด:
        1. Priority #1: จับคู่ Lot เกรดพรีเมียมเข้า SPECIALTY ก่อนเสมอ
        2. Priority #2: กระจายกระสอบที่เหลือเข้า COMMERCIAL (ใช้ price_multiplier)
           และ INSTANT (ใช้ inst_price_multiplier เพื่อกดตัดราคาสู้ DUMPER ได้อิสระ)
        """
        if target_sell_bags <= 0 or not lots or not routes:
            return []

        if inst_price_multiplier is None:
            inst_price_multiplier = price_multiplier * 0.96

        valid_mask = get_valid_route_mask(lots, routes, self.error_buffer)
        rem_lot_bags = [lot.bags for lot in lots]
        rem_route_quota = [r.quota for r in routes]
        rem_to_sell = min(target_sell_bags, sum(rem_lot_bags))

        orders: List[Dict] = []

        # --- PHASE 1: กวาดโควตา SPECIALTY ด้วย Lot ที่ผ่านสเปกและมี TV สูงสุด ---
        spec_indices = [
            j for j, r in enumerate(routes)
            if r.buyer_type == "SPECIALTY" and rem_route_quota[j] > 0
        ]
        spec_indices.sort(
            key=lambda j: estimated_ceilings.get(routes[j].route_id, routes[j].base_ceiling_mult),
            reverse=True
        )

        for j in spec_indices:
            if rem_to_sell <= 0:
                break
            route = routes[j]
            eligible_lots = [
                i for i in range(len(lots))
                if valid_mask[i, j] and rem_lot_bags[i] > 0
            ]
            eligible_lots.sort(key=lambda i: lots[i].tv, reverse=True)

            for i in eligible_lots:
                if rem_to_sell <= 0 or rem_route_quota[j] <= 0:
                    break
                alloc = min(rem_lot_bags[i], rem_route_quota[j], rem_to_sell)
                if alloc > 0:
                    ceil_mult = estimated_ceilings.get(route.route_id, route.base_ceiling_mult)
                    bid_mult = min(ceil_mult - 0.004, max(1.14, price_multiplier * 1.18))
                    orders.append({
                        "lot_id": lots[i].lot_id,
                        "lot": lots[i],
                        "route_id": route.route_id,
                        "qty": alloc,
                        "price": round(lots[i].tv * bid_mult, 2),
                        "buyer_type": "SPECIALTY"
                    })
                    rem_lot_bags[i] -= alloc
                    rem_route_quota[j] -= alloc
                    rem_to_sell -= alloc

        # --- PHASE 2: กระจายกระสอบที่เหลือให้ COMMERCIAL และ INSTANT ---
        non_spec_indices = [
            j for j, r in enumerate(routes)
            if r.buyer_type != "SPECIALTY" and rem_route_quota[j] > 0
        ]
        non_spec_indices.sort(
            key=lambda j: (0 if routes[j].buyer_type == "COMMERCIAL" else 1, -routes[j].quota)
        )

        while rem_to_sell > 0:
            progressed = False
            for j in non_spec_indices:
                if rem_to_sell <= 0 or rem_route_quota[j] <= 0:
                    continue
                route = routes[j]
                eligible_lots = [
                    i for i in range(len(lots))
                    if valid_mask[i, j] and rem_lot_bags[i] > 0
                ]
                if not eligible_lots:
                    continue

                if route.buyer_type == "INSTANT":
                    eligible_lots.sort(key=lambda i: lots[i].tv)
                else:
                    eligible_lots.sort(key=lambda i: lots[i].tv, reverse=True)

                i = eligible_lots[0]
                step_qty = min(2, rem_lot_bags[i], rem_route_quota[j], rem_to_sell)
                if step_qty > 0:
                    ceil_mult = estimated_ceilings.get(route.route_id, route.base_ceiling_mult)
                    raw_mult = price_multiplier if route.buyer_type == "COMMERCIAL" else inst_price_multiplier
                    safe_mult = min(ceil_mult - 0.003, max(0.82, raw_mult))

                    existing = next(
                        (o for o in orders if o["lot_id"] == lots[i].lot_id and o["route_id"] == route.route_id),
                        None
                    )
                    if existing:
                        existing["qty"] += step_qty
                    else:
                        orders.append({
                            "lot_id": lots[i].lot_id,
                            "lot": lots[i],
                            "route_id": route.route_id,
                            "qty": step_qty,
                            "price": round(lots[i].tv * safe_mult, 2),
                            "buyer_type": route.buyer_type
                        })

                    rem_lot_bags[i] -= step_qty
                    rem_route_quota[j] -= step_qty
                    rem_to_sell -= step_qty
                    progressed = True

            if not progressed:
                break

        return orders
