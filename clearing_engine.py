import random
import math
from typing import List, Dict, Tuple
from coffee_math import CoffeeLot, BuyerRoute, calculate_tv

class BuyerCompanyPool:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.specialty_templates = [
            {"name": "บางกอกบลูม", "min_s": 17.0, "max_g": 3.0, "max_m": 12.5, "ceil": 1.24, "base_q": 4},
            {"name": "ดอยช้างโรสเตอร์", "min_s": 16.0, "max_g": 3.0, "max_m": 12.5, "ceil": 1.22, "base_q": 3},
            {"name": "เชียงใหม่สเปเชียลตี้", "min_s": 17.0, "max_g": 2.5, "max_m": 12.0, "ceil": 1.25, "base_q": 3},
        ]
        self.commercial_templates = [
            {"name": "สยามคอฟฟี่เทรด", "min_s": 0.0, "max_g": 3.0, "max_m": 12.5, "ceil": 1.04},
            {"name": "ไทยโรสเตอร์กรุ๊ป", "min_s": 0.0, "max_g": 5.0, "max_m": 13.0, "ceil": 1.03},
            {"name": "อาราบิก้าเอ็กซ์พอร์ต", "min_s": 15.0, "max_g": 4.0, "max_m": 12.5, "ceil": 1.035},
            {"name": "บูรพาคอฟฟี่", "min_s": 0.0, "max_g": 6.0, "max_m": 13.5, "ceil": 1.025},
        ]
        self.instant_templates = [
            {"name": "อินสแตนท์อินดัสทรี", "min_s": 0.0, "max_g": 10.0, "max_m": 14.5, "ceil": 0.985},
            {"name": "โรงคั่วอุตสาหกรรม", "min_s": 0.0, "max_g": 12.0, "max_m": 15.0, "ceil": 0.975},
            {"name": "แมสเบฟเวอเรจ", "min_s": 0.0, "max_g": 9.0, "max_m": 14.0, "ceil": 0.990},
        ]

    def get_round_event_profile(self, round_idx: int, total_rounds: int) -> Tuple[float, float, int]:
        """
        ตาราง Event ตามรอบการแข่งขัน:
          - รอบ 1-5: ปกติ (demand_mult=1.0, spec_boost=1.0, phase_code=0)
          - รอบ 6-8: Specialty รับเยอะขึ้น (demand_mult=1.15, spec_boost=2.4, phase_code=1)
          - รอบ 9-12: ปกติ (demand_mult=1.0, spec_boost=1.0, phase_code=0)
          - รอบ 13-14: ความต้องการรับซื้อต่ำ (demand_mult=0.55, spec_boost=0.6, phase_code=2)
          - รอบ 15: รับซื้อต่ำมาก (demand_mult=0.28, spec_boost=0.3, phase_code=3)
          - รอบ 16 (รอบสุดท้าย): รอบขยะ บังคับรับซื้อที่ 0.80 * TV (phase_code=4)
        """
        if total_rounds < 16:
            return 1.0, 1.0, 0

        if 1 <= round_idx <= 5:
            return 1.00, 1.00, 0
        elif 6 <= round_idx <= 8:
            return 1.15, 2.40, 1
        elif 9 <= round_idx <= 12:
            return 1.00, 1.00, 0
        elif 13 <= round_idx <= 14:
            return 0.55, 0.60, 2
        elif round_idx == 15:
            return 0.28, 0.30, 3
        else:
            return 1.00, 0.00, 4

    def generate_round_routes(
        self,
        round_idx: int,
        total_rounds: int,
        total_market_bags: int,
        event_phase_offset: int = 0
    ) -> List[BuyerRoute]:
        demand_mult, spec_boost, phase_code = self.get_round_event_profile(round_idx, total_rounds)

        if total_rounds >= 16 and round_idx >= total_rounds:
            return [
                BuyerRoute(
                    route_id="R16-CLEAR",
                    company_name="ตลาดรับซื้อรอบปิดเกม (0.80)",
                    buyer_type="INSTANT",
                    quota=max(99999, total_market_bags),
                    max_defect=100.0,
                    max_moisture=100.0,
                    min_screen=0.0,
                    base_ceiling_mult=0.8000,
                )
            ]

        routes: List[BuyerRoute] = []
        auction_rounds = 15 if total_rounds >= 16 else max(1, total_rounds)
        avg_round_supply = (total_market_bags * 0.82) / auction_rounds
        target_round_demand = max(25, int(round(avg_round_supply * demand_mult)))

        if phase_code == 1:
            num_spec = self.rng.choice([2, 3])
        elif phase_code in (2, 3):
            num_spec = 1 if self.rng.random() < 0.35 else 0
        else:
            num_spec = self.rng.choice([1, 2]) if self.rng.random() < 0.75 else 0

        chosen_specs = self.rng.sample(self.specialty_templates, k=num_spec)
        spec_quota_used = 0
        for idx, tmpl in enumerate(chosen_specs):
            q = max(12, int(round(tmpl["base_q"] * 22 * spec_boost)))
            spec_quota_used += q
            routes.append(BuyerRoute(
                route_id=f"R5-0{idx+1}",
                company_name=tmpl["name"],
                buyer_type="SPECIALTY",
                quota=q,
                max_defect=tmpl["max_g"],
                max_moisture=tmpl["max_m"],
                min_screen=tmpl["min_s"],
                base_ceiling_mult=round(tmpl["ceil"] + self.rng.uniform(-0.005, 0.005), 4)
            ))

        rem_demand = max(15, target_round_demand - spec_quota_used)
        num_comm = max(2, 5 - num_spec)
        num_inst = max(2, 8 - num_spec - num_comm)

        comm_share = 0.45 if phase_code in (2, 3) else 0.65
        comm_demand = int(rem_demand * comm_share)
        inst_demand = max(6, rem_demand - comm_demand)

        for i in range(num_comm):
            tmpl = self.commercial_templates[i % len(self.commercial_templates)]
            q = max(3, comm_demand // num_comm)
            routes.append(BuyerRoute(
                route_id=f"R{i+1}-01",
                company_name=tmpl["name"],
                buyer_type="COMMERCIAL",
                quota=q,
                max_defect=tmpl["max_g"],
                max_moisture=tmpl["max_m"],
                min_screen=tmpl["min_s"],
                base_ceiling_mult=round(tmpl["ceil"] + self.rng.uniform(-0.008, 0.008), 4)
            ))

        for i in range(num_inst):
            tmpl = self.instant_templates[i % len(self.instant_templates)]
            q = max(3, inst_demand // num_inst)
            routes.append(BuyerRoute(
                route_id=f"R{num_comm+i+1}-01",
                company_name=tmpl["name"],
                buyer_type="INSTANT",
                quota=q,
                max_defect=tmpl["max_g"],
                max_moisture=tmpl["max_m"],
                min_screen=tmpl["min_s"],
                base_ceiling_mult=round(tmpl["ceil"] + self.rng.uniform(-0.008, 0.008), 4)
            ))

        return routes


class MarketClearingEngine:
    def __init__(self, grade_error_std: float = 0.005, cap_divisor: int = 5):
        self.grade_error_std = grade_error_std
        self.cap_divisor = cap_divisor  # 4 = เพดาน 1/4, 5 = เพดาน 1/5 (เช่น 25 กระสอบ ขายได้ 5/รอบ)

    def get_max_sell_per_round(self, initial_bags: int, cap_divisor: int = None) -> int:
        """กติกา: ขายได้สูงสุดไม่เกิน floor(initial_bags / cap_divisor) ต่อรอบ (รองรับทั้ง 1/4 และ 1/5)"""
        div = cap_divisor if cap_divisor is not None else self.cap_divisor
        return max(1, initial_bags // max(1, div))

    def clear_market(
        self,
        all_bids: List[Dict],
        routes: List[BuyerRoute],
        team_initial_bags: Dict[str, int],
        rng: random.Random,
        cap_divisor: int = None
    ) -> Tuple[List[Dict], Dict[str, float]]:
        div = cap_divisor if cap_divisor is not None else self.cap_divisor
        route_map = {r.route_id: r for r in routes}
        route_rem_quota = {r.route_id: r.quota for r in routes}
        team_sold_this_round = {tid: 0 for tid in team_initial_bags}

        valid_bids = []
        results = []

        for bid in all_bids:
            tid = bid["team_id"]
            rid = bid["route_id"]
            lot: CoffeeLot = bid["lot"]
            qty = bid["qty"]
            ask_price = bid["price"]

            route = route_map.get(rid)
            if not route or qty <= 0:
                continue

            is_junk_round = (rid == "R16-CLEAR")
            if is_junk_round:
                allowed_qty = qty
            else:
                max_cap = self.get_max_sell_per_round(team_initial_bags[tid], div)
                allowed_qty = min(qty, max(0, max_cap - team_sold_this_round[tid]))
                if allowed_qty <= 0:
                    results.append({**bid, "status": "CAP_EXCEEDED", "sold_qty": 0})
                    continue

            true_s = lot.s * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_m = lot.m * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_g = lot.g * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_tv = calculate_tv(true_s, true_m, true_g)

            if not is_junk_round:
                if (true_g > route.max_defect or
                    true_m > route.max_moisture or
                    true_s < route.min_screen):
                    results.append({**bid, "status": "SPEC_FAILED", "sold_qty": 0})
                    continue

            ceiling_price = true_tv * route.base_ceiling_mult
            if ask_price > ceiling_price + 1e-4:
                results.append({**bid, "status": "CEILING_EXCEEDED", "sold_qty": 0})
                continue

            bid_tv_ratio = ask_price / max(1e-6, true_tv)
            valid_bids.append({
                **bid,
                "allowed_qty": allowed_qty,
                "true_tv": true_tv,
                "bid_tv_ratio": bid_tv_ratio,
                "is_junk_round": is_junk_round,
            })

        valid_bids.sort(key=lambda x: x["bid_tv_ratio"])

        for b in valid_bids:
            tid = b["team_id"]
            rid = b["route_id"]
            if b["is_junk_round"]:
                cap_left = b["allowed_qty"]
            else:
                max_cap = self.get_max_sell_per_round(team_initial_bags[tid], div)
                cap_left = max(0, max_cap - team_sold_this_round[tid])

            quota_left = route_rem_quota[rid]
            fill_qty = min(b["allowed_qty"], cap_left, quota_left)
            if fill_qty > 0:
                team_sold_this_round[tid] += fill_qty
                route_rem_quota[rid] -= fill_qty
                status = "SOLD" if fill_qty == b["qty"] else "PARTIAL_SOLD"
                results.append({
                    **b,
                    "status": status,
                    "sold_qty": fill_qty,
                    "revenue": fill_qty * b["price"]
                })
            else:
                results.append({**b, "status": "QUOTA_FULL", "sold_qty": 0, "revenue": 0.0})

        return results, route_rem_quota
