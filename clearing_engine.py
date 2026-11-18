import random
import math
from typing import List, Dict, Tuple
from coffee_math import CoffeeLot, BuyerRoute, calculate_tv

class BuyerCompanyPool:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        # ฐานข้อมูลบริษัทผู้ซื้อทั้ง 3 กลุ่มตามสนามจริง
        self.specialty_templates = [
            {"name": "บางกอกบลูม", "min_s": 17.0, "max_g": 3.0, "max_m": 12.5, "ceil": 1.24, "base_q": 4},
            {"name": "ดอยช้างโรสเตอร์", "min_s": 16.0, "max_g": 3.0, "max_m": 12.5, "ceil": 1.22, "base_q": 2},
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

    def get_event_demand_multiplier(self, round_idx: int, total_rounds: int, event_phase_offset: int = 0) -> float:
        """
        จำลอง Market Event Wave สำหรับเกม 16 รอบ (สลับช่วงละ 4 รอบติดกัน)
        - ช่วงซบเซา (Bear Wave 4 รอบ): รับซื้อลดลง ~35% (ตัวคูณ 0.65)
        - ช่วงบูม (Bull Wave 4 รอบ): รับซื้อเพิ่มขึ้น ~35% (ตัวคูณ 1.35)
        """
        if total_rounds < 16:
            return 1.0  # โหมด 9 รอบ ตลาดปกติสม่ำเสมอ

        block = ((round_idx - 1 + event_phase_offset) // 4) % 2
        return 0.65 if block == 0 else 1.35

    def generate_round_routes(
        self,
        round_idx: int,
        total_rounds: int,
        total_market_bags: int,
        event_phase_offset: int = 0
    ) -> List[BuyerRoute]:
        routes: List[BuyerRoute] = []
        demand_mult = self.get_event_demand_multiplier(round_idx, total_rounds, event_phase_offset)

        # เป้าหมายโควตารวมของตลาดในรอบนี้ (~80% ของสต็อกเฉลี่ยต่อรอบ * event multiplier)
        avg_round_supply = (total_market_bags * 0.80) / max(1, total_rounds)
        target_round_demand = max(20, int(round(avg_round_supply * demand_mult)))

        # 1. สุ่มเปิด Route กลุ่ม SPECIALTY (0 ถึง 2 บริษัทต่อรอบ, โอกาสสูงขึ้นช่วงตลาดบูม)
        spec_prob = 0.80 if demand_mult >= 1.0 else 0.45
        num_spec = 0
        if self.rng.random() < spec_prob:
            num_spec = self.rng.choice([1, 2])

        chosen_specs = self.rng.sample(self.specialty_templates, k=num_spec)
        spec_quota_used = 0
        for idx, tmpl in enumerate(chosen_specs):
            # โควตาต่อทีมประมาณ 2-4 กระสอบ คูณจำนวนทีมแบบ scaled (สมมติ 55 ทีมแย่งกัน)
            q = max(10, int(round(tmpl["base_q"] * 18 * demand_mult)))
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

        # 2. กระจายโควตาที่เหลือให้ COMMERCIAL (65%) และ INSTANT (35%) รวมกันครบ 8 เส้นทาง
        rem_demand = max(16, target_round_demand - spec_quota_used)
        num_comm = 5 if num_spec <= 1 else 4
        num_inst = 8 - num_spec - num_comm

        comm_demand = int(rem_demand * 0.65)
        inst_demand = rem_demand - comm_demand

        for i in range(num_comm):
            tmpl = self.commercial_templates[i % len(self.commercial_templates)]
            q = max(4, comm_demand // num_comm)
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
            q = max(4, inst_demand // max(1, num_inst))
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
    def __init__(self, grade_error_std: float = 0.005):
        self.grade_error_std = grade_error_std

    @staticmethod
    def get_max_sell_per_round(initial_bags: int) -> int:
        """กติกาใหม่: ขายได้สูงสุดไม่เกิน floor(initial_bags / 4) ต่อรอบ"""
        return initial_bags // 4

    def clear_market(
        self,
        all_bids: List[Dict],
        routes: List[BuyerRoute],
        team_initial_bags: Dict[str, int],
        rng: random.Random
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        จับคู่คำสั่งซื้อขายตามกฎ Reverse Auction:
        1. เช็กเพดานขายต่อรอบ initial_bags // 4 ของแต่ละทีม
        2. จำลอง True Grade (+/- 0.5%) เพื่อตรวจสเปกจริงและคำนวณ True TV
        3. ตัดคำสั่งที่ราคา > เพดานบริษัท หรือไม่ผ่านสเปกจริง
        4. เรียงลำดับ Bid/TV จากต่ำไปสูง (ใครเสนอคุ้มค่าสุดได้ขายก่อน)
        """
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

            # บังคับเพดานขายสูงสุดต่อรอบ initial_bags // 4
            max_cap = self.get_max_sell_per_round(team_initial_bags[tid])
            allowed_qty = min(qty, max(0, max_cap - team_sold_this_round[tid]))
            if allowed_qty <= 0:
                results.append({**bid, "status": "CAP_EXCEEDED", "sold_qty": 0})
                continue

            # จำลองค่าจริงบนเซิร์ฟเวอร์ (คลาดเคลื่อน +/- 0.5%)
            true_s = lot.s * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_m = lot.m * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_g = lot.g * (1.0 + rng.uniform(-self.grade_error_std, self.grade_error_std))
            true_tv = calculate_tv(true_s, true_m, true_g)

            # ตรวจสเปกจริง
            if (true_g > route.max_defect or
                true_m > route.max_moisture or
                true_s < route.min_screen):
                results.append({**bid, "status": "SPEC_FAILED", "sold_qty": 0})
                continue

            # ตรวจเพดานราคาของบริษัท
            ceiling_price = true_tv * route.base_ceiling_mult
            if ask_price > ceiling_price:
                results.append({**bid, "status": "CEILING_EXCEEDED", "sold_qty": 0})
                continue

            bid_tv_ratio = ask_price / max(1e-6, true_tv)
            valid_bids.append({
                **bid,
                "allowed_qty": allowed_qty,
                "true_tv": true_tv,
                "bid_tv_ratio": bid_tv_ratio
            })

        # เรียงลำดับจากอัตราส่วน Price / True_TV ต่ำสุดไปสูงสุด
        valid_bids.sort(key=lambda x: x["bid_tv_ratio"])

        for b in valid_bids:
            tid = b["team_id"]
            rid = b["route_id"]
            max_cap = self.get_max_sell_per_round(team_initial_bags[tid])
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