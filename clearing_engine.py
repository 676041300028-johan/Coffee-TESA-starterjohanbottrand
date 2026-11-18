import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from coffee_math import CoffeeLot, BUYER_SPECS
from action_mask import BuyerOrder
from bag_router import BagOrderAction


@dataclass
class ClearingResult:
    round_num: int
    team_feedback: Dict[str, List[Dict]]
    team_revenue: Dict[str, float]
    team_profit_over_tv: Dict[str, float]
    market_stats: Dict[str, float]


class BuyerCompanyPool:
    COMPANIES = [
        ("1 พรีเมียมโรสเตอร์", "COMMERCIAL", 7.0, 12.5),
        ("2 โกลบอลคอฟฟี่", "COMMERCIAL", 7.0, 12.5),
        ("3 สยามคอมเมิร์ซ", "COMMERCIAL", 7.0, 12.5),
        ("4 ชุมพรคั่วบด", "COMMERCIAL", 7.0, 12.5),
        ("5 บลูเมาท์เทน", "INSTANT", 15.0, 14.0),
        ("6 ทรีอินวันฟู้ดส์", "INSTANT", 15.0, 14.0),
        ("7 เอเชียอินสแตนท์", "INSTANT", 15.0, 14.0),
        ("8 สยามเบฟเวอเรจ", "INSTANT", 15.0, 14.0),
    ]

    def __init__(self, rng: np.random.Generator, ceiling_mode: str = "FIXED"):
        self.rng = rng
        self.ceiling_mode = ceiling_mode
        self.fixed_ceilings: Dict[str, float] = {
            name: self._sample_ceiling(b_type)
            for name, b_type, _, _ in self.COMPANIES
        }

    def _sample_ceiling(self, buyer_type: str) -> float:
        spec = BUYER_SPECS[buyer_type]
        return round(
            float(self.rng.uniform(spec["ratio_min"] - 0.003, spec["ratio_max"] + 0.003)),
            4,
        )

    def plan_game_round_quotas(
        self,
        total_market_supply: int,
        total_rounds: int = 7,
        demand_ratio: float = 0.80,
    ) -> List[int]:
        total_game_quota = max(1, int(round(total_market_supply * demand_ratio)))
        weights = self.rng.dirichlet(np.full(total_rounds, 4.0))
        round_quotas = self.rng.multinomial(total_game_quota, weights)
        return [int(q) for q in round_quotas]

    def generate_round_routes(
        self,
        round_total_quota: int,
        comm_share: float = 0.45,
    ) -> List[BuyerOrder]:
        indices = self.rng.permutation(len(self.COMPANIES))
        routes: List[BuyerOrder] = []

        if round_total_quota <= 0:
            quotas = np.zeros(len(indices), dtype=int)
        else:
            weights = []
            for comp_idx in indices:
                b_type = self.COMPANIES[comp_idx][1]
                base_w = comm_share if b_type == "COMMERCIAL" else (1.0 - comm_share)
                weights.append(base_w * float(self.rng.uniform(0.4, 1.6)))
            probs = np.array(weights) / np.sum(weights)
            quotas = self.rng.multinomial(round_total_quota, probs)

        for idx_pos, comp_idx in enumerate(indices):
            name, b_type, max_def, max_moist = self.COMPANIES[comp_idx]
            ceiling = (
                self.fixed_ceilings[name]
                if self.ceiling_mode == "FIXED"
                else self._sample_ceiling(b_type)
            )
            routes.append(
                BuyerOrder(
                    order_id=f"rt{idx_pos + 1:02d}",
                    buyer_type=b_type,
                    quota_bags=int(quotas[idx_pos]),
                    company_name=name,
                    max_moisture=max_moist,
                    max_defect=max_def,
                    hidden_ceiling_ratio=ceiling,
                )
            )
        return routes


class MarketClearingEngine:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_team_lots(
        self, team_id: str, total_bags_range: Tuple[int, int] = (24, 30)
    ) -> List[CoffeeLot]:
        """
        สุ่มเมล็ดกาแฟ 4 Lots พร้อมจำลองความคลาดเคลื่อน ±0.5% ระหว่างค่าที่ผู้เล่นเห็น (s, m, g)
        กับค่าจริงฝั่งเซิร์ฟเวอร์ (s_true, m_true, g_true)
        """
        target_total = int(
            self.rng.integers(total_bags_range[0], total_bags_range[1] + 1)
        )
        splits = self.rng.multinomial(target_total - 12, [0.25, 0.25, 0.25, 0.25]) + 3
        profiles = [
            (
                float(self.rng.integers(15, 18)),
                round(float(self.rng.uniform(10.8, 12.2)), 2),
                round(float(self.rng.uniform(0.5, 2.5)), 2),
            ),
            (
                float(self.rng.integers(14, 17)),
                round(float(self.rng.uniform(11.5, 12.55)), 2),  # มีโอกาสปริ่ม 12.5% และข้ามเกรด
                round(float(self.rng.uniform(2.0, 7.05)), 2),    # มีโอกาสปริ่ม 7.0% และข้ามเกรด
            ),
            (
                float(self.rng.integers(13, 16)),
                round(float(self.rng.uniform(12.6, 13.8)), 2),
                round(float(self.rng.uniform(3.0, 9.0)), 2),
            ),
            (
                float(self.rng.integers(12, 15)),
                round(float(self.rng.uniform(12.8, 14.05)), 2),
                round(float(self.rng.uniform(6.0, 14.5)), 2),
            ),
        ]

        lots: List[CoffeeLot] = []
        for i in range(4):
            s_true, m_true, g_true = profiles[i]
            # สุ่ม error ±0.5% สำหรับค่าที่ผู้เล่นมองเห็น
            s_obs = round(s_true * (1.0 + float(self.rng.uniform(-0.005, 0.005))), 2)
            m_obs = round(m_true * (1.0 + float(self.rng.uniform(-0.005, 0.005))), 3)
            g_obs = round(g_true * (1.0 + float(self.rng.uniform(-0.005, 0.005))), 3)

            lots.append(
                CoffeeLot(
                    lot_id=f"LOT-{i+1}",
                    s=s_obs,
                    m=m_obs,
                    g=g_obs,
                    bags=int(splits[i]),
                    s_true=s_true,
                    m_true=m_true,
                    g_true=g_true,
                )
            )
        return lots

    def clear_market(
        self,
        round_num: int,
        team_inventories: Dict[str, List[CoffeeLot]],
        routes: List[BuyerOrder],
        all_orders: List[BagOrderAction],
        team_round_sell_caps: Optional[Dict[str, int]] = None,
    ) -> ClearingResult:
        """
        ตัดสินผลตลาดโดยใช้ค่าจริง (m_true, g_true, TV_true)
        และบังคับเพดานขายสูงสุดต่อรอบของแต่ละทีม (initial_bags / 6 * 1.5)
        """
        route_map = {r.order_id: r for r in routes}
        lot_lookup = {
            (t_id, lot.lot_id): lot
            for t_id, lots in team_inventories.items()
            for lot in lots
        }
        team_feedback = {t_id: [] for t_id in team_inventories}
        team_revenue = {t_id: 0.0 for t_id in team_inventories}
        team_profit_over_tv = {t_id: 0.0 for t_id in team_inventories}

        # ติดตามโควตาการขายที่เหลือในรอบนี้ของแต่ละทีม
        team_cap_left = (
            dict(team_round_sell_caps)
            if team_round_sell_caps is not None
            else {t_id: 999 for t_id in team_inventories}
        )

        valid_bids = {r.order_id: [] for r in routes}
        all_passed_ratios = []

        for order in all_orders:
            t_id = order.team_id
            route = route_map.get(order.rfq_id)
            lot = lot_lookup.get((t_id, order.lot_id))
            if route is None or lot is None or order.bags <= 0:
                continue

            # 1. ตรวจคุณภาพด้วยค่าจริงฝั่งเซิร์ฟเวอร์ (m_true, g_true)
            if lot.m_true > route.max_moisture or lot.g_true > route.max_defect:
                team_feedback[t_id].append({
                    "rfq": order.rfq_id,
                    "lot": order.lot_id,
                    "price": order.bid_price,
                    "bags_submitted": order.bags,
                    "bags_sold": 0,
                    "status": "ไม่ผ่านเกณฑ์คุณภาพ",
                })
                continue

            # 2. คำนวณ Ratio ด้วยค่า TV_true จริงฝั่งเซิร์ฟเวอร์
            true_tv = max(1e-6, lot.TV_true)
            ratio_true = order.bid_price / true_tv
            ceiling = (
                route.hidden_ceiling_ratio
                if route.hidden_ceiling_ratio is not None
                else route.max_ratio
            )
            if ratio_true > ceiling + 1e-6:
                team_feedback[t_id].append({
                    "rfq": order.rfq_id,
                    "lot": order.lot_id,
                    "price": order.bid_price,
                    "bags_submitted": order.bags,
                    "bags_sold": 0,
                    "status": "แพงเกินเพดานผู้ซื้อ",
                })
                continue

            valid_bids[order.rfq_id].append((ratio_true, float(self.rng.random()), order))

        # 3. เรียงลำดับตาม Ratio จริง (ต่ำไปสูง) และตัดยอดทั้งโควตาผู้ซื้อ + เพดานขายต่อรอบของทีม
        total_sold = 0
        for r in routes:
            rem_q = r.quota_bags
            cands = sorted(valid_bids[r.order_id], key=lambda x: (x[0], x[1]))
            for ratio_true, _, order in cands:
                t_id = order.team_id
                lot = lot_lookup[(t_id, order.lot_id)]

                # จำกัดการขายด้วย: ออเดอร์ที่ส่ง, ของในสต็อก, โควตาผู้ซื้อ, และเพดานขายต่อรอบของทีม
                sold = min(order.bags, lot.bags, rem_q, team_cap_left[t_id])

                if sold <= 0:
                    if team_cap_left[t_id] <= 0:
                        status_str = "เต็มเพดานขายต่อรอบ"
                    else:
                        status_str = "ราคาแพ้ (โควตาเต็ม)"
                elif sold < order.bags:
                    status_str = "ขายได้บางส่วน"
                else:
                    status_str = "ขายได้ทั้งหมด"

                if sold > 0:
                    lot.bags -= sold
                    rem_q -= sold
                    team_cap_left[t_id] -= sold
                    total_sold += sold
                    team_revenue[t_id] += round(order.bid_price * sold, 2)
                    team_profit_over_tv[t_id] += round((order.bid_price - lot.TV_true) * sold, 2)
                    all_passed_ratios.append(ratio_true)

                team_feedback[t_id].append({
                    "rfq": order.rfq_id,
                    "lot": order.lot_id,
                    "price": order.bid_price,
                    "bags_submitted": order.bags,
                    "bags_sold": sold,
                    "status": status_str,
                })

        stats = {
            "total_sold_bags": float(total_sold),
            "avg_clearing_ratio": float(np.mean(all_passed_ratios)) if all_passed_ratios else 0.95,
            "min_clearing_ratio": float(np.min(all_passed_ratios)) if all_passed_ratios else 0.90,
            "max_clearing_ratio": float(np.max(all_passed_ratios)) if all_passed_ratios else 1.00,
        }
        return ClearingResult(
            round_num, team_feedback, team_revenue, team_profit_over_tv, stats
        )

    def liquidate_remaining_inventory(
        self, team_inventories: Dict[str, List[CoffeeLot]], liquidation_ratio: float = 0.80
    ) -> Dict[str, Tuple[float, float]]:
        results = {}
        for t_id, lots in team_inventories.items():
            liq_rev, liq_prof = 0.0, 0.0
            for lot in lots:
                if lot.bags > 0:
                    fp = round(lot.TV_true * liquidation_ratio, 2)
                    liq_rev += fp * lot.bags
                    liq_prof += (fp - lot.TV_true) * lot.bags
                    lot.bags = 0
            results[t_id] = (round(liq_rev, 2), round(liq_prof, 2))
        return results