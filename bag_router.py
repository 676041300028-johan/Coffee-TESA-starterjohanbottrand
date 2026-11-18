from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from coffee_math import CoffeeLot
from action_mask import BuyerOrder, build_action_mask
from ceiling_estimator import OnlinePriceCeilingEstimator


@dataclass
class BagOrderAction:
    rfq_id: str
    buyer_name: str
    lot_id: str
    bags: int
    bid_price: float
    target_ratio: float
    tv: float
    team_id: str = "MY_TEAM"


class TerminalLiquidationRiskManager:
    """
    ตัวคุมความเสี่ยงรอบสุดท้ายภายใต้กติกาจำกัดยอดขายต่อรอบ (max_bags_per_round = floor(initial_bags / 6 * 1.5))
    """

    def __init__(
        self, total_rounds: int = 7, liquidation_floor_ratio: float = 0.80
    ):
        self.total_rounds = total_rounds
        self.liquidation_floor_ratio = liquidation_floor_ratio

    def compute_risk_adjustment(
        self,
        current_round: int,
        remaining_bags: int,
        total_market_demand: int,
        max_bags_per_round: int = 7,
        urgency_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        rounds_left = max(1, self.total_rounds - current_round + 1)
        time_pressure = (current_round / self.total_rounds) ** 2

        # 1. ความเสี่ยงเทียบกับกำลังซื้อตลาด
        demand_ratio = remaining_bags / max(1, total_market_demand)
        supply_pressure = min(1.0, demand_ratio * 2.5)

        # 2. ความเสี่ยงติดเพดานขายต่อรอบ (Capacity Bottleneck Risk)
        # เช่น เหลือ 3 รอบ ขายได้รอบละ 6 ใบ = ขายได้อีกไม่เกิน 18 ใบ หากของเหลือ 17 ใบต้องเร่งระบายทันที!
        max_possible_sell_left = max(1, rounds_left * max_bags_per_round)
        cap_utilization = remaining_bags / max_possible_sell_left
        capacity_pressure = min(1.0, max(0.0, (cap_utilization - 0.55) / 0.45))

        raw_urgency = (
            0.40 * time_pressure
            + 0.25 * supply_pressure
            + 0.35 * capacity_pressure
        ) * urgency_multiplier

        urgency = 1.0 if rounds_left == 1 else float(np.clip(raw_urgency, 0.0, 1.0))

        # ส่วนลด Ratio จากเพดาน (0.015 ถึง 0.085)
        ratio_discount = round(0.015 + (urgency * 0.070), 4)

        min_acceptable_ratio = max(
            self.liquidation_floor_ratio + 0.01,
            round(0.92 - (urgency * 0.11), 4),
        )

        return {
            "rounds_left": rounds_left,
            "urgency": round(urgency, 4),
            "ratio_discount": ratio_discount,
            "min_acceptable_ratio": min_acceptable_ratio,
        }


class BagLevelCombinatorialRouter:
    """
    ตัวกระจายกระสอบรายใบ (4 Lots x 8 Routes) รุ่นรองรับ Error 0.5% และเพดานขายต่อรอบ
    """

    def __init__(self, risk_manager: TerminalLiquidationRiskManager):
        self.risk_manager = risk_manager

    def route_bags(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
        action_mask: np.ndarray,
        ceiling_estimator: OnlinePriceCeilingEstimator,
        rl_aggressiveness: float = 0.5,
        ladder_spread: float = 0.015,
        urgency_multiplier: float = 1.0,
        diversification: float = 0.75,
        max_bags_per_round: Optional[int] = None,
        team_id: str = "MY_TEAM",
    ) -> List[BagOrderAction]:
        total_remaining_bags = sum(lot.bags for lot in lots)
        total_market_demand = sum(r.quota_bags for r in routes if r.quota_bags > 0)

        if total_remaining_bags <= 0 or total_market_demand <= 0:
            return []

        eff_cap = max_bags_per_round if max_bags_per_round is not None else total_remaining_bags

        risk = self.risk_manager.compute_risk_adjustment(
            current_round=current_round,
            remaining_bags=total_remaining_bags,
            total_market_demand=total_market_demand,
            max_bags_per_round=eff_cap,
            urgency_multiplier=urgency_multiplier,
        )

        # เผื่อส่วนลดความคลาดเคลื่อนของ TV (±0.5% ทำให้ TV จริงอาจต่ำกว่าที่เห็น ~1-2%)
        # ป้องกันกรณีเสนอราคาชิดเพดานแล้ว TV จริงต่ำกว่าที่คำนวณจน Ratio ทะลุเพดานผู้ซื้อ
        route_ceilings: Dict[int, float] = {}
        for j, r in enumerate(routes):
            est = ceiling_estimator.predict_max_ratio(
                company_name=r.company_name,
                buyer_type=r.buyer_type,
                safety_margin=float(np.clip(1.0 - rl_aggressiveness, 0.0, 1.0)),
            )
            route_ceilings[j] = est["recommended_ratio"] - 0.008

        lot_indices = sorted(
            range(len(lots)),
            key=lambda i: (int(np.sum(action_mask[i])), -lots[i].TV),
        )

        route_remaining_cap = {j: r.quota_bags for j, r in enumerate(routes)}
        lot_remaining_bags = {i: lots[i].bags for i in range(len(lots))}
        final_orders: List[BagOrderAction] = []

        # จำกัดจำนวนกระสอบที่ยื่นเสนอต่อรอบไม่ให้เกิน 1.5 เท่าของเพดานขายต่อรอบ (ลดการกระจายราคาเสียเปล่า)
        max_submit_budget = min(total_remaining_bags, max(eff_cap + 2, int(np.ceil(eff_cap * 1.4))))
        submitted_so_far = 0

        for i in lot_indices:
            if submitted_so_far >= max_submit_budget:
                break

            lot = lots[i]
            bags_to_allocate = min(lot_remaining_bags[i], max_submit_budget - submitted_so_far)
            if bags_to_allocate <= 0 or lot.TV <= 0:
                continue

            valid_routes = [
                j
                for j in range(len(routes))
                if action_mask[i, j] and route_remaining_cap[j] > 0
            ]
            if not valid_routes:
                continue

            total_valid_cap = sum(route_remaining_cap[j] for j in valid_routes)
            scores = []
            for j in valid_routes:
                quota_share = route_remaining_cap[j] / max(1, total_valid_cap)
                price_bonus = 1.5 if routes[j].buyer_type == "COMMERCIAL" else 1.0
                weight = (
                    diversification * quota_share
                    + (1.0 - diversification) * quota_share * price_bonus
                )
                scores.append(max(1e-6, weight))

            probs = np.array(scores) / np.sum(scores)
            raw_alloc = probs * bags_to_allocate
            alloc_counts = np.floor(raw_alloc).astype(int)
            remainder = int(bags_to_allocate - np.sum(alloc_counts))

            frac_order = np.argsort(-(raw_alloc - alloc_counts))
            for idx in frac_order:
                if remainder <= 0:
                    break
                j = valid_routes[idx]
                if alloc_counts[idx] < route_remaining_cap[j]:
                    alloc_counts[idx] += 1
                    remainder -= 1

            ordered_pairs = sorted(
                zip(valid_routes, alloc_counts),
                key=lambda x: route_ceilings[x[0]],
                reverse=True,
            )

            for j, target_qty in ordered_pairs:
                if lot_remaining_bags[i] <= 0 or submitted_so_far >= max_submit_budget:
                    break
                desired = max(int(target_qty), 0)
                if desired == 0 and lot_remaining_bags[i] > 0 and diversification < 0.95:
                    desired = min(lot_remaining_bags[i], max_submit_budget - submitted_so_far)

                send_qty = min(
                    desired,
                    lot_remaining_bags[i],
                    route_remaining_cap[j],
                    max_submit_budget - submitted_so_far,
                )
                if send_qty <= 0:
                    continue

                order = routes[j]
                base_ceiling = route_ceilings[j]
                effective_top_ratio = base_ceiling - risk["ratio_discount"] * (
                    1.0 - 0.5 * rl_aggressiveness
                )
                effective_top_ratio = max(
                    risk["min_acceptable_ratio"], effective_top_ratio
                )

                for bag_idx in range(send_qty):
                    step_drop = (
                        (bag_idx / max(1, send_qty))
                        * ladder_spread
                        * (0.5 + 0.5 * risk["urgency"])
                    )
                    bag_ratio = max(
                        risk["min_acceptable_ratio"],
                        round(effective_top_ratio - step_drop, 4),
                    )
                    bid_price = round(lot.TV * bag_ratio, 2)

                    final_orders.append(
                        BagOrderAction(
                            rfq_id=order.order_id,
                            buyer_name=order.company_name,
                            lot_id=lot.lot_id,
                            bags=1,
                            bid_price=bid_price,
                            target_ratio=bag_ratio,
                            tv=lot.TV,
                            team_id=team_id,
                        )
                    )

                lot_remaining_bags[i] -= send_qty
                route_remaining_cap[j] -= send_qty
                submitted_so_far += send_qty

        return final_orders