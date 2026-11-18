import sys
from pathlib import Path

# เพิ่มโฟลเดอร์ปัจจุบันของไฟล์นี้เข้า sys.path เพื่อให้กด Run ใน VS Code ได้จากทุกที่
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from coffee_math import CoffeeLot, BUYER_SPECS
from action_mask import BuyerOrder, build_action_mask
from ceiling_estimator import OnlinePriceCeilingEstimator
from bag_router import (
    BagOrderAction,
    TerminalLiquidationRiskManager,
    BagLevelCombinatorialRouter,
)


class BaseOpponentBot(ABC):
    """คลาสแม่แบบสำหรับบอทคู่แข่งทุกสไตล์ในตลาดจำลอง"""

    def __init__(
        self,
        team_id: str,
        total_rounds: int = 7,
        rng: Optional[np.random.Generator] = None,
    ):
        self.team_id = team_id
        self.total_rounds = total_rounds
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self):
        """รีเซ็ตความจำของบอทเมื่อเริ่มเกมใหม่"""
        pass

    def observe_feedback(
        self,
        round_feedback: List[Dict],
        rfq_to_company: Dict[str, Tuple[str, str]],
        lot_tv_map: Dict[str, float],
    ):
        """รับผลลัพธ์ท้ายรอบเพื่อปรับกลยุทธ์ในรอบถัดไป"""
        pass

    @abstractmethod
    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        """ตัดสินใจสร้างคำสั่งเสนอขายรายกระสอบ"""
        pass


# ==========================================
# 1. Dumper Bot: เน้นปล่อยของชัวร์ ตั้งราคาต่ำตัดหน้าแย่งโควตา
# ==========================================
class DumperBot(BaseOpponentBot):
    """
    บอทสายเทขาย (Dumper):
    ตั้งราคาต่ำใกล้เคียง 0.84 - 0.91 * TV เพื่อแย่งโควตาต้นๆ ของตลาด
    และยิ่งใกล้รอบสุดท้ายยิ่งกดราคาลงเข้าใกล้ 0.81 * TV
    """

    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        mask = build_action_mask(lots, routes)
        route_cap = {j: r.quota_bags for j, r in enumerate(routes)}
        orders: List[BagOrderAction] = []

        # ยิ่งรอบท้าย ยิ่งเทขายราคาต่ำลง
        round_decay = (current_round / self.total_rounds) * 0.04
        base_ratio = max(0.815, 0.89 - round_decay)

        for i, lot in enumerate(lots):
            if lot.bags <= 0 or lot.TV <= 0:
                continue

            valid_routes = [
                j for j in range(len(routes)) if mask[i, j] and route_cap[j] > 0
            ]
            # ส่งเข้า Route ที่ยังมีโควตาเหลือเยอะที่สุดก่อน
            valid_routes.sort(key=lambda j: route_cap[j], reverse=True)

            bags_left = lot.bags
            for j in valid_routes:
                if bags_left <= 0:
                    break
                send_qty = min(bags_left, route_cap[j])
                r = routes[j]

                for _ in range(send_qty):
                    noise = float(self.rng.uniform(-0.015, 0.015))
                    ratio = round(float(np.clip(base_ratio + noise, 0.81, 0.94)), 4)
                    orders.append(
                        BagOrderAction(
                            rfq_id=r.order_id,
                            buyer_name=r.company_name,
                            lot_id=lot.lot_id,
                            bags=1,
                            bid_price=round(lot.TV * ratio, 2),
                            target_ratio=ratio,
                            tv=lot.TV,
                            team_id=self.team_id,
                        )
                    )

                bags_left -= send_qty
                route_cap[j] -= send_qty

        return orders


# ==========================================
# 2. Greedy / Ceiling-Tester Bot: ตั้งราคาสูงปริ่มเพดาน ลุ้นกำไรสูงสุด
# ==========================================
class GreedyCeilingBot(BaseOpponentBot):
    """
    บอทสายโลภ (Greedy):
    ตั้งราคาชิดเพดานบนสุดของแต่ละกลุ่ม (Commercial ~ 1.028-1.036, Instant ~ 0.978-0.986)
    ยอมเสี่ยงโดนตีตก 'แพงเกินเพดาน' หรือขายไม่ออก เพื่อหวังฟันกำไรสูงสุดต่อกระสอบ
    """

    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        mask = build_action_mask(lots, routes)
        route_cap = {j: r.quota_bags for j, r in enumerate(routes)}
        orders: List[BagOrderAction] = []

        for i, lot in enumerate(lots):
            if lot.bags <= 0 or lot.TV <= 0:
                continue

            valid_routes = [
                j for j in range(len(routes)) if mask[i, j] and route_cap[j] > 0
            ]
            # เลือก Commercial ก่อน Instant เสมอเพื่อเอาเพดานสูงสุด
            valid_routes.sort(
                key=lambda j: 1 if routes[j].buyer_type == "COMMERCIAL" else 0,
                reverse=True,
            )

            bags_left = lot.bags
            for j in valid_routes:
                if bags_left <= 0:
                    break
                send_qty = min(bags_left, route_cap[j])
                r = routes[j]
                spec = BUYER_SPECS[r.buyer_type]

                for _ in range(send_qty):
                    target_ratio = round(
                        float(self.rng.uniform(spec["ratio_mid"], spec["ratio_max"] + 0.002)),
                        4,
                    )
                    orders.append(
                        BagOrderAction(
                            rfq_id=r.order_id,
                            buyer_name=r.company_name,
                            lot_id=lot.lot_id,
                            bags=1,
                            bid_price=round(lot.TV * target_ratio, 2),
                            target_ratio=target_ratio,
                            tv=lot.TV,
                            team_id=self.team_id,
                        )
                    )

                bags_left -= send_qty
                route_cap[j] -= send_qty

        return orders


# ==========================================
# 3. Quota-Proportional Bot: กระจายความเสี่ยงตามสัดส่วนโควตาผู้ซื้อ
# ==========================================
class QuotaProportionalBot(BaseOpponentBot):
    """
    บอทสายสมดุล (Quota-Proportional):
    แบ่งกระสอบกระจายไปทุก Route ที่ผ่านเกณฑ์ตามสัดส่วนโควตาที่เปิดรับ
    และตั้งราคาปานกลาง (0.93 - 0.99 * TV) เพื่อบาลานซ์โอกาสชนะกับกำไร
    """

    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        mask = build_action_mask(lots, routes)
        route_cap = {j: r.quota_bags for j, r in enumerate(routes)}
        orders: List[BagOrderAction] = []

        for i, lot in enumerate(lots):
            if lot.bags <= 0 or lot.TV <= 0:
                continue

            valid_routes = [
                j for j in range(len(routes)) if mask[i, j] and route_cap[j] > 0
            ]
            if not valid_routes:
                continue

            total_valid_quota = sum(route_cap[j] for j in valid_routes)
            weights = np.array([route_cap[j] / total_valid_quota for j in valid_routes])
            allocations = self.rng.multinomial(lot.bags, weights)

            for idx, j in enumerate(valid_routes):
                send_qty = min(int(allocations[idx]), route_cap[j])
                if send_qty <= 0:
                    continue

                r = routes[j]
                spec = BUYER_SPECS[r.buyer_type]
                mid_ceiling = spec["ratio_min"] - 0.025  # ต่ำกว่าเพดานเล็กน้อย

                for _ in range(send_qty):
                    ratio = round(float(self.rng.uniform(mid_ceiling - 0.02, mid_ceiling)), 4)
                    orders.append(
                        BagOrderAction(
                            rfq_id=r.order_id,
                            buyer_name=r.company_name,
                            lot_id=lot.lot_id,
                            bags=1,
                            bid_price=round(lot.TV * ratio, 2),
                            target_ratio=ratio,
                            tv=lot.TV,
                            team_id=self.team_id,
                        )
                    )
                route_cap[j] -= send_qty

        return orders


# ==========================================
# 4. Adaptive Competitor Bot: บอทฉลาดที่ใช้ Estimator + Router แบบเดียวกับเรา
# ==========================================
class AdaptiveCompetitorBot(BaseOpponentBot):
    """
    บอทคู่แข่งระดับสูง (Adaptive Bot):
    ใช้ OnlinePriceCeilingEstimator แกะรอยเพดาน และใช้ BagLevelCombinatorialRouter
    พร้อมปรับค่าความดุดัน (aggressiveness) อัตโนมัติตามอัตราการขายได้ในรอบที่ผ่านมา
    """

    def __init__(
        self,
        team_id: str,
        total_rounds: int = 7,
        base_aggressiveness: float = 0.55,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(team_id=team_id, total_rounds=total_rounds, rng=rng)
        self.base_aggressiveness = base_aggressiveness
        self.current_aggressiveness = base_aggressiveness
        self.estimator = OnlinePriceCeilingEstimator()
        self.risk_mgr = TerminalLiquidationRiskManager(total_rounds=total_rounds)
        self.router = BagLevelCombinatorialRouter(self.risk_mgr)

    def reset(self):
        self.current_aggressiveness = self.base_aggressiveness
        self.estimator = OnlinePriceCeilingEstimator()

    def observe_feedback(
        self,
        round_feedback: List[Dict],
        rfq_to_company: Dict[str, Tuple[str, str]],
        lot_tv_map: Dict[str, float],
    ):
        if not round_feedback:
            return

        # 1. อัปเดตเพดานราคาจากผลลัพธ์รอบล่าสุด
        self.estimator.update_from_round_results(
            round_feedback, rfq_to_company, lot_tv_map
        )

        # 2. ปรับลด/เพิ่มความดุดันตามอัตราส่วนที่ขายได้จริงในรอบล่าสุด
        submitted = sum(r.get("bags_submitted", 1) for r in round_feedback)
        sold = sum(r.get("bags_sold", 0) for r in round_feedback)
        if submitted > 0:
            win_rate = sold / submitted
            if win_rate < 0.4:
                # ขายไม่ค่อยออก -> ลดความดุดันลงเพื่อตัดราคาคู่แข่ง
                self.current_aggressiveness = max(0.10, self.current_aggressiveness - 0.15)
            elif win_rate > 0.85:
                # ขายได้ง่ายเกินไป -> ขยับราคาขึ้นใกล้เพดานเพื่อทำกำไรเพิ่ม
                self.current_aggressiveness = min(0.95, self.current_aggressiveness + 0.10)

    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        mask = build_action_mask(lots, routes)
        noise = float(self.rng.uniform(-0.05, 0.05))
        eff_aggr = float(np.clip(self.current_aggressiveness + noise, 0.05, 0.98))

        return self.router.route_bags(
            current_round=current_round,
            lots=lots,
            routes=routes,
            action_mask=mask,
            ceiling_estimator=self.estimator,
            rl_aggressiveness=eff_aggr,
            ladder_spread=0.015,
            urgency_multiplier=1.0,
            team_id=self.team_id,
        )


# ==========================================
# 5. Self-Play Checkpoint Bot: โหลดโมเดล RL ในอดีตมาแข่งกับตัวเอง
# ==========================================
class SelfPlayPolicyBot(BaseOpponentBot):
    """
    บอท Self-Play:
    รับฟังก์ชันทำนายหรือโมเดล Stable-Baselines3 (PPO / RecurrentPPO) จากอดีต
    มาแปลง State -> Action เข้า BagLevelCombinatorialRouter เพื่อแข่งกับ RL ตัวปัจจุบัน
    """

    def __init__(
        self,
        team_id: str,
        policy_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        total_rounds: int = 7,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(team_id=team_id, total_rounds=total_rounds, rng=rng)
        self.policy_predict_fn = policy_predict_fn
        self.estimator = OnlinePriceCeilingEstimator()
        self.risk_mgr = TerminalLiquidationRiskManager(total_rounds=total_rounds)
        self.router = BagLevelCombinatorialRouter(self.risk_mgr)
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def reset(self):
        self.estimator = OnlinePriceCeilingEstimator()
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def set_policy(self, policy_predict_fn: Callable[[np.ndarray], np.ndarray]):
        """อัปเดตน้ำหนักโมเดลจาก Checkpoint ล่าสุด"""
        self.policy_predict_fn = policy_predict_fn

    def observe_feedback(
        self,
        round_feedback: List[Dict],
        rfq_to_company: Dict[str, Tuple[str, str]],
        lot_tv_map: Dict[str, float],
    ):
        if not round_feedback:
            return
        self.estimator.update_from_round_results(
            round_feedback, rfq_to_company, lot_tv_map
        )
        submitted = sum(r.get("bags_submitted", 1) for r in round_feedback)
        sold = sum(r.get("bags_sold", 0) for r in round_feedback)
        exceeded = sum(
            r.get("bags_submitted", 1)
            for r in round_feedback
            if "เกินเพดาน" in r.get("status", "")
        )
        if submitted > 0:
            self.last_win_rate = sold / submitted
            self.last_exceed_rate = exceeded / submitted

    def _build_obs(
        self, current_round: int, lots: List[CoffeeLot], routes: List[BuyerOrder]
    ) -> np.ndarray:
        """สร้างเวกเตอร์ State 12 มิติให้ตรงกับ CoffeeAuctionEnv"""
        total_bags = sum(l.bags for l in lots)
        comm_bags = sum(l.bags for l in lots if l.can_sell_comm)
        inst_only_bags = sum(l.bags for l in lots if (not l.can_sell_comm and l.can_sell_inst))
        avg_tv = (sum(l.TV * l.bags for l in lots) / max(1, total_bags)) / 130.0

        total_demand = sum(r.quota_bags for r in routes if r.quota_bags > 0)
        comm_demand = sum(r.quota_bags for r in routes if r.buyer_type == "COMMERCIAL")
        inst_demand = sum(r.quota_bags for r in routes if r.buyer_type == "INSTANT")

        # สรุปสถานะเพดาน
        widths = [
            max(0.0, p.max_known_ratio - p.min_known_ratio)
            for p in self.estimator.profiles.values()
        ]
        avg_width = (float(np.mean(widths)) if widths else 0.020) / 0.020
        rand_ratio = (
            sum(1 for p in self.estimator.profiles.values() if p.is_randomized)
            / max(1, len(self.estimator.profiles))
        )

        return np.array(
            [
                current_round / max(1, self.total_rounds),
                min(2.0, total_bags / 30.0),
                comm_bags / max(1, total_bags),
                inst_only_bags / max(1, total_bags),
                avg_tv,
                min(3.0, total_demand / max(1, total_bags)),
                comm_demand / max(1, total_demand),
                inst_demand / max(1, total_demand),
                avg_width,
                rand_ratio,
                self.last_win_rate,
                self.last_exceed_rate,
            ],
            dtype=np.float32,
        )

    def act(
        self,
        current_round: int,
        lots: List[CoffeeLot],
        routes: List[BuyerOrder],
    ) -> List[BagOrderAction]:
        mask = build_action_mask(lots, routes)

        if self.policy_predict_fn is not None:
            obs = self._build_obs(current_round, lots, routes)
            raw_action = self.policy_predict_fn(obs)
            aggr = float(np.clip(raw_action[0], 0.0, 1.0))
            spread = float(np.clip(raw_action[1], 0.005, 0.030))
            urg_mult = float(np.clip(raw_action[2], 0.5, 1.5)) if len(raw_action) > 2 else 1.0
        else:
            # Fallback กรณียังไม่มี Checkpoint ให้เล่นแบบสมดุล
            aggr, spread, urg_mult = 0.5, 0.015, 1.0

        return self.router.route_bags(
            current_round=current_round,
            lots=lots,
            routes=routes,
            action_mask=mask,
            ceiling_estimator=self.estimator,
            rl_aggressiveness=aggr,
            ladder_spread=spread,
            urgency_multiplier=urg_mult,
            team_id=self.team_id,
        )


# ==========================================
# 6. Population Factory: สุ่มกองทัพคู่แข่งในแต่ละ Episode
# ==========================================
def create_opponent_population(
    num_opponents: int = 5,
    total_rounds: int = 7,
    rng: Optional[np.random.Generator] = None,
    self_play_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[BaseOpponentBot]:
    """
    สุ่มผสมบอทคู่แข่งหลายสไตล์ในแต่ละเกมเพื่อป้องกันไม่ให้ AI จำทาง (Overfitting)
    """
    rng = rng if rng is not None else np.random.default_rng()
    archetypes = ["DUMPER", "GREEDY", "PROPORTIONAL", "ADAPTIVE"]
    if self_play_fn is not None:
        archetypes.append("SELF_PLAY")

    bots: List[BaseOpponentBot] = []
    for idx in range(num_opponents):
        t_id = f"BOT_{idx + 1}"
        # รอบแรกๆ บังคับให้มีครบทุกสไตล์ ก่อนสุ่มเพิ่มเติม
        choice = archetypes[idx] if idx < len(archetypes) else str(rng.choice(archetypes))

        if choice == "DUMPER":
            bots.append(DumperBot(t_id, total_rounds=total_rounds, rng=rng))
        elif choice == "GREEDY":
            bots.append(GreedyCeilingBot(t_id, total_rounds=total_rounds, rng=rng))
        elif choice == "PROPORTIONAL":
            bots.append(QuotaProportionalBot(t_id, total_rounds=total_rounds, rng=rng))
        elif choice == "SELF_PLAY":
            bots.append(
                SelfPlayPolicyBot(
                    t_id, policy_predict_fn=self_play_fn, total_rounds=total_rounds, rng=rng
                )
            )
        else:
            aggr = float(rng.uniform(0.35, 0.75))
            bots.append(
                AdaptiveCompetitorBot(
                    t_id, total_rounds=total_rounds, base_aggressiveness=aggr, rng=rng
                )
            )

    return bots


# ==========================================
# ทดสอบรันการทำงานของกองทัพบอททั้ง 4 สไตล์ + Self-Play
# ==========================================
if __name__ == "__main__":
    rng = np.random.default_rng(123)
    sample_lots = [
        CoffeeLot("LOT-1", s=16, m=11.8, g=0.9, bags=6),
        CoffeeLot("LOT-2", s=14, m=12.5, g=4.5, bags=8),
        CoffeeLot("LOT-3", s=15, m=13.2, g=3.2, bags=7),
    ]
    sample_routes = [
        BuyerOrder("rt01", "COMMERCIAL", quota_bags=12, company_name="3 สยามคอมเมิร์ซ"),
        BuyerOrder("rt04", "INSTANT", quota_bags=20, company_name="6 ทรีอินวันฟู้ดส์"),
    ]

    # จำลองโมเดล Self-Play อย่างง่าย
    dummy_policy = lambda obs: np.array([0.75, 0.012, 1.1], dtype=np.float32)

    bots = create_opponent_population(
        num_opponents=5, total_rounds=7, rng=rng, self_play_fn=dummy_policy
    )

    print(f"{'Team ID':<8} | {'สไตล์บอท (Archetype)':<24} | {'จำนวนออเดอร์':<12} | {'Ratio เฉลี่ยที่เสนอ'}")
    print("-" * 72)
    for bot in bots:
        orders = bot.act(current_round=2, lots=sample_lots, routes=sample_routes)
        avg_r = np.mean([o.target_ratio for o in orders]) if orders else 0.0
        print(
            f"{bot.team_id:<8} | {bot.__class__.__name__:<24} | {len(orders):<12} | {avg_r:.4f}"
        )
