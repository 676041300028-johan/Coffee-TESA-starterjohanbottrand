import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List
from coffee_math import CoffeeLot, calculate_tv
from clearing_engine import BuyerCompanyPool, MarketClearingEngine
from bag_router import BagCombinatorialRouter
from action_mask import is_lot_eligible_for_route

class CoffeeAuctionEnv(gym.Env):
    def __init__(self, num_teams: int = 55, total_rounds: int = 9, seed: int = None):
        super().__init__()
        self.num_teams = num_teams
        self.total_rounds = total_rounds
        self.rng = random.Random(seed)

        self.company_pool = BuyerCompanyPool(seed=seed)
        self.clearing_engine = MarketClearingEngine(grade_error_std=0.005)
        self.router = BagCombinatorialRouter(error_buffer=0.005)

        # Action Space: [สัดส่วนการขายเทียบกับเพดาน 1/4 ของรอบ (0..1), ตัวคูณระดับราคา (0..1)]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space: 13 ตัวแปร Normalized (0.0 - 1.0)
        # ตัวที่ 13 คือ specialty_demand_ratio (สัดส่วนโควตา Specialty ที่เราขายได้ในรอบนี้)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(13,), dtype=np.float32)

    def _generate_team_lots(self) -> List[CoffeeLot]:
        lots = []
        for idx in range(4):
            bags = self.rng.randint(6, 8)
            # สุ่มเกรดให้มีโอกาสติด Specialty (s=16, 17, 18) ประมาณ 25-30%
            s = float(self.rng.choices([14, 15, 16, 17, 18], weights=[15, 30, 30, 18, 7])[0])
            m = round(self.rng.uniform(10.8, 13.4), 2)
            g = round(self.rng.uniform(0.8, 6.5), 2)
            lots.append(CoffeeLot(lot_id=f"LOT-T00-{idx+1}", bags=bags, s=s, m=m, g=g))
        return lots

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.current_round = 1
        # สุ่มจุดเริ่มต้น Event Wave (เริ่มช่วงซบเซา 4 รอบก่อน หรือเริ่มช่วงบูม 4 รอบก่อน)
        self.event_phase_offset = self.rng.choice([0, 4])

        self.team_lots = {f"T{i:02d}": self._generate_team_lots() for i in range(self.num_teams)}
        self.team_initial_bags = {
            tid: sum(l.bags for l in lots) for tid, lots in self.team_lots.items()
        }
        self.total_profit_above_tv = 0.0

        total_market_bags = sum(self.team_initial_bags.values())
        self.current_routes = self.company_pool.generate_round_routes(
            self.current_round, self.total_rounds, total_market_bags, self.event_phase_offset
        )
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        my_lots = self.team_lots["T00"]
        my_rem_bags = sum(l.bags for l in my_lots)
        my_init_bags = self.team_initial_bags["T00"]
        max_round_cap = self.clearing_engine.get_max_sell_per_round(my_init_bags)

        all_rem_bags = sum(sum(l.bags for l in lots) for lots in self.team_lots.values())
        total_demand = sum(r.quota for r in self.current_routes)

        # คำนวณสัดส่วนโควตา Specialty ที่ Lot ของเราเข้าเงื่อนไขได้จริง
        spec_eligible_quota = 0
        for r in self.current_routes:
            if r.buyer_type == "SPECIALTY":
                if any(is_lot_eligible_for_route(l, r, 0.005) for l in my_lots):
                    spec_eligible_quota += r.quota

        avg_tv_norm = np.mean([l.tv for l in my_lots if l.bags > 0]) / 135.0 if my_rem_bags > 0 else 0.85
        spec_bags = sum(l.bags for l in my_lots if l.s * 0.995 >= 16.0 and l.g * 1.005 <= 3.0 and l.m * 1.005 <= 12.5)

        obs = np.array([
            self.current_round / self.total_rounds,                       # 1. ความคืบหน้ารอบ
            (self.total_rounds - self.current_round) / self.total_rounds, # 2. รอบที่เหลือ
            my_rem_bags / max(1, my_init_bags),                           # 3. สัดส่วนสต็อกที่เหลือ
            min(1.0, my_rem_bags / max(1, max_round_cap)),                # 4. ความพร้อมขายเต็มเพดาน 1/4
            min(2.0, total_demand / max(1, all_rem_bags)),                # 5. ความตึงตัวของดีมานด์รวม (จับ Event Wave!)
            avg_tv_norm,                                                  # 6. คุณภาพเฉลี่ยของพอร์ต
            spec_bags / max(1, my_rem_bags),                              # 7. สัดส่วนของเกรด Specialty ในมือ
            min(1.0, spec_eligible_quota / max(1, max_round_cap)),        # 8. โอกาสยิงเข้า Specialty รอบนี้
            sum(r.quota for r in self.current_routes if r.buyer_type == "COMMERCIAL") / max(1, total_demand), # 9
            sum(r.quota for r in self.current_routes if r.buyer_type == "INSTANT") / max(1, total_demand),    # 10
            1.0 if self.total_rounds >= 16 else 0.0,                      # 11. โหมดเกมยาว 16 รอบ
            ((self.current_round - 1) % 4) / 4.0,                         # 12. ตำแหน่งภายในบล็อก Event 4 รอบ
            min(1.0, spec_eligible_quota / max(1, total_demand))          # 13. specialty_demand_ratio
        ], dtype=np.float32)

        return obs

    def step(self, action):
        sell_ratio = float(np.clip(action[0], 0.0, 1.0))
        price_signal = float(np.clip(action[1], 0.0, 1.0))

        my_lots = self.team_lots["T00"]
        my_init_bags = self.team_initial_bags["T00"]
        max_cap = self.clearing_engine.get_max_sell_per_round(my_init_bags)

        target_bags = int(round(sell_ratio * max_cap))
        # แปลง price_signal (0..1) เป็นตัวคูณราคาฐาน 0.96 - 1.045 (ส่วน Specialty Router จะดันขึ้น 1.21-1.235 เอง)
        price_mult = 0.96 + (price_signal * 0.085)

        ceilings = {r.route_id: r.base_ceiling_mult for r in self.current_routes}
        my_orders = self.router.route_bags(my_lots, self.current_routes, target_bags, price_mult, ceilings)

        all_bids = [{"team_id": "T00", **o} for o in my_orders]

        # จำลองบอทคู่แข่ง 54 ทีม
        for i in range(1, self.num_teams):
            tid = f"T{i:02d}"
            opp_lots = self.team_lots[tid]
            opp_cap = self.clearing_engine.get_max_sell_per_round(self.team_initial_bags[tid])
            opp_target = int(round(self.rng.uniform(0.55, 1.0) * opp_cap))
            opp_mult = self.rng.uniform(0.97, 1.035)
            opp_orders = self.router.route_bags(opp_lots, self.current_routes, opp_target, opp_mult, ceilings)
            for o in opp_orders:
                all_bids.append({"team_id": tid, **o})

        results, _ = self.clearing_engine.clear_market(
            all_bids, self.current_routes, self.team_initial_bags, self.rng
        )

        # อัปเดตสต็อกและคำนวณ Reward
        round_profit_above_tv = 0.0
        for res in results:
            if res.get("sold_qty", 0) > 0:
                tid = res["team_id"]
                lot_obj: CoffeeLot = res["lot"]
                lot_obj.bags -= res["sold_qty"]
                if tid == "T00":
                    profit = (res["price"] - lot_obj.tv) * res["sold_qty"]
                    round_profit_above_tv += profit

        self.total_profit_above_tv += round_profit_above_tv
        reward = round_profit_above_tv / 100.0  # Scale reward ให้ PPO เทรนนิ่งนิ่งขึ้น

        terminated = False
        if self.current_round >= self.total_rounds:
            terminated = True
            # บังคับขายเลหลัง 0.8 * TV สำหรับของที่เหลือในรอบสุดท้าย (ขาดทุน 20% ทันที)
            for lot in my_lots:
                if lot.bags > 0:
                    liq_loss = (0.80 * lot.tv - lot.tv) * lot.bags
                    self.total_profit_above_tv += liq_loss
                    reward += liq_loss / 100.0
                    lot.bags = 0
        else:
            self.current_round += 1
            total_market_bags = sum(self.team_initial_bags.values())
            self.current_routes = self.company_pool.generate_round_routes(
                self.current_round, self.total_rounds, total_market_bags, self.event_phase_offset
            )

        return self._get_obs(), float(reward), terminated, False, {
            "total_profit_above_tv": self.total_profit_above_tv
        }