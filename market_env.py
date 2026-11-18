import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from typing import List, Dict
from coffee_math import CoffeeLot, BuyerRoute
from clearing_engine import BuyerCompanyPool, MarketClearingEngine
from bag_router import BagCombinatorialRouter
from action_mask import is_lot_eligible_for_route

class OnlineCeilingTracker:
    """ระบบจำเพดานราคาผู้ซื้อจาก Feedback รอบก่อน (ไม่แอบดูเฉลยของเซิร์ฟเวอร์)"""
    TYPE_DEFAULTS = {
        "SPECIALTY": 1.225,
        "COMMERCIAL": 1.028,
        "INSTANT": 0.978,
    }

    def __init__(self):
        self.company_ceilings: Dict[str, float] = {}
        self.company_min_passed: Dict[str, float] = {}
        self.randomized_companies: set = set()

    def reset(self):
        self.company_ceilings.clear()
        self.company_min_passed.clear()
        self.randomized_companies.clear()

    def get_estimated_ceiling(self, route: BuyerRoute) -> float:
        if route.company_name in self.randomized_companies:
            return self.TYPE_DEFAULTS.get(route.buyer_type, 1.00)
        default_c = 1.235 if (route.buyer_type == "SPECIALTY" and route.min_screen >= 17) else self.TYPE_DEFAULTS.get(route.buyer_type, 1.00)
        return self.company_ceilings.get(route.company_name, default_c)

    def update(self, results: List[Dict], route_map: Dict[str, BuyerRoute]):
        for res in results:
            rid = res.get("route_id")
            route = route_map.get(rid)
            lot: CoffeeLot = res.get("lot")
            if not route or not lot or lot.tv <= 0:
                continue
            ratio = res["price"] / lot.tv
            status = res.get("status", "")
            cname = route.company_name

            if status == "CEILING_EXCEEDED":
                min_ok = self.company_min_passed.get(cname, 0.0)
                if ratio <= min_ok + 1e-4:
                    self.randomized_companies.add(cname)
                else:
                    prev = self.company_ceilings.get(cname, 1.30)
                    self.company_ceilings[cname] = min(prev, ratio - 0.002)
            elif status in ("SOLD", "PARTIAL_SOLD", "QUOTA_FULL"):
                known_ceil = self.company_ceilings.get(cname, 1.30)
                if ratio >= known_ceil - 1e-4:
                    self.randomized_companies.add(cname)
                else:
                    prev_ok = self.company_min_passed.get(cname, 0.0)
                    self.company_min_passed[cname] = max(prev_ok, ratio)


class OpponentArchetypeEngine:
    """จำลองคู่แข่ง 5 สายพันธุ์ตาม benchmark_new_rules.py (รู้กติกาใหม่ 100%)"""
    ARCHETYPES = ["QUOTA_V3", "QUOTA_UPGRADED", "ADAPTIVE", "DUMPER", "GREEDY"]

    def __init__(self, team_id: str, archetype: str, total_rounds: int, rng: random.Random):
        self.team_id = team_id
        self.archetype = archetype
        self.total_rounds = total_rounds
        self.rng = rng
        self.agg = 0.55
        self.ceiling_mem: Dict[str, float] = {}

    def reset(self):
        self.agg = 0.55
        self.ceiling_mem.clear()

    def observe(self, my_results: List[Dict], route_map: Dict[str, BuyerRoute]):
        if not my_results:
            return
        sub = sum(r.get("qty", 0) for r in my_results)
        sold = sum(r.get("sold_qty", 0) for r in my_results)
        if sub > 0:
            wr = sold / sub
            self.agg = float(np.clip(self.agg + (0.08 if wr > 0.7 else -0.08), 0.1, 0.95))
        for r in my_results:
            if r.get("status") == "CEILING_EXCEEDED":
                rt = route_map.get(r["route_id"])
                lot = r.get("lot")
                if rt and lot and lot.tv > 0:
                    ratio = r["price"] / lot.tv
                    self.ceiling_mem[rt.company_name] = min(self.ceiling_mem.get(rt.company_name, 1.30), ratio - 0.002)

    def act(self, current_round: int, lots: List[CoffeeLot], routes: List[BuyerRoute], init_bags: int) -> List[Dict]:
        total_rem = sum(l.bags for l in lots)
        if total_rem <= 0 or not routes:
            return []

        max_cap = max(1, init_bags // 4)
        rounds_left = max(1, self.total_rounds - current_round + 1)
        urgency = max(current_round / self.total_rounds, min(1.0, math.ceil(total_rem / max_cap) / rounds_left))
        cap_left = {r.route_id: r.quota for r in routes}
        orders: List[Dict] = []

        for lot in sorted([x for x in lots if x.bags > 0], key=lambda x: x.tv):
            use_buf = (self.archetype == "QUOTA_V3")
            safe_g = lot.g * (1.005 if use_buf else 1.0)
            safe_m = lot.m * (1.005 if use_buf else 1.0)
            safe_s = lot.s * (0.995 if use_buf else 1.0)

            avail = [
                r for r in routes
                if cap_left[r.route_id] > 0
                and safe_g <= r.max_defect
                and safe_m <= r.max_moisture
                and (r.min_screen <= 0.0 or safe_s >= r.min_screen)
            ]
            if not avail:
                continue

            avail.sort(
                key=lambda r: 1.22 if r.buyer_type == "SPECIALTY" else (1.03 if r.buyer_type == "COMMERCIAL" else 0.98),
                reverse=True
            )
            rem = lot.bags
            for r in avail:
                if rem <= 0:
                    break
                take = min(rem, cap_left[r.route_id])
                if self.archetype == "DUMPER":
                    ratio = self.rng.uniform(1.05, 1.12) if r.buyer_type == "SPECIALTY" else self.rng.uniform(0.86, 0.91)
                elif self.archetype == "GREEDY":
                    ratio = 1.235 if r.buyer_type == "SPECIALTY" else (1.032 if r.buyer_type == "COMMERCIAL" else 0.982)
                elif self.archetype == "ADAPTIVE":
                    ceil = 1.22 if r.buyer_type == "SPECIALTY" else (1.028 if r.buyer_type == "COMMERCIAL" else 0.978)
                    ratio = max(0.81, ceil - (1.0 - self.agg) * 0.04 - 0.05 * urgency)
                else:  # QUOTA_V3 & QUOTA_UPGRADED
                    if r.buyer_type == "SPECIALTY":
                        base_r = 1.225 if r.min_screen >= 17 else 1.205
                        disc = 0.020 * urgency
                    elif r.buyer_type == "COMMERCIAL":
                        base_r = 1.012 if self.archetype == "QUOTA_V3" else 1.015
                        disc = 0.042 * urgency
                    else:
                        base_r = 0.962 if self.archetype == "QUOTA_V3" else 0.965
                        disc = 0.042 * urgency
                    if self.archetype == "QUOTA_V3" and r.company_name in self.ceiling_mem:
                        base_r = min(base_r, self.ceiling_mem[r.company_name] - 0.006)
                    ratio = max(0.815, base_r - disc)

                orders.append({
                    "team_id": self.team_id,
                    "lot_id": lot.lot_id,
                    "lot": lot,
                    "route_id": r.route_id,
                    "qty": take,
                    "price": round(lot.tv * ratio, 2),
                    "buyer_type": r.buyer_type
                })
                rem -= take
                cap_left[r.route_id] -= take
        return orders


class CoffeeAuctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_teams: int = 55, total_rounds: int = 9, seed: int = None):
        super().__init__()
        self.num_teams = num_teams
        self.total_rounds = total_rounds
        self.rng = random.Random(seed)

        self.company_pool = BuyerCompanyPool(seed=seed)
        self.clearing_engine = MarketClearingEngine(grade_error_std=0.005)
        self.router = BagCombinatorialRouter(error_buffer=0.005)
        self.ceiling_tracker = OnlineCeilingTracker()

        # Action Space: [สัดส่วนการขายเทียบกับเพดาน 1/4 ของรอบ (0..1), ตัวคูณระดับราคา (0..1)]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation Space: 15 ตัวแปร Normalized (เพิ่ม Win Rate และ Ceiling Confidence)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(15,), dtype=np.float32)

        self.opponents: Dict[str, OpponentArchetypeEngine] = {}
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def _generate_team_lots(self, prefix: str) -> List[CoffeeLot]:
        lots = []
        for idx in range(4):
            bags = self.rng.randint(6, 8)
            s = float(self.rng.choices([14, 15, 16, 17, 18], weights=[15, 30, 30, 18, 7])[0])
            m = round(self.rng.uniform(10.8, 13.4), 2)
            g = round(self.rng.uniform(0.8, 6.5), 2)
            lots.append(CoffeeLot(lot_id=f"LOT-{prefix}-{idx+1}", bags=bags, s=s, m=m, g=g))
        return lots

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        self.current_round = 1
        self.event_phase_offset = self.rng.choice([0, 4])
        self.ceiling_tracker.reset()
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

        # Domain Randomization: สุ่มสัดส่วนบอทคู่แข่ง 5 สายพันธุ์ในแต่ละเกม
        weights = [self.rng.uniform(0.5, 1.5) for _ in OpponentArchetypeEngine.ARCHETYPES]
        self.opponents = {}
        for i in range(1, self.num_teams):
            tid = f"T{i:02d}"
            arch = self.rng.choices(OpponentArchetypeEngine.ARCHETYPES, weights=weights, k=1)[0]
            self.opponents[tid] = OpponentArchetypeEngine(tid, arch, self.total_rounds, self.rng)

        self.team_lots = {f"T{i:02d}": self._generate_team_lots(f"T{i:02d}") for i in range(self.num_teams)}
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

        spec_eligible_quota = 0
        for r in self.current_routes:
            if r.buyer_type == "SPECIALTY":
                if any(is_lot_eligible_for_route(l, r, 0.005) for l in my_lots):
                    spec_eligible_quota += r.quota

        avg_tv_norm = float(np.mean([l.tv for l in my_lots if l.bags > 0]) / 135.0) if my_rem_bags > 0 else 0.85
        spec_bags = sum(l.bags for l in my_lots if l.s * 0.995 >= 16.0 and l.g * 1.005 <= 3.0 and l.m * 1.005 <= 12.5)

        obs = np.array([
            self.current_round / self.total_rounds,
            (self.total_rounds - self.current_round) / self.total_rounds,
            my_rem_bags / max(1, my_init_bags),
            min(1.0, my_rem_bags / max(1, max_round_cap)),
            min(2.0, total_demand / max(1, all_rem_bags)),
            avg_tv_norm,
            spec_bags / max(1, my_rem_bags),
            min(1.0, spec_eligible_quota / max(1, max_round_cap)),
            sum(r.quota for r in self.current_routes if r.buyer_type == "COMMERCIAL") / max(1, total_demand),
            sum(r.quota for r in self.current_routes if r.buyer_type == "INSTANT") / max(1, total_demand),
            1.0 if self.total_rounds >= 16 else 0.0,
            ((self.current_round - 1) % 4) / 4.0,
            min(1.0, spec_eligible_quota / max(1, total_demand)),
            self.last_win_rate,
            self.last_exceed_rate,
        ], dtype=np.float32)

        return obs

    def step(self, action):
        sell_ratio = float(np.clip(action[0], 0.0, 1.0))
        price_signal = float(np.clip(action[1], 0.0, 1.0))

        my_lots = self.team_lots["T00"]
        my_init_bags = self.team_initial_bags["T00"]
        max_cap = self.clearing_engine.get_max_sell_per_round(my_init_bags)

        target_bags = int(round(sell_ratio * max_cap))
        price_mult = 0.94 + (price_signal * 0.10)

        # ใช้เพดานที่เรียนรู้จาก Feedback เท่านั้น (ไม่แอบดู r.base_ceiling_mult ของเซิร์ฟเวอร์)
        est_ceilings = {
            r.route_id: self.ceiling_tracker.get_estimated_ceiling(r)
            for r in self.current_routes
        }
        my_orders = self.router.route_bags(my_lots, self.current_routes, target_bags, price_mult, est_ceilings)
        all_bids = [{"team_id": "T00", **o} for o in my_orders]

        # บอทคู่แข่ง 54 ทีม (5 สายพันธุ์รู้กติกาใหม่)
        for tid, opp in self.opponents.items():
            opp_orders = opp.act(self.current_round, self.team_lots[tid], self.current_routes, self.team_initial_bags[tid])
            all_bids.extend(opp_orders)

        results, _ = self.clearing_engine.clear_market(
            all_bids, self.current_routes, self.team_initial_bags, self.rng
        )

        # แยกผลลัพธ์รายทีมเพื่ออัปเดตหน่วยความจำเพดานและคำนวณ Reward
        route_map = {r.route_id: r for r in self.current_routes}
        team_results: Dict[str, List[Dict]] = {f"T{i:02d}": [] for i in range(self.num_teams)}
        round_profit_above_tv = 0.0

        for res in results:
            tid = res["team_id"]
            if tid in team_results:
                team_results[tid].append(res)
            if res.get("sold_qty", 0) > 0:
                lot_obj: CoffeeLot = res["lot"]
                lot_obj.bags -= res["sold_qty"]
                if tid == "T00":
                    profit = (res["price"] - lot_obj.tv) * res["sold_qty"]
                    round_profit_above_tv += profit

        # อัปเดตความจำเพดานของทีมเราและคู่แข่ง
        my_res = team_results["T00"]
        self.ceiling_tracker.update(my_res, route_map)
        sub_qty = sum(r.get("qty", 0) for r in my_res)
        sold_qty = sum(r.get("sold_qty", 0) for r in my_res)
        exc_qty = sum(r.get("qty", 0) for r in my_res if r.get("status") == "CEILING_EXCEEDED")
        if sub_qty > 0:
            self.last_win_rate = sold_qty / sub_qty
            self.last_exceed_rate = exc_qty / sub_qty

        for tid, opp in self.opponents.items():
            opp.observe(team_results[tid], route_map)

        self.total_profit_above_tv += round_profit_above_tv
        reward = round_profit_above_tv / 100.0

        terminated = False
        if self.current_round >= self.total_rounds:
            terminated = True
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
