import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
from typing import List, Dict, Optional
from stable_baselines3 import PPO
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
        if route.route_id == "R16-CLEAR":
            return 0.800
        if route.company_name in self.randomized_companies:
            return self.TYPE_DEFAULTS.get(route.buyer_type, 1.00)
        default_c = 1.235 if (route.buyer_type == "SPECIALTY" and route.min_screen >= 17) else self.TYPE_DEFAULTS.get(route.buyer_type, 1.00)
        return self.company_ceilings.get(route.company_name, default_c)

    def update(self, results: List[Dict], route_map: Dict[str, BuyerRoute]):
        for res in results:
            rid = res.get("route_id")
            if rid == "R16-CLEAR":
                continue
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
    """
    คู่แข่งทุกสายพันธุ์ในสนาม 55 ทีม (อัปเกรดเผื่อ Buffer 0.5% ทุกตัว ไม่โดนตีตกสเปกฟรีๆ):
      - DUMPER: สัดส่วน 20% - 40% ของคู่แข่งทั้งหมด
      - QUOTA_V6, QUOTA_V3, QUOTA_UPGRADED, ADAPTIVE, GREEDY: เผื่อ Buffer 0.5% ครบทุกตัว
    """
    OTHER_ARCHETYPES = ["QUOTA_V6", "QUOTA_V3", "QUOTA_UPGRADED", "ADAPTIVE", "GREEDY"]

    def __init__(self, team_id: str, archetype: str, total_rounds: int, rng: random.Random):
        self.team_id = team_id
        self.archetype = archetype
        self.total_rounds = total_rounds
        self.rng = rng
        self.agg = 0.55
        self.ceiling_mem: Dict[str, float] = {}
        self.downgraded_lots: set = set()

    def reset(self):
        self.agg = 0.55
        self.ceiling_mem.clear()
        self.downgraded_lots.clear()

    def observe(self, my_results: List[Dict], route_map: Dict[str, BuyerRoute]):
        if not my_results:
            return
        sub = sum(r.get("qty", 0) for r in my_results)
        sold = sum(r.get("sold_qty", 0) for r in my_results)
        if sub > 0:
            wr = sold / sub
            self.agg = float(np.clip(self.agg + (0.08 if wr > 0.7 else -0.08), 0.1, 0.95))
        for r in my_results:
            st = r.get("status", "")
            rt = route_map.get(r["route_id"])
            lot = r.get("lot")
            if st == "CEILING_EXCEEDED" and rt and lot and lot.tv > 0:
                ratio = r["price"] / lot.tv
                self.ceiling_mem[rt.company_name] = min(self.ceiling_mem.get(rt.company_name, 1.30), ratio - 0.004)
            elif st == "SPEC_FAILED" and rt and lot and rt.buyer_type == "SPECIALTY":
                self.downgraded_lots.add(lot.lot_id)

    def _act_quota_v6(self, current_round: int, lots: List[CoffeeLot], routes: List[BuyerRoute], max_cap: int) -> List[Dict]:
        total_rem = sum(l.bags for l in lots)
        if total_rem <= 0 or max_cap <= 0 or not routes:
            return []

        effective_total = 15 if self.total_rounds >= 16 else self.total_rounds
        rounds_left = max(1, effective_total - current_round + 1)
        min_rounds_needed = math.ceil(total_rem / max_cap)
        cap_ratio = min(1.0, min_rounds_needed / rounds_left)
        progress = min(1.0, current_round / max(1, effective_total))
        urgency = max(progress, cap_ratio) if rounds_left > 1 else 1.0

        total_spec_quota = sum(r.quota for r in routes if r.buyer_type == "SPECIALTY")
        cap_left = {r.route_id: r.quota for r in routes}
        budget_left = max_cap
        orders: List[Dict] = []

        sorted_lots = sorted([l for l in lots if l.bags > 0], key=lambda x: x.tv, reverse=True)

        for lot in sorted_lots:
            if budget_left <= 0:
                break
            safe_g = lot.g * 1.005
            safe_m = lot.m * 1.005
            safe_s = lot.s * 0.995
            is_spec_lot = (
                lot.lot_id not in self.downgraded_lots
                and safe_g <= 3.0
                and safe_m <= 12.5
                and safe_s >= 16.0
            )
            lock_spec = is_spec_lot and (rounds_left > 2 or total_spec_quota >= 1)

            avail = []
            for r in routes:
                if cap_left[r.route_id] <= 0:
                    continue
                if lock_spec and r.buyer_type != "SPECIALTY":
                    continue
                if r.buyer_type == "SPECIALTY" and lot.lot_id in self.downgraded_lots:
                    continue
                if safe_g <= r.max_defect and safe_m <= r.max_moisture and (r.min_screen <= 0.0 or safe_s >= r.min_screen):
                    avail.append(r)

            if not avail:
                continue

            avail.sort(
                key=lambda r: (
                    0 if r.buyer_type == "SPECIALTY" else (1 if r.buyer_type == "COMMERCIAL" else 2),
                    -r.min_screen,
                    -cap_left[r.route_id],
                )
            )

            rem_bags = min(lot.bags, budget_left)
            for r in avail:
                if rem_bags <= 0 or budget_left <= 0:
                    break
                cap_r = cap_left[r.route_id]
                if cap_r <= 0:
                    continue

                if r.buyer_type == "SPECIALTY":
                    take = min(rem_bags, cap_r, budget_left, 2)
                else:
                    tot_q = sum(cap_left[x.route_id] for x in avail if cap_left[x.route_id] > 0)
                    share = max(1, int(round((cap_r / max(1, tot_q)) * rem_bags)))
                    take = min(rem_bags, cap_r, budget_left, share)

                if take <= 0:
                    continue

                known_ceil = self.ceiling_mem.get(r.company_name)
                if r.buyer_type == "SPECIALTY":
                    if r.min_screen >= 17:
                        top_ratio = 1.145 if cap_ratio >= 0.90 else min(1.205, 1.165 + 0.045 * progress)
                    else:
                        top_ratio = max(1.050, 1.110 - 0.050 * urgency)
                    if known_ceil:
                        top_ratio = min(top_ratio, known_ceil - 0.004)
                elif r.buyer_type == "COMMERCIAL":
                    base_r = min(1.014, known_ceil - 0.004) if known_ceil else 1.014
                    extra_cut = 0.008 if cap_ratio >= 0.90 else 0.0
                    top_ratio = max(0.955, base_r - 0.028 * urgency - extra_cut)
                else:
                    base_r = min(0.968, known_ceil - 0.004) if known_ceil else 0.968
                    extra_cut = 0.008 if cap_ratio >= 0.90 else 0.0
                    top_ratio = max(0.915, base_r - 0.028 * urgency - extra_cut)

                for step_i in range(take):
                    bid_ratio = max(0.81, round(top_ratio - 0.002 * step_i, 4))
                    orders.append({
                        "team_id": self.team_id,
                        "lot_id": lot.lot_id,
                        "lot": lot,
                        "route_id": r.route_id,
                        "qty": 1,
                        "price": round(lot.tv * bid_ratio, 2),
                        "buyer_type": r.buyer_type,
                    })

                rem_bags -= take
                budget_left -= take
                cap_left[r.route_id] -= take

        return orders

    def act(self, current_round: int, lots: List[CoffeeLot], routes: List[BuyerRoute], max_cap: int) -> List[Dict]:
        if routes and routes[0].route_id == "R16-CLEAR":
            orders = []
            for lot in lots:
                if lot.bags > 0:
                    orders.append({
                        "team_id": self.team_id,
                        "lot_id": lot.lot_id,
                        "lot": lot,
                        "route_id": "R16-CLEAR",
                        "qty": lot.bags,
                        "price": round(lot.tv * 0.795, 2),
                        "buyer_type": "INSTANT",
                    })
            return orders

        if self.archetype == "QUOTA_V6":
            return self._act_quota_v6(current_round, lots, routes, max_cap)

        total_rem = sum(l.bags for l in lots)
        if total_rem <= 0 or not routes:
            return []

        effective_total = 15 if self.total_rounds >= 16 else self.total_rounds
        rounds_left = max(1, effective_total - current_round + 1)
        urgency = max(min(1.0, current_round / max(1, effective_total)), min(1.0, math.ceil(total_rem / max(1, max_cap)) / rounds_left))
        cap_left = {r.route_id: r.quota for r in routes}
        orders: List[Dict] = []

        for lot in sorted([x for x in lots if x.bags > 0], key=lambda x: x.tv):
            safe_g = lot.g * 1.005
            safe_m = lot.m * 1.005
            safe_s = lot.s * 0.995

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
                else:
                    if r.buyer_type == "SPECIALTY":
                        base_r = 1.225 if r.min_screen >= 17 else 1.205
                        disc = 0.020 * urgency
                    elif r.buyer_type == "COMMERCIAL":
                        base_r = 1.012 if self.archetype == "QUOTA_V3" else 1.015
                        disc = 0.042 * urgency
                    else:
                        base_r = 0.962 if self.archetype == "QUOTA_V3" else 0.965
                        disc = 0.042 * urgency
                    if r.company_name in self.ceiling_mem:
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


class SelfPlayOpponentAgent:
    """บอท Self-Play: โหลดโมเดล PPO เวอร์ชันก่อนหน้ามาแข่งกับตัวปัจจุบันในสนามเดียวกัน"""
    def __init__(self, team_id: str, model: PPO, router: BagCombinatorialRouter):
        self.team_id = team_id
        self.archetype = "SELF_PLAY"
        self.model = model
        self.router = router
        self.ceiling_tracker = OnlineCeilingTracker()
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def reset(self):
        self.ceiling_tracker.reset()
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def observe(self, my_results: List[Dict], route_map: Dict[str, BuyerRoute]):
        if not my_results:
            return
        self.ceiling_tracker.update(my_results, route_map)
        sub_qty = sum(r.get("qty", 0) for r in my_results)
        sold_qty = sum(r.get("sold_qty", 0) for r in my_results)
        exc_qty = sum(r.get("qty", 0) for r in my_results if r.get("status") == "CEILING_EXCEEDED")
        if sub_qty > 0:
            self.last_win_rate = sold_qty / sub_qty
            self.last_exceed_rate = exc_qty / sub_qty

    def act(self, obs: np.ndarray, lots: List[CoffeeLot], routes: List[BuyerRoute], max_cap: int) -> List[Dict]:
        if routes and routes[0].route_id == "R16-CLEAR":
            return [
                {
                    "team_id": self.team_id,
                    "lot_id": l.lot_id,
                    "lot": l,
                    "route_id": "R16-CLEAR",
                    "qty": l.bags,
                    "price": round(l.tv * 0.795, 2),
                    "buyer_type": "INSTANT",
                }
                for l in lots if l.bags > 0
            ]

        action, _ = self.model.predict(obs, deterministic=False)
        sell_ratio = float(np.clip(action[0], 0.0, 1.0))
        comm_signal = float(np.clip(action[1], 0.0, 1.0))
        inst_signal = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else comm_signal

        comm_mult = 0.91 + (comm_signal * 0.13)
        inst_mult = 0.84 + (inst_signal * 0.145)
        target_bags = int(round(sell_ratio * max_cap))

        est_ceilings = {r.route_id: self.ceiling_tracker.get_estimated_ceiling(r) for r in routes}
        routed = self.router.route_bags(lots, routes, target_bags, comm_mult, est_ceilings, inst_price_multiplier=inst_mult)
        return [{"team_id": self.team_id, **o} for o in routed]


class CoffeeAuctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_teams: int = 55,
        total_rounds: int = 16,
        seed: int = None,
        self_play_model_path: Optional[str] = None,
        self_play_teams: int = 0,
    ):
        super().__init__()
        self.num_teams = num_teams
        self.total_rounds = total_rounds
        self.rng = random.Random(seed)

        self.company_pool = BuyerCompanyPool(seed=seed)
        self.clearing_engine = MarketClearingEngine(grade_error_std=0.005, cap_divisor=5)
        self.router = BagCombinatorialRouter(error_buffer=0.005)
        self.ceiling_tracker = OnlineCeilingTracker()

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(17,), dtype=np.float32)

        self.self_play_model: Optional[PPO] = None
        self.self_play_teams = max(0, min(num_teams - 1, self_play_teams))
        if self_play_model_path and self.self_play_teams > 0:
            self.self_play_model = PPO.load(str(self_play_model_path), device="cpu")

        self.opponents: Dict[str, OpponentArchetypeEngine] = {}
        self.self_play_agents: Dict[str, SelfPlayOpponentAgent] = {}
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0
        self.current_cap_divisor = 5

    def _create_base_game_portfolio(self) -> List[Dict]:
        num_lots = self.rng.randint(5, 8)
        total_bags = self.rng.randint(45, 56)

        base_bags = [4] * num_lots
        rem = total_bags - sum(base_bags)
        for _ in range(rem):
            base_bags[self.rng.randrange(num_lots)] += 1

        grade_presets = [
            (17.0, 11.4, 1.2),
            (16.0, 11.8, 2.2),
            (15.0, 12.1, 3.8),
            (14.0, 12.4, 5.5),
            (13.0, 13.1, 7.8),
            (12.0, 13.6, 10.5),
            (16.0, 11.6, 1.8),
            (14.0, 12.8, 6.5),
        ]
        self.rng.shuffle(grade_presets)

        template = []
        for idx in range(num_lots):
            s_b, m_b, g_b = grade_presets[idx % len(grade_presets)]
            s = float(np.clip(s_b + self.rng.choice([-1, 0, 0, 1]), 12, 18))
            m = round(float(np.clip(m_b + self.rng.uniform(-0.35, 0.35), 10.5, 14.2)), 2)
            g = round(float(np.clip(g_b + self.rng.uniform(-0.60, 0.60), 0.6, 13.5)), 2)
            template.append({"bags": base_bags[idx], "s": s, "m": m, "g": g})
        return template

    def _clone_portfolio_for_team(self, prefix: str, template: List[Dict], identical_mode: bool) -> List[CoffeeLot]:
        lots = []
        for idx, item in enumerate(template):
            if identical_mode:
                bags = item["bags"]
                s = item["s"]
                m = item["m"]
                g = item["g"]
            else:
                bags = max(3, item["bags"] + self.rng.choice([-1, 0, 0, 1]))
                s = float(np.clip(item["s"] + self.rng.choice([0, 0, 0, -1, 1]), 12, 18))
                m = round(float(np.clip(item["m"] + self.rng.uniform(-0.15, 0.15), 10.5, 14.5)), 2)
                g = round(float(np.clip(item["g"] + self.rng.uniform(-0.25, 0.25), 0.5, 14.0)), 2)
            lots.append(CoffeeLot(lot_id=f"LOT-{prefix}-{idx+1}", bags=bags, s=s, m=m, g=g))
        return lots

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.company_pool.rng.seed(seed + 10007)

        self.current_round = 1
        self.ceiling_tracker.reset()
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

        self.current_cap_divisor = self.rng.choices([5, 4], weights=[0.70, 0.30], k=1)[0]

        num_opps = self.num_teams - 1
        sp_count = self.self_play_teams if self.self_play_model is not None else 0
        rem_opps = max(0, num_opps - sp_count)

        # ปรับสัดส่วน DUMPER เป็น 20% - 40% ของคู่แข่งทั่วไป
        dumper_share = self.rng.uniform(0.20, 0.40)
        num_dumpers = int(round(rem_opps * dumper_share))

        archetype_list = ["SELF_PLAY"] * sp_count + ["DUMPER"] * num_dumpers
        other_pool = OpponentArchetypeEngine.OTHER_ARCHETYPES
        other_weights = [1.5 if a == "QUOTA_V6" else self.rng.uniform(0.7, 1.3) for a in other_pool]
        rem_count = max(0, num_opps - len(archetype_list))
        if rem_count > 0:
            archetype_list.extend(self.rng.choices(other_pool, weights=other_weights, k=rem_count))
        self.rng.shuffle(archetype_list)

        self.opponents = {}
        self.self_play_agents = {}
        for i in range(1, self.num_teams):
            tid = f"T{i:02d}"
            arch = archetype_list[i - 1]
            if arch == "SELF_PLAY" and self.self_play_model is not None:
                self.self_play_agents[tid] = SelfPlayOpponentAgent(tid, self.self_play_model, self.router)
            else:
                self.opponents[tid] = OpponentArchetypeEngine(tid, arch, self.total_rounds, self.rng)

        base_template = self._create_base_game_portfolio()
        identical_mode = (self.rng.random() < 0.5)

        self.team_lots = {
            f"T{i:02d}": self._clone_portfolio_for_team(f"T{i:02d}", base_template, identical_mode)
            for i in range(self.num_teams)
        }
        self.team_initial_bags = {
            tid: sum(l.bags for l in lots) for tid, lots in self.team_lots.items()
        }
        self.total_profit_above_tv = 0.0

        total_market_bags = sum(self.team_initial_bags.values())
        self.current_routes = self.company_pool.generate_round_routes(
            self.current_round, self.total_rounds, total_market_bags
        )
        return self._get_obs_for_team("T00", self.last_win_rate, self.last_exceed_rate), {}

    def _get_obs_for_team(self, tid: str, win_rate: float, exceed_rate: float) -> np.ndarray:
        lots = self.team_lots[tid]
        rem_bags = sum(l.bags for l in lots)
        init_bags = self.team_initial_bags[tid]
        max_round_cap = self.clearing_engine.get_max_sell_per_round(init_bags, self.current_cap_divisor)

        all_rem_bags = sum(sum(l.bags for l in t_lots) for t_lots in self.team_lots.values())
        total_demand = sum(r.quota for r in self.current_routes if r.route_id != "R16-CLEAR")

        spec_eligible_quota = 0
        for r in self.current_routes:
            if r.buyer_type == "SPECIALTY":
                if any(is_lot_eligible_for_route(l, r, 0.005) for l in lots):
                    spec_eligible_quota += r.quota

        avg_tv_norm = float(np.mean([l.tv for l in lots if l.bags > 0]) / 135.0) if rem_bags > 0 else 0.85
        spec_bags = sum(l.bags for l in lots if l.s * 0.995 >= 16.0 and l.g * 1.005 <= 3.0 and l.m * 1.005 <= 12.5)

        demand_mult, _, phase_code = self.company_pool.get_round_event_profile(self.current_round, self.total_rounds)
        auction_rounds = 15 if self.total_rounds >= 16 else self.total_rounds
        auction_rounds_left = max(0, auction_rounds - self.current_round + 1)

        min_rounds_needed = math.ceil(rem_bags / max(1, max_round_cap))
        cap_urgency = min(1.5, min_rounds_needed / max(1, auction_rounds_left))

        return np.array([
            self.current_round / max(1, self.total_rounds),
            auction_rounds_left / max(1, auction_rounds),
            rem_bags / max(1, init_bags),
            min(1.0, rem_bags / max(1, max_round_cap)),
            min(2.0, total_demand / max(1, all_rem_bags)),
            avg_tv_norm,
            spec_bags / max(1, rem_bags),
            min(1.0, spec_eligible_quota / max(1, max_round_cap)),
            sum(r.quota for r in self.current_routes if r.buyer_type == "COMMERCIAL") / max(1, total_demand),
            sum(r.quota for r in self.current_routes if r.buyer_type == "INSTANT" and r.route_id != "R16-CLEAR") / max(1, total_demand),
            1.0 if self.total_rounds >= 16 else 0.0,
            phase_code / 4.0,
            min(1.0, spec_eligible_quota / max(1, total_demand)),
            win_rate,
            exceed_rate,
            demand_mult / 1.5,
            cap_urgency,
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._get_obs_for_team("T00", self.last_win_rate, self.last_exceed_rate)

    def decode_action(self, action: np.ndarray):
        sell_ratio = float(np.clip(action[0], 0.0, 1.0))
        comm_signal = float(np.clip(action[1], 0.0, 1.0))
        inst_signal = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else comm_signal
        comm_mult = 0.91 + (comm_signal * 0.13)
        inst_mult = 0.84 + (inst_signal * 0.145)
        return sell_ratio, comm_mult, inst_mult

    def step(self, action):
        sell_ratio, comm_mult, inst_mult = self.decode_action(action)

        my_lots = self.team_lots["T00"]
        my_init_bags = self.team_initial_bags["T00"]
        is_junk_round = (self.current_routes and self.current_routes[0].route_id == "R16-CLEAR")

        if is_junk_round:
            my_orders = [
                {
                    "lot_id": l.lot_id,
                    "lot": l,
                    "route_id": "R16-CLEAR",
                    "qty": l.bags,
                    "price": round(l.tv * 0.795, 2),
                    "buyer_type": "INSTANT",
                }
                for l in my_lots if l.bags > 0
            ]
        else:
            max_cap = self.clearing_engine.get_max_sell_per_round(my_init_bags, self.current_cap_divisor)
            target_bags = int(round(sell_ratio * max_cap))
            est_ceilings = {
                r.route_id: self.ceiling_tracker.get_estimated_ceiling(r)
                for r in self.current_routes
            }
            my_orders = self.router.route_bags(
                my_lots, self.current_routes, target_bags, comm_mult, est_ceilings, inst_price_multiplier=inst_mult
            )

        all_bids = [{"team_id": "T00", **o} for o in my_orders]

        for tid, opp in self.opponents.items():
            opp_cap = self.clearing_engine.get_max_sell_per_round(self.team_initial_bags[tid], self.current_cap_divisor)
            opp_orders = opp.act(self.current_round, self.team_lots[tid], self.current_routes, opp_cap)
            all_bids.extend(opp_orders)

        for tid, sp_agent in self.self_play_agents.items():
            sp_cap = self.clearing_engine.get_max_sell_per_round(self.team_initial_bags[tid], self.current_cap_divisor)
            sp_obs = self._get_obs_for_team(tid, sp_agent.last_win_rate, sp_agent.last_exceed_rate)
            sp_orders = sp_agent.act(sp_obs, self.team_lots[tid], self.current_routes, sp_cap)
            all_bids.extend(sp_orders)

        results, _ = self.clearing_engine.clear_market(
            all_bids, self.current_routes, self.team_initial_bags, self.rng, cap_divisor=self.current_cap_divisor
        )

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

        if not is_junk_round:
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
        for tid, sp_agent in self.self_play_agents.items():
            sp_agent.observe(team_results[tid], route_map)

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
                self.current_round, self.total_rounds, total_market_bags
            )

        return self._get_obs(), float(reward), terminated, False, {
            "total_profit_above_tv": self.total_profit_above_tv
        }
