import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from typing import Any, Callable, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from coffee_math import CoffeeLot
from action_mask import BuyerOrder, build_action_mask
from ceiling_estimator import OnlinePriceCeilingEstimator
from bag_router import (
    TerminalLiquidationRiskManager,
    BagLevelCombinatorialRouter,
)
from clearing_engine import MarketClearingEngine, BuyerCompanyPool
from opponents import BaseOpponentBot, create_opponent_population


class CoffeeAuctionEnv(gym.Env):
    """
    สนามซ้อมตลาดประมูลเมล็ดกาแฟ 55 ทีม | ตลาดรับซื้อรวม 80% ของของทั้งหมด (V2.2)
    รองรับ:
      - Error ±0.5% ของค่า s, m, g ระหว่างผู้เล่นกับเซิร์ฟเวอร์
      - เพดานขายสูงสุดต่อรอบ = int(floor((initial_bags / 6) * 1.5))
      - Relative Market Outperformance Reward + File-Based Self-Play
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_rounds: int = 7,
        num_opponents: int = 54,
        market_demand_ratio: float = 0.80,
        ceiling_mode: str = "MIXED",
        self_play_model_path: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.total_rounds = total_rounds
        self.num_opponents = num_opponents
        self.market_demand_ratio = market_demand_ratio
        self.ceiling_mode = ceiling_mode
        self.self_play_model_path = self_play_model_path
        self.self_play_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.render_mode = render_mode

        if self_play_model_path:
            self.load_self_play_from_path(self_play_model_path)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-5.0, high=10.0, shape=(12,), dtype=np.float32
        )

        self.rng = np.random.default_rng()
        self.engine = MarketClearingEngine(rng=self.rng)
        self.company_pool: Optional[BuyerCompanyPool] = None
        self.opponents: List[BaseOpponentBot] = []

        self.my_team_id = "MY_TEAM"
        self.inventories: Dict[str, List[CoffeeLot]] = {}
        self.team_round_sell_caps: Dict[str, int] = {}
        self.current_round = 1
        self.round_quotas: List[int] = []
        self.current_routes: List[BuyerOrder] = []

        self.estimator = OnlinePriceCeilingEstimator()
        self.risk_mgr = TerminalLiquidationRiskManager(total_rounds=self.total_rounds)
        self.router = BagLevelCombinatorialRouter(self.risk_mgr)

        self.initial_total_bags = 0
        self.initial_total_tv = 0.0
        self.team_cumulative_profit_tv: Dict[str, float] = {}
        self.cumulative_revenue = 0.0
        self.cumulative_profit_over_tv = 0.0
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

    def load_self_play_from_path(self, model_path: str):
        path_obj = Path(model_path)
        if not path_obj.exists():
            return
        try:
            from stable_baselines3 import PPO
            sp_model = PPO.load(str(path_obj), device="cpu")
            self.self_play_model_path = str(path_obj)

            def _predict(obs: np.ndarray) -> np.ndarray:
                act, _ = sp_model.predict(obs, deterministic=True)
                return act

            self.self_play_fn = _predict
        except Exception:
            pass

    def _get_obs(self) -> np.ndarray:
        my_lots = self.inventories[self.my_team_id]
        total_bags = sum(l.bags for l in my_lots)
        comm_bags = sum(l.bags for l in my_lots if l.can_sell_comm)
        inst_only_bags = sum(
            l.bags for l in my_lots if (not l.can_sell_comm and l.can_sell_inst)
        )
        avg_tv_norm = (
            (sum(l.TV * l.bags for l in my_lots) / max(1, total_bags)) / 130.0
            if total_bags > 0
            else 0.0
        )

        total_demand = sum(r.quota_bags for r in self.current_routes if r.quota_bags > 0)
        comm_demand = sum(
            r.quota_bags for r in self.current_routes if r.buyer_type == "COMMERCIAL"
        )
        inst_demand = sum(
            r.quota_bags for r in self.current_routes if r.buyer_type == "INSTANT"
        )

        widths = [
            max(0.0, p.max_known_ratio - p.min_known_ratio)
            for p in self.estimator.profiles.values()
        ]
        avg_width_norm = (float(np.mean(widths)) if widths else 0.020) / 0.020
        rand_ratio = (
            sum(1 for p in self.estimator.profiles.values() if p.is_randomized)
            / max(1, len(self.estimator.profiles))
        )

        all_teams_rem_bags = sum(
            l.bags for lots in self.inventories.values() for l in lots
        )

        obs = np.array(
            [
                self.current_round / max(1, self.total_rounds),
                min(2.0, total_bags / 30.0),
                comm_bags / max(1, total_bags),
                inst_only_bags / max(1, total_bags),
                avg_tv_norm,
                min(5.0, total_demand / max(1, all_teams_rem_bags / max(1, 1 + self.num_opponents))),
                comm_demand / max(1, total_demand),
                inst_demand / max(1, total_demand),
                avg_width_norm,
                rand_ratio,
                self.last_win_rate,
                self.last_exceed_rate,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.engine = MarketClearingEngine(rng=self.rng)

        if self.ceiling_mode == "MIXED":
            ep_mode = "FIXED" if self.rng.random() < 0.6 else "RANDOMIZED"
        else:
            ep_mode = self.ceiling_mode

        self.company_pool = BuyerCompanyPool(rng=self.rng, ceiling_mode=ep_mode)

        self.opponents = create_opponent_population(
            num_opponents=self.num_opponents,
            total_rounds=self.total_rounds,
            rng=self.rng,
            self_play_fn=self.self_play_fn,
        )

        self.inventories = {
            self.my_team_id: self.engine.generate_team_lots(self.my_team_id)
        }
        for bot in self.opponents:
            bot.reset()
            self.inventories[bot.team_id] = self.engine.generate_team_lots(bot.team_id)

        self.team_round_sell_caps = {
            t_id: max(1, int(np.floor((sum(l.bags for l in lots) / 6.0) * 1.5)))
            for t_id, lots in self.inventories.items()
        }

        self.estimator = OnlinePriceCeilingEstimator()
        self.current_round = 1
        self.initial_total_bags = sum(l.bags for l in self.inventories[self.my_team_id])
        self.initial_total_tv = sum(
            l.TV * l.bags for l in self.inventories[self.my_team_id]
        )
        self.team_cumulative_profit_tv = {t_id: 0.0 for t_id in self.inventories}
        self.cumulative_revenue = 0.0
        self.cumulative_profit_over_tv = 0.0
        self.last_win_rate = 0.5
        self.last_exceed_rate = 0.0

        total_market_supply = sum(
            lot.bags for lots in self.inventories.values() for lot in lots
        )
        self.round_quotas = self.company_pool.plan_game_round_quotas(
            total_market_supply=total_market_supply,
            total_rounds=self.total_rounds,
            demand_ratio=self.market_demand_ratio,
        )

        self.current_routes = self.company_pool.generate_round_routes(
            round_total_quota=self.round_quotas[0]
        )

        info = {
            "ceiling_mode": ep_mode,
            "initial_bags": self.initial_total_bags,
            "max_sell_per_round": self.team_round_sell_caps[self.my_team_id],
            "initial_tv": round(self.initial_total_tv, 2),
            "total_market_supply": total_market_supply,
            "total_game_quota_80pct": sum(self.round_quotas),
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = np.clip(action, 0.0, 1.0)
        rl_aggressiveness = float(act[0])
        ladder_spread = float(0.005 + act[1] * 0.025)
        urgency_multiplier = float(0.50 + act[2] * 1.00)

        my_lots = self.inventories[self.my_team_id]
        mask = build_action_mask(my_lots, self.current_routes)

        my_orders = self.router.route_bags(
            current_round=self.current_round,
            lots=my_lots,
            routes=self.current_routes,
            action_mask=mask,
            ceiling_estimator=self.estimator,
            rl_aggressiveness=rl_aggressiveness,
            ladder_spread=ladder_spread,
            urgency_multiplier=urgency_multiplier,
            max_bags_per_round=self.team_round_sell_caps[self.my_team_id],
            team_id=self.my_team_id,
        )

        all_orders = list(my_orders)
        for bot in self.opponents:
            bot_lots = self.inventories[bot.team_id]
            all_orders.extend(
                bot.act(self.current_round, bot_lots, self.current_routes)
            )

        clearing = self.engine.clear_market(
            round_num=self.current_round,
            team_inventories=self.inventories,
            routes=self.current_routes,
            all_orders=all_orders,
            team_round_sell_caps=self.team_round_sell_caps,
        )

        rfq_to_company = {
            r.order_id: (r.company_name, r.buyer_type) for r in self.current_routes
        }
        my_tv_map = {l.lot_id: l.TV for l in my_lots}
        my_fb = clearing.team_feedback[self.my_team_id]
        self.estimator.update_from_round_results(my_fb, rfq_to_company, my_tv_map)

        opp_round_profits = []
        for t_id, prof in clearing.team_profit_over_tv.items():
            self.team_cumulative_profit_tv[t_id] += prof
            if t_id != self.my_team_id:
                opp_round_profits.append(prof)

        for bot in self.opponents:
            bot_tv_map = {l.lot_id: l.TV for l in self.inventories[bot.team_id]}
            bot.observe_feedback(
                clearing.team_feedback[bot.team_id], rfq_to_company, bot_tv_map
            )

        submitted = sum(r.get("bags_submitted", 1) for r in my_fb)
        sold = sum(r.get("bags_sold", 0) for r in my_fb)
        exceeded = sum(
            r.get("bags_submitted", 1)
            for r in my_fb
            if "เกินเพดาน" in r.get("status", "")
        )
        if submitted > 0:
            self.last_win_rate = sold / submitted
            self.last_exceed_rate = exceeded / submitted
        else:
            self.last_win_rate = 0.0
            self.last_exceed_rate = 0.0

        round_rev = clearing.team_revenue[self.my_team_id]
        round_profit_tv = clearing.team_profit_over_tv[self.my_team_id]
        self.cumulative_revenue += round_rev
        self.cumulative_profit_over_tv += round_profit_tv

        opp_mean_round_profit = float(np.mean(opp_round_profits)) if opp_round_profits else 0.0
        relative_round_edge = round_profit_tv - opp_mean_round_profit
        norm_scale = max(100.0, self.initial_total_tv * 0.05)

        step_reward = (
            0.5 * (round_profit_tv / norm_scale)
            + 0.5 * (relative_round_edge / norm_scale)
            - (exceeded * 0.025)
        )

        remaining_bags = sum(l.bags for l in my_lots)
        terminated = False
        truncated = False
        liq_loss = 0.0

        if self.current_round >= self.total_rounds or remaining_bags <= 0:
            terminated = True
            liq_res = self.engine.liquidate_remaining_inventory(
                self.inventories, liquidation_ratio=0.80
            )
            for t_id, (l_rev, l_prof) in liq_res.items():
                self.team_cumulative_profit_tv[t_id] += l_prof

            my_liq_rev, my_liq_prof = liq_res[self.my_team_id]
            self.cumulative_revenue += my_liq_rev
            self.cumulative_profit_over_tv += my_liq_prof
            liq_loss = abs(my_liq_prof)

            if remaining_bags > 0:
                step_reward += (my_liq_prof / norm_scale) * 1.2
            else:
                step_reward += 0.35

            sorted_scores = sorted(
                self.team_cumulative_profit_tv.values(), reverse=True
            )
            my_final_score = self.team_cumulative_profit_tv[self.my_team_id]
            my_rank = sorted_scores.index(my_final_score) + 1
            total_t = len(sorted_scores)
            rank_bonus = 0.50 * (1.0 - 2.0 * (my_rank - 1) / max(1, total_t - 1))
            step_reward += rank_bonus
        else:
            self.current_round += 1
            self.current_routes = self.company_pool.generate_round_routes(
                round_total_quota=self.round_quotas[self.current_round - 1]
            )

        info = {
            "round": self.current_round,
            "sold_this_round": sold,
            "exceeded_this_round": exceeded,
            "remaining_bags": sum(l.bags for l in my_lots),
            "cumulative_revenue": round(self.cumulative_revenue, 2),
            "cumulative_profit_over_tv": round(self.cumulative_profit_over_tv, 2),
            "liquidation_loss": round(liq_loss, 2),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(step_reward), terminated, truncated, info

    def render(self):
        rem = sum(l.bags for l in self.inventories[self.my_team_id])
        print(
            f"[Round {self.current_round}/{self.total_rounds}] "
            f"เหลือ: {rem} กระสอบ | รายได้สะสม: {self.cumulative_revenue:,.2f} บ. | "
            f"กำไรเหนือ TV: {self.cumulative_profit_over_tv:+,.2f} บ."
        )


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = CoffeeAuctionEnv(
        total_rounds=7,
        num_opponents=54,
        market_demand_ratio=0.80,
        ceiling_mode="MIXED",
    )
    check_env(env)
    obs, info = env.reset(seed=42)
    assert obs is not None and len(obs) == 12, "obs error!"
    print(
        f"✅ check_env ผ่านสมบูรณ์ 100%! ของเริ่มต้น={info['initial_bags']} ใบ | "
        f"เพดานขายสูงสุดต่อรอบ={info['max_sell_per_round']} ใบ"
    )