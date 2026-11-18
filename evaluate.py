import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from typing import Dict, List
import numpy as np

try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False

from stable_baselines3 import PPO
from coffee_math import CoffeeLot
from action_mask import BuyerOrder
from clearing_engine import MarketClearingEngine, BuyerCompanyPool
from opponents import (
    BaseOpponentBot,
    DumperBot,
    GreedyCeilingBot,
    QuotaProportionalBot,
    AdaptiveCompetitorBot,
    SelfPlayPolicyBot,
)


def load_trained_model(model_path: Path):
    """โหลดโมเดลที่เทรนเสร็จแล้ว (รองรับทั้ง PPO และ RecurrentPPO)"""
    if not model_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดลที่ {model_path} — กรุณารัน train.py ให้เสร็จก่อน"
        )
    if HAS_RECURRENT:
        try:
            return RecurrentPPO.load(str(model_path)), True
        except Exception:
            pass
    return PPO.load(str(model_path)), False


def make_policy_predict_fn(model, is_recurrent: bool):
    """สร้าง Wrapper สำหรับรัน Inference ทีละ Step (จัดการ LSTM State อัตโนมัติ)"""
    state = {"lstm": None, "ep_start": np.ones((1,), dtype=bool)}

    def reset_state():
        state["lstm"] = None
        state["ep_start"] = np.ones((1,), dtype=bool)

    def predict_fn(obs: np.ndarray) -> np.ndarray:
        if is_recurrent:
            action, state["lstm"] = model.predict(
                obs,
                state=state["lstm"],
                episode_start=state["ep_start"],
                deterministic=True,
            )
            state["ep_start"] = np.zeros((1,), dtype=bool)
            return action
        action, _ = model.predict(obs, deterministic=True)
        return action

    return predict_fn, reset_state


def run_55_teams_tournament(
    num_games: int = 50,
    total_teams: int = 55,
    demand_ratio: float = 0.80,
    total_rounds: int = 7,
    seed_start: int = 5000,
):
    """
    แข่งขันทัวร์นาเมนต์ 55 ทีมพร้อมกันในตลาดเดียวกัน (Head-to-Head Arena):
      - จำลอง Error ±0.5% ของ s, m, g
      - บังคับเพดานขายสูงสุดต่อรอบ = floor((initial_bags / 6) * 1.5)
      - ตลาดรับซื้อรวมตลอดเกม = 80% ของของทั้งหมดในมือทั้ง 55 ทีม
    """
    best_path = CURRENT_DIR / "models" / "best" / "best_model.zip"
    v2_final_path = CURRENT_DIR / "models" / "coffee_rl_v2_final.zip"
    v1_final_path = CURRENT_DIR / "models" / "coffee_rl_final.zip"

    if best_path.exists():
        target_path = best_path
    elif v2_final_path.exists():
        target_path = v2_final_path
    else:
        target_path = v1_final_path

    model, is_recurrent = load_trained_model(target_path)
    predict_fn, reset_lstm = make_policy_predict_fn(model, is_recurrent)

    print("=" * 105)
    print(
        f"🏆 ทัวร์นาเมนต์สนามจริง {total_teams} ทีม | ตลาดรับซื้อ {demand_ratio*100:.0f}% | "
        f"Error 0.5% + เพดานขายต่อรอบ ({num_games} เกม | โมเดล: {target_path.name})"
    )
    print("=" * 105)

    group_names = [
        "Trained_RL_AI (ทีมเรา)",
        "Adaptive_Bot_Pro",
        "Quota_Proportional",
        "Greedy_Ceiling",
        "Dumper_Bot",
        "Static_Baseline",
    ]

    group_stats: Dict[str, Dict[str, List[float]]] = {
        g_name: {
            "profit_over_tv": [],
            "revenue": [],
            "liq_loss": [],
            "ranks": [],
            "top3_hits": [],
            "top10_hits": [],
        }
        for g_name in group_names
    }

    for g in range(num_games):
        seed = seed_start + g
        rng = np.random.default_rng(seed)
        engine = MarketClearingEngine(rng=rng)

        ep_mode = "FIXED" if rng.random() < 0.6 else "RANDOMIZED"
        company_pool = BuyerCompanyPool(rng=rng, ceiling_mode=ep_mode)

        reset_lstm()
        bots: List[BaseOpponentBot] = []
        team_to_group: Dict[str, str] = {}

        bots.append(
            SelfPlayPolicyBot(
                "MY_TEAM",
                policy_predict_fn=predict_fn,
                total_rounds=total_rounds,
                rng=rng,
            )
        )
        team_to_group["MY_TEAM"] = "Trained_RL_AI (ทีมเรา)"

        other_groups = [
            "Adaptive_Bot_Pro",
            "Quota_Proportional",
            "Greedy_Ceiling",
            "Dumper_Bot",
            "Static_Baseline",
        ]
        for idx in range(1, total_teams):
            t_id = f"TEAM_{idx + 1:02d}"
            grp = other_groups[(idx - 1) % len(other_groups)]
            team_to_group[t_id] = grp

            if grp == "Adaptive_Bot_Pro":
                bots.append(
                    AdaptiveCompetitorBot(
                        t_id,
                        total_rounds=total_rounds,
                        base_aggressiveness=float(rng.uniform(0.4, 0.7)),
                        rng=rng,
                    )
                )
            elif grp == "Quota_Proportional":
                bots.append(
                    QuotaProportionalBot(t_id, total_rounds=total_rounds, rng=rng)
                )
            elif grp == "Greedy_Ceiling":
                bots.append(
                    GreedyCeilingBot(t_id, total_rounds=total_rounds, rng=rng)
                )
            elif grp == "Dumper_Bot":
                bots.append(DumperBot(t_id, total_rounds=total_rounds, rng=rng))
            else:
                bots.append(
                    SelfPlayPolicyBot(
                        t_id,
                        policy_predict_fn=lambda obs: np.array(
                            [0.55, 0.40, 0.50], dtype=np.float32
                        ),
                        total_rounds=total_rounds,
                        rng=rng,
                    )
                )

        # แจกของให้ครบทั้ง 55 ทีม
        inventories: Dict[str, List[CoffeeLot]] = {}
        game_rev = {b.team_id: 0.0 for b in bots}
        game_prof = {b.team_id: 0.0 for b in bots}
        game_liq = {b.team_id: 0.0 for b in bots}

        for b in bots:
            b.reset()
            inventories[b.team_id] = engine.generate_team_lots(b.team_id)

        # คำนวณเพดานขายสูงสุดต่อรอบของแต่ละทีม = int(floor((initial_bags / 6) * 1.5))
        team_round_sell_caps = {
            t_id: max(1, int(np.floor((sum(l.bags for l in lots) / 6.0) * 1.5)))
            for t_id, lots in inventories.items()
        }

        total_supply = sum(l.bags for lots in inventories.values() for l in lots)
        round_quotas = company_pool.plan_game_round_quotas(
            total_market_supply=total_supply,
            total_rounds=total_rounds,
            demand_ratio=demand_ratio,
        )

        # แข่งขันครบ 7 รอบ
        for rnd in range(1, total_rounds + 1):
            routes: List[BuyerOrder] = company_pool.generate_round_routes(
                round_total_quota=round_quotas[rnd - 1]
            )
            all_orders = []
            for b in bots:
                all_orders.extend(b.act(rnd, inventories[b.team_id], routes))

            clearing = engine.clear_market(
                round_num=rnd,
                team_inventories=inventories,
                routes=routes,
                all_orders=all_orders,
                team_round_sell_caps=team_round_sell_caps,
            )
            rfq_to_company = {
                r.order_id: (r.company_name, r.buyer_type) for r in routes
            }

            for b in bots:
                t_id = b.team_id
                tv_map = {l.lot_id: l.TV for l in inventories[t_id]}
                b.observe_feedback(
                    clearing.team_feedback[t_id], rfq_to_company, tv_map
                )
                game_rev[t_id] += clearing.team_revenue[t_id]
                game_prof[t_id] += clearing.team_profit_over_tv[t_id]

        # รอบบังคับเทขาย 0.80 * TV_true
        liq_res = engine.liquidate_remaining_inventory(
            inventories, liquidation_ratio=0.80
        )
        for t_id, (l_rev, l_prof) in liq_res.items():
            game_rev[t_id] += l_rev
            game_prof[t_id] += l_prof
            game_liq[t_id] = abs(l_prof)

        sorted_teams = sorted(
            game_prof.keys(), key=lambda t: game_prof[t], reverse=True
        )
        rank_map = {t_id: rank + 1 for rank, t_id in enumerate(sorted_teams)}

        for b in bots:
            t_id = b.team_id
            grp = team_to_group[t_id]
            rk = rank_map[t_id]
            group_stats[grp]["profit_over_tv"].append(game_prof[t_id])
            group_stats[grp]["revenue"].append(game_rev[t_id])
            group_stats[grp]["liq_loss"].append(game_liq[t_id])
            group_stats[grp]["ranks"].append(rk)
            group_stats[grp]["top3_hits"].append(1.0 if rk <= 3 else 0.0)
            group_stats[grp]["top10_hits"].append(1.0 if rk <= 10 else 0.0)

    summary_order = sorted(
        group_names,
        key=lambda g: np.mean(group_stats[g]["profit_over_tv"]),
        reverse=True,
    )

    print(
        f"{'อันดับ':<6} | {'กลุ่มกลยุทธ์':<23} | {'กำไรเหนือ TV เฉลี่ย':<19} | "
        f"{'อันดับเฉลี่ย (จาก 55)':<20} | {'ติด Top 10':<11} | {'โดนปรับรอบ 7'}"
    )
    print("-" * 105)

    for idx, grp in enumerate(summary_order, start=1):
        avg_p = float(np.mean(group_stats[grp]["profit_over_tv"]))
        avg_rk = float(np.mean(group_stats[grp]["ranks"]))
        top10_pct = float(np.mean(group_stats[grp]["top10_hits"])) * 100.0
        avg_lq = float(np.mean(group_stats[grp]["liq_loss"]))
        marker = "👑" if "Trained_RL_AI" in grp else "  "
        print(
            f"{marker} #{idx:<3} | {grp:<23} | {avg_p:+15.2f} บ. | "
            f"{avg_rk:13.1f} / 55   | {top10_pct:7.1f}%   | {avg_lq:10.2f} บ."
        )
    print("=" * 105)


if __name__ == "__main__":
    run_55_teams_tournament(num_games=50, total_teams=55, demand_ratio=0.80)