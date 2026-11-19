import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from market_env import CoffeeAuctionEnv

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def evaluate_rl_tournament(
    model_path: Path,
    total_rounds: int = 16,
    num_games: int = 100,
    seed_start: int = 7000,
):
    if not model_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path}")

    model = PPO.load(str(model_path))
    env = CoffeeAuctionEnv(num_teams=55, total_rounds=total_rounds, seed=seed_start)

    archetypes = ["RL_AI (ทีมเรา)", "QUOTA_V6", "QUOTA_V3", "QUOTA_UPGRADED", "ADAPTIVE", "DUMPER", "GREEDY"]
    stats = {
        a: {"profit": [], "rank": [], "top1": [], "top3": [], "top10": [], "leftover_bags": []}
        for a in archetypes
    }

    print("=" * 105)
    print(f"🏆 วัดผลทัวร์นาเมนต์ 55 ทีม | โหมด {total_rounds} รอบ (Action 3D + เพดาน 1/5 & 1/4) | จำนวน {num_games} เกม")
    print(f"📦 โมเดลที่ทดสอบ: {model_path}")
    print("=" * 105)

    for g in range(num_games):
        obs, _ = env.reset(seed=seed_start + g)
        done = False

        team_arch = {"T00": "RL_AI (ทีมเรา)"}
        for tid, opp in env.opponents.items():
            team_arch[tid] = opp.archetype

        init_tv_value = {
            tid: sum(l.bags * l.tv for l in lots)
            for tid, lots in env.team_lots.items()
        }
        team_revenue = {tid: 0.0 for tid in team_arch}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            sell_ratio, comm_mult, inst_mult = env.decode_action(action)

            my_lots = env.team_lots["T00"]
            my_init_bags = env.team_initial_bags["T00"]
            is_junk = (env.current_routes and env.current_routes[0].route_id == "R16-CLEAR")

            if is_junk:
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
                max_cap = env.clearing_engine.get_max_sell_per_round(my_init_bags, env.current_cap_divisor)
                target_bags = int(round(sell_ratio * max_cap))
                est_ceilings = {
                    r.route_id: env.ceiling_tracker.get_estimated_ceiling(r)
                    for r in env.current_routes
                }
                my_orders = env.router.route_bags(
                    my_lots, env.current_routes, target_bags, comm_mult, est_ceilings, inst_price_multiplier=inst_mult
                )

            all_bids = [{"team_id": "T00", **o} for o in my_orders]
            for tid, opp in env.opponents.items():
                opp_cap = env.clearing_engine.get_max_sell_per_round(env.team_initial_bags[tid], env.current_cap_divisor)
                opp_orders = opp.act(env.current_round, env.team_lots[tid], env.current_routes, opp_cap)
                all_bids.extend(opp_orders)

            results, _ = env.clearing_engine.clear_market(
                all_bids, env.current_routes, env.team_initial_bags, env.rng, cap_divisor=env.current_cap_divisor
            )

            route_map = {r.route_id: r for r in env.current_routes}
            team_results = {tid: [] for tid in team_arch}

            for res in results:
                tid = res["team_id"]
                if tid in team_results:
                    team_results[tid].append(res)
                if res.get("sold_qty", 0) > 0:
                    lot_obj = res["lot"]
                    lot_obj.bags -= res["sold_qty"]
                    team_revenue[tid] += res["price"] * res["sold_qty"]

            if not is_junk:
                my_res = team_results["T00"]
                env.ceiling_tracker.update(my_res, route_map)
                sub_qty = sum(r.get("qty", 0) for r in my_res)
                sold_qty = sum(r.get("sold_qty", 0) for r in my_res)
                exc_qty = sum(r.get("qty", 0) for r in my_res if r.get("status") == "CEILING_EXCEEDED")
                if sub_qty > 0:
                    env.last_win_rate = sold_qty / sub_qty
                    env.last_exceed_rate = exc_qty / sub_qty

            for tid, opp in env.opponents.items():
                opp.observe(team_results[tid], route_map)

            if env.current_round >= env.total_rounds:
                done = True
                for tid, lots in env.team_lots.items():
                    for l in lots:
                        if l.bags > 0:
                            team_revenue[tid] += (0.80 * l.tv) * l.bags
            else:
                env.current_round += 1
                total_market_bags = sum(env.team_initial_bags.values())
                env.current_routes = env.company_pool.generate_round_routes(
                    env.current_round, env.total_rounds, total_market_bags
                )
                obs = env._get_obs()

        game_profit = {
            tid: round(team_revenue[tid] - init_tv_value[tid], 2)
            for tid in team_arch
        }
        sorted_teams = sorted(game_profit.keys(), key=lambda t: game_profit[t], reverse=True)
        rank_map = {tid: idx + 1 for idx, tid in enumerate(sorted_teams)}

        for tid, arch in team_arch.items():
            rk = rank_map[tid]
            stats[arch]["profit"].append(game_profit[tid])
            stats[arch]["rank"].append(rk)
            stats[arch]["top1"].append(1.0 if rk == 1 else 0.0)
            stats[arch]["top3"].append(1.0 if rk <= 3 else 0.0)
            stats[arch]["top10"].append(1.0 if rk <= 10 else 0.0)
            stats[arch]["leftover_bags"].append(sum(l.bags for l in env.team_lots[tid]))

    ordered = sorted(
        [a for a in archetypes if len(stats[a]["profit"]) > 0],
        key=lambda a: np.mean(stats[a]["profit"]),
        reverse=True,
    )

    print(f"{'อันดับ':<6} | {'สายพันธุ์บอท':<18} | {'กำไรเหนือ TV เฉลี่ย':<19} | {'อันดับเฉลี่ย':<13} | {'แชมป์ #1':<9} | {'ติด Top 3':<9} | {'ติด Top 10'}")
    print("-" * 105)
    for idx, arch in enumerate(ordered, 1):
        mark = "👑" if "RL_AI" in arch else "  "
        print(
            f"{mark}#{idx:<4} | {arch:<18} | {np.mean(stats[arch]['profit']):+15.2f} บ. | "
            f"{np.mean(stats[arch]['rank']):6.1f} / 55   | "
            f"{np.mean(stats[arch]['top1'])*100:6.1f}%  | "
            f"{np.mean(stats[arch]['top3'])*100:6.1f}%  | "
            f"{np.mean(stats[arch]['top10'])*100:6.1f}%"
        )
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=16, help="จำนวนรอบ (16 หรือ 9)")
    parser.add_argument("--games", type=int, default=100, help="จำนวนเกมที่จำลอง")
    parser.add_argument("--model", type=str, default="", help="ระบุ path ไฟล์ .zip")
    args = parser.parse_args()

    if args.model:
        target = Path(args.model)
    elif args.rounds == 16:
        target = MODELS_DIR / "coffee_rl_16rounds_event" / "best" / "best_model.zip"
    else:
        target = MODELS_DIR / "coffee_rl_9rounds" / "best" / "best_model.zip"

    evaluate_rl_tournament(target, total_rounds=args.rounds, num_games=args.games)
