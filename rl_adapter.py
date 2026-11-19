"""RL Model Loader, Observation Builder, and Test-Time Monte Carlo Lookahead Adapter."""
import copy
import io
import math
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from coffee_math import CoffeeLot, BuyerRoute
from bag_router import BagCombinatorialRouter
from action_mask import is_lot_eligible_for_route
from history_tracker import classify_segment, extract_history_stats
from fallback_rules import bag_weight_kg, get_lot_avail_kg, get_rfq_wanted_kg, get_rfq_min_kg

HERE = Path(__file__).resolve().parent

DEFECT_BUFFER_NORMAL = 0.65
DEFECT_BUFFER_ENDGAME = 0.25
MOISTURE_BUFFER = 0.25

# พารามิเตอร์ Test-Time Rollout (Monte Carlo Lookahead)
ROLLOUT_NUM_CANDIDATES = 15
ROLLOUT_MC_SAMPLES = 12


def _load_model(folder_name: str, zip_name: str):
    p_dir = HERE / "models" / folder_name
    p_zip = HERE / "models" / zip_name
    if p_dir.exists() and p_dir.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in p_dir.iterdir():
                if f.is_file():
                    zf.write(f, f.name)
        buf.seek(0)
        print(f"✅ โหลดโมเดล RL สำเร็จ (จากโฟลเดอร์): {p_dir}", flush=True)
        return PPO.load(buf, device="cpu")
    if p_zip.exists():
        print(f"✅ โหลดโมเดล RL สำเร็จ (จากไฟล์): {p_zip}", flush=True)
        return PPO.load(str(p_zip), device="cpu")
    print(f"⚠️ ไม่พบโมเดล: {p_dir} หรือ {p_zip}", flush=True)
    return None


RL_MODEL_16 = _load_model("rl_16", "coffee_rl_16rounds_event_final.zip")
RL_MODEL_9 = _load_model("rl_9", "coffee_rl_9rounds_final.zip")
ROUTER = BagCombinatorialRouter(error_buffer=0.0)


def get_round_progress(state: dict, ctx: Any) -> tuple[int, int]:
    rnd_obj = state.get("round") or {}
    cur_rnd = int(rnd_obj.get("round_num") or rnd_obj.get("number") or rnd_obj.get("index") or 0)
    if cur_rnd <= 0 and "round_id" in rnd_obj:
        digits = "".join(ch for ch in str(rnd_obj["round_id"]) if ch.isdigit())
        cur_rnd = int(digits) if digits else 1

    tot_rnd = int(
        rnd_obj.get("total_rounds")
        or ctx.rules.get("total_rounds")
        or (len(ctx.calendar) if isinstance(ctx.calendar, list) and len(ctx.calendar) > 0 else 0)
        or 9
    )
    return max(1, cur_rnd), max(cur_rnd, tot_rnd)


def _estimate_event_phase(cur_rnd: int, tot_rnd: int, ctx: Any) -> tuple[float, int]:
    if isinstance(ctx.calendar, list) and len(ctx.calendar) >= cur_rnd:
        entry = ctx.calendar[cur_rnd - 1]
        if isinstance(entry, dict):
            ev = str(entry.get("event") or entry.get("theme") or entry.get("phase") or "").lower()
            if any(w in ev for w in ("festival", "เทศกาล", "specialty", "peak", "high")):
                return 1.15, 1
            if any(w in ev for w in ("low", "ซบเซา", "slow")):
                return 0.55, 2
            if any(w in ev for w in ("final", "clear", "ปิดตลาด", "junk")):
                return 1.00, 4

    if tot_rnd < 16:
        return 1.0, 0
    if 6 <= cur_rnd <= 8:
        return 1.15, 1
    if 13 <= cur_rnd <= 14:
        return 0.55, 2
    if cur_rnd == 15:
        return 0.28, 3
    if cur_rnd >= 16:
        return 1.00, 4
    return 1.00, 0


def _get_lot_init_kg(lot: dict) -> float:
    for k in ("kg_total", "initial_kg", "total_kg", "kg"):
        if k in lot and lot[k] is not None:
            return float(lot[k])
    return get_lot_avail_kg(lot)


def _is_clearance_round(cur_rnd: int, tot_rnd: int, routes: list[BuyerRoute]) -> bool:
    if tot_rnd >= 16 and cur_rnd >= tot_rnd:
        return True
    if routes and all(
        "0.80" in r.company_name or "CLEAR" in r.route_id.upper() or "ปิดเกม" in r.company_name
        for r in routes
    ):
        return True
    return False


def _generate_candidate_actions(
    base_sell_ratio: float,
    base_comm_mult: float,
    base_inst_mult: float,
    is_endgame: bool,
    cap_urgency: float,
    last_win: float,
    last_exc: float,
) -> list[tuple[str, float, float, float]]:
    """
    สร้าง 15 แผนยุทธวิธีรอบจุดที่โมเดล RL แนะนำ:
      - ปรับปริมาณการขาย (sell_ratio): อนุรักษนิยม / ตามโมเดล / เร่งระบายเต็มเพดาน
      - ปรับราคา (comm_mult, inst_mult): ดันกำไรสูงสุด / สมดุล / กดตัดราคาสู้คู่แข่ง
    """
    if is_endgame:
        sell_grid = [1.0]
    elif cap_urgency >= 0.85 or last_win < 0.45:
        sell_grid = [
            float(np.clip(base_sell_ratio, 0.25, 1.0)),
            float(np.clip(base_sell_ratio + 0.20, 0.35, 1.0)),
            1.0,
        ]
    else:
        sell_grid = [
            float(np.clip(base_sell_ratio - 0.18, 0.15, 1.0)),
            float(np.clip(base_sell_ratio, 0.15, 1.0)),
            float(np.clip(base_sell_ratio + 0.22, 0.25, 1.0)),
        ]

    # ลดโอกาสติดเพดานซ้ำหากรอบที่แล้วโดน CEILING_EXCEEDED สูง
    up_shift = 0.004 if last_exc > 0.25 else 0.014

    price_shifts = [
        ("rl_base", 0.000, 0.000),
        ("agg_profit", up_shift, up_shift * 0.8),
        ("slight_cut", -0.012, -0.018),
        ("deep_cut", -0.025, -0.038),
        ("split_guard", 0.006, -0.028),
    ]

    candidates: list[tuple[str, float, float, float]] = []
    seen = set()
    for sr in sell_grid:
        for label, d_comm, d_inst in price_shifts:
            c_m = round(float(np.clip(base_comm_mult + d_comm, 0.88, 1.035)), 4)
            i_m = round(float(np.clip(base_inst_mult + d_inst, 0.82, 0.982)), 4)
            key = (round(sr, 3), c_m, i_m)
            if key not in seen:
                seen.add(key)
                candidates.append((f"{label}@sr={sr:.2f}", sr, c_m, i_m))
            if len(candidates) >= ROLLOUT_NUM_CANDIDATES:
                return candidates
    return candidates


def _evaluate_orders_monte_carlo(
    orders: list[dict],
    my_lots: list[CoffeeLot],
    routes: list[BuyerRoute],
    est_ceilings: dict[str, float],
    cur_rnd: int,
    tot_rnd: int,
    max_round_cap: int,
    my_rem_bags: int,
    last_win: float,
    rng: random.Random,
) -> float:
    """
    จำลองการแข่งกับคู่แข่งสายต่างๆ (QUOTA_V6, ADAPTIVE, DUMPER, SNIPER) แบบ Monte Carlo
    เพื่อคำนวณ Expected Value (กำไรเหนือ TV ที่คาดว่าจะขายได้จริง + มูลค่าของเหลือที่ถือไปรอบหน้า)
    """
    if not orders:
        return -9999.0

    route_map = {r.route_id: r for r in routes}
    lot_map = {l.lot_id: l for l in my_lots}
    rounds_left = max(1, tot_rnd - cur_rnd + 1)
    auction_rounds_left = max(1, ((tot_rnd - 1) if tot_rnd >= 16 else tot_rnd) - cur_rnd + 1)
    min_rounds_needed = math.ceil(my_rem_bags / max(1, max_round_cap))
    urgency = min(1.2, min_rounds_needed / auction_rounds_left)

    # ประเมินมูลค่าโอกาสของกระสอบที่ยังไม่ถูกขายในรอบนี้ (Continuation Value)
    is_boom_upcoming = (tot_rnd >= 16 and cur_rnd < 6)
    if rounds_left <= 1:
        hold_factor = 0.80  # รอบสุดท้ายถ้าขายไม่ออกจะโดนบังคับเทขาย 0.80
    elif urgency >= 0.90:
        hold_factor = 0.86  # โควตาตึงตัวมาก ถ้าไม่ขายรอบนี้เสี่ยงไปติดคอขวดท้ายเกม
    elif is_boom_upcoming:
        hold_factor = 1.04  # รอบ 1-5 เก็บ Specialty ไว้รอขายรอบ 6-8 มีมูลค่าสูง
    else:
        hold_factor = 0.94 - (0.04 * min(1.0, cur_rnd / max(1, tot_rnd)))

    total_score = 0.0
    for _ in range(ROLLOUT_MC_SAMPLES):
        # จำลองเพดานจริงของแต่ละ Route รอบจุดประมาณการ
        sim_ceilings: dict[str, float] = {}
        for r in routes:
            est_c = est_ceilings.get(r.route_id, r.base_ceiling_mult)
            sim_ceilings[r.route_id] = est_c + rng.uniform(-0.008, 0.006)

        # จำลองราคาตัดหน้าของคู่แข่งในแต่ละกลุ่มตลาด (อิงจาก Win Rate รอบล่าสุด)
        comp_pressure = max(0.0, 0.65 - last_win) * 0.04
        dumper_share = rng.uniform(0.20, 0.40)

        sim_profit = 0.0
        sold_bags_total = 0
        sold_spec_bags = 0

        for item in orders:
            rid = item["route_id"]
            lid = item["lot_id"]
            qty = int(item["qty"])
            price = float(item["price"])
            r = route_map.get(rid)
            l = lot_map.get(lid)
            if not r or not l or l.tv <= 0:
                continue

            bid_ratio = price / l.tv
            true_ceil = sim_ceilings[rid]

            # 1. ตรวจเพดานราคา
            if bid_ratio > true_ceil:
                # โดนตีตก CEILING_EXCEEDED -> ขายได้ 0 และเสียโอกาสเก็บของ
                continue

            # 2. จำลองการแย่งโควตากับคู่แข่งใน Segment นั้น
            if r.buyer_type == "SPECIALTY":
                # คู่แข่งสาย QUOTA_V6 / SNIPER ยื่นแถว 1.15 - 1.21
                opp_median = rng.uniform(1.155, 1.205) - comp_pressure
                fill_prob = 1.0 if bid_ratio <= opp_median else max(0.15, 1.0 - (bid_ratio - opp_median) * 14.0)
            elif r.buyer_type == "COMMERCIAL":
                # คู่แข่งสาย ADAPTIVE / QUOTA ยื่นแถว 0.96 - 1.01
                opp_median = rng.uniform(0.965, 1.008) - (0.025 * min(1.0, urgency)) - comp_pressure
                fill_prob = 1.0 if bid_ratio <= opp_median else max(0.10, 1.0 - (bid_ratio - opp_median) * 16.0)
            else:
                # ตลาด INSTANT มี DUMPER 20-40% กดแถว 0.86 - 0.91
                if rng.random() < dumper_share:
                    opp_median = rng.uniform(0.875, 0.925) - comp_pressure
                else:
                    opp_median = rng.uniform(0.925, 0.965) - comp_pressure
                fill_prob = 1.0 if bid_ratio <= opp_median else max(0.08, 1.0 - (bid_ratio - opp_median) * 15.0)

            expected_fill = qty * fill_prob
            sold_bags_total += expected_fill
            if r.buyer_type == "SPECIALTY":
                sold_spec_bags += expected_fill

            # กำไรเทียบกับ TV
            sim_profit += (price - l.tv) * expected_fill

        # คำนวณมูลค่าของที่เหลือในมือหลังจบรอบนี้
        unsold_bags = max(0.0, my_rem_bags - sold_bags_total)
        avg_tv = float(np.mean([l.tv for l in my_lots if l.bags > 0]))
        continuation_val = unsold_bags * avg_tv * (hold_factor - 1.0)

        # โบนัสพิเศษหากขาย Specialty ได้สำเร็จในรอบนี้
        spec_bonus = sold_spec_bags * 3.5
        total_score += (sim_profit + continuation_val + spec_bonus)

    return total_score / ROLLOUT_MC_SAMPLES


def run_rl_agent(state: dict, ctx: Any) -> tuple[list[dict], str]:
    cur_rnd, tot_rnd = get_round_progress(state, ctx)
    model = RL_MODEL_9 if (tot_rnd <= 11 and RL_MODEL_9 is not None) else (RL_MODEL_16 or RL_MODEL_9)
    tag = "ppo_rl_9r" if model is RL_MODEL_9 else "ppo_rl_16r"
    if model is None:
        return [], "no_model"

    bag_kg = bag_weight_kg(ctx.rules)
    raw_lots = state.get("lots") or []
    raw_rfqs = state.get("rfqs") or []

    rfq_map = {r["rfq_id"]: r for r in raw_rfqs if r.get("rfq_id")}
    last_win, last_exc, est_ceilings, banned_segments = extract_history_stats(state, ctx.memory, ctx.tv, rfq_map)

    rounds_left = max(1, tot_rnd - cur_rnd + 1)
    is_endgame = (rounds_left <= 2)

    my_lots: list[CoffeeLot] = []
    my_init_bags = 0
    for l in raw_lots:
        lid = l.get("lot_id")
        if not lid:
            continue
        avail_bags = int(get_lot_avail_kg(l) // bag_kg)
        init_bags = max(avail_bags, int(_get_lot_init_kg(l) // bag_kg))
        my_init_bags += init_bags
        sample = l.get("sample") or l
        tv_val = float(ctx.tv.get(lid, 100.0))

        raw_s = float(sample.get("screen") or sample.get("screen_size") or 15.0)
        raw_m = float(sample.get("moisture_pct") or sample.get("moisture") or 12.0)
        raw_g = float(sample.get("defect_pct") or sample.get("defect") or 5.0)

        obs_m = (raw_m + MOISTURE_BUFFER) if (raw_m <= 13.5 and not is_endgame) else raw_m
        obs_g = (raw_g + (DEFECT_BUFFER_ENDGAME if is_endgame else DEFECT_BUFFER_NORMAL)) if raw_g <= 7.0 else raw_g

        banned = banned_segments.get(lid, set())
        if "COMMERCIAL" in banned:
            obs_g = max(obs_g, 8.5)
        elif "SPECIALTY" in banned:
            obs_g = max(obs_g, 3.8)

        my_lots.append(CoffeeLot(
            lot_id=lid,
            bags=avail_bags,
            s=raw_s,
            m=obs_m,
            g=obs_g,
            tv=tv_val,
        ))

    my_rem_bags = sum(l.bags for l in my_lots)
    if my_rem_bags <= 0 or not raw_rfqs:
        return [], tag

    routes: list[BuyerRoute] = []
    for r in raw_rfqs:
        rid = r.get("rfq_id")
        if not rid:
            continue
        seg = classify_segment(r)
        spec = r.get("spec") or r.get("requirements") or r
        def_d = 3.0 if seg == "SPECIALTY" else (7.0 if seg == "COMMERCIAL" else 15.0)
        def_m = 12.5 if seg in ("SPECIALTY", "COMMERCIAL") else 15.0
        buyer = str(r.get("buyer") or r.get("buyer_name") or rid)
        def_s = 17.0 if "บางกอกบลูม" in buyer else (16.0 if seg == "SPECIALTY" else 0.0)

        routes.append(BuyerRoute(
            route_id=rid,
            company_name=buyer,
            buyer_type=seg,
            quota=int(get_rfq_wanted_kg(r, ctx.rules) // bag_kg),
            max_defect=float(spec.get("max_defect_pct") or spec.get("max_defect") or def_d),
            max_moisture=float(spec.get("max_moisture_pct") or spec.get("max_moisture") or def_m),
            min_screen=float(spec.get("min_screen") or spec.get("min_size") or def_s),
            min_bags_per_bid=max(1, int(math.ceil(get_rfq_min_kg(r, ctx.rules) / bag_kg))),
        ))

    round_cap_kg = float(state.get("capacity_left_kg") or state.get("round_capacity_kg") or (my_rem_bags * bag_kg))
    max_round_cap = max(1, int(round_cap_kg // bag_kg))

    # โหมดรอบปิดเกม (เช่น รอบ 16 รับขยะ 0.80 * TV): เทขายของเหลือทั้งหมดที่ 0.798 * TV
    if _is_clearance_round(cur_rnd, tot_rnd, routes):
        clear_offers: list[dict] = []
        rem_cap_bags = max_round_cap
        sorted_routes = sorted(routes, key=lambda x: x.quota, reverse=True)
        for l in my_lots:
            if l.bags <= 0 or rem_cap_bags <= 0:
                continue
            for r in sorted_routes:
                if l.bags <= 0 or r.quota <= 0 or rem_cap_bags <= 0:
                    continue
                min_req = max(1, r.min_bags_per_bid)
                take = min(l.bags, r.quota, rem_cap_bags)
                if take >= min_req:
                    clear_offers.append({
                        "rfq_id": r.route_id,
                        "lot_id": l.lot_id,
                        "kg": round(take * bag_kg, 2),
                        "price_per_kg": round(l.tv * 0.798, 2),
                    })
                    l.bags -= take
                    r.quota -= take
                    rem_cap_bags -= take
        if clear_offers:
            ctx.notes.append(f"CLEARANCE R{cur_rnd}/{tot_rnd} | offers={len(clear_offers)}")
            return clear_offers, f"{tag}_clear"

    total_demand = sum(r.quota for r in routes)
    if total_demand > my_rem_bags * 4:
        num_teams_est = max(1, int(state.get("num_teams") or len(state.get("standings") or []) or 55))
        all_rem_bags_est = max(my_rem_bags, my_rem_bags * num_teams_est)
    else:
        all_rem_bags_est = max(1, my_rem_bags)

    spec_eligible_quota = 0
    for r in routes:
        if r.buyer_type == "SPECIALTY":
            if any(is_lot_eligible_for_route(l, r, 0.0) for l in my_lots if l.bags > 0):
                spec_eligible_quota += r.quota

    avg_tv_norm = float(np.mean([l.tv for l in my_lots if l.bags > 0]) / 135.0)
    spec_bags = sum(
        l.bags for l in my_lots
        if l.bags > 0 and l.s >= 16.0 and l.g <= 3.0 and l.m <= 12.5
    )

    demand_mult, phase_code = _estimate_event_phase(cur_rnd, tot_rnd, ctx)
    auction_rounds = (tot_rnd - 1) if tot_rnd >= 16 else tot_rnd
    auction_rounds_left = max(0, auction_rounds - cur_rnd + 1)
    min_rounds_needed = math.ceil(my_rem_bags / max(1, max_round_cap))
    cap_urgency = min(1.5, min_rounds_needed / max(1, auction_rounds_left))

    obs = np.array([
        cur_rnd / max(1, tot_rnd),
        auction_rounds_left / max(1, auction_rounds),
        my_rem_bags / max(1, my_init_bags),
        min(1.0, my_rem_bags / max(1, max_round_cap)),
        min(2.0, total_demand / max(1, all_rem_bags_est)),
        avg_tv_norm,
        spec_bags / max(1, my_rem_bags),
        min(1.0, spec_eligible_quota / max(1, max_round_cap)),
        sum(r.quota for r in routes if r.buyer_type == "COMMERCIAL") / max(1, total_demand),
        sum(r.quota for r in routes if r.buyer_type == "INSTANT") / max(1, total_demand),
        1.0 if tot_rnd >= 16 else 0.0,
        phase_code / 4.0,
        min(1.0, spec_eligible_quota / max(1, total_demand)),
        last_win,
        last_exc,
        demand_mult / 1.5,
        cap_urgency,
    ], dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    base_sell_ratio = float(np.clip(action[0], 0.0, 1.0))
    comm_signal = float(np.clip(action[1], 0.0, 1.0))
    inst_signal = float(np.clip(action[2], 0.0, 1.0)) if len(action) > 2 else comm_signal

    base_comm_mult = 0.91 + (comm_signal * 0.13)
    base_inst_mult = 0.84 + (inst_signal * 0.145)

    # =========================================================================
    # TEST-TIME ROLLOUT: จำลอง 15 แผนยุทธวิธีรอบจุดที่โมเดลแนะนำ เพื่อเลือกแผนที่ดีที่สุด
    # =========================================================================
    candidates = _generate_candidate_actions(
        base_sell_ratio, base_comm_mult, base_inst_mult, is_endgame, cap_urgency, last_win, last_exc
    )

    sim_rng = ctx.rng if hasattr(ctx, "rng") and ctx.rng is not None else random.Random(cur_rnd * 997)
    best_routed: list[dict] = []
    best_score = -1e18
    best_label = "rl_base"

    for label, sr, c_mult, i_mult in candidates:
        t_bags = (
            max(min(my_rem_bags, max_round_cap), int(round(sr * max_round_cap)))
            if is_endgame
            else int(round(sr * max_round_cap))
        )
        cand_routed = ROUTER.route_bags(
            my_lots, routes, t_bags, c_mult, est_ceilings, inst_price_multiplier=i_mult
        )
        if not cand_routed:
            continue

        score = _evaluate_orders_monte_carlo(
            cand_routed,
            my_lots,
            routes,
            est_ceilings,
            cur_rnd,
            tot_rnd,
            max_round_cap,
            my_rem_bags,
            last_win,
            sim_rng,
        )
        if score > best_score:
            best_score = score
            best_routed = cand_routed
            best_label = label

    routed = best_routed

    # Endgame Sweep: กวาดกระสอบที่เหลือช่วง 2 รอบสุดท้ายไม่ให้ค้างสต็อก
    if is_endgame:
        routed_bags_by_lot: dict[str, int] = {}
        used_quota_by_route: dict[str, int] = {}
        total_routed_bags = 0
        for item in routed:
            q = int(item.get("qty") or 0)
            routed_bags_by_lot[item["lot_id"]] = routed_bags_by_lot.get(item["lot_id"], 0) + q
            used_quota_by_route[item["route_id"]] = used_quota_by_route.get(item["route_id"], 0) + q
            total_routed_bags += q

        rem_cap_bags = max_round_cap - total_routed_bags
        if rem_cap_bags > 0:
            for l in my_lots:
                rem_l = l.bags - routed_bags_by_lot.get(l.lot_id, 0)
                if rem_l <= 0 or rem_cap_bags <= 0:
                    continue
                for r in sorted(routes, key=lambda x: x.quota, reverse=True):
                    rem_q = r.quota - used_quota_by_route.get(r.route_id, 0)
                    min_req = max(1, r.min_bags_per_bid)
                    if rem_q < min_req or rem_l < min_req or rem_cap_bags < min_req:
                        continue
                    if l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_screen:
                        take = min(rem_l, rem_q, rem_cap_bags)
                        ceil_m = est_ceilings.get(r.route_id, 0.978)
                        dump_mult = min(ceil_m - 0.003, 0.835)
                        routed.append({
                            "lot_id": l.lot_id,
                            "route_id": r.route_id,
                            "qty": take,
                            "price": round(l.tv * dump_mult, 2),
                        })
                        rem_l -= take
                        rem_cap_bags -= take
                        used_quota_by_route[r.route_id] = used_quota_by_route.get(r.route_id, 0) + take

    ctx.notes.append(
        f"RL+Rollout R{cur_rnd}/{tot_rnd} | best={best_label} (EV={best_score:+.1f}) | banned={banned_segments}"
    )

    offers: list[dict] = []
    for item in routed:
        qty_bags = int(item.get("qty") or 0)
        if qty_bags <= 0:
            continue
        offers.append({
            "rfq_id": item["route_id"],
            "lot_id": item["lot_id"],
            "kg": round(qty_bags * bag_kg, 2),
            "price_per_kg": round(float(item["price"]), 2),
        })
    return offers, f"{tag}_rollout"
