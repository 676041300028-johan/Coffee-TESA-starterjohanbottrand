import math
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


# ==============================================================================
# 1. โครงสร้างข้อมูลและคณิตศาสตร์ประเมินราคากาแฟ (Stand-alone)
# ==============================================================================
GRADE_TABLE = [
    (1.0, "PREMIUM", 1.08),
    (3.0, "GRADE_1", 1.04),
    (5.0, "GRADE_2", 1.00),
    (7.0, "GRADE_3", 0.96),
    (15.0, "BELOW_STANDARD", 0.85),
    (float("inf"), "REJECT", 0.60),
]


def calc_tv(s: float, m: float, g: float, p0: float = 120.0) -> float:
    grade_mult = 0.60
    for max_d, _, g_mult in GRADE_TABLE:
        if g <= max_d:
            grade_mult = g_mult
            break
    G = round((1.0 - g / 100.0) * grade_mult, 4)

    if m <= 12.5:
        M = 1.0000
    elif m <= 15.0:
        M = round((1.0 - m / 100.0) / 0.875 - 0.02 * (m - 12.5), 4)
    else:
        M = 0.6000

    s_int = int(round(s))
    if s_int <= 12:
        S = 0.92
    elif s_int == 13:
        S = 0.97
    elif s_int in (14, 15):
        S = 1.00
    elif s_int in (16, 17):
        S = 1.03
    else:
        S = 1.06

    return round(p0 * G * M * S, 2)


@dataclass
class SimLot:
    lot_id: str
    s: float
    m: float
    g: float
    bags: int
    s_true: float
    m_true: float
    g_true: float
    TV: float = field(init=False)
    TV_true: float = field(init=False)

    def __post_init__(self):
        self.TV = calc_tv(self.s, self.m, self.g)
        self.TV_true = calc_tv(self.s_true, self.m_true, self.g_true)


@dataclass
class SimRoute:
    order_id: str
    company_name: str
    buyer_type: str  # SPECIALTY / COMMERCIAL / INSTANT
    quota_bags: int
    max_defect: float
    max_moisture: float
    min_size: float
    hidden_ceiling_ratio: float


@dataclass
class SimOrder:
    team_id: str
    rfq_id: str
    lot_id: str
    bags: int
    bid_price: float


# ==============================================================================
# 2. บอททุกสายพันธุ์ในสนาม 55 ทีม (อัปเกรดให้รู้กติกาใหม่เท่ากันทุกตัว 100%)
# ==============================================================================
class BaseBot:
    def __init__(self, team_id: str, total_rounds: int, rng: np.random.Generator):
        self.team_id = team_id
        self.total_rounds = total_rounds
        self.rng = rng
        self.initial_total_bags = 0

    def reset(self):
        self.initial_total_bags = 0

    def observe_feedback(self, feedback: List[Dict], route_map: Dict[str, SimRoute], tv_map: Dict[str, float]):
        pass

    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        raise NotImplementedError


class QuotaProportionalV3Bot(BaseBot):
    """
    [ทีมเรา] อัปเกรดเต็มระบบ:
      - รู้จัก Specialty (1.22 - 1.24) + คุมเพดานขาย 1/4 ต่อรอบ
      - เผื่อ Error 0.5% ของ m, g (ไม่โดนตีตกสเปก) + จำเพดานราคาจาก Feedback
      - ยื่นกระจายครบทุกใบเพื่อชนเพดาน 1/4 เต็มโควตาทุกรอบ
    """
    def __init__(self, team_id: str, total_rounds: int, rng: np.random.Generator):
        super().__init__(team_id, total_rounds, rng)
        self.ceiling_memory: Dict[str, float] = {}

    def reset(self):
        super().reset()
        self.ceiling_memory = {}

    def observe_feedback(self, feedback: List[Dict], route_map: Dict[str, SimRoute], tv_map: Dict[str, float]):
        for fb in feedback:
            r = route_map.get(fb["rfq"])
            if not r:
                continue
            tv = tv_map.get(fb["lot"], 100.0)
            ratio = fb["price"] / max(1e-6, tv)
            if "เกินเพดาน" in fb["status"]:
                prev = self.ceiling_memory.get(r.company_name, 1.30)
                self.ceiling_memory[r.company_name] = min(prev, ratio - 0.002)

    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        if current_round == 1 or self.initial_total_bags == 0:
            self.initial_total_bags = sum(l.bags for l in lots)

        total_rem = sum(l.bags for l in lots)
        if total_rem <= 0 or not routes:
            return []

        max_sell_cap = max(1, self.initial_total_bags // 4)
        rounds_left = max(1, self.total_rounds - current_round + 1)
        min_rounds_needed = math.ceil(total_rem / max_sell_cap)

        # เร่งสปีดอัตโนมัติเมื่อจำนวนรอบที่เหลือใกล้เคียงจำนวนรอบขั้นต่ำที่ต้องใช้ขาย
        progress = current_round / self.total_rounds
        cap_urgency = min(1.0, min_rounds_needed / rounds_left)
        urgency = max(progress, cap_urgency) if rounds_left > 1 else 1.0

        cap_left = {r.order_id: r.quota_bags for r in routes}
        orders: List[SimOrder] = []

        sorted_lots = sorted([x for x in lots if x.bags > 0], key=lambda x: x.TV)

        for l in sorted_lots:
            safe_g = l.g * 1.005
            safe_m = l.m * 1.005

            avail = [
                r for r in routes
                if cap_left[r.order_id] > 0
                and safe_g <= r.max_defect
                and safe_m <= r.max_moisture
                and l.s >= r.min_size
            ]
            if not avail:
                continue

            spec_routes = [r for r in avail if r.buyer_type == "SPECIALTY"]
            norm_routes = [r for r in avail if r.buyer_type != "SPECIALTY"]

            rem_bags = l.bags
            alloc = []

            for r in spec_routes:
                if rem_bags <= 0:
                    break
                take = min(rem_bags, cap_left[r.order_id])
                if take > 0:
                    alloc.append((r, take))
                    rem_bags -= take
                    cap_left[r.order_id] -= take

            if rem_bags > 0 and norm_routes:
                tot_q = sum(cap_left[r.order_id] for r in norm_routes)
                for r in norm_routes:
                    if rem_bags <= 0:
                        break
                    share = max(1, int(round((cap_left[r.order_id] / max(1, tot_q)) * rem_bags)))
                    take = min(share, rem_bags, cap_left[r.order_id])
                    if take > 0:
                        alloc.append((r, take))
                        rem_bags -= take
                        cap_left[r.order_id] -= take

            for r, qty in alloc:
                if r.buyer_type == "SPECIALTY":
                    base_r = 1.225 if r.min_size >= 17 else 1.205
                    discount = 0.020 * urgency
                elif r.buyer_type == "COMMERCIAL":
                    base_r = 1.012
                    discount = 0.042 * urgency
                else:
                    base_r = 0.962
                    discount = 0.042 * urgency

                known = self.ceiling_memory.get(r.company_name)
                if known:
                    base_r = min(base_r, known - 0.006)

                ratio = max(0.815, base_r - discount)
                for i in range(qty):
                    orders.append(
                        SimOrder(self.team_id, r.order_id, l.lot_id, 1, round(l.TV * (ratio - 0.002 * i), 2))
                    )

        return orders


class QuotaProportionalUpgradedBot(BaseBot):
    """
    [คู่แข่งสายสมดุล] อัปเกรดรู้กติกาใหม่:
      - รู้จัก Specialty (1.22-1.24) + รู้เพดานขาย 1/4 ต่อรอบ
      - แต่ไม่ได้เผื่อ Buffer Error 0.5% และไม่มีระบบจำเพดาน
    """
    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        if current_round == 1 or self.initial_total_bags == 0:
            self.initial_total_bags = sum(l.bags for l in lots)

        total_rem = sum(l.bags for l in lots)
        if total_rem <= 0:
            return []

        max_sell_cap = max(1, self.initial_total_bags // 4)
        rounds_left = max(1, self.total_rounds - current_round + 1)
        urgency = max(current_round / self.total_rounds, min(1.0, math.ceil(total_rem / max_sell_cap) / rounds_left))

        cap_left = {r.order_id: r.quota_bags for r in routes}
        orders: List[SimOrder] = []

        for l in sorted([x for x in lots if x.bags > 0], key=lambda x: x.TV):
            avail = [
                r for r in routes
                if cap_left[r.order_id] > 0 and l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_size
            ]
            if not avail:
                continue
            tot_q = sum(cap_left[r.order_id] for r in avail)
            for r in avail:
                share = max(1, int(round((cap_left[r.order_id] / tot_q) * l.bags)))
                take = min(share, l.bags, cap_left[r.order_id])
                if r.buyer_type == "SPECIALTY":
                    base_r = 1.215
                elif r.buyer_type == "COMMERCIAL":
                    base_r = 1.015
                else:
                    base_r = 0.965
                ratio = max(0.82, base_r - 0.04 * urgency)
                for i in range(take):
                    orders.append(SimOrder(self.team_id, r.order_id, l.lot_id, 1, round(l.TV * (ratio - 0.002 * i), 2)))
                cap_left[r.order_id] -= take
        return orders


class AdaptiveUpgradedBot(BaseBot):
    """[คู่แข่งปรับตัว] อัปเกรดรู้จัก Specialty + เพดาน 1/4 + ปรับราคาตาม Win Rate"""
    def __init__(self, team_id: str, total_rounds: int, rng: np.random.Generator):
        super().__init__(team_id, total_rounds, rng)
        self.agg = 0.55

    def observe_feedback(self, feedback: List[Dict], route_map: Dict[str, SimRoute], tv_map: Dict[str, float]):
        sub = sum(x["bags_submitted"] for x in feedback)
        sold = sum(x["bags_sold"] for x in feedback)
        if sub > 0:
            wr = sold / sub
            self.agg = float(np.clip(self.agg + (0.08 if wr > 0.7 else -0.08), 0.1, 0.95))

    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        if current_round == 1 or self.initial_total_bags == 0:
            self.initial_total_bags = sum(l.bags for l in lots)

        cap_left = {r.order_id: r.quota_bags for r in routes}
        orders: List[SimOrder] = []
        urgency = (current_round / self.total_rounds) ** 1.4
        for l in lots:
            if l.bags <= 0:
                continue
            avail = [
                r for r in routes
                if cap_left[r.order_id] > 0 and l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_size
            ]
            avail.sort(
                key=lambda r: 1.22 if r.buyer_type == "SPECIALTY" else (1.03 if r.buyer_type == "COMMERCIAL" else 0.98),
                reverse=True,
            )
            rem = l.bags
            for r in avail:
                if rem <= 0:
                    break
                take = min(rem, cap_left[r.order_id])
                ceil = 1.22 if r.buyer_type == "SPECIALTY" else (1.028 if r.buyer_type == "COMMERCIAL" else 0.978)
                ratio = max(0.81, ceil - (1.0 - self.agg) * 0.04 - 0.05 * urgency)
                for _ in range(take):
                    orders.append(SimOrder(self.team_id, r.order_id, l.lot_id, 1, round(l.TV * ratio, 2)))
                rem -= take
                cap_left[r.order_id] -= take
        return orders


class DumperUpgradedBot(BaseBot):
    """[คู่แข่งสายทุ่มราคา] อัปเกรดรู้จัก Specialty + เพดาน 1/4"""
    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        cap_left = {r.order_id: r.quota_bags for r in routes}
        orders: List[SimOrder] = []
        for l in lots:
            if l.bags <= 0:
                continue
            rem = l.bags
            for r in routes:
                if rem <= 0 or cap_left[r.order_id] <= 0:
                    continue
                if l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_size:
                    take = min(rem, cap_left[r.order_id])
                    ratio = float(self.rng.uniform(1.05, 1.12)) if r.buyer_type == "SPECIALTY" else float(self.rng.uniform(0.86, 0.91))
                    for _ in range(take):
                        orders.append(SimOrder(self.team_id, r.order_id, l.lot_id, 1, round(l.TV * ratio, 2)))
                    rem -= take
                    cap_left[r.order_id] -= take
        return orders


class GreedyUpgradedBot(BaseBot):
    """[คู่แข่งสายโลภ] อัปเกรดรู้จัก Specialty (ตั้ง 1.235 / 1.032 / 0.982) + เพดาน 1/4"""
    def act(self, current_round: int, lots: List[SimLot], routes: List[SimRoute]) -> List[SimOrder]:
        cap_left = {r.order_id: r.quota_bags for r in routes}
        orders: List[SimOrder] = []
        for l in lots:
            if l.bags <= 0:
                continue
            rem = l.bags
            for r in routes:
                if rem <= 0 or cap_left[r.order_id] <= 0:
                    continue
                if l.g <= r.max_defect and l.m <= r.max_moisture and l.s >= r.min_size:
                    take = min(rem, cap_left[r.order_id])
                    ceil = 1.235 if r.buyer_type == "SPECIALTY" else (1.032 if r.buyer_type == "COMMERCIAL" else 0.982)
                    for _ in range(take):
                        orders.append(SimOrder(self.team_id, r.order_id, l.lot_id, 1, round(l.TV * ceil, 2)))
                    rem -= take
                    cap_left[r.order_id] -= take
        return orders


# ==============================================================================
# 3. สนามจำลองกติกาใหม่ (Specialty + Error 0.5% + เพดาน 1/4 + 9/16 รอบ)
# ==============================================================================
class NewRuleArena:
    COMPANIES = [
        ("2 บางกอกบลูม", "SPECIALTY", 3.0, 12.5, 17.0, 1.24),
        ("1 ดอยช้างโรสเตอร์", "SPECIALTY", 3.0, 12.5, 16.0, 1.22),
        ("1 พรีเมียมโรสเตอร์", "COMMERCIAL", 7.0, 12.5, 0.0, 1.03),
        ("2 โกลบอลคอฟฟี่", "COMMERCIAL", 7.0, 12.5, 0.0, 1.03),
        ("3 สยามคอมเมิร์ซ", "COMMERCIAL", 7.0, 12.5, 0.0, 1.03),
        ("5 บลูเมาท์เทน", "INSTANT", 15.0, 14.0, 0.0, 0.98),
        ("6 ทรีอินวันฟู้ดส์", "INSTANT", 15.0, 14.0, 0.0, 0.98),
        ("7 เอเชียอินสแตนท์", "INSTANT", 15.0, 14.0, 0.0, 0.98),
    ]

    def __init__(self, rng: np.random.Generator, total_rounds: int, event_wave: bool):
        self.rng = rng
        self.total_rounds = total_rounds
        self.event_wave = event_wave
        self.ceilings = {c[0]: round(c[5] + float(rng.uniform(-0.003, 0.003)), 4) for c in self.COMPANIES}

    def generate_lots(self) -> List[SimLot]:
        tot = int(self.rng.integers(24, 31))
        splits = self.rng.multinomial(tot - 12, [0.25, 0.25, 0.25, 0.25]) + 3
        profiles = [
            (float(self.rng.choice([16.0, 17.0, 18.0])), round(float(self.rng.uniform(10.8, 12.3)), 2), round(float(self.rng.uniform(0.8, 3.05)), 2)),
            (float(self.rng.integers(14, 17)), round(float(self.rng.uniform(11.5, 12.55)), 2), round(float(self.rng.uniform(2.5, 7.05)), 2)),
            (float(self.rng.integers(13, 16)), round(float(self.rng.uniform(12.6, 13.8)), 2), round(float(self.rng.uniform(3.5, 9.5)), 2)),
            (float(self.rng.integers(12, 15)), round(float(self.rng.uniform(12.8, 14.05)), 2), round(float(self.rng.uniform(6.5, 14.5)), 2)),
        ]
        lots = []
        for i, (st, mt, gt) in enumerate(profiles):
            s_obs = float(round(st * (1.0 + float(self.rng.uniform(-0.005, 0.005)))))
            m_obs = round(mt * (1.0 + float(self.rng.uniform(-0.005, 0.005))), 2)
            g_obs = round(gt * (1.0 + float(self.rng.uniform(-0.005, 0.005))), 2)
            lots.append(SimLot(f"LOT-{i+1}", s_obs, m_obs, g_obs, int(splits[i]), st, mt, gt))
        return lots

    def plan_quotas(self, total_supply: int, demand_ratio: float = 0.80) -> List[int]:
        tot_q = int(round(total_supply * demand_ratio))
        if not self.event_wave:
            w = self.rng.dirichlet(np.full(self.total_rounds, 4.0))
        else:
            raw = [0.55 if ((r // 4) % 2 == 0) else 1.45 for r in range(self.total_rounds)]
            probs = np.array(raw) / np.sum(raw)
            w = self.rng.dirichlet(probs * 25.0)
        return [int(q) for q in self.rng.multinomial(tot_q, w)]

    def generate_routes(self, round_q: int) -> List[SimRoute]:
        sq1 = int(self.rng.integers(1, 6)) if round_q >= 10 else 0
        sq2 = int(self.rng.integers(1, 4)) if round_q >= 10 else 0
        rem_q = max(0, round_q - sq1 - sq2)
        other_q = self.rng.multinomial(rem_q, [0.16, 0.16, 0.16, 0.18, 0.17, 0.17]) if rem_q > 0 else np.zeros(6, dtype=int)
        all_q = [sq1, sq2] + list(other_q)
        return [
            SimRoute(f"R5-{i+1:02d}", c[0], c[1], int(all_q[i]), c[2], c[3], c[4], self.ceilings[c[0]])
            for i, c in enumerate(self.COMPANIES)
        ]

    def clear_market(self, inventories: Dict[str, List[SimLot]], routes: List[SimRoute], orders: List[SimOrder], caps: Dict[str, int]):
        r_map = {r.order_id: r for r in routes}
        l_map = {(t, l.lot_id): l for t, lots in inventories.items() for l in lots}
        fb = {t: [] for t in inventories}
        prof = {t: 0.0 for t in inventories}
        cap_left = dict(caps)
        bids = {r.order_id: [] for r in routes}

        for o in orders:
            r = r_map.get(o.rfq_id)
            lot = l_map.get((o.team_id, o.lot_id))
            if not r or not lot or o.bags <= 0:
                continue
            if lot.m_true > r.max_moisture or lot.g_true > r.max_defect or lot.s_true < r.min_size:
                fb[o.team_id].append({"rfq": o.rfq_id, "lot": o.lot_id, "price": o.bid_price, "bags_submitted": o.bags, "bags_sold": 0, "status": "ไม่ผ่านสเปก"})
                continue
            ratio_true = o.bid_price / max(1e-6, lot.TV_true)
            if ratio_true > r.hidden_ceiling_ratio + 1e-6:
                fb[o.team_id].append({"rfq": o.rfq_id, "lot": o.lot_id, "price": o.bid_price, "bags_submitted": o.bags, "bags_sold": 0, "status": "แพงเกินเพดานผู้ซื้อ"})
                continue
            bids[o.rfq_id].append((ratio_true, float(self.rng.random()), o))

        for r in routes:
            rem_q = r.quota_bags
            for ratio_true, _, o in sorted(bids[r.order_id], key=lambda x: (x[0], x[1])):
                lot = l_map[(o.team_id, o.lot_id)]
                sold = min(o.bags, lot.bags, rem_q, cap_left[o.team_id])
                if sold > 0:
                    lot.bags -= sold
                    rem_q -= sold
                    cap_left[o.team_id] -= sold
                    prof[o.team_id] += round((o.bid_price - lot.TV_true) * sold, 2)
                fb[o.team_id].append({
                    "rfq": o.rfq_id,
                    "lot": o.lot_id,
                    "price": o.bid_price,
                    "bags_submitted": o.bags,
                    "bags_sold": sold,
                    "status": "ขายได้ทั้งหมด" if sold > 0 else "เต็มโควตา/เพดานรอบ",
                })
        return fb, prof


def run_benchmark(total_rounds: int, event_wave: bool, num_games: int = 40):
    mode_label = f"{total_rounds} รอบ" + (" (อีเวนต์ 4 รอบติด)" if event_wave else " (ตลาดปกติ)")
    print("=" * 108)
    print(f"🧪 ทดสอบบอท 5 สายพันธุ์ (ทุกตัวรู้กติกาใหม่เหมือนกัน 100%) | สนาม 55 ทีม: {mode_label} ({num_games} เกม)")
    print("=" * 108)

    groups = [
        "Quota_Prop_V3 (ทีมเรา)",
        "Quota_Prop_Upgraded",
        "Adaptive_Upgraded",
        "Dumper_Upgraded",
        "Greedy_Upgraded",
    ]
    stats = {g: {"prof": [], "rank": [], "top10": [], "liq": [], "rejects": []} for g in groups}

    for g in range(num_games):
        rng = np.random.default_rng(8000 + g)
        arena = NewRuleArena(rng=rng, total_rounds=total_rounds, event_wave=event_wave)

        bots: List[BaseBot] = []
        team_grp: Dict[str, str] = {}
        for idx in range(55):
            t_id = f"T{idx+1:02d}"
            grp = groups[idx % len(groups)]
            team_grp[t_id] = grp
            if grp == "Quota_Prop_V3 (ทีมเรา)":
                bots.append(QuotaProportionalV3Bot(t_id, total_rounds, rng))
            elif grp == "Quota_Prop_Upgraded":
                bots.append(QuotaProportionalUpgradedBot(t_id, total_rounds, rng))
            elif grp == "Adaptive_Upgraded":
                bots.append(AdaptiveUpgradedBot(t_id, total_rounds, rng))
            elif grp == "Dumper_Upgraded":
                bots.append(DumperUpgradedBot(t_id, total_rounds, rng))
            else:
                bots.append(GreedyUpgradedBot(t_id, total_rounds, rng))

        inventories = {}
        game_prof = {b.team_id: 0.0 for b in bots}
        game_rej = {b.team_id: 0 for b in bots}
        for b in bots:
            b.reset()
            inventories[b.team_id] = arena.generate_lots()

        caps = {t: max(1, sum(l.bags for l in lots) // 4) for t, lots in inventories.items()}
        tot_supply = sum(l.bags for lots in inventories.values() for l in lots)
        quotas = arena.plan_quotas(tot_supply, demand_ratio=0.80)

        for rnd in range(1, total_rounds + 1):
            routes = arena.generate_routes(quotas[rnd - 1])
            orders = []
            for b in bots:
                orders.extend(b.act(rnd, inventories[b.team_id], routes))

            fb, rnd_prof = arena.clear_market(inventories, routes, orders, caps)
            r_map = {r.order_id: r for r in routes}
            for b in bots:
                t = b.team_id
                tv_map = {l.lot_id: l.TV for l in inventories[t]}
                b.observe_feedback(fb[t], r_map, tv_map)
                game_prof[t] += rnd_prof[t]
                game_rej[t] += sum(1 for x in fb[t] if "ไม่ผ่านสเปก" in x["status"])

        game_liq = {}
        for t, lots in inventories.items():
            liq_loss = 0.0
            for l in lots:
                if l.bags > 0:
                    fp = round(l.TV_true * 0.80, 2)
                    prof_l = (fp - l.TV_true) * l.bags
                    game_prof[t] += prof_l
                    liq_loss += abs(prof_l)
                    l.bags = 0
            game_liq[t] = liq_loss

        sorted_t = sorted(game_prof.keys(), key=lambda x: game_prof[x], reverse=True)
        rank_map = {t: rk + 1 for rk, t in enumerate(sorted_t)}

        for b in bots:
            t = b.team_id
            grp = team_grp[t]
            stats[grp]["prof"].append(game_prof[t])
            stats[grp]["rank"].append(rank_map[t])
            stats[grp]["top10"].append(1.0 if rank_map[t] <= 10 else 0.0)
            stats[grp]["liq"].append(game_liq[t])
            stats[grp]["rejects"].append(game_rej[t])

    ordered = sorted(groups, key=lambda g: np.mean(stats[g]["prof"]), reverse=True)
    print(f"{'อันดับ':<6} | {'สายพันธุ์บอท':<26} | {'กำไรเหนือ TV':<15} | {'อันดับเฉลี่ย':<13} | {'ติด Top 10':<10} | {'โดนปรับจบเกม':<12} | {'โดนตีตกสเปก'}")
    print("-" * 108)
    for i, grp in enumerate(ordered, 1):
        marker = "👑" if "ทีมเรา" in grp else "  "
        print(
            f"{marker}#{i:<4} | {grp:<26} | {np.mean(stats[grp]['prof']):+11.2f} บ. | "
            f"{np.mean(stats[grp]['rank']):6.1f} / 55   | {np.mean(stats[grp]['top10'])*100:6.1f}%   | "
            f"{np.mean(stats[grp]['liq']):8.2f} บ. | {np.mean(stats[grp]['rejects']):6.1f} ใบ"
        )
    print("=" * 108 + "\n")


if __name__ == "__main__":
    run_benchmark(total_rounds=9, event_wave=False, num_games=400)
    run_benchmark(total_rounds=16, event_wave=True, num_games=40)