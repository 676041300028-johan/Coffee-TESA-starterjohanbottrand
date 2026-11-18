from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BagOrderAction:
  rfq_id: str
  buyer_name: str
  lot_id: str
  bags: int
  bid_price: float
  target_ratio: float
  tv: float


class TerminalLiquidationRiskManager:
  """
    ตัวคุมความเสี่ยงรอบสุดท้าย (รอบ 1 ถึง total_rounds):
    คำนวณแรงกดดันในการระบายของ (Urgency) เพื่อปรับลดราคาเสนอขายหนีรอบ Liquidation (0.80 * TV)
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
  ) -> Dict[str, float]:
    """
        คืนค่า:
          - urgency (0.0 ถึง 1.0): 0 = สบายๆ, 1 = วิกฤตต้องระบายของให้หมดเดี๋ยวนี้
          - ratio_discount: ส่วนลด Ratio ที่ต้องหักออกจากเพดานเพื่อตัดราคาคู่แข่ง
          - min_acceptable_ratio: อัตราส่วนต่ำสุดที่ยอมขายได้ในรอบนี้ (ต้องมากกว่า 0.80 เสมอ)
        """
    rounds_left = max(1, self.total_rounds - current_round + 1)

    # 1. ความเสี่ยงด้านเวลา (Time Decay Risk)
    # รอบแรกๆ = ต่ำ, รอบรองสุดท้าย = สูง, รอบสุดท้าย = 1.0
    time_pressure = (current_round / self.total_rounds) ** 2

    # 2. ความเสี่ยงด้านปริมาณของล้นมือเทียบกับกำลังซื้อตลาดในรอบนี้
    demand_ratio = remaining_bags / max(1, total_market_demand)
    supply_pressure = min(1.0, demand_ratio * 2.5)  # ถ้าของเรากินสัดส่วน >40% ของตลาด ถือว่ากดดันสูง

    # รวมความเร่งด่วน (0.0 - 1.0)
    urgency = min(1.0, 0.65 * time_pressure + 0.35 * supply_pressure)

    # ในรอบสุดท้าย (rounds_left == 1) บังคับ urgency = 1.0 เพื่อหนี 0.80 * TV
    if rounds_left == 1:
      urgency = 1.0

    # คำนวณส่วนลด Ratio จากเพดาน (สูงสุดไม่เกิน 0.08 เพื่อไม่ให้ต่ำกว่า 0.85)
    ratio_discount = round(0.005 + (urgency * 0.065), 4)

    # ราคาพื้นของรอบนี้ (ยิ่งใกล้รอบสุดท้าย ยิ่งยอมลดเข้าใกล้ 0.81 แต่ต้องชนะ 0.80)
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
    ตัวกระจายกระสอบรายใบ (4 Lots x 8 Routes) ด้วยหลักการ Opportunity-Cost Greedy Allocator
    และ Price Laddering (ซอยราคาขั้นบันได)
    """

  def __init__(self, risk_manager: TerminalLiquidationRiskManager):
    self.risk_manager = risk_manager

  def route_bags(
      self,
      current_round: int,
      lots: List,  # List[CoffeeLot]
      routes: List,  # List[BuyerOrder]
      action_mask: np.ndarray,  # Boolean Matrix (4 Lots x 8 Routes)
      ceiling_estimator,  # OnlinePriceCeilingEstimator
      rl_aggressiveness: float = 0.5,  # ค่าที่รับมาจาก RL Agent [0.0 = เน้นขายชัวร์, 1.0 = เน้นกำไรชิดเพดาน]
      ladder_spread: float = 0.015,  # ช่วงกระจายราคาขั้นบันได (เช่น 1.5%)
  ) -> List[BagOrderAction]:
    """
        จับคู่กระสอบทั้ง 24-30 ใบ เข้ากับ rt01 - rt08 พร้อมคำนวณราคาเสนอขาย (บาท/กก.)
        """
    total_remaining_bags = sum(lot.bags for lot in lots)
    total_market_demand = sum(
        r.quota_bags for r in routes if r.quota_bags > 0
    )

    # 1. ประเมินความเสี่ยงรอบสุดท้าย
    risk = self.risk_manager.compute_risk_adjustment(
        current_round=current_round,
        remaining_bags=total_remaining_bags,
        total_market_demand=total_market_demand,
    )

    # 2. ประเมินเพดาน Ratio ของแต่ละ Route ล่วงหน้า
    route_ceilings = {}
    for j, r in enumerate(routes):
      est = ceiling_estimator.predict_max_ratio(
          company_name=getattr(r, "company_name", r.order_id),
          buyer_type=r.buyer_type,
          safety_margin=(
              1.0 - rl_aggressiveness
          ),  # ถ้า RL สั่งดุดัน (1.0) margin จะเป็น 0.0 (ชิดเพดานบน)
      )
      route_ceilings[j] = est["recommended_ratio"]

    # 3. จัดลำดับความสำคัญของ Lot:
    # ส่ง Lot ที่ "มีทางเลือกน้อยกว่า" (เช่น ขายได้เฉพาะ Instant) หรือ "TV สูงสุด" ไปจองโควตาก่อน
    # เพื่อป้องกันไม่ให้ Lot พรีเมียมไปแย่งโควตา Instant แล้วเหลือ Lot เกรดรองค้างมือ
    lot_indices = list(range(len(lots)))
    lot_indices.sort(
        key=lambda i: (
            int(np.sum(action_mask[i])),  # ทางเลือกน้อยจัดสรรให้ก่อน
            -lots[i].TV,  # ถ้าทางเลือกเท่ากัน เอา TV สูงลงก่อนเพื่อทำกำไร
        )
    )

    # ติดตามโควตาคงเหลือของแต่ละ Route (ไม่ส่งเกินโควตาผู้ซื้อรายนั้น)
    route_remaining_cap = {j: r.quota_bags for j, r in enumerate(routes)}
    lot_remaining_bags = {i: lots[i].bags for i in range(len(lots))}

    final_orders: List[BagOrderAction] = []

    # 4. วนลูปจัดสรรกระสอบเข้า Route ที่ให้ผลตอบแทน (Expected Price) สูงสุด
    for i in lot_indices:
      lot = lots[i]
      if lot_remaining_bags[i] <= 0:
        continue

      # หา Route ทั้งหมดที่ Lot นี้ผ่าน Action Mask และยังมีโควตาเหลือ
      valid_routes = [
          j
          for j in range(len(routes))
          if action_mask[i, j] and route_remaining_cap[j] > 0
      ]

      if not valid_routes:
        continue

      # เรียง Route ตามเพดาน Ratio จากมากไปน้อย (เช่น Commercial 1.03 มาก่อน Instant 0.98)
      valid_routes.sort(key=lambda j: route_ceilings[j], reverse=True)

      for j in valid_routes:
        if lot_remaining_bags[i] <= 0:
          break

        order = routes[j]
        # จำกัดจำนวนกระสอบที่ส่งให้ Route นี้ ไม่ให้เกินโควตาผู้ซื้อ (และกระจายความเสี่ยงไม่ทุ่มหมดหน้าตักถ้ามีหลายเจ้า)
        max_send = min(lot_remaining_bags[i], route_remaining_cap[j])
        if max_send <= 0:
          continue

        # คำนวณเพดานเป้าหมายหลังหักส่วนลดความเสี่ยงรอบท้าย (Liquidation Risk)
        base_ceiling = route_ceilings[j]
        # ถ้า RL สั่งเน้นขายชัวร์ (aggressiveness ต่ำ) หรือใกล้รอบสุดท้าย (urgency สูง) ราคาจะถูกกดลงมาสู้คู่แข่ง
        effective_top_ratio = base_ceiling - risk["ratio_discount"] * (
            1.0 - 0.5 * rl_aggressiveness
        )
        effective_top_ratio = max(
            risk["min_acceptable_ratio"], effective_top_ratio
        )

        # 5. สร้าง Price Laddering (ซอยราคาเป็นขั้นบันไดสำหรับกระสอบแต่ละใบในล็อต)
        # เพื่อป้องกันการโดนตัดราคาทั้งล็อตพร้อมกัน
        for bag_idx in range(max_send):
          # ใบแรกตั้งราคาสูงสุด ใบถัดไปลดหลั่นลงมาตาม ladder_spread
          step_drop = (
              (bag_idx / max(1, max_send))
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
                  buyer_name=getattr(order, "company_name", order.order_id),
                  lot_id=lot.lot_id,
                  bags=1,  # แยกราย 1 กระสอบแบบเดียวกับ R3 เพื่อความยืดหยุ่นสูงสุดในการจับคู่
                  bid_price=bid_price,
                  target_ratio=bag_ratio,
                  tv=lot.TV,
              )
          )

        lot_remaining_bags[i] -= max_send
        route_remaining_cap[j] -= max_send

    return final_orders


# ==========================================
# ทดสอบการทำงานร่วมกันทั้งหมด (End-to-End Test)
# ==========================================
if __name__ == "__main__":
  from dataclasses import dataclass

  # จำลองคลาสแบบย่อสำหรับทดสอบรัน
  @dataclass
  class MockLot:
    lot_id: str
    TV: float
    bags: int

  @dataclass
  class MockRoute:
    order_id: str
    company_name: str
    buyer_type: str
    quota_bags: int

  class MockEstimator:

    def predict_max_ratio(self, company_name, buyer_type, safety_margin):
      return {
          "recommended_ratio": 1.028 if buyer_type == "COMMERCIAL" else 0.978
      }

  lots = [
      MockLot("LOT-1 (Premium)", TV=135.0, bags=6),  # ผ่านหมด (2 routes)
      MockLot("LOT-2 (InstOnly)", TV=110.0, bags=8),  # ผ่านเฉพาะ Instant (1 route)
  ]
  routes = [
      MockRoute("R6-01", "3 สยามคอมเมิร์ซ", "COMMERCIAL", quota_bags=5),
      MockRoute("R6-02", "6 ทรีอินวันฟู้ดส์", "INSTANT", quota_bags=10),
  ]
  # LOT-1 ส่งได้ทั้ง 2 เจ้า, LOT-2 ส่งได้เฉพาะ R6-02
  mask = np.array([[True, True], [False, True]])

  risk_mgr = TerminalLiquidationRiskManager(total_rounds=7)
  router = BagLevelCombinatorialRouter(risk_mgr)

  # ทดสอบสถานการณ์รอบที่ 6 (ใกล้รอบสุดท้าย ของยังเหลือ 14 กระสอบ)
  orders = router.route_bags(
      current_round=6,
      lots=lots,
      routes=routes,
      action_mask=mask,
      ceiling_estimator=MockEstimator(),
      rl_aggressiveness=0.3,  # เน้นระบายของหนีรอบ 7
  )

  print(
      f"{'RFQ':<7} | {'ผู้ซื้อ':<18} | {'LOT':<16} | {'กระสอบ':<6} | {'TV':<7} |"
      f" {'Ratio':<7} | {'ราคาเสนอ (บ./กก.)'}"
  )
  print("-" * 85)
  for o in orders:
    print(
        f"{o.rfq_id:<7} | {o.buyer_name:<18} | {o.lot_id:<16} | {o.bags:<6} |"
        f" {o.tv:<7.2f} | {o.target_ratio:<7.4f} | {o.bid_price:.2f}"
    )