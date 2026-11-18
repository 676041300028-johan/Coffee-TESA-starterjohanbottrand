from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class CompanyCeilingProfile:
  company_name: str
  buyer_type: str  # "Commercial" หรือ "Instant"
  min_known_ratio: float
  max_known_ratio: float
  is_randomized: bool = False
  conflict_count: int = 0
  history: List[Tuple[str, float, float, str]] = field(
      default_factory=list
  )  # (rfq_id, price, ratio, status)


class OnlinePriceCeilingEstimator:
  """
    ระบบแกะรอยเพดาน Ratio (Bid / TV) จากผลลัพธ์รอบล่าสุด (บาท/กก.)
    แยกตาม 'ชื่อบริษัท' และ 'ประเภทผู้ซื้อ' พร้อมตรวจจับการสุ่มเพดานใหม่ทุกรอบ
    """

  TYPE_PRIORS = {
      "Commercial": {
          "init_min": 1.020,
          "init_max": 1.040,
          "safe_fallback": 1.025,  # ค่าปลอดภัยเมื่อพบว่าสุ่มเพดานใหม่ทุกรอบ
          "mean": 1.030,
      },
      "Instant": {
          "init_min": 0.970,
          "init_max": 0.990,
          "safe_fallback": 0.975,  # ค่าปลอดภัยเมื่อพบว่าสุ่มเพดานใหม่ทุกรอบ
          "mean": 0.980,
      },
  }

  def __init__(self, tolerance: float = 1e-4):
    self.profiles: Dict[str, CompanyCeilingProfile] = {}
    self.tolerance = tolerance
    self.type_passed_ratios: Dict[str, List[float]] = {
        "Commercial": [],
        "Instant": [],
    }

  def _normalize_type(self, buyer_type: str) -> str:
    return (
        "Commercial"
        if "commercial" in buyer_type.lower() or "comm" in buyer_type.lower()
        else "Instant"
    )

  def _get_profile(
      self, company_name: str, buyer_type: str
  ) -> CompanyCeilingProfile:
    b_type = self._normalize_type(buyer_type)
    if company_name not in self.profiles:
      prior = self.TYPE_PRIORS[b_type]
      self.profiles[company_name] = CompanyCeilingProfile(
          company_name=company_name,
          buyer_type=b_type,
          min_known_ratio=prior["init_min"],
          max_known_ratio=prior["init_max"],
      )
    return self.profiles[company_name]

  def update_from_round_results(
      self,
      my_results: List[Dict],
      rfq_to_company: Dict[str, Tuple[str, str]],
      lot_tv_map: Dict[str, float],
  ):
    """
        อัปเดตข้อมูลจากตาราง 'ผลรอบล่าสุด'
        - my_results: list ของ dict เช่น {"rfq": "R3-02", "lot": "LOT-T25-1", "price": 125.2, "status": "ขายได้ทั้งหมด"}
        - rfq_to_company: dict จับคู่ RFQ กับ (ชื่อบริษัท, ประเภท) เช่น {"R3-02": ("3 สยามคอมเมิร์ซ", "Commercial")}
        - lot_tv_map: dict ค่า TV ของแต่ละ Lot เช่น {"LOT-T25-1": 122.00}
        """
    for row in my_results:
      rfq_id = row["rfq"]
      lot_id = row["lot"]
      bid_price = float(row["price"])
      status = row["status"].strip()

      if rfq_id not in rfq_to_company or lot_id not in lot_tv_map:
        continue

      company_name, raw_type = rfq_to_company[rfq_id]
      tv = lot_tv_map[lot_id]
      ratio = round(bid_price / tv, 5)

      prof = self._get_profile(company_name, raw_type)
      prof.history.append((rfq_id, bid_price, ratio, status))

      # แยกกลุ่มสถานะตามข้อความจริงจากระบบ
      is_passed = any(
          k in status
          for k in ["ขายได้", "ขายได้ทั้งหมด", "ขายได้บางส่วน", "ราคาแพ้"]
      )
      is_exceeded = "แพงเกินเพดาน" in status or "เกินเพดาน" in status

      if not is_passed and not is_exceeded:
        continue  # กรณี "ไม่ผ่านเกณฑ์คุณภาพ" ไม่นำมาคำนวณเพดานราคา

      if is_passed:
        # ตรวจจับ Regime Change: เคยโดนตีตกที่ Ratio ต่ำกว่านี้ แต่รอบนี้กลับขายผ่าน
        if ratio >= prof.max_known_ratio - self.tolerance:
          prof.conflict_count += 1
          prof.is_randomized = True
          prior = self.TYPE_PRIORS[prof.buyer_type]
          prof.min_known_ratio = prior["init_min"]
          prof.max_known_ratio = prior["init_max"]
        else:
          prof.min_known_ratio = max(prof.min_known_ratio, ratio)
          self.type_passed_ratios[prof.buyer_type].append(ratio)

      elif is_exceeded:
        # ตรวจจับ Regime Change: เคยขายผ่านที่ Ratio สูงกว่านี้ แต่รอบนี้กลับโดนตีตก
        if ratio <= prof.min_known_ratio + self.tolerance:
          prof.conflict_count += 1
          prof.is_randomized = True
          prior = self.TYPE_PRIORS[prof.buyer_type]
          prof.min_known_ratio = prior["init_min"]
          prof.max_known_ratio = prior["init_max"]
        else:
          prof.max_known_ratio = min(prof.max_known_ratio, ratio)

  def predict_max_ratio(
      self, company_name: str, buyer_type: str, safety_margin: float = 0.25
  ) -> Dict:
    """
        คืนค่า Ratio สูงสุดที่แนะนำให้เสนอขายสำหรับบริษัทและประเภทนั้นในรอบถัดไป
        safety_margin: 0.0 (ชิดขอบบน Max Known) ถึง 1.0 (ชิดขอบล่าง Min Known)
        """
    prof = self._get_profile(company_name, buyer_type)
    prior = self.TYPE_PRIORS[prof.buyer_type]

    # หากตรวจพบว่าบริษัทนี้สุ่มเพดานใหม่ทุกรอบ -> สลับมาอิงค่าเฉลี่ย/ค่าปลอดภัยตามประเภท
    if prof.is_randomized:
      obs = self.type_passed_ratios[prof.buyer_type]
      type_avg = float(np.mean(obs)) if len(obs) >= 3 else prior["safe_fallback"]
      rec_ratio = min(type_avg, prior["mean"])
      return {
          "company": company_name,
          "type": prof.buyer_type,
          "mode": "TYPE_FALLBACK (สุ่มเพดานใหม่ทุกรอบ)",
          "min_known_ratio": prior["init_min"],
          "max_known_ratio": prior["init_max"],
          "recommended_ratio": round(rec_ratio, 4),
      }

    # โหมดติดตามรายบริษัท (Interval Estimation)
    width = max(0.0, prof.max_known_ratio - prof.min_known_ratio)
    rec_ratio = prof.min_known_ratio + (1.0 - safety_margin) * width

    return {
        "company": company_name,
        "type": prof.buyer_type,
        "mode": "COMPANY_TRACKING (ติดตามเพดานรายบริษัท)",
        "min_known_ratio": round(prof.min_known_ratio, 4),
        "max_known_ratio": round(prof.max_known_ratio, 4),
        "recommended_ratio": round(rec_ratio, 4),
    }


# ==========================================
# ทดสอบรันด้วยข้อมูลจริงจากรอบ R3 ของคุณ
# ==========================================
if __name__ == "__main__":
  estimator = OnlinePriceCeilingEstimator()

  # สมมติค่า TV ของล็อตในรอบ R3 (ดึงมาจาก find tv lot.py)
  lot_tv_r3 = {
      "LOT-T25-1": 122.00,
      "LOT-T25-4": 135.00,
  }

  # การจับคู่ RFQ ของรอบ R3 กับชื่อบริษัทในตาราง RFQ
  rfq_map_r3 = {
      "R3-02": ("3 สยามคอมเมิร์ซ", "Commercial"),
  }

  # ข้อมูลผลรอบล่าสุด R3 ตามตารางที่คุณส่งมาเป๊ะๆ
  r3_feedback = [
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-4",
          "price": 150.0,
          "status": "แพงเกินเพดานผู้ซื้อ",
      },
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-1",
          "price": 140.0,
          "status": "แพงเกินเพดานผู้ซื้อ",
      },
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-1",
          "price": 120.0,
          "status": "ขายได้ทั้งหมด",
      },
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-1",
          "price": 125.2,
          "status": "ขายได้ทั้งหมด",
      },
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-1",
          "price": 122.0,
          "status": "ขายได้ทั้งหมด",
      },
      {
          "rfq": "R3-02",
          "lot": "LOT-T25-1",
          "price": 118.0,
          "status": "ขายได้ทั้งหมด",
      },
  ]

  estimator.update_from_round_results(r3_feedback, rfq_map_r3, lot_tv_r3)

  print("ผลลัพธ์การแกะรอยเพดานหลังจบรอบ R3:")
  print(estimator.predict_max_ratio("3 สยามคอมเมิร์ซ", "Commercial"))
  print(estimator.predict_max_ratio("4 ชุมพรคั่วบด", "Commercial"))
  print(estimator.predict_max_ratio("6 ทรีอินวันฟู้ดส์", "Instant"))