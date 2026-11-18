from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from coffee_math import BUYER_SPECS


@dataclass
class CompanyCeilingProfile:
    company_name: str
    buyer_type: str  # "COMMERCIAL" หรือ "INSTANT"
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
    ดึงค่าตั้งต้นจาก BUYER_SPECS ใน coffee_math.py เพื่อให้เป็นมาตรฐานเดียวกันทั้งระบบ
    """

    TYPE_PRIORS = {
        "COMMERCIAL": {
            "init_min": 1.020,
            "init_max": 1.040,
            "safe_fallback": BUYER_SPECS["COMMERCIAL"]["ratio_min"],  # 1.025
            "mean": BUYER_SPECS["COMMERCIAL"]["ratio_mid"],           # 1.030
        },
        "INSTANT": {
            "init_min": 0.970,
            "init_max": 0.990,
            "safe_fallback": BUYER_SPECS["INSTANT"]["ratio_min"],     # 0.975
            "mean": BUYER_SPECS["INSTANT"]["ratio_mid"],              # 0.980
        },
    }

    def __init__(self, tolerance: float = 1e-4):
        self.profiles: Dict[str, CompanyCeilingProfile] = {}
        self.tolerance = tolerance
        self.type_passed_ratios: Dict[str, List[float]] = {
            "COMMERCIAL": [],
            "INSTANT": [],
        }

    def _normalize_type(self, buyer_type: str) -> str:
        return (
            "COMMERCIAL"
            if "comm" in buyer_type.strip().lower()
            else "INSTANT"
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
        - rfq_to_company: dict จับคู่ RFQ กับ (ชื่อบริษัท, ประเภท) เช่น {"R3-02": ("3 สยามคอมเมิร์ซ", "COMMERCIAL")}
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
            if tv <= 0:
                continue
            ratio = round(bid_price / tv, 5)

            prof = self._get_profile(company_name, raw_type)
            prof.history.append((rfq_id, bid_price, ratio, status))

            # แยกกลุ่มสถานะตามข้อความจริงจากระบบและ Simulator
            is_passed = any(
                k in status
                for k in ["ขายได้", "ขายได้ทั้งหมด", "ขายได้บางส่วน", "ราคาแพ้", "Sold", "Partially", "Outbid"]
            )
            is_exceeded = any(
                k in status
                for k in ["แพงเกินเพดาน", "เกินเพดาน", "Exceeds"]
            )

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

    def get_summary_features(self) -> Tuple[float, float, float]:
        """
        สรุปค่าสถิติสำหรับส่งเข้า Observation Space ของ RL Agent (Normalize แล้ว):
        คืนค่า: (avg_comm_width, avg_inst_width, randomized_ratio)
        """
        comm_widths = []
        inst_widths = []
        rand_count = 0

        for prof in self.profiles.values():
            w = max(0.0, prof.max_known_ratio - prof.min_known_ratio)
            if prof.buyer_type == "COMMERCIAL":
                comm_widths.append(w)
            else:
                inst_widths.append(w)
            if prof.is_randomized:
                rand_count += 1

        avg_comm_w = float(np.mean(comm_widths)) if comm_widths else 0.020
        avg_inst_w = float(np.mean(inst_widths)) if inst_widths else 0.020
        rand_ratio = (rand_count / len(self.profiles)) if self.profiles else 0.0

        return round(avg_comm_w, 5), round(avg_inst_w, 5), round(rand_ratio, 4)


if __name__ == "__main__":
    estimator = OnlinePriceCeilingEstimator()

    lot_tv_r3 = {
        "LOT-T25-1": 122.00,
        "LOT-T25-4": 135.00,
    }
    rfq_map_r3 = {
        "R3-02": ("3 สยามคอมเมิร์ซ", "COMMERCIAL"),
    }
    r3_feedback = [
        {"rfq": "R3-02", "lot": "LOT-T25-4", "price": 150.0, "status": "แพงเกินเพดานผู้ซื้อ"},
        {"rfq": "R3-02", "lot": "LOT-T25-1", "price": 140.0, "status": "แพงเกินเพดานผู้ซื้อ"},
        {"rfq": "R3-02", "lot": "LOT-T25-1", "price": 125.2, "status": "ขายได้ทั้งหมด"},
    ]

    estimator.update_from_round_results(r3_feedback, rfq_map_r3, lot_tv_r3)
    print("ผลลัพธ์หลังรอบ R3:", estimator.predict_max_ratio("3 สยามคอมเมิร์ซ", "COMMERCIAL"))
    print("ฟีเจอร์สำหรับ RL:", estimator.get_summary_features())
