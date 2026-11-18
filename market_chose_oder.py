from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import List
import numpy as np
from find_tv_lot import CoffeeLot, analyze_4_lots
# ==========================================
# 1. ดึง CoffeeLot มาจากไฟล์ "find tv lot.py"
# ==========================================
file_path = Path(__file__).parent / "find tv lot.py"
spec = importlib.util.spec_from_file_location("find_tv_lot", file_path)
find_tv_lot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(find_tv_lot)

CoffeeLot = find_tv_lot.CoffeeLot  # ดึงคลาส CoffeeLot มาใช้งาน


# ==========================================
# 2. โครงสร้างข้อมูลคำสั่งซื้อผู้ซื้อ (BuyerOrder)
# ==========================================
@dataclass
class BuyerOrder:
  order_id: str
  buyer_type: str  # "COMMERCIAL" หรือ "INSTANT"
  max_moisture: float  # เกณฑ์ความชื้นสูงสุดที่รับ
  max_defect: float  # เกณฑ์เมล็ดเสียสูงสุดที่รับ
  quota_bags: int  # จำนวนกระสอบที่ต้องการซื้อ
  max_ratio: float  # เพดานราคา Ratio


# ==========================================
# 3. ตัวกรองเงื่อนไข (Action Masking)
# ==========================================
def build_action_mask(
    lots: List[CoffeeLot], routes: List[BuyerOrder]
) -> np.ndarray:
  num_lots = len(lots)  # 4 Lots
  num_routes = len(routes)  # 8 Routes (rt01 - rt08)
  mask = np.zeros((num_lots, num_routes), dtype=bool)

  for i, lot in enumerate(lots):
    if lot.bags <= 0:
      continue

    for j, order in enumerate(routes):
      if order.quota_bags <= 0:
        continue

      # ใช้ชื่อฟิลด์ moisture และ defect ตามไฟล์ find tv lot.py[cite: 1]
      if lot.moisture > order.max_moisture or lot.defect > order.max_defect:
        continue

      mask[i, j] = True

  return mask


# ==========================================
# 4. ทดสอบรัน
# ==========================================
if __name__ == "__main__":
  # ส่งค่าใช้ชื่อพารามิเตอร์ screen, moisture, defect ตาม find tv lot.py[cite: 1]
  my_lots = [
      CoffeeLot("LOT-1", screen=16, moisture=11.8, defect=0.9, bags=6),
      CoffeeLot("LOT-2", screen=14, moisture=12.5, defect=4.5, bags=8),
      CoffeeLot("LOT-3", screen=15, moisture=13.2, defect=3.2, bags=7),
      CoffeeLot("LOT-4", screen=13, moisture=13.9, defect=11.5, bags=0),
  ]

  market_routes = [
      BuyerOrder(
          "rt01",
          "COMMERCIAL",
          max_moisture=12.5,
          max_defect=7.0,
          quota_bags=15,
          max_ratio=1.03,
      ),
      BuyerOrder(
          "rt02",
          "COMMERCIAL",
          max_moisture=12.5,
          max_defect=7.0,
          quota_bags=10,
          max_ratio=1.03,
      ),
      BuyerOrder(
          "rt03",
          "COMMERCIAL",
          max_moisture=12.5,
          max_defect=7.0,
          quota_bags=0,
          max_ratio=1.03,
      ),
      BuyerOrder(
          "rt04",
          "INSTANT",
          max_moisture=14.0,
          max_defect=15.0,
          quota_bags=20,
          max_ratio=0.98,
      ),
      BuyerOrder(
          "rt05",
          "INSTANT",
          max_moisture=14.0,
          max_defect=15.0,
          quota_bags=25,
          max_ratio=0.98,
      ),
      BuyerOrder(
          "rt06",
          "INSTANT",
          max_moisture=14.0,
          max_defect=15.0,
          quota_bags=15,
          max_ratio=0.98,
      ),
      BuyerOrder(
          "rt07",
          "COMMERCIAL",
          max_moisture=12.5,
          max_defect=3.0,
          quota_bags=10,
          max_ratio=1.04,
      ),
      BuyerOrder(
          "rt08",
          "INSTANT",
          max_moisture=14.0,
          max_defect=15.0,
          quota_bags=10,
          max_ratio=0.98,
      ),
  ]

  action_mask = build_action_mask(my_lots, market_routes)

  print("ตารางสิทธิ์การส่งออก (Action Mask: 1 = ส่งได้, 0 = บล็อก):")
  header = "LOT \\ ROUTE | " + " | ".join([r.order_id for r in market_routes])
  print(header)
  print("-" * len(header))
  for i, lot in enumerate(my_lots):
    row_str = " | ".join([f"  {int(val)} " for val in action_mask[i]])
    print(f"{lot.lot_id:<11} | {row_str}")