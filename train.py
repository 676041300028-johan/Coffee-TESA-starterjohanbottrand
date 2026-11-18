from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from market_env import CoffeeAuctionEnv

def make_env(total_rounds: int, rank: int):
    def _init():
        return CoffeeAuctionEnv(num_teams=55, total_rounds=total_rounds, seed=100 + rank)
    return _init

def train_profile(total_rounds: int, save_name: str, timesteps: int = 300_000):
    print(f"=== เริ่มเทรนโมเดลโหมด {total_rounds} รอบ -> {save_name}.zip ===")
    num_cpu = 4
    env = SubprocVecEnv([make_env(total_rounds, i) for i in range(num_cpu)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,  # ค่า gamma สูงเพื่อให้ AI มองการณ์ไกลข้าม Event Wave 4 รอบได้
        ent_coef=0.01,
        verbose=1
    )
    model.learn(total_timesteps=timesteps)
    model.save(save_name)
    env.close()
    print(f"=== บันทึก {save_name}.zip เรียบร้อย! ===\n")

if __name__ == "__main__":
    # 1. เทรนโหมดปัจจุบัน (9 รอบ)
    train_profile(total_rounds=9, save_name="model_9rounds", timesteps=300_000)

    # 2. เทรนโหมดรอบใหญ่ (16 รอบ พร้อม Event Wave สลับช่วงละ 4 รอบ)
    train_profile(total_rounds=16, save_name="model_16rounds_event", timesteps=400_000)