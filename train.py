import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from market_env import CoffeeAuctionEnv

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def make_env(total_rounds: int, rank: int, is_eval: bool = False):
    def _init():
        seed = (9000 + rank) if is_eval else (100 + rank)
        env = CoffeeAuctionEnv(num_teams=55, total_rounds=total_rounds, seed=seed)
        return Monitor(env)
    return _init


def train_profile(total_rounds: int, profile_name: str, timesteps: int = 300_000, set_as_default_best: bool = False):
    profile_dir = MODELS_DIR / profile_name
    best_dir = profile_dir / "best"
    ckpt_dir = profile_dir / "checkpoints"
    log_dir = profile_dir / "logs"

    for d in (best_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 เริ่มเทรนโมเดลโหมด {total_rounds} รอบ | Profile: {profile_name} ({timesteps:,} steps)")
    print(f"📁 โฟลเดอร์เก็บโมเดล: {profile_dir}")
    print("=" * 80)

    num_cpu = 4
    train_env = SubprocVecEnv([make_env(total_rounds, i, is_eval=False) for i in range(num_cpu)])
    eval_env = SubprocVecEnv([make_env(total_rounds, 0, is_eval=True)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(5000 // num_cpu, 1000),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    ckpt_callback = CheckpointCallback(
        save_freq=max(50_000 // num_cpu, 5000),
        save_path=str(ckpt_dir),
        name_prefix=profile_name,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,  # มองการณ์ไกลข้ามช่วง Event Wave 4 รอบ
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps, callback=[eval_callback, ckpt_callback])

    final_path = MODELS_DIR / f"{profile_name}_final.zip"
    model.save(str(final_path))

    # คัดลอก best_model.zip ไปไว้ที่ models/best/best_model.zip เพื่อให้ evaluate.py โหลดได้ทันที
    if set_as_default_best:
        default_best_dir = MODELS_DIR / "best"
        default_best_dir.mkdir(parents=True, exist_ok=True)
        best_src = best_dir / "best_model.zip"
        if best_src.exists():
            import shutil
            shutil.copy2(best_src, default_best_dir / "best_model.zip")
            shutil.copy2(best_src, MODELS_DIR / "coffee_rl_v2_final.zip")
            print(f"⭐ อัปเดต Default Best Model -> {default_best_dir / 'best_model.zip'}")

    train_env.close()
    eval_env.close()
    print(f"✅ เสร็จสิ้นการเทรน {profile_name} -> บันทึกที่ {final_path}
")


if __name__ == "__main__":
    # 1. เทรนโหมดปัจจุบัน (9 รอบ) และตั้งเป็น Default Best Model
    train_profile(
        total_rounds=9,
        profile_name="coffee_rl_9rounds",
        timesteps=300_000,
        set_as_default_best=True,
    )

    # 2. เทรนโหมดรอบใหญ่ (16 รอบ พร้อม Event Wave สลับช่วงละ 4 รอบ)
    train_profile(
        total_rounds=16,
        profile_name="coffee_rl_16rounds_event",
        timesteps=400_000,
        set_as_default_best=False,
    )
