import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from market_env import CoffeeAuctionEnv


def make_env(
    rank: int,
    seed: int = 0,
    total_rounds: int = 7,
    num_opponents: int = 54,
    market_demand_ratio: float = 0.80,
):
    """สร้าง Environment ย่อยสำหรับรันขนานหลาย CPU (55 ทีม | ตลาดรับซื้อ 80%)"""
    def _init():
        env = CoffeeAuctionEnv(
            total_rounds=total_rounds,
            num_opponents=num_opponents,
            market_demand_ratio=market_demand_ratio,
            ceiling_mode="MIXED",
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class FileBasedSelfPlayCallback(BaseCallback):
    """
    อัปเดตโมเดล Self-Play บน Windows โดยเซฟเป็นไฟล์ .zip ลงดิสก์
    แล้วส่ง String Path ให้แต่ละ Process โหลดแทนการส่งฟังก์ชัน (แก้ปัญหา Pickle บน Windows 100%)
    """
    def __init__(self, save_path: Path, update_freq: int = 12_500, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.update_freq == 0:
            self.model.save(str(self.save_path))
            self.training_env.env_method("load_self_play_from_path", str(self.save_path))
            if self.verbose > 0:
                print(f"\n🔄 [Step {self.num_timesteps:,}] อัปเดตโมเดล Self-Play ({self.save_path.name}) เข้าทั้ง 4 Cores สำเร็จ!")
        return True


def train_bot_v2(
    total_timesteps: int = 500_000,
    num_envs: int = 4,
    total_rounds: int = 7,
    num_opponents: int = 54,
    market_demand_ratio: float = 0.80,
):
    models_dir = CURRENT_DIR / "models"
    logs_dir = CURRENT_DIR / "tb_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🚀 เริ่มต้นการเทรน AI Trading Bot V2 ({total_timesteps:,} timesteps | "
        f"{1 + num_opponents} ทีม | ตลาดรับซื้อ {market_demand_ratio*100:.0f}%)"
    )

    if num_envs > 1:
        train_env = SubprocVecEnv(
            [
                make_env(i, seed=200, total_rounds=total_rounds, num_opponents=num_opponents, market_demand_ratio=market_demand_ratio)
                for i in range(num_envs)
            ]
        )
    else:
        train_env = DummyVecEnv(
            [
                make_env(0, seed=200, total_rounds=total_rounds, num_opponents=num_opponents, market_demand_ratio=market_demand_ratio)
            ]
        )

    eval_env = DummyVecEnv(
        [
            make_env(99, seed=999, total_rounds=total_rounds, num_opponents=num_opponents, market_demand_ratio=market_demand_ratio)
        ]
    )

    # ปรับจูน PPO V2: ลด ent_coef=0.001 เพื่อบีบ std ให้แคบและตัดสินใจนิ่งขึ้น
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        tensorboard_log=str(logs_dir),
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(logs_dir),
        eval_freq=max(5_000, 20_000 // num_envs),
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000, 50_000 // num_envs),
        save_path=str(models_dir / "checkpoints"),
        name_prefix="coffee_rl_v2",
    )

    selfplay_callback = FileBasedSelfPlayCallback(
        save_path=models_dir / "selfplay_latest.zip",
        update_freq=max(12_500, 50_000 // num_envs),
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, selfplay_callback],
        progress_bar=True,
    )

    final_path = models_dir / "coffee_rl_v2_final.zip"
    model.save(str(final_path))
    print(f"\n✅ เทรน V2 เสร็จสมบูรณ์! บันทึกโมเดลสุดท้ายที่: {final_path}")
    print(f"🏆 โมเดลที่ทำคะแนนสูงสุดถูกบันทึกไว้ที่: {models_dir / 'best' / 'best_model.zip'}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    # เทรน 500,000 steps บน 4 Cores (ใช้เวลาประมาณ 2 ชั่วโมง)
    train_bot_v2(
        total_timesteps=500_000,
        num_envs=4,
        total_rounds=7,
        num_opponents=54,
        market_demand_ratio=0.80,
    )