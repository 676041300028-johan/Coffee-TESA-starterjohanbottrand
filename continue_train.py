import argparse
import shutil
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from market_env import CoffeeAuctionEnv

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def make_env(total_rounds: int, rank: int, is_eval: bool = False, sp_path: str = None, sp_teams: int = 0):
    def _init():
        seed = (19000 + rank) if is_eval else (5000 + rank)
        env = CoffeeAuctionEnv(
            num_teams=55,
            total_rounds=total_rounds,
            seed=seed,
            self_play_model_path=sp_path,
            self_play_teams=sp_teams,
        )
        return Monitor(env)
    return _init


def continue_training(
    total_rounds: int = 16,
    extra_steps: int = 500_000,
    lr: float = 1e-4,
    ent_coef: float = 0.003,
    self_play_teams: int = 8,
    model_path: str = "",
):
    profile_name = "coffee_rl_16rounds_event" if total_rounds == 16 else "coffee_rl_9rounds"
    profile_dir = MODELS_DIR / profile_name
    best_dir = profile_dir / "best"
    ckpt_dir = profile_dir / "checkpoints"
    log_dir = profile_dir / "logs"

    for d in (best_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    load_target = Path(model_path) if model_path else (best_dir / "best_model.zip")
    if not load_target.exists():
        load_target = MODELS_DIR / f"{profile_name}_final.zip"
    if not load_target.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลตั้งต้นสำหรับเทรนต่อ: {load_target}")

    # สร้าง Snapshot ของโมเดลตั้งต้นเพื่อส่งลงไปเป็นคู่แข่ง Self-Play ในสนาม
    sp_snapshot = ckpt_dir / f"{profile_name}_sp_opponent.zip"
    shutil.copy2(load_target, sp_snapshot)

    print("=" * 85)
    print(f"🔄 โหลดโมเดลเดิมมาเทรนต่อ (Self-Play {self_play_teams} ทีม): {load_target}")
    print(f"⚙️  โหมด: {total_rounds} รอบ | เพิ่มอีก {extra_steps:,} steps | LR={lr} | ent_coef={ent_coef}")
    print("=" * 85)

    num_cpu = 4
    sp_arg = str(sp_snapshot) if self_play_teams > 0 else None
    train_env = SubprocVecEnv([
        make_env(total_rounds, i, is_eval=False, sp_path=sp_arg, sp_teams=self_play_teams)
        for i in range(num_cpu)
    ])
    eval_env = SubprocVecEnv([
        make_env(total_rounds, 0, is_eval=True, sp_path=sp_arg, sp_teams=self_play_teams)
    ])

    model = PPO.load(
        str(load_target),
        env=train_env,
        learning_rate=lr,
        ent_coef=ent_coef,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(5000 // num_cpu, 1000),
        n_eval_episodes=30,
        deterministic=True,
    )

    ckpt_callback = CheckpointCallback(
        save_freq=max(50_000 // num_cpu, 5000),
        save_path=str(ckpt_dir),
        name_prefix=f"{profile_name}_ft",
    )

    model.learn(
        total_timesteps=extra_steps,
        callback=[eval_callback, ckpt_callback],
        reset_num_timesteps=False,
    )

    final_path = MODELS_DIR / f"{profile_name}_final.zip"
    model.save(str(final_path))

    if total_rounds == 16 and (best_dir / "best_model.zip").exists():
        default_best = MODELS_DIR / "best"
        default_best.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_dir / "best_model.zip", default_best / "best_model.zip")
        shutil.copy2(best_dir / "best_model.zip", MODELS_DIR / "coffee_rl_v2_final.zip")

    train_env.close()
    eval_env.close()
    print(f"✅ เทรนต่อ Self-Play เสร็จสมบูรณ์! บันทึกตัวเก่งสุดไว้ที่: {best_dir / 'best_model.zip'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=16, help="โหมดรอบที่จะเทรนต่อ (16 หรือ 9)")
    parser.add_argument("--steps", type=int, default=500_000, help="จำนวน steps ที่ต้องการเทรนเพิ่ม")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate สำหรับจูนละเอียด")
    parser.add_argument("--selfplay", type=int, default=8, help="จำนวนทีมคู่แข่ง Self-Play ในสนาม (0 = ปิด)")
    parser.add_argument("--model", type=str, default="", help="ระบุ path โมเดลตั้งต้นเอง")
    args = parser.parse_args()

    continue_training(
        total_rounds=args.rounds,
        extra_steps=args.steps,
        lr=args.lr,
        self_play_teams=args.selfplay,
        model_path=args.model,
    )
