import shutil
from datetime import datetime
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
        seed = (9000 + rank) if is_eval else (100 + rank)
        env = CoffeeAuctionEnv(
            num_teams=55,
            total_rounds=total_rounds,
            seed=seed,
            self_play_model_path=sp_path,
            self_play_teams=sp_teams,
        )
        return Monitor(env)
    return _init


def run_overnight_selfplay_loop(
    total_rounds: int = 16,
    profile_name: str = "coffee_rl_16rounds_event",
    phase1_steps: int = 200_000,
    selfplay_cycle_steps: int = 250_000,
    self_play_teams: int = 8,
    set_as_default_best: bool = True,
):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = MODELS_DIR / profile_name
    best_dir = profile_dir / f"best_{run_tag}"
    ckpt_dir = profile_dir / "checkpoints"
    log_dir = profile_dir / "logs"

    for d in (best_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_cpu = 4

    print("=" * 90)
    print(f"[PHASE 1] เทรนโหมด {total_rounds} รอบ เจอคู่แข่ง Buffer 0.5% ทุกตัว (Dumper 20-40%) | {phase1_steps:,} steps")
    print("=" * 90)

    train_env_p1 = SubprocVecEnv([make_env(total_rounds, i, is_eval=False) for i in range(num_cpu)])
    eval_env_p1 = SubprocVecEnv([make_env(total_rounds, 0, is_eval=True)])

    eval_cb_p1 = EvalCallback(
        eval_env_p1,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(5000 // num_cpu, 1000),
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env_p1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        ent_coef=0.012,
        verbose=1,
    )
    model.learn(total_timesteps=phase1_steps, callback=[eval_cb_p1])

    current_champion = ckpt_dir / f"{profile_name}_gen0_{run_tag}.zip"
    best_src = best_dir / "best_model.zip"
    if best_src.exists():
        shutil.copy2(best_src, current_champion)
    else:
        model.save(str(current_champion))

    train_env_p1.close()
    eval_env_p1.close()

    generation = 1
    while True:
        lr_gen = max(5e-5, 1.5e-4 * (0.85 ** (generation - 1)))
        ent_gen = max(0.002, 0.005 * (0.85 ** (generation - 1)))

        print("=" * 90)
        print(f"[SELF-PLAY GEN #{generation}] อัปเกรดคู่แข่งเป็นตัวเก่งล่าสุด ({self_play_teams} ทีม) | LR={lr_gen:.6f} | {selfplay_cycle_steps:,} steps")
        print("เซฟทับอัตโนมัติตลอดทาง - ตื่นมาตอนไหนกด Ctrl+C หยุดได้เลยครับ!")
        print("=" * 90)

        train_env_sp = SubprocVecEnv([
            make_env(total_rounds, i, is_eval=False, sp_path=str(current_champion), sp_teams=self_play_teams)
            for i in range(num_cpu)
        ])
        eval_env_sp = SubprocVecEnv([
            make_env(total_rounds, 0, is_eval=True, sp_path=str(current_champion), sp_teams=self_play_teams)
        ])

        model_sp = PPO.load(
            str(current_champion),
            env=train_env_sp,
            learning_rate=lr_gen,
            ent_coef=ent_gen,
        )

        eval_cb_sp = EvalCallback(
            eval_env_sp,
            best_model_save_path=str(best_dir),
            log_path=str(log_dir),
            eval_freq=max(5000 // num_cpu, 1000),
            n_eval_episodes=25,
            deterministic=True,
        )
        ckpt_cb_sp = CheckpointCallback(
            save_freq=max(50_000 // num_cpu, 5000),
            save_path=str(ckpt_dir),
            name_prefix=f"{profile_name}_gen{generation}_{run_tag}",
        )

        model_sp.learn(
            total_timesteps=selfplay_cycle_steps,
            callback=[eval_cb_sp, ckpt_cb_sp],
            reset_num_timesteps=False,
        )

        next_champion = ckpt_dir / f"{profile_name}_gen{generation}_{run_tag}.zip"
        if (best_dir / "best_model.zip").exists():
            shutil.copy2(best_dir / "best_model.zip", next_champion)
        else:
            model_sp.save(str(next_champion))
        current_champion = next_champion

        final_path = MODELS_DIR / f"{profile_name}_final.zip"
        shutil.copy2(current_champion, final_path)

        latest_best_dir = profile_dir / "best"
        latest_best_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(current_champion, latest_best_dir / "best_model.zip")
        except PermissionError:
            pass

        if set_as_default_best:
            default_best = MODELS_DIR / "best"
            default_best.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(current_champion, default_best / "best_model.zip")
                shutil.copy2(current_champion, MODELS_DIR / "coffee_rl_v2_final.zip")
            except PermissionError:
                pass

        train_env_sp.close()
        eval_env_sp.close()
        print(f"จบ Self-Play Gen #{generation} -> อัปเดตตัวเก่งสุดเรียบร้อยที่ {final_path}")
        print()
        generation += 1


if __name__ == "__main__":
    run_overnight_selfplay_loop(
        total_rounds=16,
        profile_name="coffee_rl_16rounds_event",
        phase1_steps=200_000,
        selfplay_cycle_steps=250_000,
        self_play_teams=8,
        set_as_default_best=True,
    )
