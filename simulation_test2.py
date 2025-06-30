# ==============================================================================
# 司令塔：メインスクリプト (main_optimization.py)
# ==============================================================================
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import ansys.rocky.core as pyrocky

# --- 1. 実験データの準備 ---
# キャリブレーションの目標となる実験結果の値を定義
# 例：実験で得られた安息角が35度、ホッパーからの排出流量が 1.2 kg/s など
EXPERIMENTAL_TARGET_VALUE = 35.0 

# --- 2. 探索するパラメータの範囲を定義 ---
# (名前, (下限, 上限)) の形式で指定
# 例：静摩擦係数 (0.1 ~ 0.8)、転がり摩擦係数 (0.01 ~ 0.5)
search_space = [
    Real(0.1, 0.8, name='static_friction'),
    Real(0.01, 0.5, name='rolling_friction')
]

# --- 3. 目的関数 (Objective Function) の作成 ---
#    これがループの中核。ベイズ最適化がこの関数を何度も呼び出す。
@use_named_args(search_space)
def objective_function(**params):
    """
    与えられたDEMパラメータでRockyシミュレーションを実行し、
    実験結果との誤差を返す関数。
    
    Args:
        **params (dict): {'static_friction': 0.5, 'rolling_friction': 0.2} のような辞書
    
    Returns:
        float: 実験結果とシミュレーション結果の誤差（この値を最小化する）
    """
    static_friction = params['static_friction']
    rolling_friction = params['rolling_friction']
    
    print(f"\n--- Iteration Start ---")
    print(f"Trying Parameters: Static Friction = {static_friction:.4f}, Rolling Friction = {rolling_friction:.4f}")

    # ===== ここから PyRocky の処理 =====
    # 3-1. Rockyを起動し、パラメータを設定してシミュレーションを実行
    try:
        # この部分は実際のRockyプロジェクトに合わせて実装
        simulation_result = run_rocky_simulation_and_get_result(
            static_friction,
            rolling_friction
        )
    except Exception as e:
        print(f"!!! Rocky Simulation Failed: {e}")
        # シミュレーションが失敗した場合は、非常に大きな誤差を返してペナルティを与える
        return 1e10
    # ===== PyRocky の処理ここまで =====

    # 3-2. 誤差を計算
    #    （例：二乗誤差。他にも絶対誤差など、目的に応じて設定）
    error = (simulation_result - EXPERIMENTAL_TARGET_VALUE)**2
    
    print(f"Simulation Result = {simulation_result:.4f}, Target = {EXPERIMENTAL_TARGET_VALUE}, Error = {error:.4f}")
    print(f"--- Iteration End ---\n")
    
    return error

# --- 4. PyRockyの具体的な処理を関数として定義 ---
def run_rocky_simulation_and_get_result(friction1, friction2):
    """
    PyRockyを使ってシミュレーションを実行し、結果を返す部分。
    この関数を、ご自身のRockyプロジェクトに合わせて具体的に記述する必要があります。
    
    Returns:
        float: シミュレーションから得られた評価値（例：安息角、排出流量など）
    """
    rocky = None # エラーハンドリングのため
    try:
        # Rockyを起動
        rocky = pyrocky.launch_rocky()
        
        # ベースとなるプロジェクトファイルを開く
        rocky.api.OpenProject("base_project.rocky")
        
        # パラメータを設定
        # 例：デフォルトの材料ペアの静摩擦係数と転がり摩擦係数を設定
        rocky.api.GetStudy().GetElement("Default Material Interaction").GetProperty("Friction Coefficient").SetValue(friction1)
        rocky.api.GetStudy().GetElement("Default Material Interaction").GetProperty("Rolling Friction Coefficient").SetValue(friction2)
        
        # シミュレーションを実行し、完了まで待つ
        rocky.api.StartSimulation(wait_for_completion=True)
        
        # 結果を抽出 (例：安息角を計算するカスタムスクリプトを実行)
        # ここは最もカスタマイズが必要な部分
        # result_value = rocky.api.GetAngleOfRepose() # このような関数は仮のものです
        # 例として、最後の時間ステップの粒子の平均速度を返す
        last_timestep = rocky.api.GetStudy().GetCurve(-1) # 最後のタイムステップのカーブを取得
        result_value = last_timestep.GetProperty("Average Particle Translational Velocity").GetMaximum() # 仮の例
        
        return result_value
        
    finally:
        # 成功しても失敗しても、必ずRockyセッションを閉じる
        if rocky:
            rocky.close()

# --- 5. ベイズ最適化の実行 ---
# n_calls: ループを回す回数（シミュレーションの試行回数）
# x0:      パラメータの初期値（指定しない場合はランダムに選ばれる）
print("==========================================")
print("Starting Bayesian Optimization...")
print("==========================================")

result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=20,  # 例えば、合計20回のシミュレーションを実行
    n_initial_points=5, # 最初の5回はランダムに探索
    random_state=123
)

# --- 6. 結果の表示 ---
print("\n==========================================")
print("Optimization Finished!")
print("==========================================")
print(f"Best Parameters Found:")
print(f"  - Static Friction: {result.x[0]:.4f}")
print(f"  - Rolling Friction: {result.x[1]:.4f}")
print(f"Minimum Error (Squared): {result.fun:.4f}")