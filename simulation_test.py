# rotating_drum_simulation.py

import ansys.rocky.core as pyrocky
from pathlib import Path
import os

# ==============================================================================
# 1. 初期設定 (ユーザーが変更する部分)
# ==============================================================================

# ベースとなるRockyプロジェクトファイルのパス
# このスクリプトと同じ階層にあると仮定
BASE_PROJECT_PATH = Path(__file__).parent / "rotating_drum_base.rocky"

# 結果を保存するフォルダ
RESULTS_DIR = Path(__file__).parent / "results"

# 今回のシミュレーションで設定するドラムの回転数 (RPM)
ROTATION_SPEED_RPM = 50.0

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    """
    PyRockyを使用して回転ドラムのシミュレーションを実行し、結果を取得するメイン関数
    """
    
    # 結果保存用フォルダがなければ作成
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 新しいプロジェクトファイル名を設定
    new_project_name = f"drum_{ROTATION_SPEED_RPM}rpm.rocky"
    new_project_path = RESULTS_DIR / new_project_name
    
    rocky = None  # finallyブロックで使うため、tryの外で定義
    try:
        # ----------------------------------------------------------------------
        # 2. プリプロセス (Pre-processing)
        # ----------------------------------------------------------------------
        print(">>> Launching Ansys Rocky...")
        rocky = pyrocky.launch_rocky()
        
        print(f">>> Opening base project: {BASE_PROJECT_PATH}")
        rocky.api.OpenProject(str(BASE_PROJECT_PATH))
        
        # スタディオブジェクトを取得（以降の操作で頻繁に使う）
        study = rocky.api.GetStudy()
        
        print(f">>> Modifying parameters: Setting rotation speed to {ROTATION_SPEED_RPM} RPM")
        # ジオメトリ名"Drum"を探し、その角速度プロパティを変更
        drum_geometry = study.GetElement("Drum")
        if not drum_geometry:
            raise RuntimeError("Geometry named 'Drum' not found in the project.")
            
        # 角速度の値を設定 (RPMからrad/sに変換する必要がある場合も考慮)
        angular_velocity_prop = drum_geometry.GetProperty("Angular Velocity")
        # RockyではY軸周りの回転を負の値で定義することが多い
        angular_velocity_prop.SetValue(f"0; -{ROTATION_SPEED_RPM} rpm; 0") 

        print(f">>> Saving modified project to: {new_project_path}")
        rocky.api.SaveProject(str(new_project_path))

        # ----------------------------------------------------------------------
        # 3. ソルブ (Solving)
        # ----------------------------------------------------------------------
        print("\n>>> Starting simulation... (This may take some time)")
        # wait_for_completion=True で計算完了までPythonスクリプトの実行を待機させる
        rocky.api.StartSimulation(wait_for_completion=True)
        print(">>> Simulation finished!")

        # ----------------------------------------------------------------------
        # 4. ポストプロセス (Post-processing)
        # ----------------------------------------------------------------------
        print("\n>>> Extracting results...")
        
        # 結果①：ドラムのY軸周りの平均駆動電力を取得
        # "Boundaries" -> "Drum" -> "Power" という名前のカーブを探す
        power_curve = study.GetCurve("Boundaries", "Drum", "Power")
        power_table = power_curve.GetProperty("Y-Axis Torque Power").GetTable()
        
        # テーブルの最後の行の最後の値が最終的な平均電力
        # テーブル形式: [[time1, value1], [time2, value2], ...]
        if power_table:
            last_power_value_watt = power_table[-1][1]
            print(f"  - Average Driving Power (Y-Axis): {last_power_value_watt:.4f} W")
        else:
            print("  - Could not retrieve power data.")

        # 結果②：最後の時間ステップにおける、全粒子の平均速度を取得
        last_timestep_data = study.GetCurve(-1) # -1で最後のタイムステップを指定
        avg_particle_velocity = last_timestep_data.GetProperty("Average Particle Translational Velocity").GetValue()

        print(f"  - Average Particle Velocity at last step: {avg_particle_velocity:.4f} m/s")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        # ----------------------------------------------------------------------
        # 5. クリーンアップ (Cleanup)
        # ----------------------------------------------------------------------
        # エラーが発生しても、必ずRockyセッションを閉じてライセンスを解放する
        if rocky:
            print("\n>>> Closing Ansys Rocky session.")
            rocky.close()

if __name__ == "__main__":
    main()