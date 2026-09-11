# 台大四個 Demo 實作計畫(2026-09-11)

對應設計:`docs/superpowers/specs/2026-09-11-ntu-demo-4-apps-design.md`
畫質閘門結果:zero_dce / dncnn_color_blind / real_esrgan_x2 三個皆保留(zero_dce 加強度混合避免過曝)。

## 共通做法
- 每個資料夾:`hailo_vdevice.py`(單例)、主程式、純邏輯模組 + `test_*.py`(不需裝置可跑)。
- 視覺 .hef 用硬連結自 `C:\Hailo_ModelZoo`;GenAI 用 `../models/`。
- 實機驗證:subprocess 跑主程式 `--selftest`(載模型、抓一幀/跑一次、離開),之後 `hailortcli scan` 確認裝置仍在。
- 完成一個就加 `launchers\<Name>.bat` 與桌面捷徑,並 commit。

## 任務
1. WakeUp:`squat_counter.py`(狀態機)+ 測試 → `wakeup.py`(時鐘/鬧鈴/計次/過關)→ 實機 → bat/捷徑 → commit
2. OfflineChat:`llm_engine.py`(串流、tok/s、切模型、think 摺疊)+ 測試 → `offline_chat.py`(PySide6)→ 實機 → bat/捷徑 → commit
3. Calories:`nutrition.py`(prompt/JSON 解析/今日累計)+ 測試 → `calories.py`(OpenCV + 面板)→ 實機 → bat/捷徑 → commit
4. PhotoRestore:`restore_engine.py`(分塊/融合/三模型)+ 測試 → `photo_restore.py`(PySide6 + 對比滑桿)→ 實機 → bat/捷徑 → commit
5. 冷啟動全跑一輪、README 補四個 app、最終回報;WakeUp 真人驗收交給使用者
