# 台大場四個新 Demo 設計(2026-09-11)

目的:2026-09-15 台大推廣場(產品推廣 + 課程合作 + 競賽宣傳),講者在台上投影操作,可能邀學生上台。
交付:四個資料夾 + `launchers\*.bat` + 桌面捷徑,雙擊即跑,不需打包 exe。
預算:兩個工作天(約 16 h)。

## 共通規範

| 項目 | 決定 |
|---|---|
| 位置 | `C:\ugen300-demos\{WakeUp,Calories,OfflineChat,PhotoRestore}\` |
| 裝置 | 每個 demo 獨立程序;用各資料夾的 `hailo_vdevice.py` 單例;結束不呼叫 `VDevice.release()`(見 2026-09-10 掉線修正) |
| 模型路徑 | 視覺模型從 `C:\Hailo_ModelZoo\` 硬連結/複製進各資料夾(`.hef` 已被 gitignore);GenAI 模型用 `../models/` |
| 畫面 | 深色底、大字(投影可讀)、右上角「離線 · UGen300」徽章;`q` 離開、`r` 重來 |
| 啟動 | `launchers\<Name>.bat`(chcp 65001、PYTHONUTF8=1、cd 到資料夾、python 主程式、pause);桌面捷徑指向 bat |
| 錯誤處理 | 模型載入失敗 → 視窗顯示中文錯誤與建議(裝置未插/檔案缺),不直接崩潰 |
| 測試 | 每個 demo 用 subprocess 實機跑一次(裝置存活、畫面出現、核心路徑通);#7 的真人深蹲由使用者驗收 |

## 1. WakeUp —— 早八起床驗證(#7)

- 技術:`yolov8s_pose.hef` + `hailo_pose.PoseEstimator`(複製自 Y5),膝角計算沿用 `yoga_logic._angle`。
- 流程:時鐘畫面(顯示 07:59 → 08:00)→ 響鈴(winsound 循環,獨立執行緒)+ 大字「做 N 下深蹲才能關鬧鐘」→ 偵測到人 → 狀態機:膝角 < 100° 進入「蹲下」、> 160° 回「站立」且計 1 下 → 大數字計次 → 達 N → 停鈴、勝利音效、「驗證通過,早安!」。
- 參數:`--reps 5 --source 0 --conf 0.3`。
- 主程式 `wakeup.py`;邏輯 `squat_counter.py`(純函式,可離線單元測試)。

## 2. Calories —— 離線食物熱量估算(#6)

- 技術:`vlm_client.py`(複製自 V3,含 hailo_vdevice 單例)+ `../models/Qwen2-VL-2B-Instruct.hef`。
- 流程:鏡頭預覽 → 空白鍵 → 3 秒倒數拍照 → VLM prompt 要求 JSON `[{"item","portion","kcal","protein_g"}]` → 解析(容錯:抓第一個 `[...]`,失敗則顯示原文)→ 右側面板表格 + 總熱量大字 + 「今日累計」(存 `today.json`,依日期歸零)。畫面固定顯示「AI 估算 ±30%」。
- 主程式 `calories.py`;prompt 與解析 `nutrition.py`(純函式,可單元測試)。

## 3. OfflineChat —— 離線 ChatGPT(#9)

- 技術:`hailo_platform.genai.LLM` 直接載 `../models/Qwen3-1.7B-Instruct.hef`(不走 hailo-ollama,其 qwen3 blob 需上網 pull);PySide6 視窗。
- 流程:啟動即載模型(顯示載入進度)→ 深色聊天介面(氣泡、輸入框、Enter 送出)→ `generate()` 逐 token 串流顯示 → 標頭顯示「已離線 · 0 元/月 · X token/秒」。
- 模型切換:下拉可選 `Qwen3-1.7B` / `Llama3.2-1B` / `Qwen2.5-1.5B`(切換 = 釋放模型層物件再載入,VDevice 不動)。
- Qwen3 的 `<think>` 區塊:預設摺疊顯示為「思考中…」,可展開。
- 主程式 `offline_chat.py`;LLM 包裝 `llm_engine.py`(串流、tok/s 統計、模型切換)。

## 4. PhotoRestore —— 老照片修復站(#15)

- 技術:`zero_dce.hef`(提亮 400×600)、`dncnn_color_blind.hef`(去雜訊 321×481)、`real_esrgan_x2.hef`(512→1024);PySide6 視窗。
- 前置閘門:先用三張測試圖各跑一次,量化畫質不可接受者直接移除該按鈕。
- 流程:拖曳/開啟圖片 → 勾選要套用的處理(可疊加,順序:提亮 → 去雜訊 → 放大)→ 分塊推論(重疊 16 px、線性融合)→ 前後對比滑桿(拖動分界線)→ 另存。
- 主程式 `photo_restore.py`;推論與分塊 `restore_engine.py`(純函式部分可離線測試)。

## 順序與檢查點

1. PhotoRestore 畫質閘門(30 分)→ 決定保留哪些模型
2. WakeUp → 3. OfflineChat → 4. Calories → 5. PhotoRestore 完成
6. 四個 launchers + 桌面捷徑 → 全部冷啟動跑一輪 → 回報

## 不做

- 打包 exe、自動更新、多語 UI、雲端任何東西。
