# MMK 詳細分析メモ 1 (MMK Analysis Memo 1)

## 1. 特徴量貢献度 (Feature Contributions)

MMKモデルが学習した、各特徴量の予測への貢献度ランキングです。

````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Feature_Importance.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Feature_Importance.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Feature_Importance.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Feature_Importance.png)
````

---

## 2. 統合サイクルグラフ (Unified Cyclic Graphs)

Sin/Cos で入力された循環特徴量（曜日、月）を統合し、「時間軸」として可視化したグラフです。

### 曜日 (Weekday)
````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Contribution_Weekday_Unified.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Contribution_Weekday_Unified.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Contribution_Weekday_Unified.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Contribution_Weekday_Unified.png)
````

### 月 (Month)
````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Contribution_Month_Unified.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Contribution_Month_Unified.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Contribution_Month_Unified.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Contribution_Month_Unified.png)
````

---

## 3. その他の特徴量グラフ (Other Feature Graphs)

他の量的変数・カテゴリ変数の貢献度グラフ（2次関数近似）です。

### 過去の電力 (Past Power)
````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Contribution_Past_Power_Normalized.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Contribution_Past_Power_Normalized.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Contribution_Past_Power_Normalized.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Contribution_Past_Power_Normalized.png)
````

### 気温 (Temperature)
````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Contribution_Temperature.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Contribution_Temperature.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Contribution_Temperature.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Contribution_Temperature.png)
````

### 日射量 (Solar Radiation)
````carousel
![春](file:///c:/Users/2213144/practice/picture_analysis/MMK_spring_Contribution_Solar_Radiation.png)
<!-- slide -->
![夏](file:///c:/Users/2213144/practice/picture_analysis/MMK_summer_Contribution_Solar_Radiation.png)
<!-- slide -->
![秋](file:///c:/Users/2213144/practice/picture_analysis/MMK_autumn_Contribution_Solar_Radiation.png)
<!-- slide -->
![冬](file:///c:/Users/2213144/practice/picture_analysis/MMK_winter_Contribution_Solar_Radiation.png)
````

---

## 4. 学習された関数・近似式一覧 (Learned Equations)

各特徴量・エキスパートごとの学習済み関数の近似式一覧です。

| 季節 | 特徴量 | コンポーネント | 推定タイプ | 近似式 |
|---|---|---|---|---|
| 春 | 過去の電力 (正規化) | **合計** | Taylor/Jacobi型 (2次) | `$-0.04x² - 0.07x - 0.02$` |
| | | エキスパート 1 | Wavelet型 | `$-0.48sin(0.37x - 1.09) - 0.42$` |
| | | エキスパート 2 | Wavelet型 | `$-1.03sin(0.34x - 1.12) - 0.92$` |
| | | エキスパート 3 | Wavelet型 | `$0.69sin(0.31x - 1.13) + 0.63$` |
| | | エキスパート 4 | Wavelet型 | `$0.27sin(0.48x - 1.04) + 0.22$` |
| 春 | 気温 | **合計** | Taylor/Jacobi型 (2次) | `$0.02x² + 0.04x + 1.23e-03$` |
| | | エキスパート 1 | Wavelet型 | `$0.56sin(0.34x - 1.14) + 0.51$` |
| | | エキスパート 2 | Spline型 (複合) | `$-5.07e-04x³ + 2.83e-03x² + 1.33e-03x - 0.02 (近似)$` |
| | | エキスパート 3 | Wavelet型 | `$1.58sin(0.12x - 1.43) + 1.56$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.04x + 6.10e-04$` |
| 春 | 降水量 | **合計** | Taylor/Jacobi型 (2次) | `$8.96e-03x² + 0.01x + 9.06e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.04x + 0.01$` |
| | | エキスパート 2 | Wavelet型 | `$0.02sin(0.98x - 0.41) + 8.62e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.03x² - 0.07x + 4.96e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.05x - 8.72e-03$` |
| 春 | 日射量 | **合計** | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.02x - 0.01$` |
| | | エキスパート 1 | Wavelet型 | `$-0.38sin(-0.30x - 2.00) - 0.36$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-4.93e-03x² - 9.28e-03x + 7.43e-03$` |
| | | エキスパート 3 | Wavelet型 | `$-0.01sin(0.65x - 0.27) - 0.02$` |
| | | エキスパート 4 | Wavelet型 | `$1.17sin(0.25x - 1.20) + 1.10$` |
| 春 | 曜日 (Sin) | **合計** | Taylor/Jacobi型 (2次) | `$0.01x² + 0.03x - 2.37e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.04x - 9.09e-03$` |
| | | エキスパート 2 | Taylor/Jacobi型 (3次) | `$-1.53e-03x³ + 1.84e-03x² + 0.02x + 7.71e-03$` |
| | | エキスパート 3 | Wavelet型 | `$0.02sin(0.71x - 0.82) + 0.01$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.07x + 0.01$` |
| 春 | 曜日 (Cos) | **合計** | Wavelet型 | `$-0.03sin(0.83x - 0.54) - 0.02$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$-0.03x² - 0.09x - 0.01$` |
| | | エキスパート 2 | Spline型 (複合) | `$1.85e-03x³ + 7.30e-03x² + 5.57e-03x - 0.01 (近似)$` |
| | | エキスパート 3 | Wavelet型 | `$0.11sin(0.46x - 1.01) + 0.09$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.04x - 0.01$` |
| 春 | 月 (Sin) | **合計** | Taylor/Jacobi型 (2次) | `$-6.83e-03x² - 0.01x - 4.02e-03$` |
| | | エキスパート 1 | Wavelet型 | `$-0.08sin(0.67x - 0.59) - 0.05$` |
| | | エキスパート 2 | Wavelet型 | `$0.04sin(0.91x - 0.37) + 0.02$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$7.65e-03x² + 0.02x + 6.68e-03$` |
| | | エキスパート 4 | Spline型 (複合) | `$-7.62e-04x³ - 0.01x² - 0.01x + 0.02 (近似)$` |
| 春 | 月 (Cos) | **合計** | Taylor/Jacobi型 (2次) | `$0.02x² + 0.04x + 2.49e-03$` |
| | | エキスパート 1 | Wavelet型 | `$-0.18sin(0.43x - 0.97) - 0.15$` |
| | | エキスパート 2 | Wavelet型 | `$-0.82sin(-0.27x - 1.95) - 0.77$` |
| | | エキスパート 3 | Wavelet型 | `$2.73sin(0.14x - 1.36) + 2.68$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.04x + 9.84e-03$` |
| 春 | 祝日フラグ | **合計** | Taylor/Jacobi型 (3次) | `$-3.81e-03x³ - 7.15e-03x² + 7.78e-03x - 4.69e-03$` |
| | | エキスパート 1 | Wavelet型 | `$1.28sin(0.24x - 1.24) + 1.22$` |
| | | エキスパート 2 | Spline型 (複合) | `$-2.69e-03x³ - 2.02e-03x² + 0.01x + 8.21e-03 (近似)$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.04x + 2.41e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-9.82e-03x² - 0.03x + 2.40e-03$` |
| 春 | 休日フラグ | **合計** | Taylor/Jacobi型 (2次) | `$0.02x² + 0.05x + 0.01$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.07x + 0.01$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.04x + 4.31e-03$` |
| | | エキスパート 3 | Wavelet型 | `$-0.30sin(0.42x - 0.94) - 0.25$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.03x² + 0.08x + 0.01$` |
| 夏 | 過去の電力 (正規化) | **合計** | Taylor/Jacobi型 (3次) | `$-3.98e-03x³ - 0.01x² - 2.52e-03x - 5.76e-03$` |
| | | エキスパート 1 | Wavelet型 | `$-0.23sin(0.55x - 0.86) - 0.17$` |
| | | エキスパート 2 | Wavelet型 | `$1.75sin(0.21x - 1.29) + 1.68$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-7.35e-03x² - 8.28e-03x + 0.01$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.05x + 0.01$` |
| 夏 | 気温 | **合計** | Spline型 (複合) | `$-1.79e-03x³ - 6.63e-04x² + 2.97e-03x + 7.81e-03 (近似)$` |
| | | エキスパート 1 | Wavelet型 | `$0.15sin(0.56x - 0.76) + 0.11$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-0.03x² - 0.09x + 5.28e-03$` |
| | | エキスパート 3 | Wavelet型 | `$0.22sin(0.45x - 0.98) + 0.18$` |
| | | エキスパート 4 | Spline型 (複合) | `$-1.94e-03x³ + 1.46e-04x² + 3.03e-03x - 9.63e-03 (近似)$` |
| 夏 | 降水量 | **合計** | Wavelet型 | `$0.05sin(0.67x - 1.00) + 0.05$` |
| | | エキスパート 1 | Wavelet型 | `$0.52sin(0.37x - 1.05) + 0.47$` |
| | | エキスパート 2 | Wavelet型 | `$-0.12sin(0.54x - 1.05) - 0.09$` |
| | | エキスパート 3 | Wavelet型 | `$-0.96sin(-0.22x - 1.89) - 0.93$` |
| | | エキスパート 4 | Spline型 (複合) | `$1.54e-03x³ + 6.54e-04x² - 5.13e-03x + 3.02e-03 (近似)$` |
| 夏 | 日射量 | **合計** | Spline型 (複合) | `$1.58e-04x³ - 1.24e-03x² - 1.69e-03x - 1.98e-03 (近似)$` |
| | | エキスパート 1 | Spline型 (複合) | `$1.53e-03x³ + 2.40e-03x² - 3.40e-03x + 2.70e-03 (近似)$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.04x - 4.95e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-7.44e-03x² - 0.02x + 2.14e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.03x + 1.95e-03$` |
| 夏 | 曜日 (Sin) | **合計** | Taylor/Jacobi型 (2次) | `$3.25e-03x² + 7.92e-03x - 2.76e-03$` |
| | | エキスパート 1 | Wavelet型 | `$-0.04sin(0.88x - 0.29) - 0.02$` |
| | | エキスパート 2 | Wavelet型 | `$0.78sin(0.19x - 1.30) + 0.75$` |
| | | エキスパート 3 | Wavelet型 | `$0.24sin(0.36x - 1.08) + 0.21$` |
| | | エキスパート 4 | Spline型 (複合) | `$4.92e-04x³ - 3.97e-03x² - 0.01x - 2.94e-03 (近似)$` |
| 夏 | 曜日 (Cos) | **合計** | Taylor/Jacobi型 (3次) | `$-2.15e-03x³ - 5.25e-03x² + 6.60e-04x - 1.68e-03$` |
| | | エキスパート 1 | Spline型 (複合) | `$6.55e-04x³ + 1.97e-03x² + 2.37e-03x + 2.86e-03 (近似)$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-6.71e-03x² - 0.01x + 3.87e-03$` |
| | | エキスパート 3 | Wavelet型 | `$-0.01sin(1.77x - 0.31) + 5.00e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.06x - 3.75e-03$` |
| 夏 | 月 (Sin) | **合計** | Wavelet型 | `$0.04sin(0.64x - 0.50) + 0.02$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.04x² + 0.10x + 1.25e-03$` |
| | | エキスパート 2 | Wavelet型 | `$-0.58sin(-0.28x - 1.96) - 0.55$` |
| | | エキスパート 3 | Wavelet型 | `$0.88sin(0.29x - 1.17) + 0.81$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-5.72e-03x² - 0.02x - 7.28e-03$` |
| 夏 | 月 (Cos) | **合計** | Taylor/Jacobi型 (2次) | `$-9.87e-03x² - 0.02x + 4.05e-03$` |
| | | エキスパート 1 | Spline型 (複合) | `$-8.21e-04x³ + 4.82e-03x² + 0.01x + 3.97e-03 (近似)$` |
| | | エキスパート 2 | Wavelet型 | `$-0.02sin(1.32x - 0.52) + 4.60e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.06x - 1.75e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.03x + 3.28e-03$` |
| 夏 | 祝日フラグ | **合計** | Taylor/Jacobi型 (2次) | `$0.01x² + 0.03x - 6.27e-03$` |
| | | エキスパート 1 | Wavelet型 | `$0.07sin(0.60x - 1.09) + 0.06$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$0.03x² + 0.09x + 5.89e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-2.04e-03x² - 0.01x - 0.01$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.06x + 9.60e-03$` |
| 夏 | 休日フラグ | **合計** | Spline型 (複合) | `$1.81e-03x³ + 5.89e-04x² - 0.02x + 1.80e-03 (近似)$` |
| | | エキスパート 1 | Wavelet型 | `$-0.25sin(0.40x - 0.95) - 0.22$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.06x - 1.62e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.07x - 9.89e-03$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.05x + 0.02$` |
| 秋 | 過去の電力 (正規化) | **合計** | Taylor/Jacobi型 (2次) | `$-0.03x² - 0.06x - 0.02$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.03x² + 0.07x + 5.07e-03$` |
| | | エキスパート 2 | Wavelet型 | `$-0.59sin(-0.41x - 2.13) - 0.50$` |
| | | エキスパート 3 | Wavelet型 | `$0.69sin(0.30x - 1.19) + 0.64$` |
| | | エキスパート 4 | Wavelet型 | `$-1.59sin(0.21x - 1.29) - 1.53$` |
| 秋 | 気温 | **合計** | Taylor/Jacobi型 (2次) | `$0.01x² + 0.03x + 2.73e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.04x² + 0.09x + 4.04e-03$` |
| | | エキスパート 2 | Wavelet型 | `$-0.22sin(-0.39x - 2.11) - 0.19$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.07x + 8.57e-03$` |
| | | エキスパート 4 | Spline型 (複合) | `$-1.84e-03x³ - 1.50e-05x² + 9.24e-03x - 1.25e-03 (近似)$` |
| 秋 | 降水量 | **合計** | Taylor/Jacobi型 (2次) | `$-7.39e-03x² - 0.01x - 0.01$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$1.03e-03x² + 0.01x + 0.01$` |
| | | エキスパート 2 | Wavelet型 | `$0.34sin(0.36x - 1.07) + 0.29$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.05x - 0.02$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-2.37e-03x² - 0.01x - 9.06e-03$` |
| 秋 | 日射量 | **合計** | Taylor/Jacobi型 (2次) | `$-8.72e-03x² - 0.02x - 8.23e-04$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$-8.68e-03x² - 0.02x + 2.78e-06$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-9.91e-03x² - 0.02x + 3.77e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.01x² - 0.03x - 0.01$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$5.99e-03x² + 0.02x + 0.01$` |
| 秋 | 曜日 (Sin) | **合計** | Taylor/Jacobi型 (2次) | `$1.83e-03x² + 0.01x - 4.95e-03$` |
| | | エキスパート 1 | Spline型 (複合) | `$1.77e-04x³ + 1.06e-03x² - 2.59e-04x - 0.01 (近似)$` |
| | | エキスパート 2 | Wavelet型 | `$0.83sin(0.31x - 1.12) + 0.77$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-4.79e-03x² - 0.03x - 0.02$` |
| | | エキスパート 4 | Wavelet型 | `$0.04sin(0.76x - 0.68) + 0.03$` |
| 秋 | 曜日 (Cos) | **合計** | Taylor/Jacobi型 (2次) | `$0.01x² + 9.69e-03x + 5.18e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.09x - 0.02$` |
| | | エキスパート 2 | Wavelet型 | `$0.39sin(0.32x - 1.21) + 0.36$` |
| | | エキスパート 3 | Wavelet型 | `$1.55sin(0.21x - 1.29) + 1.49$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$6.25e-03x² + 0.02x + 2.33e-03$` |
| 秋 | 月 (Sin) | **合計** | Taylor/Jacobi型 (3次) | `$2.84e-03x³ + 6.71e-03x² - 4.60e-03x + 6.56e-03$` |
| | | エキスパート 1 | Spline型 (複合) | `$6.34e-05x³ - 3.94e-04x² + 1.86e-04x + 0.01 (近似)$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.05x - 6.54e-03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-9.79e-03x² - 0.03x + 1.23e-03$` |
| | | エキスパート 4 | Wavelet型 | `$0.15sin(0.58x - 0.87) + 0.11$` |
| 秋 | 月 (Cos) | **合計** | Spline型 (複合) | `$-7.11e-04x³ + 3.03e-03x² + 7.03e-03x + 4.26e-03 (近似)$` |
| | | エキスパート 1 | Wavelet型 | `$0.23sin(0.42x - 1.09) + 0.20$` |
| | | エキスパート 2 | Wavelet型 | `$0.54sin(0.29x - 1.21) + 0.50$` |
| | | エキスパート 3 | Wavelet型 | `$-1.71sin(0.21x - 1.30) - 1.64$` |
| | | エキスパート 4 | Taylor/Jacobi型 (2次) | `$-3.98e-03x² - 0.01x + 2.27e-03$` |
| 秋 | 祝日フラグ | **合計** | Taylor/Jacobi型 (2次) | `$0.02x² + 0.04x + 7.36e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.04x + 0.01$` |
| | | エキスパート 2 | Spline型 (複合) | `$1.83e-03x³ + 2.70e-04x² - 1.89e-03x + 7.23e-03 (近似)$` |
| | | エキスパート 3 | Spline型 (複合) | `$-4.26e-04x³ + 2.97e-04x² + 6.58e-03x + 1.21e-03 (近似)$` |
| | | エキスパート 4 | Wavelet型 | `$0.70sin(0.35x - 1.12) + 0.63$` |
| 秋 | 休日フラグ | **合計** | Taylor/Jacobi型 (2次) | `$4.19e-03x² + 0.01x + 9.41e-04$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$3.41e-03x² + 0.02x + 6.43e-03$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$6.23e-03x² + 0.02x - 0.01$` |
| | | エキスパート 3 | Wavelet型 | `$0.70sin(0.33x - 1.11) + 0.65$` |
| | | エキスパート 4 | Wavelet型 | `$-0.59sin(0.34x - 1.12) - 0.54$` |
| 冬 | 過去の電力 (正規化) | **合計** | Taylor/Jacobi型 (3次) | `$-0.09x³ + 0.09x² + 0.90x - 0.04$` |
| | | エキスパート 1 | Taylor/Jacobi型 (3次) | `$-0.08x³ + 0.12x² + 0.86x + 0.04$` |
| | | エキスパート 2 | Taylor/Jacobi型 (3次) | `$-0.09x³ + 0.13x² + 0.92x - 0.19$` |
| | | エキスパート 3 | Wavelet型 | `$1.26sin(0.69x - 0.52) + 0.97$` |
| | | エキスパート 4 | Wavelet型 | `$1.78sin(0.59x - 0.82) + 1.48$` |
| 冬 | 気温 | **合計** | Taylor/Jacobi型 (3次) | `$0.02x³ - 0.03x² - 0.22x + 0.01$` |
| | | エキスパート 1 | Taylor/Jacobi型 (3次) | `$0.03x³ - 0.02x² - 0.26x - 0.08$` |
| | | エキスパート 2 | Spline型 (複合) | `$2.90e-03x³ - 0.05x² - 0.10x + 7.31e-03 (近似)$` |
| | | エキスパート 3 | Spline型 (複合) | `$-0.02x³ - 0.01x² - 0.01x - 0.29 (近似)$` |
| | | エキスパート 4 | Spline型 (複合) | `$0.06x³ + 4.41e-03x² - 0.34x + 0.21 (近似)$` |
| 冬 | 降水量 | **合計** | Taylor/Jacobi型 (3次) | `$-3.04e-03x³ - 0.01x² + 6.82e-03x + 0.05$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$-0.03x² - 0.04x + 0.11$` |
| | | エキスパート 2 | Wavelet型 | `$0.09sin(1.34x - 0.12) + 0.03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-1.25e-03x² + 0.04x + 0.07$` |
| | | エキスパート 4 | Wavelet型 | `$0.08sin(1.42x + 0.15) + 0.02$` |
| 冬 | 日射量 | **合計** | Taylor/Jacobi型 (2次) | `$2.21e-03x² + 0.01x - 8.16e-03$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$0.01x² + 0.02x - 0.03$` |
| | | エキスパート 2 | Wavelet型 | `$0.30sin(0.44x - 0.88) + 0.25$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$0.02x² + 0.06x + 0.02$` |
| | | エキスパート 4 | Taylor/Jacobi型 (3次) | `$-4.88e-03x³ + 4.90e-03x² + 0.04x - 0.02$` |
| 冬 | 曜日 (Sin) | **合計** | Taylor/Jacobi型 (2次) | `$-0.02x² - 0.10x + 0.05$` |
| | | エキスパート 1 | Wavelet型 | `$-0.29sin(0.76x - 0.67) - 0.14$` |
| | | エキスパート 2 | Wavelet型 | `$-0.30sin(1.05x + 0.02) - 0.18$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$-0.06x² - 0.26x + 0.04$` |
| | | エキスパート 4 | Wavelet型 | `$0.20sin(-1.89x + 2.15) - 0.02$` |
| 冬 | 曜日 (Cos) | **合計** | Spline型 (複合) | `$-1.19e-03x³ + 5.64e-03x² + 0.01x - 0.02 (近似)$` |
| | | エキスパート 1 | Spline型 (複合) | `$0.01x³ + 0.02x² - 0.01x - 8.64e-03 (近似)$` |
| | | エキスパート 2 | Spline型 (複合) | `$9.41e-05x³ + 1.06e-03x² - 3.63e-03x + 0.01 (近似)$` |
| | | エキスパート 3 | Spline型 (複合) | `$3.85e-03x³ + 0.02x² - 7.29e-04x - 0.06 (近似)$` |
| | | エキスパート 4 | Spline型 (複合) | `$-3.71e-04x³ + 0.04x² + 0.04x - 0.11 (近似)$` |
| 冬 | 月 (Sin) | **合計** | Spline型 (複合) | `$-1.09e-03x³ + 7.60e-04x² + 0.01x + 7.01e-03 (近似)$` |
| | | エキスパート 1 | Taylor/Jacobi型 (3次) | `$3.92e-03x³ - 9.50e-03x² - 0.02x + 0.09$` |
| | | エキスパート 2 | Wavelet型 | `$-0.12sin(0.93x - 0.92) + 0.03$` |
| | | エキスパート 3 | Taylor/Jacobi型 (2次) | `$0.03x² + 0.06x - 0.11$` |
| | | エキスパート 4 | Wavelet型 | `$0.10sin(1.34x - 0.80) - 0.11$` |
| 冬 | 月 (Cos) | **合計** | Taylor/Jacobi型 (3次) | `$-0.02x³ - 4.04e-03x² + 0.10x - 0.01$` |
| | | エキスパート 1 | Taylor/Jacobi型 (3次) | `$-0.01x³ + 0.03x² + 0.12x - 0.13$` |
| | | エキスパート 2 | Spline型 (複合) | `$-0.01x³ - 0.01x² + 0.02x + 0.03 (近似)$` |
| | | エキスパート 3 | Wavelet型 | `$0.21sin(1.11x - 0.33) + 0.08$` |
| | | エキスパート 4 | Wavelet型 | `$0.39sin(0.66x - 0.68) + 0.28$` |
| 冬 | 祝日フラグ | **合計** | Taylor/Jacobi型 (2次) | `$0.01x² + 0.02x - 0.01$` |
| | | エキスパート 1 | Wavelet型 | `$0.10sin(0.82x - 0.77) + 0.05$` |
| | | エキスパート 2 | Wavelet型 | `$0.64sin(0.26x - 1.27) + 0.60$` |
| | | エキスパート 3 | Taylor/Jacobi型 (3次) | `$1.95e-03x³ + 4.40e-03x² + 4.07e-03x + 4.85e-03$` |
| | | エキスパート 4 | Wavelet型 | `$-0.05sin(0.68x - 0.51) - 0.04$` |
| 冬 | 休日フラグ | **合計** | Taylor/Jacobi型 (3次) | `$3.55e-03x³ - 4.40e-03x² - 0.03x + 0.01$` |
| | | エキスパート 1 | Taylor/Jacobi型 (2次) | `$-8.59e-03x² - 0.02x + 1.48e-03$` |
| | | エキスパート 2 | Taylor/Jacobi型 (2次) | `$-5.14e-03x² - 0.02x - 0.02$` |
| | | エキスパート 3 | Spline型 (複合) | `$-1.99e-03x³ + 1.74e-03x² + 4.11e-03x - 0.03 (近似)$` |
| | | エキスパート 4 | Wavelet型 | `$-0.11sin(0.92x - 0.78) - 2.78e-03$` |
