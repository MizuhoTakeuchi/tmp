# C++版 map_composer テスト結果

対象: `cpp/` 実装（PCL + Eigen）。`./data/pcd` の2種類（円型・L字型）点群を用いて、`map1` 座標系への統合結果を検証した。

- 実行環境: リポジトリ同梱のビルドとCLIを使用
- 検証観点:
  - 各マップ間（mapKとmap1）の変換パラメータ（並進 t, 回転角 θ）
  - `map1` 座標系へ変換後の全点群オーバレイプロット

## 実行手順（再現手順）
- ビルド: `cmake -S cpp -B build && cmake --build build -j`
- 円型: `./build/map_composer_cli --pcd-dir data/pcd/circle --write-out out/pcd_circle_in1`
- L字型: `./build/map_composer_cli --pcd-dir data/pcd/L --write-out out/pcd_L_in1`
- 可視化（PNG生成）:
  - `./scripts/plot_pcd_overlay.py out/pcd_circle_in1 out/plots/pcd_circle_overlay.png`
  - `./scripts/plot_pcd_overlay.py out/pcd_L_in1 out/plots/pcd_L_overlay.png`

## 結果: 変換パラメータ（src→map1）
- 円型（data/pcd/circle）
  - map1→map1: t=(0.00000, 0.00000), θ= 0.000°
  - map2→map1: t=(-4.38134, 0.896575), θ=-75.000°
  - map3→map1: t=(+0.939693, +0.342020), θ=+20.000°
  - 参考（doc/validation.md と比較するための逆変換 map1→mapK）
    - map1→map2（逆変換）: t≈(+2.0008, +3.9981), θ≈+75.000°
    - map1→map3（逆変換）: t≈(-1.0000, +0.0000), θ≈-20.000°
    - 期待値（doc/validation.md）: map1→map2: t=(2,4), θ=+75°／map1→map3: t=(-1,0), θ=-20°
    - 結果は数値誤差範囲内で一致

- L字型（data/pcd/L）
  - map1→map1: t=(0.00000, 0.00000), θ= 0.000°
  - map2→map1: t=(-4.38134, 0.896575), θ=-75.000°
  - map3→map1: t=(+0.939693, +0.342020), θ=+20.000°
  - 注: 単一Lによる±180°曖昧性に対し、端点整合スコア＋BFS時の候補比較（訪問済み点群とのロバスト距離）で解消済み。

## 結果: プロット（map1座標系に変換後の全点群）
- 円型: `out/plots/pcd_circle_overlay.png`
- L字型: `out/plots/pcd_L_overlay.png`

![circle overlay](../out/plots/pcd_circle_overlay.png)

![L overlay](../out/plots/pcd_L_overlay.png)

## まとめ
- 円型・L字型の両データセットで、map1座標系への統合後に各点群が重畳し、推定変換は妥当。
- L字型のmap3に見られた±180°不定性は、端点整合スコアとBFSでの候補比較により解消された。
- 追加の数値評価（例: 変換後の最短距離の中央値/平均）も容易に算出可能。必要であれば追記する。
