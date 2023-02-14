 # SIGNATE StudentCup 2021秋の分析記録
 
## タイトル
**SIGNATE Student Cup 2021秋： オペレーション最適化に向けたシェアサイクルの利用予測**
- [コンペティション全体サイト](https://signate.jp/competitions/549)
- [予測部門のサイト](https://signate.jp/competitions/550)

## テーマ
「オペレーション最適化に向けたシェアサイクルの利用予測」

## タスク説明
各ステーションで記録された自転車の台数状況、サービス利用者の移動履歴、ステーション情報（所在地や最大駐輪数）、および気象情報をもとに、特定の日時・ステーションにおける利用可能な自転車数の予測にチャレンジして頂きます。
 
<img width="681" alt="Screenshot 2023-02-15 at 0 15 26" src="https://user-images.githubusercontent.com/78187015/218779203-e4a98bd6-c220-4fac-b8ca-2016b57500ed.png">

## 解法
- 特徴量エンジニアリング
    - ラベルエンコーディング
    - 標準化
- 使用モデル
    - LightGBM
- validation方法
    - 月ごとに検証データを作成し、1カ月ごとずらしていく
