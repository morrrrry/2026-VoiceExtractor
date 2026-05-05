# Voice Extractor — 使用マニュアル

音楽と声が混在した音声ファイルから、音楽のみのリファレンスを用いてボーカル成分を抽出するツール。

## windows向けexe
Pythonベースをexeファイルにしたものでライブラリ群が含まれて容量が大きいのでGoogleドライブ配布となります
v1.0.0
https://drive.google.com/file/d/1K3VNw1KPZOfhOsGZ3ETmsDeR23ynqhw7/view?usp=drive_link

---

## 目次

1. [概要・処理方針](#概要処理方針)
2. [機能一覧](#機能一覧)
3. [動作環境](#動作環境)
4. [インストール・環境構築](#インストール環境構築)
5. [起動方法](#起動方法)
6. [PyInstaller によるスタンドアロン化](#pyinstaller-によるスタンドアロン化)
7. [画面説明と操作手順](#画面説明と操作手順)
8. [出力ファイル仕様](#出力ファイル仕様)
9. [アルゴリズム詳細](#アルゴリズム詳細)
10. [パラメータ調整ガイド](#パラメータ調整ガイド)
11. [トラブルシューティング](#トラブルシューティング)

---

## 概要・処理方針

| 項目 | 内容 |
|------|------|
| 入力 A | Mix ファイル（音楽＋ボーカル） |
| 入力 B | Music ファイル（音楽のみ） |
| 出力 | Voice（ボーカル成分）、Music aligned、Mix — いずれもステレオ |

**重要な設計方針：**

- **Mix（音楽＋声）は一切加工しない。** 声の質感・位相を原音のまま保護する。
- **Music（音楽のみ）を Mix に近づける方向で補正する。** 時間軸・音量・周波数特性の3段階で補正。
- 最終的に `Voice = Mix − 補正済み Music` として差分を取り出す。
- **L/R チャンネルを独立処理**することでステレオの空間情報を保持する。

---

## 機能一覧

### 標準解析（Step 1〜5）

| ステップ | 内容 |
|----------|------|
| Step 1 | 相互相関による粗いタイムオフセット検出 |
| Step 2 | サンプル単位の精密オフセット検索 |
| Step 3 | ストレッチ補正（サンプリングレート差・再生速度差の吸収）※オプション |
| Step 4 | 最小二乗法による音量スケーリング |
| Step 5 | ステレオ差分抽出（L/R 独立） |

### 精密補正タブ（オプション）

- 波形上のドラッグ操作で「音楽のみ区間」を指定
- 指定区間の STFT インパルス応答を推定し、Music 全体に適用して周波数特性の差を補正
- L/R チャンネル独立処理
- 補正前/補正後のステレオ再生で聴き比べ
- 「元に戻す」で標準解析結果に一発リバート

### 保存

| モード | 内容 |
|--------|------|
| 2ch×3ファイル | `voice_stereo.wav` / `music_stereo.wav` / `mix_stereo.wav` を個別に保存 |
| 6ch マルチトラック | 1ファイルに Ch1-6 を収録（DAW へのダイレクトインポート用） |

6ch チャンネルアサイン：

```
Ch1: Voice L    Ch2: Voice R
Ch3: Music L    Ch4: Music R    （アライメント・補正済み）
Ch5: Mix L      Ch6: Mix R
```

---

## 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11、macOS 12+、Ubuntu 20.04+ |
| Python | 3.9 以上（3.11 推奨） |
| オーディオデバイス | ステレオ出力対応（プレビュー再生時） |

---

## インストール・環境構築

### 1. Python のインストール

[python.org](https://www.python.org/downloads/) から Python 3.11 をダウンロードしてインストール。

Windows の場合はインストーラの **「Add Python to PATH」にチェック**を入れること。

### 2. 仮想環境の作成（推奨）

```bash
# プロジェクトフォルダを作成して移動
mkdir voice_extractor
cd voice_extractor

# 仮想環境を作成・有効化
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. 依存ライブラリのインストール

```bash
pip install --upgrade pip

pip install numpy scipy matplotlib soundfile librosa sounddevice PyQt5 pyinstaller
```

各パッケージの役割：

| パッケージ | 用途 |
|-----------|------|
| `numpy` | 配列演算 |
| `scipy` | 相互相関・STFT |
| `matplotlib` | 波形グラフ描画 |
| `soundfile` | WAV 読み書き（24-bit PCM 出力） |
| `librosa` | MP3 読み込み・リサンプリング・ストレッチ |
| `sounddevice` | ステレオリアルタイム再生 |
| `PyQt5` | GUI フレームワーク |
| `pyinstaller` | スタンドアロン EXE 化 |

#### Windows で MP3 読み込みを使う場合（libsndfile 追加）

librosa は内部で `soundfile` → `audioread` とフォールバックするが、確実に MP3 を扱うには `ffmpeg` のパスが通っていることを確認する。

```bash
# Windows（winget）
winget install Gyan.FFmpeg

# macOS（Homebrew）
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### 4. スクリプトの配置

```
voice_extractor/
├── venv/
└── voice_extractor.py
```

---

## 起動方法

```bash
# 仮想環境を有効化してから実行
python voice_extractor.py
```

---

## PyInstaller によるスタンドアロン化

仮想環境が有効な状態で以下を実行する。

### Windows（単一 EXE）

```bash
pyinstaller ^
  --onefile ^
  --windowed ^
  --name VoiceExtractor ^
  --hidden-import sounddevice ^
  --hidden-import soundfile ^
  --hidden-import librosa ^
  --hidden-import scipy.signal ^
  --hidden-import PyQt5.QtCore ^
  --hidden-import PyQt5.QtWidgets ^
  --hidden-import matplotlib.backends.backend_qt5agg ^
  voice_extractor.py
```

### macOS（.app バンドル）

```bash
pyinstaller \
  --onefile \
  --windowed \
  --name VoiceExtractor \
  --hidden-import sounddevice \
  --hidden-import soundfile \
  --hidden-import librosa \
  --hidden-import scipy.signal \
  --hidden-import PyQt5.QtCore \
  --hidden-import PyQt5.QtWidgets \
  --hidden-import matplotlib.backends.backend_qt5agg \
  voice_extractor.py
```

### Linux（単一バイナリ）

```bash
pyinstaller \
  --onefile \
  --name voice_extractor \
  --hidden-import sounddevice \
  --hidden-import soundfile \
  --hidden-import librosa \
  --hidden-import scipy.signal \
  --hidden-import PyQt5.QtCore \
  --hidden-import PyQt5.QtWidgets \
  --hidden-import matplotlib.backends.backend_qt5agg \
  voice_extractor.py
```

ビルド完了後、`dist/` フォルダに実行ファイルが生成される。

### よくあるビルドエラーと対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `ModuleNotFoundError: No module named 'sounddevice'` | hidden-import 漏れ | `--hidden-import sounddevice` を追加 |
| `OSError: PortAudio library not found` | PortAudio DLL が同梱されていない | `sounddevice` の DLL を `--add-binary` で追加、または `sounddevice` を再インストール |
| `libsndfile not found` | soundfile の DLL が見つからない | `--add-binary "venv/Lib/site-packages/soundfile.libs/*:soundfile.libs"` を追加（Windows） |
| EXE が起動後すぐ落ちる | コンソールが隠れてエラーが見えない | `--windowed` を外して `--console` でデバッグ |
| フォントが文字化けする | Yu Gothic が EXE に含まれない | `matplotlib.rc` の `font.family` を `'sans-serif'` に変更してビルド |

---

## 画面説明と操作手順

### 基本的な使い方（標準解析）

**① ファイル選択**

「Mix（音楽＋声）」と「Music（音楽のみ）」の各「選択…」ボタンを押してファイルを指定する。
対応フォーマット：MP3 / WAV / FLAC / OGG / AAC / M4A

**② パラメータ設定**

| パラメータ | 説明 | 推奨値 |
|------------|------|--------|
| SR | サンプリングレート。入力ファイルと合わせる | 44100 Hz |
| 粗い検索区間 | 相互相関に使う区間長（秒）。長いほど精度が上がるが遅い | 10 秒 |
| ストレッチ補正 | 再生速度差がある場合にチェック（CDリッピング差など） | ON |
| 補正幅± | ストレッチ率の探索範囲 | 0.002 |

**③ 解析実行**

「▶ 解析・抽出を実行」を押す。進行状況がプログレスバーで表示される。

**④ グラフ確認**

「波形・相関グラフ」タブで以下を確認する。

- 相互相関グラフ：ピークが鋭いほど位置合わせ精度が高い
- アライメント後の重ね合わせ：Mix と Music Aligned が近いほど差分が小さくなる
- 抽出された Voice：音楽成分の漏れが少ないか確認する

**⑤ 保存**

「📂 2ch×3ファイル保存」または「🎛 6ch マルチトラック保存」で出力する。

---

### 精密補正タブの使い方

標準解析で音楽成分が残っている場合に使用する。

**① 音楽のみ区間を選択**

「精密補正」タブの Mix 波形上で、**声が入っていない区間**（音楽のみの部分）をマウスでドラッグして選択する。黄色のハイライトで範囲が表示される。

選択が難しい場合は「開始」「終了」のスピンボックスに数値を直接入力できる。

**② 再生確認**

「▶ Mix（選択範囲）」で選択区間の音を確認し、声が含まれていないことを確かめる。

**③ 補正オプション設定**

| オプション | 説明 |
|------------|------|
| フレーム長 | STFT のフレームサイズ。長いほど周波数分解能が高いが時間分解能が低下。4096 から試す |
| フレーム集約方法 | `median`：外れ値に強く安定（推奨）。`mean`：全フレーム平均。`best`：最も相関が高い1フレームのみ使用 |

**④ 適用**

「🔬 精密補正を適用（L/R 独立）」を押す。ログに L/R それぞれの補正前後の相関係数が表示される。

相関係数が **改善** していれば補正成功。**悪化** していればフレーム長・集約方法・選択区間を変えて再試行する。

**⑤ 聴き比べ**

「▶ 補正前 Voice」「▶ 補正後 Voice」ボタンでステレオ再生して比較する。

**⑥ 保存**

精密補正後に保存ボタンを押すと補正済みの結果が出力される。
「↩ 元の結果に戻す」でいつでも標準解析結果に戻せる。

---

## 出力ファイル仕様

すべての出力ファイルは **24-bit PCM WAV** 形式。

| ファイル | 内容 | 備考 |
|----------|------|------|
| `voice_stereo.wav` | ボーカル成分 (L/R) | Mix − Music aligned |
| `music_stereo.wav` | 補正済み音楽トラック (L/R) | 位置・音量・周波数補正済み |
| `mix_stereo.wav` | 入力 Mix (L/R) | 無加工コピー |
| `multitrack_6ch.wav` | 上記3素材を2ch×3で収録 | Ch1-6、DAW 取り込み用 |

クリッピング防止処理：ピーク振幅が 0.99 を超える場合は全体を正規化する。

---

## アルゴリズム詳細

### オフセット定義

```
offset > 0 : music[offset:] が mix[0:] に対応（music が遅れて始まる）
offset < 0 : mix[-offset:] が music[0:] に対応（mix が遅れて始まる）
```

### Step 1：粗いオフセット検索

各ファイルの RMS が最大値の10%以上の有効区間から中心付近のセグメントを切り出し、`scipy.signal.correlate` で正規化相互相関を計算する。

### Step 2：精密オフセット検索

Step 1 の結果を中心に ±1秒 の範囲をサンプル単位で走査し、正規化相互相関が最大となる offset を確定する。

### Step 3：ストレッチ補正

`scipy.optimize.minimize_scalar` で相関係数を最大化する再生速度比率を探索し、`librosa.resample`（soxr_hq）で Music を伸縮する。ステレオは L/R 同一比率で処理する。

### Step 4：音量スケーリング

重複区間（最大30秒）での `min |mix - scale × music|²` を最小二乗法で解き、スケール係数を求める。

### Step 5：差分抽出

```
music_aligned[ms : ms+N] = music[mu : mu+N] × scale
voice = mix − music_aligned    （L/R それぞれ）
```

### 精密補正：STFT フィルタ推定

指定区間（音楽のみ）で STFT を計算し、フレームごとに伝達関数 `H[f,t]` を推定する。

```
H[f, t] = conj(Music[f,t]) × Mix[f,t] / (|Music[f,t]|² + ε)
```

これを時間軸で集約（median / mean / best）して時間不変フィルタ `H[f]` を求め、Music 全体に適用する。Mix は変更しない。

---

## パラメータ調整ガイド

### 相関係数が低い（0.8 未満）場合

1. ストレッチ補正を ON にして補正幅を広げる（0.005 など）
2. 粗い検索区間を長くする（30秒）
3. SR を入力ファイルの実際のサンプリングレートに合わせる

### 精密補正後に声が増えた場合

1. 選択区間を変える（声が完全に入っていない区間にする）
2. フレーム長を短くする（2048 など）
3. 集約方法を `median` → `best` に変える
4. 区間を短くして安定した音楽区間のみに絞る

### 精密補正後も音楽が残る場合

1. フレーム長を長くする（8192 など）
2. 集約方法を `median` → `mean` に変える
3. 別の区間（音楽の種類が変わる前後）で再試行する

### ステレオ L/R の補正量が大きく異なる場合

L と R で録音条件が異なる可能性がある。6ch マルチトラックで個別確認し、DAW 側でさらに調整することを推奨する。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| 起動時に `No module named 'PyQt5'` | ライブラリ未インストール | `pip install PyQt5` |
| 起動時に `No module named 'sounddevice'` | ライブラリ未インストール | `pip install sounddevice` |
| MP3 読み込みエラー | ffmpeg が未インストール | ffmpeg をインストールして PATH を通す |
| 再生音が出ない | デフォルト出力デバイスが無効 | OS のサウンド設定でデフォルトデバイスを確認 |
| グラフの日本語が文字化けする | Yu Gothic が見つからない | `matplotlib.rc('font', family='IPAexGothic')` など環境に合わせて変更 |
| 精密補正が遅い | フレーム長が大きすぎる / 区間が長い | フレーム長を 2048 に下げる、または区間を10秒程度に絞る |
| EXE 化後に起動しない | DLL が同梱されていない | `--console` でビルドして起動時のエラーを確認する |
