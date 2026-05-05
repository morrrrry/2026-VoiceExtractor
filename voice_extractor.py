#!/usr/bin/env python3
"""
Voice Extractor - 音楽と声が混在したファイルから声を抽出するツール
音楽のみのファイルを参照として、位置合わせ・音量合わせ・差分抽出を行う

処理方針:
  - Mix（音楽＋声）は一切処理しない（声の質感を保護）
  - Music（音楽のみ）を Mix に近づけるよう補正する
  - 補正済み Music を Mix から引いて Voice を抽出
  - L/R チャンネル独立処理でステレオ出力

オフセット符号の定義:
  offset > 0 → music[offset:] が mix[0:] に対応する（music が遅れている）
  offset < 0 → mix[-offset:] が music[0:] に対応する（mix が遅れている）
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
matplotlib.rc('font', family='Yu Gothic')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy import signal
from scipy.optimize import minimize_scalar
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QGroupBox, QTextEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QMessageBox, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import sounddevice as sd


# ─────────────────────────────────────────────
#  Audio Loading
# ─────────────────────────────────────────────
def load_audio_mono(path: str, target_sr: int = 44100):
    """MP3/WAV を読み込んでモノラルに変換（内部処理用）"""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr


def load_audio_stereo(path: str, target_sr: int = 44100):
    """ステレオ (shape: [samples, 2]) で読み込む。モノラル素材は複製してステレオ化。"""
    y, sr = librosa.load(path, sr=target_sr, mono=False)
    if y.ndim == 1:
        stereo = np.stack([y, y], axis=1)
    elif y.shape[0] == 1:
        stereo = np.stack([y[0], y[0]], axis=1)
    else:
        stereo = y[:2].T
    return stereo.astype(np.float32), sr


# ─────────────────────────────────────────────
#  相関計算ユーティリティ
# ─────────────────────────────────────────────
def compute_correlation(mix: np.ndarray, music: np.ndarray, offset: int,
                        eval_sec: float = 10.0, sr: int = 44100) -> float:
    """
    指定オフセットでの正規化相互相関（モノラル）
    offset > 0: music[offset:] vs mix[0:]
    offset < 0: music[0:]      vs mix[-offset:]
    """
    if offset >= 0:
        mix_start, mu_start = 0, offset
    else:
        mix_start, mu_start = -offset, 0

    eval_len = int(eval_sec * sr)
    actual_len = min(len(mix) - mix_start, len(music) - mu_start, eval_len)
    if actual_len < 100:
        return -1.0

    m = mix  [mix_start : mix_start + actual_len]
    n = music[mu_start  : mu_start  + actual_len]
    denom = np.sqrt(np.dot(m, m) * np.dot(n, n))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(m, n) / denom)


# ─────────────────────────────────────────────
#  STFT フィルタ推定・適用（1ch）
# ─────────────────────────────────────────────
def estimate_stft_filter_mono(mix_seg: np.ndarray, music_seg: np.ndarray,
                               frame_len: int = 4096, aggregate: str = 'median',
                               reg_percentile: float = 5.0,
                               log_fn=None) -> np.ndarray:
    """
    1チャンネル分の STFT フィルタ H[f] を推定する。
    H[f,t] = conj(Music[f,t]) * Mix[f,t] / (|Music[f,t]|^2 + eps)
    を時間軸で集約して返す。

    aggregate: 'mean' / 'median' / 'best'
    Returns: H (complex, shape=(frame_len//2+1,))
    """
    from scipy.signal import stft as _stft

    hop = frame_len // 2
    _, _, Zmix   = _stft(mix_seg,   fs=1, window='hann',
                          nperseg=frame_len, noverlap=frame_len - hop)
    _, _, Zmusic = _stft(music_seg, fs=1, window='hann',
                          nperseg=frame_len, noverlap=frame_len - hop)

    avg_power = np.mean(np.abs(Zmusic)**2, axis=1, keepdims=True)
    eps = np.percentile(avg_power, reg_percentile) * 0.05 + 1e-12

    H_frames = np.conj(Zmusic) * Zmix / (np.abs(Zmusic)**2 + eps)
    n_frames = H_frames.shape[1]

    if log_fn:
        log_fn(f"    フレーム長: {frame_len}  フレーム数: {n_frames}  集約: {aggregate}")

    if aggregate == 'mean':
        H_avg = np.mean(H_frames, axis=1)
    elif aggregate == 'median':
        H_avg = (np.median(np.real(H_frames), axis=1) +
                 1j * np.median(np.imag(H_frames), axis=1))
    elif aggregate == 'best':
        approx = Zmusic * H_frames
        num    = np.sum(np.real(np.conj(approx) * Zmix), axis=0)
        den    = (np.sqrt(np.sum(np.abs(approx)**2, axis=0)) *
                  np.sqrt(np.sum(np.abs(Zmix)**2,   axis=0)) + 1e-12)
        corrs  = num / den
        best_i = int(np.argmax(corrs))
        H_avg  = H_frames[:, best_i]
        if log_fn:
            log_fn(f"    best フレーム: #{best_i}  スペクトル相関: {corrs[best_i]:.4f}")
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    return H_avg


def apply_stft_filter_mono(sig: np.ndarray, H: np.ndarray,
                            frame_len: int = 4096) -> np.ndarray:
    """H（時間不変フィルタ）を信号 sig 全体に適用して返す（1ch）"""
    from scipy.signal import stft as _stft, istft as _istft

    hop = frame_len // 2
    _, _, Zsig = _stft(sig, fs=1, window='hann',
                        nperseg=frame_len, noverlap=frame_len - hop)
    Zfilt = Zsig * H[:, None]
    _, out = _istft(Zfilt, fs=1, window='hann',
                     nperseg=frame_len, noverlap=frame_len - hop)
    result = np.zeros(len(sig), dtype=np.float32)
    copy = min(len(sig), len(out))
    result[:copy] = out[:copy].astype(np.float32)
    return result


def refine_channel(mix_ch: np.ndarray, music_ch: np.ndarray,
                   sr: int, s: int, e: int,
                   frame_len: int, aggregate: str,
                   log_fn=None) -> np.ndarray:
    """
    1チャンネル分の精密補正。
    music_ch を補正して mix_ch との差分が最小になるようにする。
    Returns: music_refined (同長)
    """
    mix_seg   = mix_ch  [s:e]
    music_seg = music_ch[s:e]

    # フレーム長自動縮小
    fl = frame_len
    while fl > len(mix_seg) // 4 and fl > 256:
        fl //= 2

    H = estimate_stft_filter_mono(mix_seg, music_seg,
                                   frame_len=fl, aggregate=aggregate,
                                   log_fn=log_fn)

    # 全体に適用
    music_work = apply_stft_filter_mono(music_ch, H, frame_len=fl)

    # 指定区間でスケール再推定（最小二乗）
    denom = np.dot(music_work[s:e], music_work[s:e])
    scale_r = float(np.dot(mix_ch[s:e], music_work[s:e]) / denom) \
              if denom > 1e-10 else 1.0
    music_work *= scale_r

    return music_work.astype(np.float32)


def refine_extraction_stereo(mix_st: np.ndarray, music_aligned_st: np.ndarray,
                              sr: int, start_sec: float, end_sec: float,
                              frame_len: int = 4096,
                              aggregate: str = 'median',
                              log_fn=None):
    """
    ステレオ（shape: [N, 2]）で精密補正を行う。
    L/R 独立に STFT フィルタを推定・適用。
    mix_st は一切変更しない（声の質感保護）。

    Returns: (voice_refined_st, music_refined_st, info_dict)
    """
    s = int(start_sec * sr)
    e = int(end_sec   * sr)
    N = len(mix_st)
    s = max(0, min(s, N - 2))
    e = max(s + sr, min(e, N))

    if log_fn:
        log_fn(f"  推定区間: {s/sr:.2f}〜{e/sr:.2f}秒 ({(e-s)/sr:.2f}秒)")

    music_ref_st = np.zeros_like(music_aligned_st)
    corr_before_list = []
    corr_after_list  = []

    for ch, ch_name in enumerate(['L', 'R']):
        if log_fn:
            log_fn(f"  ── チャンネル {ch_name} ──")
        mix_ch   = mix_st          [:, ch]
        music_ch = music_aligned_st[:, ch]

        c_before = compute_correlation(mix_ch[s:e], music_ch[s:e], 0, sr=sr)
        if log_fn:
            log_fn(f"    補正前 相関（区間内）: {c_before:.4f}")

        music_refined_ch = refine_channel(
            mix_ch, music_ch, sr, s, e,
            frame_len=frame_len, aggregate=aggregate,
            log_fn=log_fn)

        c_after = compute_correlation(mix_ch[s:e], music_refined_ch[s:e], 0, sr=sr)
        if log_fn:
            log_fn(f"    補正後 相関（区間内）: {c_after:.4f}")

        music_ref_st[:, ch] = music_refined_ch
        corr_before_list.append(c_before)
        corr_after_list.append(c_after)

    voice_ref_st = mix_st - music_ref_st

    if log_fn:
        def rms_db(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-10)
        log_fn(f"  補正前 voice RMS (L): {rms_db(mix_st[:,0] - music_aligned_st[:,0]):.1f} dBFS")
        log_fn(f"  補正後 voice RMS (L): {rms_db(voice_ref_st[:,0]):.1f} dBFS")
        log_fn(f"  補正前 voice RMS (R): {rms_db(mix_st[:,1] - music_aligned_st[:,1]):.1f} dBFS")
        log_fn(f"  補正後 voice RMS (R): {rms_db(voice_ref_st[:,1]):.1f} dBFS")

    info = {
        'frame_len': frame_len, 'aggregate': aggregate,
        'corr_before_L': corr_before_list[0], 'corr_after_L': corr_after_list[0],
        'corr_before_R': corr_before_list[1], 'corr_after_R': corr_after_list[1],
        'corr_before': np.mean(corr_before_list),
        'corr_after':  np.mean(corr_after_list),
    }
    return voice_ref_st, music_ref_st, info


# ─────────────────────────────────────────────
#  Step 1: 粗い時間オフセット検出（モノラルで実行）
# ─────────────────────────────────────────────
def coarse_offset_search(mix: np.ndarray, music: np.ndarray, sr: int,
                          segment_sec: float = 10.0, log_fn=None):
    """
    scipy.signal.correlate でサンプル単位の粗いオフセットを検出（モノラル）

    offset 定義: music[offset:] が mix[0:] に対応するサンプル数
    """
    seg_len = int(segment_sec * sr)

    def center_of_valid(sig, seg_l):
        rms = np.array([np.sqrt(np.mean(sig[i:i+sr]**2))
                        for i in range(0, len(sig)-sr, sr)])
        valid = np.where(rms > rms.max() * 0.1)[0]
        center = valid[len(valid)//2] if len(valid) > 0 else len(rms)//2
        start = max(0, center * sr - seg_l // 2)
        start = min(start, max(0, len(sig) - seg_l))
        return start

    m_start  = center_of_valid(mix, seg_len)
    ms_start = center_of_valid(music, seg_len)
    mix_seg   = mix  [m_start  : m_start  + min(seg_len, len(mix)   - m_start)]
    music_seg = music[ms_start : ms_start + min(seg_len, len(music) - ms_start)]

    corr = signal.correlate(mix_seg, music_seg, mode='full')
    lags = signal.correlation_lags(len(mix_seg), len(music_seg), mode='full')

    norm = np.sqrt(np.sum(mix_seg**2) * np.sum(music_seg**2)) + 1e-10
    corr_norm = corr / norm

    best_idx = np.argmax(np.abs(corr_norm))
    raw_lag  = lags[best_idx]
    offset   = ms_start - raw_lag - m_start

    if log_fn:
        log_fn(f"  mix セグメント: {m_start/sr:.2f}〜{(m_start+len(mix_seg))/sr:.2f}秒")
        log_fn(f"  music セグメント: {ms_start/sr:.2f}〜{(ms_start+len(music_seg))/sr:.2f}秒")
        log_fn(f"  生ラグ: {raw_lag:+d} samples ({raw_lag/sr:+.3f}秒)")
        log_fn(f"  → オフセット: {offset:+d} samples ({offset/sr:+.3f}秒)")

    lags_sec = (lags + m_start - ms_start) / sr
    return offset, lags_sec, corr_norm


# ─────────────────────────────────────────────
#  Step 2: 精密オフセット検索
# ─────────────────────────────────────────────
def fine_offset_search(mix: np.ndarray, music: np.ndarray, sr: int,
                       coarse_offset: int, search_range_sec: float = 1.0,
                       log_fn=None):
    """粗いオフセット周辺をサンプル精度で精密化（モノラル）"""
    search_range = int(search_range_sec * sr)
    best_corr   = -np.inf
    best_offset = coarse_offset

    step = max(1, search_range // 100)
    for off in range(coarse_offset - search_range,
                     coarse_offset + search_range, step):
        c = compute_correlation(mix, music, off, sr=sr)
        if c > best_corr:
            best_corr, best_offset = c, off

    for delta in range(-step * 2, step * 2 + 1):
        off = best_offset + delta
        c = compute_correlation(mix, music, off, sr=sr)
        if c > best_corr:
            best_corr, best_offset = c, off

    if log_fn:
        log_fn(f"  精密オフセット: {best_offset:+d} samples ({best_offset/sr:+.3f}秒)")
        log_fn(f"  最大相関係数: {best_corr:.4f}")
    return best_offset, best_corr


# ─────────────────────────────────────────────
#  Step 3: ストレッチ比率の最適化（モノラル）
# ─────────────────────────────────────────────
def optimize_stretch(mix: np.ndarray, music: np.ndarray, sr: int,
                     offset: int, stretch_range=(0.998, 1.002), log_fn=None):
    """music の再生速度を微調整"""
    def neg_corr(ratio):
        n_target = max(1, int(len(music) / ratio))
        s = librosa.resample(music, orig_sr=len(music), target_sr=n_target,
                             res_type='soxr_hq')
        return -compute_correlation(mix, s, offset, sr=sr)

    result = minimize_scalar(neg_corr, bounds=stretch_range, method='bounded',
                             options={'xatol': 1e-6})
    ratio = result.x
    n_target = max(1, int(len(music) / ratio))
    stretched = librosa.resample(music, orig_sr=len(music), target_sr=n_target,
                                  res_type='soxr_hq')
    corr_after = compute_correlation(mix, stretched, offset, sr=sr)

    if log_fn:
        log_fn(f"  ストレッチ比率: {ratio:.6f} ({(ratio-1)*100:+.4f}%)")
        log_fn(f"  補正後相関: {corr_after:.4f}")
    return stretched, ratio


def stretch_stereo(music_st: np.ndarray, ratio: float) -> np.ndarray:
    """ステレオ音源にストレッチを適用"""
    result_channels = []
    for ch in range(2):
        ch_data = music_st[:, ch]
        n_target = max(1, int(len(ch_data) / ratio))
        stretched = librosa.resample(ch_data, orig_sr=len(ch_data),
                                     target_sr=n_target, res_type='soxr_hq')
        result_channels.append(stretched)
    # 長さを揃える
    min_len = min(len(c) for c in result_channels)
    return np.stack([c[:min_len] for c in result_channels], axis=1).astype(np.float32)


# ─────────────────────────────────────────────
#  Step 4: 音量スケーリング
# ─────────────────────────────────────────────
def estimate_volume_scale(mix: np.ndarray, music: np.ndarray, sr: int,
                           offset: int, log_fn=None) -> float:
    """music × scale ≈ mix の最小二乗スケールを推定（モノラル）"""
    mix_start, mu_start = (0, offset) if offset >= 0 else (-offset, 0)
    actual_len = min(len(mix) - mix_start, len(music) - mu_start, int(30.0 * sr))
    if actual_len <= 0:
        return 1.0

    m = mix  [mix_start : mix_start + actual_len]
    n = music[mu_start  : mu_start  + actual_len]
    denom = np.dot(n, n)
    scale = float(np.dot(m, n) / denom) if denom > 1e-10 else 1.0

    if log_fn:
        log_fn(f"  スケール係数: {scale:.4f} ({20*np.log10(abs(scale)+1e-10):.2f} dB)")
    return scale


# ─────────────────────────────────────────────
#  Step 5: アライメント・差分抽出（ステレオ）
# ─────────────────────────────────────────────
def align_stereo(music_st: np.ndarray, offset: int, scale: float,
                 out_len: int) -> np.ndarray:
    """
    ステレオ music_st をオフセット・スケールで mix に合わせて配置する。
    Returns: aligned (shape: [out_len, 2], float32)
    """
    aligned = np.zeros((out_len, 2), dtype=np.float32)
    ms = 0   if offset >= 0 else -offset
    mu = offset if offset >= 0 else 0
    copy_len = min(out_len - ms, len(music_st) - mu)
    if copy_len > 0:
        aligned[ms : ms + copy_len] = music_st[mu : mu + copy_len] * scale
    return aligned


def extract_voice_stereo(mix_st: np.ndarray,
                          music_aligned_st: np.ndarray) -> np.ndarray:
    """Mix から Music を引いて Voice を抽出（ステレオ）"""
    out_len = min(len(mix_st), len(music_aligned_st))
    return (mix_st[:out_len] - music_aligned_st[:out_len]).astype(np.float32)


# ─────────────────────────────────────────────
#  Worker Thread
# ─────────────────────────────────────────────
class ProcessWorker(QThread):
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, mix_path, music_path, sr, do_stretch, coarse_seg, stretch_range):
        super().__init__()
        self.mix_path   = mix_path
        self.music_path = music_path
        self.sr         = sr
        self.do_stretch = do_stretch
        self.coarse_seg = coarse_seg
        self.stretch_range = stretch_range

    def run(self):
        try:
            self.log.emit("━━━ Step 0: ファイル読み込み ━━━")
            # モノラル（オフセット検出・相関計算用）
            mix_mono,   sr = load_audio_mono(self.mix_path,   self.sr)
            self.log.emit(f"  Mix:   {len(mix_mono)/sr:.2f}秒 ({len(mix_mono):,} samples @ {sr} Hz)")
            self.progress.emit(8)

            music_mono, _  = load_audio_mono(self.music_path, self.sr)
            self.log.emit(f"  Music: {len(music_mono)/sr:.2f}秒 ({len(music_mono):,} samples @ {sr} Hz)")

            # ステレオ読み込み
            mix_st,   _ = load_audio_stereo(self.mix_path,   self.sr)
            music_st, _ = load_audio_stereo(self.music_path, self.sr)
            self.progress.emit(18)

            self.log.emit("\n━━━ Step 1: 粗いオフセット検索（相互相関） ━━━")
            coarse_off, lags, corr = coarse_offset_search(
                mix_mono, music_mono, sr, self.coarse_seg, log_fn=self.log.emit)
            self.progress.emit(35)

            self.log.emit("\n━━━ Step 2: 精密オフセット検索 ━━━")
            fine_off, peak_corr = fine_offset_search(
                mix_mono, music_mono, sr, coarse_off, log_fn=self.log.emit)
            self.progress.emit(50)

            stretch_ratio = 1.0
            music_mono_work = music_mono
            music_st_work   = music_st

            if self.do_stretch:
                self.log.emit("\n━━━ Step 3: ストレッチ補正 ━━━")
                music_mono_work, stretch_ratio = optimize_stretch(
                    mix_mono, music_mono, sr, fine_off,
                    stretch_range=self.stretch_range,
                    log_fn=self.log.emit)
                # ステレオにも同じ ratio を適用
                music_st_work = stretch_stereo(music_st, stretch_ratio)
                # ストレッチ後の微調整
                fine_off, peak_corr = fine_offset_search(
                    mix_mono, music_mono_work, sr, fine_off,
                    search_range_sec=0.2, log_fn=None)
            else:
                self.log.emit("\n[Step 3 スキップ]")
            self.progress.emit(65)

            self.log.emit("\n━━━ Step 4: 音量スケーリング ━━━")
            scale = estimate_volume_scale(mix_mono, music_mono_work, sr, fine_off,
                                          log_fn=self.log.emit)
            self.progress.emit(78)

            self.log.emit("\n━━━ Step 5: ステレオ差分抽出 ━━━")
            out_len = len(mix_st)
            music_aligned_st = align_stereo(music_st_work, fine_off, scale, out_len)
            voice_st         = extract_voice_stereo(mix_st, music_aligned_st)

            def rms_db(x): return 20*np.log10(np.sqrt(np.mean(x**2))+1e-10)
            self.log.emit(f"  Mix RMS (L/R): {rms_db(mix_st[:,0]):.1f} / {rms_db(mix_st[:,1]):.1f} dBFS")
            self.log.emit(f"  Music Aligned RMS (L/R): {rms_db(music_aligned_st[:,0]):.1f} / {rms_db(music_aligned_st[:,1]):.1f} dBFS")
            self.log.emit(f"  Voice RMS (L/R): {rms_db(voice_st[:,0]):.1f} / {rms_db(voice_st[:,1]):.1f} dBFS")
            self.progress.emit(95)

            self.finished.emit({
                'mix_st':           mix_st,           # [N, 2]
                'music_st':         music_st_work,    # [N, 2] ストレッチ済み・スケール前
                'music_aligned_st': music_aligned_st, # [N, 2] アライメント+スケール済み
                'voice_st':         voice_st,         # [N, 2]
                # モノラル（グラフ用）
                'mix_mono':   mix_mono,
                'music_mono': music_mono_work,
                'lags':  lags,
                'corr':  corr,
                # パラメータ
                'sr':           sr,
                'offset':       fine_off,
                'scale':        scale,
                'stretch_ratio': stretch_ratio,
                'peak_corr':    peak_corr,
            })
            self.progress.emit(100)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class VoiceExtractorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result = None
        self.mix_path = self.music_path = None
        self.setWindowTitle("Voice Extractor — ステレオ音楽差分による声抽出")
        self.setMinimumSize(1200, 840)
        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0f1923; color: #cdd6e0; }
            QGroupBox { border: 1px solid #1e3a52; border-radius: 6px;
                        margin-top: 10px; font-weight: bold; color: #7eb8d4; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #1a2c3d; border: 1px solid #2a4a6a;
                          border-radius: 4px; padding: 6px 14px; color: #7eb8d4; }
            QPushButton:hover { background: #2a4a6a; color: #fff; }
            QPushButton:disabled { color: #445; border-color: #222; }
            QPushButton#run_btn  { background: #1a2c3d; border: 2px solid #3a9dd4;
                                    color: #3a9dd4; font-weight: bold; font-size: 13px; }
            QPushButton#run_btn:hover  { background: #3a9dd4; color: #000; }
            QPushButton#save_btn { background: #0f2a1a; border: 1px solid #3aaf5a;
                                    color: #3aaf5a; font-weight: bold; }
            QPushButton#save_btn:hover { background: #3aaf5a; color: #000; }
            QTextEdit { background: #080e14; border: 1px solid #1e3a52;
                        color: #88b4c8; font-family: monospace; font-size: 11px; }
            QLabel { color: #9ab0c0; }
            QLabel#file_label { color: #5a7a8a; font-size: 11px; }
            QProgressBar { border: 1px solid #1e3a52; border-radius: 3px;
                           background: #080e14; height: 12px; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                  stop:0 #1a6aaa, stop:1 #3ad4d4); border-radius: 2px; }
            QSpinBox, QDoubleSpinBox { background: #080e14; border: 1px solid #1e3a52;
                                       color: #cdd6e0; padding: 3px; }
            QCheckBox { color: #9ab0c0; spacing: 6px; }
            QTabWidget::pane { border: 1px solid #1e3a52; }
            QTabBar::tab { background: #1a2c3d; color: #607888; padding: 7px 18px;
                           border-radius: 3px 3px 0 0; }
            QTabBar::tab:selected { background: #0f1923; color: #7eb8d4;
                                    border-bottom: 2px solid #3a9dd4; }
            QPushButton#multi_btn { background: #1a1a2a; border: 1px solid #b07ad4;
                                     color: #b07ad4; font-weight: bold; }
            QPushButton#multi_btn:hover { background: #b07ad4; color: #000; }
            QPushButton#refine_btn { background: #1a2a1a; border: 1px solid #80d4a0;
                                      color: #80d4a0; font-weight: bold; }
            QPushButton#refine_btn:hover { background: #80d4a0; color: #000; }
            QPushButton#revert_btn { background: #2a1a1a; border: 1px solid #d48080;
                                      color: #d48080; }
            QPushButton#revert_btn:hover { background: #d48080; color: #000; }
            QComboBox { background: #0d0d1a; color: #cdd6e0; border: 1px solid #1e3a52; padding: 2px; }
            QComboBox QAbstractItemView { background: #0d0d1a; color: #cdd6e0; }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ファイル選択
        fg = QGroupBox("入力ファイル")
        fl = QVBoxLayout(fg); fl.setSpacing(4)
        for attr, label, key in [
            ('mix_label',   'Mix   (音楽＋声):', 'mix'),
            ('music_label', 'Music (音楽のみ):', 'music'),
        ]:
            lbl = QLabel("未選択"); lbl.setObjectName("file_label")
            setattr(self, attr, lbl)
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(lbl, 1)
            btn = QPushButton("選択…")
            btn.clicked.connect(lambda _, k=key: self._pick(k))
            row.addWidget(btn)
            fl.addLayout(row)
        root.addWidget(fg)

        # パラメータ
        pg = QGroupBox("パラメータ")
        pl = QHBoxLayout(pg); pl.setSpacing(12)
        pl.addWidget(QLabel("SR:"))
        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 96000); self.sr_spin.setValue(44100)
        self.sr_spin.setSuffix(" Hz"); self.sr_spin.setFixedWidth(100)
        pl.addWidget(self.sr_spin)

        pl.addWidget(QLabel("粗い検索区間:"))
        self.seg_spin = QDoubleSpinBox()
        self.seg_spin.setRange(2.0, 60.0); self.seg_spin.setValue(10.0)
        self.seg_spin.setSuffix(" 秒"); self.seg_spin.setFixedWidth(90)
        pl.addWidget(self.seg_spin)

        self.stretch_chk = QCheckBox("ストレッチ補正")
        self.stretch_chk.setChecked(True); pl.addWidget(self.stretch_chk)

        pl.addWidget(QLabel("補正幅±:"))
        self.stretch_spin = QDoubleSpinBox()
        self.stretch_spin.setRange(0.0005, 0.05); self.stretch_spin.setValue(0.002)
        self.stretch_spin.setDecimals(4); self.stretch_spin.setSingleStep(0.0005)
        self.stretch_spin.setFixedWidth(90)
        pl.addWidget(self.stretch_spin)
        pl.addStretch()
        root.addWidget(pg)

        # ボタン行
        br = QHBoxLayout()
        self.run_btn = QPushButton("▶  解析・抽出を実行")
        self.run_btn.setObjectName("run_btn"); self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run); br.addWidget(self.run_btn)

        self.save_btn = QPushButton("📂  2ch×3ファイル保存")
        self.save_btn.setObjectName("save_btn"); self.save_btn.setMinimumHeight(36)
        self.save_btn.setEnabled(False); self.save_btn.clicked.connect(self._save)
        br.addWidget(self.save_btn)

        self.save_multi_btn = QPushButton("🎛  6ch マルチトラック保存")
        self.save_multi_btn.setObjectName("multi_btn"); self.save_multi_btn.setMinimumHeight(36)
        self.save_multi_btn.setEnabled(False)
        self.save_multi_btn.clicked.connect(self._save_multitrack)
        br.addWidget(self.save_multi_btn)
        root.addLayout(br)

        self.progress = QProgressBar(); root.addWidget(self.progress)

        # タブ
        self.tabs = QTabWidget()

        # グラフタブ
        gw = QWidget(); gv = QVBoxLayout(gw); gv.setContentsMargins(0,0,0,0)
        self.figure  = plt.figure(figsize=(14, 10), facecolor='#080e14')
        self.canvas  = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, gw)
        self.toolbar.setStyleSheet("background:#1a2c3d; color:#7eb8d4;")
        gv.addWidget(self.toolbar); gv.addWidget(self.canvas)
        self.tabs.addTab(gw, "📊 波形・相関グラフ")

        # ログタブ
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Monospace", 10))
        self.tabs.addTab(self.log_view, "📝 処理ログ")

        # 精密補正タブ
        self.refine_tab = self._build_refine_tab()
        self.tabs.addTab(self.refine_tab, "🔬 精密補正（オプション）")
        self.tabs.setTabEnabled(2, False)
        root.addWidget(self.tabs, 1)

        self.statusBar().showMessage("ファイルを選択して「解析・抽出を実行」を押してください")

    def _pick(self, target):
        path, _ = QFileDialog.getOpenFileName(
            self, "音声ファイルを選択", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.aac *.m4a);;All Files (*)")
        if path:
            if target == 'mix':
                self.mix_path = path; self.mix_label.setText(os.path.basename(path))
            else:
                self.music_path = path; self.music_label.setText(os.path.basename(path))

    def _run(self):
        if not self.mix_path or not self.music_path:
            QMessageBox.warning(self, "エラー", "2つのファイルを選択してください"); return
        self.run_btn.setEnabled(False); self.save_btn.setEnabled(False)
        self.save_multi_btn.setEnabled(False)
        self.log_view.clear(); self.progress.setValue(0)
        self.figure.clear(); self.canvas.draw()

        r = self.stretch_spin.value()
        self.worker = ProcessWorker(
            self.mix_path, self.music_path, self.sr_spin.value(),
            self.stretch_chk.isChecked(), self.seg_spin.value(),
            (1.0 - r, 1.0 + r))
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_view.append)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_err(self, msg):
        self.log_view.append("\n[ERROR]\n" + msg)
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage("エラーが発生しました")

    def _on_done(self, res):
        self.result = res
        # 元結果を保存（精密補正の revert 用）
        self.result_original = {k: v.copy() if isinstance(v, np.ndarray) else v
                                 for k, v in res.items()}
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.save_multi_btn.setEnabled(True)
        self._draw(res)

        dur = len(res['mix_st']) / res['sr']
        self.refine_start.setMaximum(dur - 0.1)
        self.refine_end.setMaximum(dur)
        self.refine_end.setValue(min(dur, self.refine_end.value() or dur))
        self.tabs.setTabEnabled(2, True)
        self._draw_refine_waveform()

        off, sr = res['offset'], res['sr']
        self.statusBar().showMessage(
            f"完了  |  オフセット: {off/sr:+.3f}秒 ({off:+d} samples)  "
            f"スケール: {res['scale']:.4f} ({20*np.log10(abs(res['scale'])+1e-10):.2f}dB)  "
            f"相関: {res['peak_corr']:.4f}  ストレッチ: {res['stretch_ratio']:.6f}")
        self.tabs.setCurrentIndex(0)
        self.log_view.append("\n━━━ 完了 ━━━")

    # ─────────────────────────────────────────────
    #  グラフ描画
    # ─────────────────────────────────────────────
    def _draw(self, res):
        sr   = res['sr']
        mix  = res['mix_mono']
        mus  = res['music_mono']
        mal  = res['music_aligned_st'][:, 0]  # L ch for display
        voc  = res['voice_st'][:, 0]           # L ch for display
        lags = res['lags']
        corr = res['corr']
        off  = res['offset']

        self.figure.clear()
        self.figure.set_facecolor('#080e14')
        C = dict(mix='#3ab4e0', music='#e05a3a', align='#a060d0',
                 voice='#3acf70', corr='#f0c030', bg='#080e14',
                 grid='#12263a', txt='#607888')

        def sax(ax, title, xl='時間 (秒)', yl='振幅'):
            ax.set_facecolor(C['bg'])
            ax.set_title(title, color='#7eb8d4', fontsize=10, pad=4)
            ax.set_xlabel(xl, color=C['txt'], fontsize=8)
            ax.set_ylabel(yl, color=C['txt'], fontsize=8)
            ax.tick_params(colors=C['txt'], labelsize=8)
            for s in ax.spines.values(): s.set_color('#1e3a52')
            ax.grid(True, color=C['grid'], lw=0.5, ls='--', alpha=0.7)

        def wplot(ax, y, sr, color, label, N=10000):
            t = np.linspace(0, len(y)/sr, len(y))
            s = max(1, len(y)//N)
            ax.plot(t[::s], y[::s], color=color, lw=0.6, alpha=0.9, label=label)

        gs = gridspec.GridSpec(4, 2, figure=self.figure,
                               hspace=0.55, wspace=0.3,
                               left=0.07, right=0.97, top=0.96, bottom=0.05)

        ax0 = self.figure.add_subplot(gs[0, 0])
        wplot(ax0, mix, sr, C['mix'], 'Mix (mono)'); sax(ax0, "Mix（音楽＋声）— モノラル参照")
        ax0.legend(fontsize=7, loc='upper right', framealpha=0.3)

        ax1 = self.figure.add_subplot(gs[0, 1])
        wplot(ax1, mus, sr, C['music'], 'Music (mono)'); sax(ax1, "Music Only — モノラル参照")
        ax1.legend(fontsize=7, loc='upper right', framealpha=0.3)

        ax2 = self.figure.add_subplot(gs[1, :])
        s = max(1, len(corr)//16000)
        ax2.plot(lags[::s], corr[::s], color=C['corr'], lw=0.7, alpha=0.9, label='相互相関')
        ax2.axvline(off/sr, color='#ff6060', lw=1.5, ls='--',
                    label=f'検出オフセット: {off/sr:+.3f}秒')
        sax(ax2, "粗い相互相関", yl='正規化相関係数')
        ax2.legend(fontsize=8, loc='upper right', framealpha=0.4)

        ax3 = self.figure.add_subplot(gs[2, :])
        v_sec = min(30.0, len(mix)/sr); v_smp = int(v_sec*sr)
        sv = max(1, v_smp//10000); tv = np.linspace(0, v_sec, v_smp)
        ax3.plot(tv[::sv], mix[:v_smp:sv], color=C['mix'], lw=0.7, alpha=0.85, label='Mix (mono)')
        ax3.plot(tv[::sv], mal[:v_smp:sv], color=C['align'], lw=0.7, alpha=0.85,
                 label=f'Music Aligned L (×{res["scale"]:.3f})')
        sax(ax3, f"アライメント後の重ね合わせ（先頭 {v_sec:.0f}秒）")
        ax3.legend(fontsize=8, loc='upper right', framealpha=0.4)

        ax4 = self.figure.add_subplot(gs[3, :])
        wplot(ax4, voc, sr, C['voice'], '抽出声成分 (L)')
        sax(ax4, "抽出された声成分 — Voice L (Mix − Aligned Music)")
        ax4.legend(fontsize=8, loc='upper right', framealpha=0.4)

        self.canvas.draw()

    # ─────────────────────────────────────────────
    #  保存
    # ─────────────────────────────────────────────
    def _save(self):
        """2ch WAV × 3ファイル（Voice / Music / Mix）を保存"""
        if not self.result:
            return
        folder = QFileDialog.getExistingDirectory(self, "保存フォルダを選択")
        if not folder:
            return

        res     = self.result
        sr      = res['sr']
        voice   = res['voice_st']
        music_a = res['music_aligned_st']
        mix_out = res['mix_st'][:len(voice)]

        files = {
            'voice_stereo.wav': voice,
            'music_stereo.wav': music_a[:len(voice)],
            'mix_stereo.wav':   mix_out,
        }
        for fname, data in files.items():
            data = data.copy().astype(np.float32)
            peak = np.max(np.abs(data))
            if peak > 0.99:
                data = data / peak * 0.99
            fpath = os.path.join(folder, fname)
            sf.write(fpath, data, sr, subtype='PCM_24')
            self.log_view.append(f"[保存] {fpath}")

        self.statusBar().showMessage(f"3ファイル保存完了: {folder}")

    def _save_multitrack(self):
        """6ch WAV を出力"""
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "6ch マルチトラック保存先", "multitrack_6ch.wav",
            "WAV Files (*.wav);;All Files (*)")
        if not path:
            return

        res     = self.result
        sr      = res['sr']
        n       = len(res['voice_st'])
        voice   = res['voice_st']
        music_a = res['music_aligned_st'][:n]
        mix_out = res['mix_st'][:n]

        six_ch = np.stack([
            voice  [:, 0],  # Ch1 Voice L
            voice  [:, 1],  # Ch2 Voice R
            music_a[:, 0],  # Ch3 Music L
            music_a[:, 1],  # Ch4 Music R
            mix_out[:, 0],  # Ch5 Mix L
            mix_out[:, 1],  # Ch6 Mix R
        ], axis=1).astype(np.float32)

        peak = np.max(np.abs(six_ch))
        if peak > 0.99:
            six_ch = six_ch / peak * 0.99
            self.log_view.append(f"[6ch] クリッピング防止: peak={peak:.3f}")

        sf.write(path, six_ch, sr, subtype='PCM_24')
        dur = n / sr
        self.log_view.append(
            f"[6ch 保存完了] {path}\n"
            f"  Ch1/2: Voice L/R  |  Ch3/4: Music Aligned L/R  |  Ch5/6: Mix L/R\n"
            f"  {dur:.2f}秒  {sr} Hz  24-bit PCM  6ch")
        self.statusBar().showMessage(f"6ch 保存完了: {path}")

    # ─────────────────────────────────────────────
    #  精密補正タブ
    # ─────────────────────────────────────────────
    def _build_refine_tab(self) -> QWidget:
        self._sel_start  = 0.0
        self._sel_end    = 0.0
        self._drag_x0    = None
        self._play_stream = None
        self._play_pos   = 0
        self._play_data  = None
        self._play_sr    = 44100
        self._play_offset_sec = 0.0
        self._voice_before_st = None
        self._voice_after_st  = None

        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setSpacing(6); vl.setContentsMargins(8, 8, 8, 8)

        desc = QLabel(
            "波形上でドラッグして「音楽のみ区間」を選択。"
            "その区間の L/R 各チャンネルで STFT フィルタを推定し、Music 全体に適用して差分を再計算します。"
            "Mix は一切変更しません（声の質感保護）。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#7eb8d4; font-size:11px; padding:4px 6px; "
                           "border:1px solid #1e3a52; border-radius:4px;")
        vl.addWidget(desc)

        # 波形キャンバス
        self.ref_fig = plt.figure(figsize=(12, 3.5), facecolor='#080e14')
        self.ref_canvas = FigureCanvas(self.ref_fig)
        self.ref_canvas.setMinimumHeight(180)
        self.ref_canvas.mpl_connect('button_press_event',   self._refine_mouse_press)
        self.ref_canvas.mpl_connect('motion_notify_event',  self._refine_mouse_move)
        self.ref_canvas.mpl_connect('button_release_event', self._refine_mouse_release)
        vl.addWidget(self.ref_canvas, 3)

        # 選択区間表示
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("選択区間:"))
        self.sel_start_lbl = QLabel("0.00 秒")
        self.sel_end_lbl   = QLabel("0.00 秒")
        self.sel_start_lbl.setStyleSheet("color:#f0c030; font-weight:bold;")
        self.sel_end_lbl.setStyleSheet(  "color:#f0c030; font-weight:bold;")
        sel_row.addWidget(self.sel_start_lbl)
        sel_row.addWidget(QLabel("〜"))
        sel_row.addWidget(self.sel_end_lbl)
        sel_row.addSpacing(20)

        self.refine_start = QDoubleSpinBox()
        self.refine_start.setRange(0, 9999); self.refine_start.setDecimals(2)
        self.refine_start.setSuffix(" 秒"); self.refine_start.setFixedWidth(110)
        self.refine_start.valueChanged.connect(self._spinbox_to_sel)
        self.refine_end = QDoubleSpinBox()
        self.refine_end.setRange(0, 9999); self.refine_end.setDecimals(2)
        self.refine_end.setSuffix(" 秒"); self.refine_end.setFixedWidth(110)
        self.refine_end.valueChanged.connect(self._spinbox_to_sel)
        sel_row.addWidget(QLabel("開始:")); sel_row.addWidget(self.refine_start)
        sel_row.addWidget(QLabel("終了:")); sel_row.addWidget(self.refine_end)
        sel_row.addStretch()
        vl.addLayout(sel_row)

        # 再生コントロール（ステレオ対応）
        pg = QGroupBox("再生コントロール（ステレオ）")
        pl = QHBoxLayout(pg); pl.setSpacing(8)
        self.pb_mix_range = QPushButton("▶  Mix（選択範囲）")
        self.pb_mix_full  = QPushButton("▶  Mix（全体）")
        self.pb_before    = QPushButton("▶  補正前 Voice")
        self.pb_after     = QPushButton("▶  補正後 Voice")
        self.pb_stop      = QPushButton("⏹  停止")
        self.pb_before.setEnabled(False)
        self.pb_after.setEnabled(False)
        for b, fn in [
            (self.pb_mix_range,  lambda: self._play('mix',    scoped=True)),
            (self.pb_mix_full,   lambda: self._play('mix',    scoped=False)),
            (self.pb_before,     lambda: self._play('before', scoped=True)),
            (self.pb_after,      lambda: self._play('after',  scoped=True)),
            (self.pb_stop,       self._stop_play),
        ]:
            b.setMinimumHeight(30); pl.addWidget(b); b.clicked.connect(fn)
        pl.addStretch()
        vl.addWidget(pg)

        # 補正オプション
        og = QGroupBox("補正オプション（STFT インパルス応答推定 — L/R 独立処理）")
        ol = QVBoxLayout(og)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("フレーム長:"))
        self.fir_taps = QSpinBox()
        self.fir_taps.setRange(256, 65536); self.fir_taps.setValue(4096)
        self.fir_taps.setSingleStep(1024); self.fir_taps.setFixedWidth(90)
        row1.addWidget(self.fir_taps); row1.addWidget(QLabel("samples"))
        self._fl_ms_lbl = QLabel("(93ms @ 44.1kHz)")
        self._fl_ms_lbl.setStyleSheet("color:#607888; font-size:10px;")
        row1.addWidget(self._fl_ms_lbl)
        self.fir_taps.valueChanged.connect(
            lambda v: self._fl_ms_lbl.setText(
                f"({v/self.sr_spin.value()*1000:.1f}ms @ {self.sr_spin.value()}Hz)"))
        row1.addStretch()
        ol.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("フレーム集約方法:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(["median（外れ値に強い・推奨）",
                                  "mean（全フレーム平均）",
                                  "best（最高相関フレーム1本）"])
        self.agg_combo.setFixedWidth(260)
        row2.addWidget(self.agg_combo); row2.addStretch()
        ol.addLayout(row2)
        vl.addWidget(og)

        # アクションボタン
        bl = QHBoxLayout()
        self.refine_btn = QPushButton("🔬  精密補正を適用（L/R 独立）")
        self.refine_btn.setObjectName("refine_btn"); self.refine_btn.setMinimumHeight(34)
        self.refine_btn.clicked.connect(self._apply_refine)
        self.revert_btn = QPushButton("↩  元の結果に戻す")
        self.revert_btn.setObjectName("revert_btn"); self.revert_btn.setMinimumHeight(34)
        self.revert_btn.clicked.connect(self._revert_refine)
        bl.addWidget(self.refine_btn); bl.addWidget(self.revert_btn); bl.addStretch()
        vl.addLayout(bl)

        self.refine_log = QTextEdit()
        self.refine_log.setReadOnly(True)
        self.refine_log.setFont(QFont("Monospace", 10))
        self.refine_log.setMaximumHeight(130)
        vl.addWidget(self.refine_log)

        # 再生ヘッド更新タイマー
        self._play_timer = QTimer()
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._update_play_head)

        return w

    # ── 波形描画
    def _draw_refine_waveform(self):
        if not self.result:
            return
        res = self.result
        sr  = res['sr']
        mix_l  = res['mix_st'][:, 0]
        voc_l  = res['voice_st'][:, 0]

        self.ref_fig.clear()
        self.ref_fig.set_facecolor('#080e14')
        gs2 = gridspec.GridSpec(2, 1, figure=self.ref_fig,
                                hspace=0.15, left=0.05, right=0.99,
                                top=0.93, bottom=0.12)
        C = dict(mix='#3ab4e0', voice='#3acf70',
                 bg='#080e14', grid='#12263a', txt='#607888')

        def sax2(ax, title):
            ax.set_facecolor(C['bg'])
            ax.set_title(title, color='#7eb8d4', fontsize=9, pad=2)
            ax.tick_params(colors=C['txt'], labelsize=8)
            for sp in ax.spines.values(): sp.set_color('#1e3a52')
            ax.grid(True, color=C['grid'], lw=0.4, ls='--', alpha=0.6)

        def wplot2(ax, y, sr, color, N=12000):
            t = np.linspace(0, len(y)/sr, len(y))
            s = max(1, len(y)//N)
            ax.plot(t[::s], y[::s], color=color, lw=0.6, alpha=0.9)

        self._ref_ax_mix = self.ref_fig.add_subplot(gs2[0])
        wplot2(self._ref_ax_mix, mix_l, sr, C['mix'])
        sax2(self._ref_ax_mix, 'Mix L — ドラッグで音楽のみ区間を選択')
        self._ref_ax_mix.set_xlim(0, len(mix_l)/sr)

        self._ref_ax_voc = self.ref_fig.add_subplot(gs2[1], sharex=self._ref_ax_mix)
        wplot2(self._ref_ax_voc, voc_l, sr, C['voice'])
        sax2(self._ref_ax_voc, 'Voice L — 現在の抽出結果')
        self._ref_ax_voc.set_xlabel('時間 (秒)', color=C['txt'], fontsize=8)

        self._play_line = None
        if self._sel_end > self._sel_start:
            self._draw_selection()

        self.ref_canvas.draw()

    def _draw_selection(self):
        s, e = self._sel_start, self._sel_end
        if not hasattr(self, '_ref_ax_mix'):
            return
        for ax in (self._ref_ax_mix, self._ref_ax_voc):
            ax.axvspan(s, e, alpha=0.18, color='#f0c030', zorder=2)
            ax.axvline(s, color='#f0c030', lw=1.2, ls='--', alpha=0.8, zorder=3)
            ax.axvline(e, color='#f0c030', lw=1.2, ls='--', alpha=0.8, zorder=3)
        self.ref_canvas.draw_idle()

    def _clear_and_redraw_selection(self):
        self._draw_refine_waveform()

    # ── マウスイベント
    def _refine_mouse_press(self, event):
        if event.inaxes not in (getattr(self, '_ref_ax_mix', None),
                                 getattr(self, '_ref_ax_voc', None)):
            return
        if event.button != 1:
            return
        self._drag_x0 = event.xdata
        self._sel_start = self._sel_end = event.xdata

    def _refine_mouse_move(self, event):
        if self._drag_x0 is None or event.xdata is None:
            return
        x = event.xdata
        self._sel_start = min(self._drag_x0, x)
        self._sel_end   = max(self._drag_x0, x)
        self.ref_canvas.draw_idle()

    def _refine_mouse_release(self, event):
        if self._drag_x0 is None:
            return
        if event.xdata is not None:
            self._sel_start = min(self._drag_x0, event.xdata)
            self._sel_end   = max(self._drag_x0, event.xdata)
        self._drag_x0 = None

        dur = len(self.result['mix_st']) / self.result['sr'] if self.result else 9999
        self._sel_start = max(0.0, min(self._sel_start, dur))
        self._sel_end   = max(0.0, min(self._sel_end,   dur))

        self.refine_start.blockSignals(True)
        self.refine_end.blockSignals(True)
        self.refine_start.setValue(self._sel_start)
        self.refine_end.setValue(self._sel_end)
        self.refine_start.blockSignals(False)
        self.refine_end.blockSignals(False)

        self.sel_start_lbl.setText(f"{self._sel_start:.2f} 秒")
        self.sel_end_lbl.setText(  f"{self._sel_end:.2f} 秒")
        self._clear_and_redraw_selection()

    def _spinbox_to_sel(self):
        self._sel_start = self.refine_start.value()
        self._sel_end   = self.refine_end.value()
        self.sel_start_lbl.setText(f"{self._sel_start:.2f} 秒")
        self.sel_end_lbl.setText(  f"{self._sel_end:.2f} 秒")
        self._clear_and_redraw_selection()

    # ── 再生（ステレオ対応）
    def _play(self, track: str, scoped: bool):
        self._stop_play()
        if not self.result:
            return
        sr = self.result['sr']

        if track == 'mix':
            data_st = self.result['mix_st']
        elif track == 'before':
            data_st = self._voice_before_st if self._voice_before_st is not None \
                      else self.result['voice_st']
        else:
            data_st = self._voice_after_st  if self._voice_after_st  is not None \
                      else self.result['voice_st']

        if scoped and self._sel_end > self._sel_start + 0.05:
            s = int(self._sel_start * sr)
            e = int(self._sel_end   * sr)
            data_st = data_st[s:e]
            self._play_offset_sec = self._sel_start
        else:
            self._play_offset_sec = 0.0

        # ステレオ float32
        play_data = data_st.astype(np.float32).copy()
        peak = np.max(np.abs(play_data))
        if peak > 0.98:
            play_data = play_data / peak * 0.98

        self._play_data   = play_data
        self._play_sr     = sr
        self._play_pos    = 0
        self._play_stream = sd.OutputStream(
            samplerate=sr, channels=2, dtype='float32',
            callback=self._audio_callback,
            finished_callback=self._play_finished)
        self._play_stream.start()
        self._play_timer.start()

    def _audio_callback(self, outdata, frames, time_info, status):
        remaining = len(self._play_data) - self._play_pos
        if remaining <= 0:
            outdata[:] = 0
            raise sd.CallbackStop()
        n = min(frames, remaining)
        outdata[:n, :] = self._play_data[self._play_pos : self._play_pos + n]
        if n < frames:
            outdata[n:] = 0
        self._play_pos += n

    def _play_finished(self):
        self._play_timer.stop()
        if hasattr(self, '_play_line') and self._play_line and \
                hasattr(self, '_ref_ax_mix'):
            try:
                self._play_line.remove()
            except Exception:
                pass
            self._play_line = None
            self.ref_canvas.draw_idle()

    def _stop_play(self):
        if hasattr(self, '_play_stream') and self._play_stream:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception:
                pass
            self._play_stream = None
        self._play_timer.stop()

    def _update_play_head(self):
        if not self.result or not hasattr(self, '_ref_ax_mix'):
            return
        pos_sec = self._play_offset_sec + self._play_pos / self._play_sr
        if hasattr(self, '_play_line') and self._play_line:
            try: self._play_line.remove()
            except Exception: pass
        self._play_line = self._ref_ax_mix.axvline(
            pos_sec, color='#ff4444', lw=1.5, alpha=0.9, zorder=5)
        if hasattr(self, '_ref_ax_voc'):
            self._ref_ax_voc.axvline(
                pos_sec, color='#ff4444', lw=1.5, alpha=0.9, zorder=5)
        self.ref_canvas.draw_idle()

    # ── 精密補正適用
    def _apply_refine(self):
        if not self.result:
            return
        res = self.result
        sr  = res['sr']
        s   = self._sel_start
        e   = self._sel_end
        if e - s < 1.0:
            QMessageBox.warning(self, "区間エラー",
                                "1秒以上の区間をドラッグで選択してください")
            return

        self.refine_log.clear()
        self.refine_log.append("━━━ 精密補正（ステレオ L/R 独立） 開始 ━━━")
        self.refine_btn.setEnabled(False)

        agg_map = {0: 'median', 1: 'mean', 2: 'best'}
        agg = agg_map.get(self.agg_combo.currentIndex(), 'median')

        try:
            self._voice_before_st = res['voice_st'].copy()

            voice_ref_st, music_ref_st, info = refine_extraction_stereo(
                res['mix_st'],
                res['music_aligned_st'],
                sr, s, e,
                frame_len=self.fir_taps.value(),
                aggregate=agg,
                log_fn=self.refine_log.append)

            self._voice_after_st = voice_ref_st

            # result を更新（mix_st は変えない）
            res['voice_st']         = voice_ref_st
            res['music_aligned_st'] = music_ref_st

            self.pb_before.setEnabled(True)
            self.pb_after.setEnabled(True)

            cb = info['corr_before']
            ca = info['corr_after']
            self.refine_log.append("━━━ 完了 ━━━")
            self.refine_log.append(
                f"平均相関: {cb:.4f} → {ca:.4f}  "
                f"({'改善' if ca > cb else '悪化 ← 設定変更推奨'})")
            self.refine_log.append(
                f"L: {info['corr_before_L']:.4f} → {info['corr_after_L']:.4f}  "
                f"R: {info['corr_before_R']:.4f} → {info['corr_after_R']:.4f}")
            self.refine_log.append("▶ 補正前 / 補正後 Voice ボタンで聴き比べできます")

            self._draw_refine_waveform()
            self._draw(res)
            self.statusBar().showMessage(
                f"精密補正適用済み  L: {info['corr_before_L']:.4f}→{info['corr_after_L']:.4f}  "
                f"R: {info['corr_before_R']:.4f}→{info['corr_after_R']:.4f}")

        except Exception:
            import traceback
            self.refine_log.append("[ERROR]\n" + traceback.format_exc())
        finally:
            self.refine_btn.setEnabled(True)

    def _revert_refine(self):
        if not hasattr(self, 'result_original') or not self.result_original:
            return
        self._stop_play()
        for k, v in self.result_original.items():
            self.result[k] = v.copy() if isinstance(v, np.ndarray) else v
        self._voice_before_st = None
        self._voice_after_st  = None
        self.pb_before.setEnabled(False)
        self.pb_after.setEnabled(False)
        self.refine_log.append("↩ 元の結果に戻しました")
        self._draw_refine_waveform()
        self._draw(self.result)
        self.statusBar().showMessage("標準解析結果に戻しました")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Extractor")
    win = VoiceExtractorWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
