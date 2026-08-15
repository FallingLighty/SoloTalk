#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoloTalk 1.8 — 单机版听说模考编辑器
依赖安装：pip install PyQt5 pyttsx3 sounddevice vosk python-vlc numpy scipy fastembed edge-tts
（edge-tts 为高质量神经语音，需联网；未安装或断网时自动回退系统 SAPI5 语音）
所有控制台输出（print / 异常栈 / 崩溃栈）都会同步写入 _solo_diag.log（打包后写用户主目录），方便无控制台环境排查。

Vosk 英语模型请下载并解压到本脚本同级目录，或修改 MODEL_PATH 变量。
bge-small-en-v1.5 模型：
  方法A - 手动下载：
      $env:HF_ENDPOINT="https://hf-mirror.com"
hf download Xenova/bge-small-en-v1.5 --local-dir "脚本同级目录/bge-small-en-v1.5"
      额外步骤：将 onnx/model.onnx 复制一份为 model_optimized.onnx 放到根目录
               （fastembed 需要此文件名，程序启动时会自动尝试复制）
    方法B - 自动下载（不知道，没试过，可能不行）：
      pip install fastembed 后首次运行会自动从 HuggingFace 下载。

如不使用此功能（或模型目录不存在），Part B 将回退到 Jaccard 相似度，Part C 回退到关键词匹配，不影响基本使用。
"""

import sys
import os
import json
import zipfile
import tempfile
import wave
import datetime
import time
import threading
import ctypes
from pathlib import Path

# 确保 VLC 可被找到（双击运行时工作目录可能不在脚本同级）
# 打包后支持 one-folder 与 one-file 两种模式：
#  - one-folder：exe 与 VLC/ 并排，取 exe 所在目录
#  - one-file：PyInstaller 解压到临时 _MEIPASS，VLC/ 被一并发到该目录
if getattr(sys, "frozen", False):
    _BASE_DIR = os.path.dirname(sys.executable)        # one-folder
    if hasattr(sys, "_MEIPASS"):                        # one-file
        _BASE_DIR = sys._MEIPASS
    # PyInstaller 6.x 在部分环境下 sys.executable 可能指向 _internal 内部，
    # 导致 _BASE_DIR 多一层嵌套。检测并修正。
    if os.path.basename(_BASE_DIR) == "_internal":
        _BASE_DIR = os.path.dirname(_BASE_DIR)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_DIR = _BASE_DIR
_VLC_DIR = os.path.join(_BASE_DIR, "VLC")
_INTERNAL_DIR = os.path.join(_BASE_DIR, "_internal")  # PyInstaller one-folder 数据目录
# PyInstaller 6.x 可能把数据文件放到 _internal/ 下，而不是 exe 同级
if os.path.isdir(_INTERNAL_DIR) and not os.path.isdir(_VLC_DIR):
    _VLC_DIR = os.path.join(_INTERNAL_DIR, "VLC")

# ===== DLL 搜索路径（必须在任何第三方 import 之前注册）=====
# ===== 模块级 DLL 目录 cookie（防止 os.add_dll_directory 被 GC 移除）=====
_dll_dir_cookies = []

if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.AddDllDirectory(_VLC_DIR)
    except Exception:
        pass
    try:
        ctypes.windll.kernel32.AddDllDirectory(_SCRIPT_DIR)
    except Exception:
        pass
    if getattr(sys, "frozen", False) and os.path.isdir(_INTERNAL_DIR):
        try:
            ctypes.windll.kernel32.AddDllDirectory(_INTERNAL_DIR)
        except Exception:
            pass
        # 注意：不将 _INTERNAL_DIR 加入 os.add_dll_directory！
        # _internal/ 包含大量其他包的 DLL（PyQt5、numpy、scipy），可能与 onnxruntime
        # 产生 CRT 版本冲突。只添加 onnxruntime 自己的目录。
        _ort_dll_dir = os.path.join(_INTERNAL_DIR, "onnxruntime", "capi")
        if os.path.isdir(_ort_dll_dir):
            try:
                ctypes.windll.kernel32.AddDllDirectory(_ort_dll_dir)
            except Exception:
                pass
            if hasattr(os, "add_dll_directory"):
                try:
                    _dll_dir_cookies.append(os.add_dll_directory(_ort_dll_dir))
                except Exception as e:
                    pass

def _pick_log_path():
    """决定 _solo_diag.log 的位置：
    - 源码运行：脚本同级目录
    - 打包运行：exe 同级目录优先（用户期望日志就在 exe 旁边）；
      若 exe 所在目录不可写（如 Program Files），回退到用户主目录"""
    if not getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_solo_diag.log")
    exe_log = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "_solo_diag.log")
    try:
        with open(exe_log, "a", encoding="utf-8"):
            pass
        return exe_log
    except Exception:
        return os.path.join(os.path.expanduser("~"), "_solo_diag.log")


# 日志路径：开发时脚本同级；打包后 exe 同级（不可写则回退用户主目录）
_diag_project_log = _pick_log_path()

# ---------- 控制台输出全量落盘：所有 print() 同时写入控制台和 _solo_diag.log ----------
# windowed 模式（无控制台，sys.stdout 为 None）下只写日志，控制台部分自动跳过。
class _ConsoleTee:
    """替换 sys.stdout/stderr：保留原控制台输出，同时把每一行写入日志文件。"""

    def __init__(self, stream, log_path, is_error=False):
        self._stream = stream          # 原始流，windowed 模式下为 None
        self._log_path = log_path
        self._is_error = is_error
        self._buf = []                 # 行缓冲：攒完整行再写日志，避免半行/时间戳错位
        self._lock = threading.Lock()

    def write(self, data):
        if not data:
            return
        try:
            if self._stream is not None:
                self._stream.write(data)
                self._stream.flush()
        except Exception:
            pass
        with self._lock:
            self._buf.append(data)
            joined = "".join(self._buf)
            parts = joined.split("\n")
            self._buf = [parts[-1]] if parts[-1] else []
            for line in parts[:-1]:
                self._flush_line(line)

    def flush(self):
        try:
            if self._stream is not None:
                self._stream.flush()
        except Exception:
            pass
        with self._lock:
            tail = "".join(self._buf)
            if tail:
                self._flush_line(tail)
            self._buf = []

    def _flush_line(self, line):
        if not line:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = "STDERR" if self._is_error else "OUT"
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}][{tag}] {line}\n")
        except Exception:
            pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def fileno(self):
        return -1

    def writable(self):
        return True


def _install_console_tee():
    sys.stdout = _ConsoleTee(sys.stdout, _diag_project_log)
    sys.stderr = _ConsoleTee(sys.stderr, _diag_project_log, is_error=True)


_install_console_tee()

# ---------- 全局异常钩子：把未捕获异常写入日志 ----------
# windowed 模式（console=False）下无控制台，异常直接静默闪退。
# 自定义 sys.excepthook 后，Python 异常（含 Qt 槽函数里的）都会落盘到 _solo_diag.log，
# 便于定位崩溃根因。
def _log_exception(exc_type, exc_value, exc_tb):
    try:
        import traceback
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    except Exception:
        tb_text = f"{exc_type}: {exc_value}"
    msg = f"\n[EXCEPTION] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{tb_text}"
    for _p in ({os.path.join(_SCRIPT_DIR, '_solo_diag.log'), _diag_project_log}
               if getattr(sys, "frozen", False) else
               {os.path.join(_SCRIPT_DIR, '_solo_diag.log')}):
        try:
            with open(_p, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass
    # SystemExit 正常退出，不额外处理；其余异常走默认处理（windowed 下无输出，无害）
    if exc_type is not SystemExit:
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

sys.excepthook = _log_exception

# faulthandler：捕获 C 层崩溃（如 VLC 原生库段错误/访问冲突）。
# 崩溃时把各线程的 Python 调用栈 dump 到日志文件，Python 异常钩子抓不到的
# 底层崩溃靠它留痕。
try:
    import faulthandler
    _fh_path = (os.path.join(os.path.expanduser("~"), "_solo_diag.log")
                if getattr(sys, "frozen", False)
                else os.path.join(_SCRIPT_DIR, "_solo_diag.log"))
    _fh_file = open(_fh_path, "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_fh_file, all_threads=True)
except Exception:
    pass
print(f"=== SoloTalk startup ===")
print(f"log_path={_diag_project_log}")
print(f"frozen={getattr(sys, 'frozen', False)}")
print(f"executable={sys.executable}")
print(f"_BASE_DIR={_BASE_DIR}")
print(f"_INTERNAL_DIR={_INTERNAL_DIR}")
print(f"_INTERNAL_DIR exists={os.path.isdir(_INTERNAL_DIR)}")
if os.path.isdir(_INTERNAL_DIR):
    print(f"_INTERNAL_DIR children (first 30): {sorted(os.listdir(_INTERNAL_DIR))[:30]}")
# onnxruntime DLL 预加载由 pyi_rth_aaaa_preload_ort.py (runtime hook) 完成
# — 必须在任何 Python import 之前执行，否则 CRT 版本冲突导致 error 1114


def _resolve_model_dir(name):
    """查找模型/资源目录，按优先级返回第一个存在的路径。
    one-folder 打包时 PyInstaller 把 datas 放到 _internal/，源码运行时在脚本同级。
    优先级：① _BASE_DIR/name（exe 同级）② _INTERNAL_DIR/name（_internal 子目录）③ cwd/name"""
    for base in (_BASE_DIR, _INTERNAL_DIR, os.getcwd()):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(_BASE_DIR, name)  # 都不存在返回默认路径，让调用方自行处理


def _resolve_resource_file(name):
    """查找资源文件（图标等），与 _resolve_model_dir 同优先级。找不到返回 None。"""
    for base in (_BASE_DIR, _INTERNAL_DIR, os.getcwd()):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None

# 切换工作目录到脚本所在目录
if os.getcwd() != _SCRIPT_DIR:
    try:
        os.chdir(_SCRIPT_DIR)
    except Exception:
        pass

if sys.platform == "win32":
    # 设置插件路径环境变量，让 libvlc 能找到 plugins 目录
    if os.path.isdir(_VLC_DIR):
        # exe 同级存在 VLC/（打包携带或手动复制）
        os.environ["VLC_PLUGIN_PATH"] = _VLC_DIR
        _vlc_dll = os.path.join(_VLC_DIR, "libvlc.dll")
        if os.path.isfile(_vlc_dll):
            os.environ["PYTHON_VLC_LIB_PATH"] = _vlc_dll
    else:
        # 回退：从注册表读取 VLC 安装路径，显式设置插件路径
        # python-vlc 内部通过 ctypes 找 DLL 时会查注册表，但找到 DLL 后
        # 不一定能正确定位 plugins 目录（尤其是 portable 版 VLC）。
        # 手动读出路径并设置 VLC_PLUGIN_PATH，确保 vlc.Instance() 不失败。
        _reg_vlc = None
        try:
            import winreg
            for _hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for _sub in (r"SOFTWARE\VideoLAN\VLC",
                             r"SOFTWARE\WOW6432Node\VideoLAN\VLC"):
                    try:
                        _key = winreg.OpenKey(_hive, _sub)
                        # VLC 注册表存的是 "InstallDir"，不是 (Default) 值
                        _reg_vlc, _ = winreg.QueryValueEx(_key, "InstallDir")
                        winreg.CloseKey(_key)
                        if _reg_vlc and os.path.isdir(_reg_vlc):
                            break
                    except OSError:
                        continue
                if _reg_vlc:
                    break
        except Exception:
            pass
        if _reg_vlc and os.path.isdir(_reg_vlc):
            _vlc_dll = os.path.join(_reg_vlc, "libvlc.dll")
            if os.path.isfile(_vlc_dll):
                os.environ["VLC_PLUGIN_PATH"] = _reg_vlc
                os.environ["PYTHON_VLC_LIB_PATH"] = _vlc_dll

import pyttsx3
import sounddevice as sd
import numpy as np
from scipy.io import wavfile as wav_write
import vlc

# ---------- BGE 嵌入模型（Part B/C 语义相似度，可选） ----------
_BGE_MODEL_DIR = _resolve_model_dir("bge-small-en-v1.5")
_bge_embedder = None  # 延迟加载，首次使用时初始化

try:
    import onnxruntime
    print(f"onnxruntime import OK, version={onnxruntime.__version__}")
except Exception as e:
    print(f"onnxruntime import FAILED: {e}")
    import traceback as _tb
    print(_tb.format_exc())

try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
    print("fastembed import OK")
except ImportError as e:
    _FASTEMBED_AVAILABLE = False
    print(f"fastembed import FAILED: {e}")


def _get_bge_model():
    """获取 BGE 嵌入模型实例（懒加载单例）。
    模型目录存在且 fastembed 已安装时返回 TextEmbedding 对象，
    否则返回 None（调用方应回退到原始评估方法）。
    完全离线运行：不连接 HuggingFace 或任何外部服务。"""
    global _bge_embedder
    if _bge_embedder is not None:
        if isinstance(_bge_embedder, TextEmbedding):
            print("[BGE] 返回已缓存的模型实例")
            return _bge_embedder
        else:
            print("[BGE] 已标记为不可用，跳过加载")
            return None

    print("[BGE] === 开始加载 BGE 嵌入模型 ===")
    print(f"[BGE] 解析到的模型目录: {_BGE_MODEL_DIR}")
    print(f"[BGE] 路径解析逻辑: _BASE_DIR={_BASE_DIR}, _INTERNAL_DIR={_INTERNAL_DIR}")
    print(f"[BGE] 目录存在: {os.path.isdir(_BGE_MODEL_DIR)}")
    print(f"[BGE] fastembed 可用: {_FASTEMBED_AVAILABLE}")

    if not _FASTEMBED_AVAILABLE:
        print("[BGE] ❌ fastembed 未安装，回退简单匹配")
        _bge_embedder = []
        return None
    if not os.path.isdir(_BGE_MODEL_DIR):
        print(f"[BGE] ❌ 模型目录不存在: {_BGE_MODEL_DIR}")
        # 列出可能的位置帮助排查
        for d in (_BASE_DIR, _INTERNAL_DIR, os.getcwd()):
            p = os.path.join(d, "bge-small-en-v1.5")
            print(f"[BGE]   检查 {d}/bge-small-en-v1.5 → {'√ 存在' if os.path.isdir(p) else '✗ 不存在'}")
        _bge_embedder = []
        return None

    # 检查必需的 ONNX 模型文件是否存在
    onnx_file = os.path.join(_BGE_MODEL_DIR, "model_optimized.onnx")
    print(f"[BGE] model_optimized.onnx: {'√' if os.path.isfile(onnx_file) else '✗ 不存在'}")
    if not os.path.isfile(onnx_file):
        # 尝试从 onnx/ 子目录复制
        src = os.path.join(_BGE_MODEL_DIR, "onnx", "model.onnx")
        if os.path.isfile(src):
            print("[BGE] 尝试从 onnx/model.onnx 复制...")
            try:
                import shutil
                shutil.copy2(src, onnx_file)
                print("[BGE] 复制成功")
            except Exception as e:
                print(f"[BGE] 复制失败: {e}")
        if not os.path.isfile(onnx_file):
            print(f"[BGE] ❌ ONNX 文件缺失: {onnx_file}")
            _bge_embedder = []
            return None

    # 强制离线模式
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cache_dir = os.path.dirname(_BGE_MODEL_DIR)
    print(f"[BGE] fastembed cache_dir: {cache_dir}")
    print(f"[BGE] fastembed 将在此目录下查找 bge-small-en-v1.5/ 子目录")

    try:
        _bge_embedder = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        print("[BGE] TextEmbedding 创建成功，预热中...")
        list(_bge_embedder.embed(["warmup"]))
        print("[BGE] ✅ BGE 嵌入模型加载成功！Part B/C 使用语义相似度。")
        return _bge_embedder
    except Exception as e:
        print(f"[BGE] ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        _bge_embedder = []
        return None


def _embedding_cosine_similarity(text1, text2, embedder):
    """用 BGE 嵌入计算两句英文的余弦相似度。返回 0.0~1.0 的浮点数。"""
    if not text1.strip() or not text2.strip():
        return 0.0
    embs = list(embedder.embed([text1, text2]))
    a, b = embs[0], embs[1]
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(dot / (norm_a * norm_b))
    # BGE 相似度范围约 [-1, 1]，归一化到 [0, 1]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ------------------ Windows 键屏蔽（模考全屏锁定用）------------------
# Python 3.14 的 ctypes.wintypes 不再保证暴露 LRESULT/WPARAM/HHOOK 等别名，
# 这里用与平台位宽一致的基础类型自行定义，避免 AttributeError/OverflowError。
def _wintype(name, fallback):
    return getattr(ctypes.wintypes, name, fallback)

_LRESULT = _wintype("LRESULT", ctypes.c_ssize_t)
_WPARAM = _wintype("WPARAM", ctypes.c_size_t)
_LPARAM = _wintype("LPARAM", ctypes.c_ssize_t)
_HHOOK = _wintype("HHOOK", ctypes.c_void_p)
_HINSTANCE = _wintype("HINSTANCE", ctypes.c_void_p)
_DWORD = _wintype("DWORD", ctypes.c_uint32)
_BOOL = _wintype("BOOL", ctypes.c_int)
_ULONG_PTR = ctypes.c_size_t

class WindowsKeyBlocker:
    """通过底层键盘钩子屏蔽系统级快捷键，仅用于模考模式锁定：
    - 左右 Windows 键（含 Win+L）
    - Alt 键（含 Alt+Tab / Alt+Esc / Alt+F4 / 菜单激活）
    - Tab 键（防止焦点跳出与 Alt+Tab 切换窗口）
    注意：钩子在安装它的线程（主线程）的消息循环中生效，屏蔽期间均不可触发，
    考试结束/退出时必须 uninstall。"""
    _VK_LWIN = 0x5B
    _VK_RWIN = 0x5C
    _VK_MENU = 0x12   # Alt
    _VK_TAB = 0x09    # Tab
    _WH_KEYBOARD_LL = 13
    _BLOCK_KEYS = (_VK_LWIN, _VK_RWIN, _VK_MENU, _VK_TAB)

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("vkCode", _DWORD),
            ("scanCode", _DWORD),
            ("flags", _DWORD),
            ("time", _DWORD),
            ("dwExtraInfo", ctypes.POINTER(_ULONG_PTR)),
        ]

    def __init__(self):
        self._user32 = ctypes.windll.user32
        # 显式声明 API 返回类型，避免 64 位下默认 c_int 截断导致 OverflowError
        self._user32.SetWindowsHookExW.restype = _HHOOK
        self._user32.UnhookWindowsHookEx.restype = _BOOL
        self._user32.UnhookWindowsHookEx.argtypes = [_HHOOK]
        self._user32.CallNextHookEx.restype = _LRESULT
        self._user32.CallNextHookEx.argtypes = [_HHOOK, ctypes.c_int, _WPARAM, _LPARAM]
        self._hook = None
        self._proc = None

    def _callback(self, nCode, wParam, lParam):
        try:
            if nCode == 0:
                kbd = ctypes.cast(lParam, ctypes.POINTER(self.KBDLLHOOKSTRUCT)).contents
                if kbd.vkCode in self._BLOCK_KEYS:
                    return 1  # 拦截，阻止系统响应
        except Exception:
            pass
        # hhk 参数在现代 Windows 上被忽略，传 None 避免 self._hook int 溢出
        return self._user32.CallNextHookEx(None, nCode, wParam, lParam)

    def install(self):
        if self._hook:
            return
        # 64 位下回调返回类型必须是 LRESULT（c_ssize_t），不能是 c_int
        self._proc = ctypes.WINFUNCTYPE(
            _LRESULT, ctypes.c_int, _WPARAM, _LPARAM
        )(self._callback)
        self._hook = self._user32.SetWindowsHookExW(self._WH_KEYBOARD_LL, self._proc, None, 0)
        if not self._hook:
            raise ctypes.WinError()

    def uninstall(self):
        if self._hook:
            self._user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self._proc = None

# ------------------ 全局配置 ------------------
MODEL_PATH = _resolve_model_dir("vosk-model-small-en-us-0.15")
RECORDINGS_DIR = "recordings"
HISTORY_DIR = "history"
SAMPLE_RATE = 16000
TTS_TIMEOUT = 30

# ------------------ TTS 工具（每次创建独立引擎）------------------
# 优先使用 edge-tts（微软神经网络语音，自然、有停顿、听得清）；
# 不可用（未安装 / 无网络）时回退 Windows 自带 SAPI5（pyttsx3）。
try:
    import edge_tts as _edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

TTS_EDGE_VOICE = "en-US-JennyNeural"   # 英文女声，清晰自然
TTS_EDGE_DIR = os.path.join(tempfile.gettempdir(), "solotalk_tts")  # 生成音频的缓存目录


def tts_generate_audio(text, out_path):
    """用 edge-tts 把文本合成 mp3 到 out_path，成功返回 True。需联网。"""
    if not EDGE_TTS_AVAILABLE or not (text and text.strip()):
        return False
    try:
        import asyncio
        async def _gen():
            com = _edge_tts.Communicate(text.strip(), voice=TTS_EDGE_VOICE)
            await asyncio.wait_for(com.save(out_path), timeout=TTS_TIMEOUT)
        asyncio.run(_gen())
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        print(f"edge-tts 失败，回退系统语音: {e}")
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return False


def _tts_split_sentences(text):
    """按句子边界切分，逐句合成/朗读，避免整段含糊、无停顿。"""
    import re
    parts = re.split(r'(?<=[.!?;:])\s+', text.strip())
    return [p for p in parts if p.strip()]


def tts_speak_blocking(text, stop_event=None):
    """回退方案：Windows SAPI5（pyttsx3），优先英文语音，逐句朗读并加句间停顿。"""
    engine = None
    try:
        engine = pyttsx3.init()
        # 优先选英文语音（如果有）
        try:
            voices = engine.getProperty('voices')
            for v in voices:
                vid = str(getattr(v, 'id', '') or '')
                if 'en' in vid.lower():
                    engine.setProperty('voice', v.id)
                    break
        except Exception:
            pass
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 1.0)
        if stop_event:
            def timeout_monitor():
                if stop_event.wait(TTS_TIMEOUT):
                    try:
                        engine.stop()
                    except:
                        pass
            threading.Thread(target=timeout_monitor, daemon=True).start()
        for sent in (_tts_split_sentences(text) or [text]):
            if stop_event and stop_event.is_set():
                break
            engine.say(sent)
            engine.runAndWait()
            if stop_event and stop_event.is_set():
                break
            time.sleep(0.18)  # 句间停顿，让朗读更自然
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        if engine:
            try:
                engine.stop()
                del engine
            except:
                pass

# ------------------ 工具函数 ------------------
def levenshtein_wer(ref, hyp):
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dp = [[0] * (len(hyp_words) + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    wer = dp[n][len(hyp_words)] / n
    return wer, 1.0 - wer

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def _cosine01(a, b):
    """两向量的余弦相似度，归一化到 [0, 1]（BGE 原始值约 [-1,1]，映射到 (x+1)/2）。"""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))


def _partC_point_scores(recognized, key_points_str, embedder):
    """Part C 逐条要点评分（来自 _partC_semantic.md 算法）。

    对每一条要点 p：
      1. 词袋命中率 wb = 要点中去重词的命中数 / 总去重词数（识别文本里出现即命中）
      2. 语义相似度 sem = 要点向量 vs 识别文本各滑动窗口（40 词窗、步长 20）的 cosine 最大值
      3. 本条得分 = max(wb, sem)
    聚合：
      匹配度 = 各条得分的平均值
      覆盖   = 得分 >= 0.80 的要点条数

    返回 (每条得分列表, 是否使用了语义相似度)。
    无 embedder 时 sem 取 0，退化为纯词袋命中率。
    """
    points = [p.strip() for p in key_points_str.split(',') if p.strip()]
    if not points:
        return [], False
    if not recognized.strip():
        return [0.0] * len(points), False
    hyp_words = recognized.lower().split()
    # 识别文本的滑动窗口（40 词窗、步长 20）
    windows = []
    for i in range(0, max(1, len(hyp_words)), 20):
        win = ' '.join(hyp_words[i:i + 40])
        if win.strip():
            windows.append(win)
    if not windows:
        windows = [recognized.lower()]

    use_sem = embedder is not None
    win_embs = None
    point_embs = None
    if use_sem:
        try:
            win_embs = list(embedder.embed(windows))
            point_embs = list(embedder.embed(points))
        except Exception:
            use_sem = False

    scores = []
    for idx, p in enumerate(points):
        pw = [w for w in p.lower().split() if w]
        if not pw:
            scores.append(0.0)
            continue
        distinct = set(pw)
        hit = sum(1 for w in distinct if w in hyp_words)
        wb = hit / len(distinct)                 # 词袋命中率
        sem = 0.0
        if use_sem and win_embs is not None:
            pe = point_embs[idx]
            sims = [_cosine01(pe, we) for we in win_embs]
            sem = max(sims) if sims else 0.0      # 语义相似度（滑动窗口最大值）
        scores.append(max(wb, sem))              # 取较大值
    # 字数惩罚：防止只说 1~2 词就拿高分（阈值 20 词以上才满分）
    word_count = max(1, len(hyp_words))
    length_penalty = min(1.0, (word_count / 20.0) ** 2)
    if length_penalty < 1.0:
        scores = [s * length_penalty for s in scores]
    return scores, use_sem


def recognize_audio_file(filepath, model):
    if not model or not filepath or not isinstance(filepath, str) or not os.path.exists(filepath):
        return ""
    try:
        wf = wave.open(filepath, 'rb')
    except Exception:
        return ""
    try:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            return "[格式不符]"
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        result_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                result_text += res.get("text", "") + " "
        res = json.loads(rec.FinalResult())
        result_text += res.get("text", "")
        return result_text.strip()
    except Exception:
        return "[识别失败]"
    finally:
        try:
            wf.close()
        except Exception:
            pass


def _partB_slots_as_ordered(partB_raw):
    """把 recordings.partB 统一成 (Q路径列表, A路径列表)，兼容 dict / list 两种存储。
    - dict（当前格式）：键为 PartB_Q1…/PartB_A1…，按实际键计数；
    - list（早期格式）：固定前 3 个为三问、其后为五答（与旧 reevaluate 行为一致）。"""
    if isinstance(partB_raw, dict):
        n3 = len([k for k in partB_raw if k.startswith("PartB_Q")])
        n5 = len([k for k in partB_raw if k.startswith("PartB_A")])
        b3 = [partB_raw.get(f"PartB_Q{i+1}") for i in range(n3)]
        b5 = [partB_raw.get(f"PartB_A{i+1}") for i in range(n5)]
    elif isinstance(partB_raw, list):
        b3 = partB_raw[:3]
        b5 = partB_raw[3:]
    else:
        b3, b5 = [], []
    return b3, b5


def evaluate_recordings(recs, pkg, model, existing_eval=None, selection=None):
    """对一份录音（recordings 字典）做离线批改，返回 eval_result。
    统一了「考试结束实时批改」与「历史记录重新批改」两条路径，避免逻辑分叉。
    Part B 跳过/未录音的空槽位记 0.0 相似度，不抛异常。
    当 BGE 嵌入模型可用时，Part B/C 使用语义余弦相似度；否则回退到 Jaccard/关键词匹配。

    参数：
      existing_eval: 历史重新批改时传入的旧评估结果，作为基础 dict，
                     只有被选中的槽位会被重新计算，其余保留原值，避免无谓的 Vosk 识别开销。
      selection: dict，指定要重新批改的槽位，格式：
                 {'partA': bool,
                  'partB_three': {0: bool, 1: bool, ...},   # 按题目序号
                  'partB_five':  {0: bool, ...},
                  'partC': bool}
                 为 None 时表示「全部重新批改」（实时批改场景）。"""
    eval_result = dict(existing_eval) if existing_eval else {}
    _all = (selection is None)

    # 尝试加载 BGE 嵌入模型（懒加载，仅首次调用时初始化）
    bge = _get_bge_model()
    if bge:
        print("[BGE] 评分将使用 BGE 语义余弦相似度")
    else:
        print("[BGE] 评分将回退到 Jaccard/关键词匹配")

    # ── Part A ──
    if _all or (selection.get('partA')):
        if recs.get('partA'):
            hyp = recognize_audio_file(recs['partA'], model)
            wer, acc = levenshtein_wer(pkg.partA_hidden_text, hyp)
            eval_result['partA'] = {'recognized': hyp, 'wer': wer, 'accuracy': acc}
        else:
            eval_result['partA'] = {'recognized': '', 'wer': 1.0, 'accuracy': 0.0}
    elif 'partA' not in eval_result:
        eval_result['partA'] = {'recognized': '', 'wer': 1.0, 'accuracy': 0.0}

    b3, b5 = _partB_slots_as_ordered(recs.get('partB', {}))
    n3 = len(pkg.partB_three_questions)
    n5 = len(pkg.partB_five_answers)
    sel3 = selection.get('partB_three') if selection else None
    sel5 = selection.get('partB_five') if selection else None

    # ── Part B 三问 ──
    old3 = eval_result.get('partB_three', [])
    new3 = []
    for i, rec in enumerate(b3):
        if _all or (sel3 and sel3.get(i)):
            ref = pkg.partB_three_questions[i].get('hidden_answer', '') if i < n3 else ''
            hyp = recognize_audio_file(rec, model) if rec else ""
            if bge and rec and ref.strip() and hyp.strip():
                sim = _embedding_cosine_similarity(ref, hyp, bge)
            else:
                sim = jaccard_similarity(ref, hyp) if rec else 0.0
            new3.append({'recognized': hyp, 'similarity': sim})
        else:
            new3.append(old3[i] if i < len(old3) else {'recognized': '', 'similarity': 0.0})
    eval_result['partB_three'] = new3

    # ── Part B 五答 ──
    old5 = eval_result.get('partB_five', [])
    new5 = []
    for i, rec in enumerate(b5):
        if _all or (sel5 and sel5.get(i)):
            ref = pkg.partB_five_answers[i].get('hidden_answer', '') if i < n5 else ''
            hyp = recognize_audio_file(rec, model) if rec else ""
            if bge and rec and ref.strip() and hyp.strip():
                sim = _embedding_cosine_similarity(ref, hyp, bge)
            else:
                sim = jaccard_similarity(ref, hyp) if rec else 0.0
            new5.append({'recognized': hyp, 'similarity': sim})
        else:
            new5.append(old5[i] if i < len(old5) else {'recognized': '', 'similarity': 0.0})
    eval_result['partB_five'] = new5

    # ── Part C ──
    if _all or (selection.get('partC')):
        if recs.get('partC'):
            hyp = recognize_audio_file(recs['partC'], model)
            scores, use_sem = _partC_point_scores(hyp, pkg.partC_key_points, bge)
            total = len(scores)
            matched = sum(1 for s in scores if s >= 0.80)   # 得分 >= 0.80 的要点条数
            matching = (sum(scores) / total) if total else 0.0
            c_result = {
                'recognized': hyp,
                'coverage': f"{matched}/{total}",
                'semantic_sim': matching,        # 匹配度 = 各条得分平均（同时作为得分依据）
                'points': scores,
            }
            eval_result['partC'] = c_result
        else:
            eval_result['partC'] = {'recognized': '', 'coverage': '0/0', 'semantic_sim': 0.0}
    elif 'partC' not in eval_result:
        eval_result['partC'] = {'recognized': '', 'coverage': '0/0', 'semantic_sim': 0.0}

    return eval_result

# ------------------ 数据模型 ------------------
class SoloPackage:
    def __init__(self):
        self.meta = {
            "name": "Untitled",
            "version": "1.8",
            "created": datetime.datetime.now().isoformat(),
            "author": "",
            "anonymous": False
        }
        self.partA_video_path = None
        self.partA_hidden_text = ""
        self.partB_situation = ""
        self.partB_listening_text = ""
        self.partB_listening_audio = None   # 听力文本：可选音频文件（绝对路径）
        self.partB_three_questions = []
        self.partB_five_answers = []
        self.partC_summary = ""
        self.partC_keywords = ""
        self.partC_source_type = "tts"
        self.partC_tts_text = ""
        self.partC_audio_path = None
        self.partC_key_points = ""

    def to_dict(self):
        return {
            "meta": self.meta,
            "partA": {
                "video": os.path.basename(self.partA_video_path) if self.partA_video_path else None,
                "hidden_text": self.partA_hidden_text
            },
            "partB": {
                "situation": self.partB_situation,
                "listening_text": self.partB_listening_text,
                "listening_audio": os.path.basename(self.partB_listening_audio) if self.partB_listening_audio else None,
                "three_questions": [
                    {**q, "en_audio": os.path.basename(q["en_audio"]) if q.get("en_audio") else None}
                    for q in self.partB_three_questions
                ],
                "five_answers": [
                    {**q, "q_audio": os.path.basename(q["q_audio"]) if q.get("q_audio") else None}
                    for q in self.partB_five_answers
                ]
            },
            "partC": {
                "summary": self.partC_summary,
                "keywords": self.partC_keywords,
                "source_type": self.partC_source_type,
                "tts_text": self.partC_tts_text,
                "audio": os.path.basename(self.partC_audio_path) if self.partC_audio_path else None,
                "key_points": self.partC_key_points
            }
        }

    def from_dict(self, data, base_dir=None):
        self.meta = data.get("meta", self.meta)
        pa = data.get("partA", {})
        self.partA_hidden_text = pa.get("hidden_text", "")
        if pa.get("video") and base_dir:
            self.partA_video_path = os.path.join(base_dir, pa["video"])
        pb = data.get("partB", {})
        self.partB_situation = pb.get("situation", "")
        self.partB_listening_text = pb.get("listening_text", "")
        if pb.get("listening_audio") and base_dir:
            self.partB_listening_audio = os.path.join(base_dir, pb["listening_audio"])
        self.partB_three_questions = [
            {**q, "en_audio": os.path.join(base_dir, q["en_audio"]) if q.get("en_audio") and base_dir else q.get("en_audio")}
            for q in pb.get("three_questions", [])
        ]
        self.partB_five_answers = [
            {**q, "q_audio": os.path.join(base_dir, q["q_audio"]) if q.get("q_audio") and base_dir else q.get("q_audio")}
            for q in pb.get("five_answers", [])
        ]
        pc = data.get("partC", {})
        self.partC_summary = pc.get("summary", "")
        self.partC_keywords = pc.get("keywords", "")
        self.partC_source_type = pc.get("source_type", "tts")
        self.partC_tts_text = pc.get("tts_text", "")
        if pc.get("audio") and base_dir:
            self.partC_audio_path = os.path.join(base_dir, pc["audio"])
        self.partC_key_points = pc.get("key_points", "")

class PracticeSession:
    def __init__(self, package_name):
        self.package_name = package_name
        self.timestamp = datetime.datetime.now()
        self.partA_recording = None
        self.partB_slots = {}      # 以 "PartB_Q1" / "PartB_A1" 为键，重录时覆盖而非追加
        self.partC_recording = None
        self.evaluation = {}

# ------------------ 主窗口 ------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoloTalk 1.8 - 单机版听说模考编辑器")
        self.setMinimumSize(1024, 700)

        # 窗口图标（打包后图标可能在 _internal/ 下）
        _icon_path = _resolve_resource_file("xixi.ico")
        if _icon_path:
            self.setWindowIcon(QIcon(_icon_path))
        else:
            print("[ICON] xixi.ico 未找到，跳过窗口图标设置")

        self.current_package = SoloPackage()
        self.current_session = None

        self.central = QStackedWidget()
        self.setCentralWidget(self.central)

        self.home_page = HomePage(self)
        self.editor_page = EditorPage(self)
        self.practice_page = PracticePage(self)
        self.history_page = HistoryPage(self)

        self.central.addWidget(self.home_page)
        self.central.addWidget(self.editor_page)
        self.central.addWidget(self.practice_page)
        self.central.addWidget(self.history_page)
        self.central.setCurrentWidget(self.home_page)
        self.show()

    def go_to(self, page):
        self.central.setCurrentWidget(page)

    def closeEvent(self, event):
        # 如果当前在练习页面且考试中，需确认中途退出
        if self.central.currentWidget() == self.practice_page:
            if not self.practice_page.request_exit():
                event.ignore()
                return
        event.accept()

class HomePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel("SoloTalk")
        title.setStyleSheet("font-size:42px; font-weight:bold; color:#2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("单机版英语听说模考编辑器 · 完全离线")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:16px; color:#7f8c8d; margin-bottom:30px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        btn_style = """
            QPushButton { font-size:20px; padding:18px 50px; border:2px solid #3498db;
                          border-radius:10px; background:white; color:#2c3e50; }
            QPushButton:hover { background:#ecf0f1; }
        """
        btn_editor = QPushButton("✏️  编辑模式")
        btn_editor.setStyleSheet(btn_style)
        btn_editor.clicked.connect(lambda: self.main.go_to(self.main.editor_page))
        btn_practice = QPushButton("🎧  练习模式")
        btn_practice.setStyleSheet(btn_style)
        btn_practice.setToolTip("可跳过 / 上一步，开头试音也能跳过")
        btn_practice.clicked.connect(lambda: self.load_and_start("practice"))
        btn_exam = QPushButton("📝  模考模式")
        btn_exam.setStyleSheet(btn_style)
        btn_exam.setToolTip("不可跳过 / 上一步，试音也不能跳过")
        btn_exam.clicked.connect(lambda: self.load_and_start("exam"))
        btn_history = QPushButton("📋  历史记录")
        btn_history.setStyleSheet(btn_style)
        btn_history.clicked.connect(lambda: self.main.go_to(self.main.history_page))
        layout.addWidget(btn_editor, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_practice, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_exam, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_history, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def load_and_start(self, mode):
        path, _ = QFileDialog.getOpenFileName(self, "选择 .solo 文件", "", "SoloTalk 文件 (*.solo)")
        if not path:
            return
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                with zf.open('data.json') as f:
                    data = json.load(f)
                temp_dir = tempfile.mkdtemp(prefix="solotalk_")
                zf.extractall(temp_dir)
                self.main.current_package.from_dict(data, base_dir=temp_dir)
                self.main.current_package.meta['temp_dir'] = temp_dir
            # 先把题目包交给练习页，确保进度文件路径能取到正确的包名
            self.main.practice_page.pkg = self.main.current_package
            self.main.practice_page.set_mode(mode)
            self.main.go_to(self.main.practice_page)
            # 进入后先检查该模式 + 该题包是否有“一半”的进度，再让用户选择继续或重新开始
            self.main.practice_page._check_and_prompt_progress()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

# ------------------ 编辑模式 ------------------
class EditorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.pkg = self.main.current_package
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        top_bar = QHBoxLayout()
        self.name_edit = QLineEdit(self.pkg.meta.get("name", ""))
        self.name_edit.setPlaceholderText("题目包名称")
        self.author_edit = QLineEdit(self.pkg.meta.get("author", ""))
        self.author_edit.setPlaceholderText("题目作者（可选匿名）")
        self.anonymous_check = QCheckBox("匿名")
        self.anonymous_check.setChecked(self.pkg.meta.get("anonymous", False))
        if self.pkg.meta.get("author"):
            self.author_edit.setReadOnly(True)
            self.anonymous_check.setEnabled(False)
        btn_new = QPushButton("新建")
        btn_open = QPushButton("打开 .solo")
        btn_save = QPushButton("保存")
        btn_back = QPushButton("返回主页")
        top_bar.addWidget(QLabel("名称："))
        top_bar.addWidget(self.name_edit)
        top_bar.addWidget(QLabel("作者："))
        top_bar.addWidget(self.author_edit)
        top_bar.addWidget(self.anonymous_check)
        top_bar.addWidget(btn_new)
        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_save)
        top_bar.addStretch()
        top_bar.addWidget(btn_back)
        btn_new.clicked.connect(self.new_package)
        btn_open.clicked.connect(self.open_package)
        btn_save.clicked.connect(self.save_package)
        btn_back.clicked.connect(lambda: self.main.go_to(self.main.home_page))
        layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        self.partA_widget = PartAEditor(self.pkg)
        self.partB_widget = PartBEditor(self.pkg)
        self.partC_widget = PartCEditor(self.pkg)
        self.tabs.addTab(self.partA_widget, "Part A 模仿朗读")
        self.tabs.addTab(self.partB_widget, "Part B 角色扮演")
        self.tabs.addTab(self.partC_widget, "Part C 故事复述")
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def new_package(self):
        self.main.current_package = SoloPackage()
        self.pkg = self.main.current_package
        self.name_edit.setText("")
        self.author_edit.setText("")
        self.anonymous_check.setChecked(False)
        self.author_edit.setReadOnly(False)
        self.anonymous_check.setEnabled(True)
        self.partA_widget.pkg = self.pkg
        self.partB_widget.pkg = self.pkg
        self.partC_widget.pkg = self.pkg
        self.partA_widget.refresh()
        self.partB_widget.refresh()
        self.partC_widget.refresh()

    def open_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开 .solo 文件", "", "SoloTalk 文件 (*.solo)")
        if path:
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    with zf.open('data.json') as f:
                        data = json.load(f)
                    temp_dir = tempfile.mkdtemp(prefix="solotalk_")
                    zf.extractall(temp_dir)
                    self.pkg.from_dict(data, base_dir=temp_dir)
                    self.pkg.meta['temp_dir'] = temp_dir
                self.name_edit.setText(self.pkg.meta.get("name", ""))
                self.author_edit.setText(self.pkg.meta.get("author", ""))
                self.anonymous_check.setChecked(self.pkg.meta.get("anonymous", False))
                if self.pkg.meta.get("author"):
                    self.author_edit.setReadOnly(True)
                    self.anonymous_check.setEnabled(False)
                else:
                    self.author_edit.setReadOnly(False)
                    self.anonymous_check.setEnabled(True)
                self.partA_widget.refresh()
                self.partB_widget.refresh()
                self.partC_widget.refresh()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开失败：{str(e)}")

    def save_package(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 .solo 文件", "", "SoloTalk 文件 (*.solo)")
        if not path:
            return
        self.pkg.meta["name"] = self.name_edit.text() or "Untitled"
        self.pkg.meta["author"] = self.author_edit.text() if not self.anonymous_check.isChecked() else "匿名"
        self.pkg.meta["anonymous"] = self.anonymous_check.isChecked()
        self.partA_widget.save_to_pkg()
        self.partB_widget.save_to_pkg()
        self.partC_widget.save_to_pkg()
        try:
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('data.json', json.dumps(self.pkg.to_dict(), indent=2))
                if self.pkg.partA_video_path and os.path.exists(self.pkg.partA_video_path):
                    zf.write(self.pkg.partA_video_path, os.path.basename(self.pkg.partA_video_path))
                # 收集 Part B/C 引用的音频（Part C 仅“上传音频”模式；Part B 9 处均可选），去重后写入
                audio_files, seen = [], set()
                def _add_audio(p):
                    if p and os.path.exists(p) and p not in seen:
                        seen.add(p)
                        audio_files.append(p)
                if self.pkg.partC_source_type == "audio":
                    _add_audio(self.pkg.partC_audio_path)
                _add_audio(self.pkg.partB_listening_audio)
                for q in self.pkg.partB_three_questions:
                    _add_audio(q.get("en_audio"))
                for a in self.pkg.partB_five_answers:
                    _add_audio(a.get("q_audio"))
                for ap in audio_files:
                    zf.write(ap, os.path.basename(ap))
            QMessageBox.information(self, "成功", f"题目包已保存至 {path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

class PartAEditor(QWidget):
    def __init__(self, pkg):
        super().__init__()
        self.pkg = pkg
        layout = QFormLayout()
        self.video_label = QLabel("未选择视频")
        btn_video = QPushButton("选择视频")
        btn_video.clicked.connect(self.upload_video)
        video_box = QHBoxLayout()
        video_box.addWidget(self.video_label)
        video_box.addWidget(btn_video)
        layout.addRow("视频文件：", video_box)
        self.hidden_edit = QTextEdit()
        self.hidden_edit.setPlaceholderText("输入隐藏原文（仅用于考后批改）")
        layout.addRow("原文文本(隐藏)：", self.hidden_edit)
        self.setLayout(layout)

    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.pkg.partA_video_path = path
            self.video_label.setText(os.path.basename(path))

    def save_to_pkg(self):
        self.pkg.partA_hidden_text = self.hidden_edit.toPlainText()

    def refresh(self):
        if self.pkg.partA_video_path:
            self.video_label.setText(os.path.basename(self.pkg.partA_video_path))
        else:
            self.video_label.setText("未选择视频")
        self.hidden_edit.setPlainText(self.pkg.partA_hidden_text)

class PartBEditor(QWidget):
    def __init__(self, pkg):
        super().__init__()
        self.pkg = pkg
        self._listening_audio_path = None
        self._listening_audio_label = None
        self._three_audio = []       # 每问"英文机答"的可选音频（绝对路径或 None）
        self._three_audio_labels = []
        self._five_audio = []        # 每答"英文问题"的可选音频
        self._five_audio_labels = []
        self.setup_ui()

    def _make_audio_row(self):
        """生成一行「文件名 + 选择音频 + 清除」，返回 (layout, label, btn_choose, btn_clear)。"""
        label = QLabel("未选择音频")
        label.setStyleSheet("color:#666;")
        btn_choose = QPushButton("选择音频")
        btn_clear = QPushButton("清除")
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(btn_choose)
        row.addWidget(btn_clear)
        return row, label, btn_choose, btn_clear

    def _choose_audio(self, label):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频文件 (*.wav *.mp3 *.ogg *.m4a)")
        if path:
            label.setText(os.path.basename(path))
        return path

    def _clear_audio(self, label):
        label.setText("未选择音频")
        return None

    def setup_ui(self):
        scroll = QScrollArea()
        widget = QWidget()
        form = QFormLayout()
        self.situation_edit = QTextEdit()
        self.situation_edit.setPlaceholderText("情景介绍（学生自读）")
        form.addRow("情景介绍：", self.situation_edit)
        self.listen_edit = QTextEdit()
        self.listen_edit.setPlaceholderText("三问前听力文本（电脑 TTS 朗读，也可上传音频）")
        form.addRow("听力文本：", self.listen_edit)
        l_row, self._listening_audio_label, l_btn, l_clear = self._make_audio_row()
        l_btn.clicked.connect(lambda: self._set_listening_audio(self._choose_audio(self._listening_audio_label)))
        l_clear.clicked.connect(lambda: self._clear_listening_audio(self._clear_audio(self._listening_audio_label)))
        form.addRow("听力音频（可选）：", l_row)

        self.three_group = QGroupBox("三问设置（每题）")
        three_layout = QVBoxLayout()
        self.three_widgets = []
        self._three_audio = []
        self._three_audio_labels = []
        for i in range(3):
            g = QGroupBox(f"第{i+1}问")
            inner = QFormLayout()
            cn = QLineEdit(); cn.setPlaceholderText("中文提示")
            en = QLineEdit(); en.setPlaceholderText("英文机答 (TTS)")
            hid = QLineEdit(); hid.setPlaceholderText("标准答案(隐藏)")
            inner.addRow("中文提示：", cn)
            inner.addRow("英文机答：", en)
            a_row, a_label, a_btn, a_clear = self._make_audio_row()
            a_btn.clicked.connect(lambda _, i=i, lb=a_label: self._set_three_audio(i, self._choose_audio(lb)))
            a_clear.clicked.connect(lambda _, i=i, lb=a_label: self._clear_three_audio(i, self._clear_audio(lb)))
            inner.addRow("机答音频（可选）：", a_row)
            inner.addRow("标准答案：", hid)
            g.setLayout(inner)
            three_layout.addWidget(g)
            self.three_widgets.append((cn, en, hid))
            self._three_audio.append(None)
            self._three_audio_labels.append(a_label)
        self.three_group.setLayout(three_layout)
        form.addRow(self.three_group)

        self.five_group = QGroupBox("五答设置（每题）")
        five_layout = QVBoxLayout()
        self.five_widgets = []
        self._five_audio = []
        self._five_audio_labels = []
        for i in range(5):
            g = QGroupBox(f"第{i+1}答")
            inner = QFormLayout()
            q = QLineEdit(); q.setPlaceholderText("英文问题 (TTS)")
            hid = QLineEdit(); hid.setPlaceholderText("标准答案(隐藏)")
            inner.addRow("英文问题：", q)
            a_row, a_label, a_btn, a_clear = self._make_audio_row()
            a_btn.clicked.connect(lambda _, i=i, lb=a_label: self._set_five_audio(i, self._choose_audio(lb)))
            a_clear.clicked.connect(lambda _, i=i, lb=a_label: self._clear_five_audio(i, self._clear_audio(lb)))
            inner.addRow("问题音频（可选）：", a_row)
            inner.addRow("标准答案：", hid)
            g.setLayout(inner)
            five_layout.addWidget(g)
            self.five_widgets.append((q, hid))
            self._five_audio.append(None)
            self._five_audio_labels.append(a_label)
        self.five_group.setLayout(five_layout)
        form.addRow(self.five_group)
        widget.setLayout(form)
        scroll.setWidget(widget)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    # ---- 音频存取（绝对路径保存在内存，序列化时转 basename）----
    def _set_listening_audio(self, path):
        self._listening_audio_path = path

    def _clear_listening_audio(self, _):
        self._listening_audio_path = None

    def _set_three_audio(self, i, path):
        self._three_audio[i] = path

    def _clear_three_audio(self, i, _):
        self._three_audio[i] = None

    def _set_five_audio(self, i, path):
        self._five_audio[i] = path

    def _clear_five_audio(self, i, _):
        self._five_audio[i] = None

    def save_to_pkg(self):
        self.pkg.partB_situation = self.situation_edit.toPlainText()
        self.pkg.partB_listening_text = self.listen_edit.toPlainText()
        self.pkg.partB_listening_audio = self._listening_audio_path
        self.pkg.partB_three_questions = []
        for i, (cn, en, hid) in enumerate(self.three_widgets):
            self.pkg.partB_three_questions.append({
                "cn_prompt": cn.text(), "en_answer": en.text(), "hidden_answer": hid.text(),
                "en_audio": self._three_audio[i]
            })
        self.pkg.partB_five_answers = []
        for i, (q, hid) in enumerate(self.five_widgets):
            self.pkg.partB_five_answers.append({
                "en_question": q.text(), "hidden_answer": hid.text(),
                "q_audio": self._five_audio[i]
            })

    def refresh(self):
        self.situation_edit.setPlainText(self.pkg.partB_situation)
        self.listen_edit.setPlainText(self.pkg.partB_listening_text)
        self._listening_audio_path = self.pkg.partB_listening_audio
        self._listening_audio_label.setText(os.path.basename(self.pkg.partB_listening_audio)
                                            if self.pkg.partB_listening_audio else "未选择音频")
        for i, (cn, en, hid) in enumerate(self.three_widgets):
            if i < len(self.pkg.partB_three_questions):
                q = self.pkg.partB_three_questions[i]
                cn.setText(q.get("cn_prompt","")); en.setText(q.get("en_answer","")); hid.setText(q.get("hidden_answer",""))
                self._three_audio[i] = q.get("en_audio")
            else:
                cn.setText(""); en.setText(""); hid.setText("")
                self._three_audio[i] = None
            self._three_audio_labels[i].setText(os.path.basename(self._three_audio[i]) if self._three_audio[i] else "未选择音频")
        for i, (q, hid) in enumerate(self.five_widgets):
            if i < len(self.pkg.partB_five_answers):
                a = self.pkg.partB_five_answers[i]
                q.setText(a.get("en_question","")); hid.setText(a.get("hidden_answer",""))
                self._five_audio[i] = a.get("q_audio")
            else:
                q.setText(""); hid.setText("")
                self._five_audio[i] = None
            self._five_audio_labels[i].setText(os.path.basename(self._five_audio[i]) if self._five_audio[i] else "未选择音频")

class PartCEditor(QWidget):
    def __init__(self, pkg):
        super().__init__()
        self.pkg = pkg
        layout = QFormLayout()
        self.summary_edit = QTextEdit()
        self.summary_edit.setPlaceholderText("故事梗概")
        self.keywords_edit = QLineEdit()
        self.keywords_edit.setPlaceholderText("关键词提示（逗号分隔）")
        layout.addRow("故事梗概：", self.summary_edit)
        layout.addRow("关键词：", self.keywords_edit)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["电脑 TTS 朗读", "上传音频文件"])
        self.source_combo.currentIndexChanged.connect(self.toggle_source)
        layout.addRow("故事音频来源：", self.source_combo)

        self.tts_edit = QTextEdit()
        self.tts_label = QLabel("故事全文：")
        self.tts_edit.setPlaceholderText("输入故事全文，电脑 TTS 朗读")
        layout.addRow(self.tts_label, self.tts_edit)

        self.audio_label = QLabel("未选择音频")
        self.audio_btn = QPushButton("选择音频")
        self.audio_btn.clicked.connect(self.upload_audio)
        audio_box = QHBoxLayout()
        audio_box.addWidget(self.audio_label)
        audio_box.addWidget(self.audio_btn)
        layout.addRow("音频文件：", audio_box)

        self.key_points_edit = QTextEdit()
        self.key_points_edit.setPlaceholderText("答案要点（隐藏，用于批改）")
        layout.addRow("答案要点：", self.key_points_edit)
        self.setLayout(layout)

    def toggle_source(self):
        is_tts = self.source_combo.currentIndex() == 0
        self.audio_label.setVisible(not is_tts)
        self.audio_btn.setVisible(not is_tts)
        # TTS 模式：故事全文必填；上传音频模式：故事全文选填（仅参考）
        if is_tts:
            self.tts_label.setText("故事全文：")
            self.tts_edit.setPlaceholderText("输入故事全文，电脑 TTS 朗读")
        else:
            self.tts_label.setText("故事全文（选填）：")
            self.tts_edit.setPlaceholderText("故事全文（选填，仅作参考，不用于播放）")

    def upload_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频文件 (*.wav *.mp3 *.ogg)")
        if path:
            self.pkg.partC_audio_path = path
            self.audio_label.setText(os.path.basename(path))

    def save_to_pkg(self):
        self.pkg.partC_summary = self.summary_edit.toPlainText()
        self.pkg.partC_keywords = self.keywords_edit.text()
        self.pkg.partC_source_type = "tts" if self.source_combo.currentIndex() == 0 else "audio"
        self.pkg.partC_tts_text = self.tts_edit.toPlainText()
        self.pkg.partC_key_points = self.key_points_edit.toPlainText()

    def refresh(self):
        self.summary_edit.setPlainText(self.pkg.partC_summary)
        self.keywords_edit.setText(self.pkg.partC_keywords)
        if self.pkg.partC_source_type == "audio":
            self.source_combo.setCurrentIndex(1)
            if self.pkg.partC_audio_path:
                self.audio_label.setText(os.path.basename(self.pkg.partC_audio_path))
        else:
            self.source_combo.setCurrentIndex(0)
        self.tts_edit.setPlainText(self.pkg.partC_tts_text)
        self.key_points_edit.setPlainText(self.pkg.partC_key_points)
        self.toggle_source()

# ------------------ 练习 / 模考 模式 ------------------
# 练习模式：可跳过 / 上一步，开头试音也能跳过
# 模考模式：跳过 / 上一步 均禁用，试音也不能跳过
class PracticePage(QWidget):
    signal_update_display = pyqtSignal(str, str)
    signal_tts_ready = pyqtSignal()
    signal_tts_next = pyqtSignal()
    signal_tts_file_ready = pyqtSignal(str)   # edge-tts 生成的音频文件就绪（主线程播放）
    signal_finished = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.pkg = None
        self.session = None
        # 关闭硬件解码与 OSD，减少 Windows 下嵌入 VLC 时 Direct3D 与主线程消息循环冲突导致的"未响应"
        self.vlc_instance = vlc.Instance(
            "--no-video-title-show --no-osd --no-snapshot-preview --avcodec-hw=none"
        )
        if self.vlc_instance is None:
            raise RuntimeError(
                "VLC 初始化失败：无法创建 vlc.Instance()。\n"
                "请确认 G:\\emm\\VLC\\ 目录存在且包含完整的 VLC 插件（plugins 文件夹）。\n"
                "如果根目录有 libvlc.dll，请尝试将其删除或重命名，让程序使用 VLC\\ 子目录下的完整版本。"
            )
        self.player = self.vlc_instance.media_player_new()
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_tick)
        self._current_phase = None
        self._phase_end_time = 0
        self._timer_callback = None
        self._teardown_done = False  # 守卫标志：teardown 后所有延迟回调应跳过
        self._audio_frames = []
        self._stream = None
        self._is_recording = False
        self._tts_stop_event = None
        self._tts_thread = None
        self._tts_next_callback = None
        self._navigating = False
        self._exam_active = False
        self.mode = "practice"
        self._b_moments = []
        self._win_key_blocker = None
        self._fullscreen = False
        # VLC 视频嵌入到 Qt 的 video_frame（set_hwnd）；点击由 _disable_vlc_input 禁用其输入
        self._vlc_parent_hwnd = None
        self.player.event_manager().event_attach(vlc.EventType.MediaPlayerEndReached, self._on_vlc_end)
        # 每次真正开始播放后，重新禁用视频子窗口输入（VLC 可能在 play 时重建窗口）
        self.player.event_manager().event_attach(
            vlc.EventType.MediaPlayerPlaying,
            lambda e: QTimer.singleShot(50, self._disable_vlc_input))

        self.setup_ui()
        self.signal_update_display.connect(self._update_display)
        self.signal_tts_ready.connect(self._prepare_tts_done)
        self.signal_tts_next.connect(self._on_tts_next)
        self.signal_tts_file_ready.connect(self._on_tts_file_ready)
        self.signal_finished.connect(self._on_exam_finished)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.main_label = QLabel("准备开始练习")
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setStyleSheet("font-size:28px; font-weight:bold;")
        self.sub_label = QLabel("")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setStyleSheet("font-size:18px; color:gray;")
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size:20px; color:#e74c3c; font-weight:bold;")
        self.countdown_label.hide()

        # 视频区：用 video_frame 作为 VLC 的嵌入画布（set_hwnd 把视频渲染到这个原生窗口）。
        # 点击问题在 _disable_vlc_input 里通过禁用视频子窗口输入解决，不再单独开窗口。
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background:black;")
        _grid = QGridLayout(self.video_container)
        _grid.setContentsMargins(0, 0, 0, 0)
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background:black;")
        _grid.addWidget(self.video_frame, 0, 0)
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("font-size:18px;")
        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.video_container)
        self.display_stack.addWidget(self.text_display)

        # 录音指示：作为 video_container 的子控件居中显示（最开始的版本）
        self.recording_overlay = QLabel("🔴 录音中...", self.video_container)
        self.recording_overlay.setAlignment(Qt.AlignCenter)
        self.recording_overlay.setStyleSheet("color:white; font-size:28px; background:rgba(0,0,0,150); border-radius:10px;")
        self.recording_overlay.setFixedSize(200, 60)
        self.recording_overlay.move((self.video_container.width() - 200) // 2, 20)
        self.recording_overlay.hide()

        layout.addWidget(self.main_label)
        layout.addWidget(self.sub_label)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.display_stack)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始练习")
        self.start_btn.clicked.connect(self.start_exam)
        self.skip_btn = QPushButton("⏭  跳过")
        self.skip_btn.setStyleSheet(
            "QPushButton { background:#f39c12; color:white; border:1px solid #e67e22; font-weight:bold; }"
            "QPushButton:disabled { background:#bdc3c7; color:#ecf0f1; }"
            "QPushButton:hover:!disabled { background:#e67e22; }"
        )
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self._skip_current)
        self.prev_btn = QPushButton("⏮  上一步")
        self.prev_btn.setStyleSheet(
            "QPushButton { background:#3498db; color:white; border:1px solid #2980b9; font-weight:bold; }"
            "QPushButton:disabled { background:#bdc3c7; color:#ecf0f1; }"
            "QPushButton:hover:!disabled { background:#2980b9; }"
        )
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self._prev_current)
        btn_back = QPushButton("返回主页")
        btn_back.clicked.connect(self._request_back)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.skip_btn)
        btn_layout.addWidget(btn_back)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.video_container.resizeEvent = self._resize_video

    def _resize_video(self, event):
        """video_container 尺寸变化时，重新居中录音提示。"""
        self.recording_overlay.move((self.video_container.width() - 200) // 2, 20)

    # ---------- 退出 / 收尾 ----------
    def _request_back(self):
        """返回主页按钮：考试中需确认，确认后结束并回家（不记录历史）。"""
        if self.request_exit():
            self.main.go_to(self.main.home_page)

    def _confirm_abort(self):
        """考试中途退出的确认提示。返回 True 表示确认退出。"""
        reply = QMessageBox.warning(
            self, "确认退出",
            "中途退出将保存当前进度与已录制音频，下次可从本次位置继续。\n确定要退出吗？",
            QMessageBox.Yes | QMessageBox.No)
        return reply == QMessageBox.Yes

    def request_exit(self):
        """返回 True 表示允许退出（已确认或无需确认）。"""
        if self._exam_active:
            if not self._confirm_abort():
                return False
        self._teardown_exam()
        return True

    def _teardown_exam(self):
        """停止一切声画/计时/录音，保存当前进度与正在录的音，然后回到开考前初始界面。
        中途退出调用；正常结束不经过这里。"""
        self._teardown_done = True  # 先设标志，让所有 pending 回调立即跳过
        self.timer.stop()
        # 强制停止 TTS 线程
        self._cancel_tts()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=0.5)
        # 同步停止 VLC
        self._stop_video()
        self.player.audio_set_volume(100)
        # 先把正在录的音保存下来，再把进度写入文件
        self._finalize_current_recording()
        if self._exam_active and self._current_phase:
            self._save_progress()
        self.stop_recording()
        self._reset_page_state()

    def _reset_page_state(self):
        """放弃当前考试全部内容，回到开考前/练习前的初始界面。
        中途退出、正常结束后再次进入都靠它恢复成“刚打开”的状态。"""
        self._exam_active = False
        self._teardown_done = False
        self._navigating = False
        self._current_phase = None
        self._step_index = 0
        self._b_step = 0
        self._c_step = 0
        self._b_moments = []
        self._timer_callback = None
        self._tts_next_callback = None
        self._audio_callback = None
        self._is_recording = False
        self._audio_frames = []
        self.countdown_label.hide()
        self.recording_overlay.hide()
        self.player.stop()
        self.display_stack.setCurrentWidget(self.text_display)
        self.main_label.setText("准备开始" + ("模考" if self.mode == "exam" else "练习"))
        self.sub_label.setText("")
        self._stop_parse_timer()
        self._parse_media = None
        self.start_btn.setEnabled(True)
        self._update_nav_buttons()
        self._restore_window()
        self._unblock_windows_key()

    def abort_exam(self):
        """兼容旧调用（如关闭窗口时）。等同 _request_back 的收尾逻辑。"""
        self._teardown_exam()

    def _block_windows_key(self):
        """模考模式：安装底层键盘钩子屏蔽 Windows 键。"""
        if sys.platform != "win32":
            return
        try:
            if self._win_key_blocker is None:
                self._win_key_blocker = WindowsKeyBlocker()
            self._win_key_blocker.install()
        except Exception as e:
            print("Windows 键屏蔽安装失败：", e)
            self._win_key_blocker = None

    def _unblock_windows_key(self):
        """卸载 Windows 键屏蔽（考试结束/退出时调用）。"""
        if self._win_key_blocker is not None:
            try:
                self._win_key_blocker.uninstall()
            except Exception:
                pass
            self._win_key_blocker = None

    # ---------- 进度保存 / 恢复 ----------
    def _progress_file_path(self):
        """返回当前模式 + 当前题目包对应的进度文件路径。
        模式与题目包分别编码进文件名，练习/模考进度天然隔离、不会混用。"""
        pkg = self.pkg or getattr(self.main, "current_package", None)
        pkg_name = pkg.meta.get("name", "untitled") if pkg else "untitled"
        safe = "".join(c for c in pkg_name if c.isalnum() or c in (" ", "-", "_")).rstrip().replace(" ", "_")
        if not safe:
            safe = "untitled"
        return os.path.join(HISTORY_DIR, f"progress_{self.mode}_{safe}.json")

    def _save_progress(self):
        """把当前考试/练习进度写入文件，包含已保存的录音路径与当前检查点。
        中途退出时调用，以便下次从该检查点继续。"""
        if not self.pkg or not self.session:
            return
        try:
            data = {
                "mode": self.mode,
                "package": self.pkg.meta.get("name", ""),
                "phase": self._current_phase,
                "step_index": getattr(self, "_step_index", 0),
                "b_step": getattr(self, "_b_step", 0),
                "b_moments": getattr(self, "_b_moments", []),
                "c_step": getattr(self, "_c_step", 0),
                "mic_test_active": self._current_phase == "mic" and getattr(self, "_is_recording", False),
                "prepare_text": getattr(self, "_prepare_text", ""),
                "video_duration": getattr(self, "video_duration", 60),
                "recordings": {
                    "partA": self.session.partA_recording,
                    "partB": self.session.partB_slots,
                    "partC": self.session.partC_recording,
                },
                "timestamp": datetime.datetime.now().isoformat(),
            }
            os.makedirs(HISTORY_DIR, exist_ok=True)
            with open(self._progress_file_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print("保存进度失败：", e)

    def _load_progress(self):
        """读取当前模式 + 当前题目包的进度文件。不存在或损坏时返回 None。"""
        path = self._progress_file_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("读取进度失败：", e)
            return None

    def _clear_progress(self, delete_recordings=False):
        """删除进度文件（正常完成或用户选择重新开考）。
        delete_recordings=True 时（用户主动“重新开始”），会先删除进度中记录的全部已录音
        WAV 文件，避免废弃音频残留占用磁盘；正常完成考试时调用不带此参数，保留录音用于
        线下批改与历史。"""
        if delete_recordings:
            try:
                data = self._load_progress()
            except Exception:
                data = None
            if data:
                recs = data.get("recordings", {}) or {}
                paths = []
                if recs.get("partA"):
                    paths.append(recs["partA"])
                if recs.get("partC"):
                    paths.append(recs["partC"])
                for v in (recs.get("partB") or {}).values():
                    if v:
                        paths.append(v)
                for p in paths:
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                            print("已删除录音：", p)
                    except Exception as e:
                        print("删除录音失败：", e)
        path = self._progress_file_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print("清理进度文件失败：", e)

    def _describe_progress(self, data):
        """把进度数据转成给用户看的简短摘要（用于恢复对话框）。"""
        mode_text = "模考" if data.get("mode") == "exam" else "练习"
        phase_map = {"mic": "试音阶段", "partA": "A 篇", "partB": "B 篇", "partC": "C 篇"}
        phase = phase_map.get(data.get("phase", ""), data.get("phase", "未知"))
        ts = data.get("timestamp", "")
        ts_show = ts[:19].replace("T", " ") if ts else "未知"
        recs = data.get("recordings", {}) or {}
        n = 0
        if recs.get("partA"):
            n += 1
        if recs.get("partC"):
            n += 1
        n += len([v for v in (recs.get("partB") or {}).values() if v])
        return (f"模式：{mode_text}\n当前位置：{phase}\n"
                f"已录音：{n} 段\n保存时间：{ts_show}")

    def _check_and_prompt_progress(self):
        """进入练习/模考页面后调用：先检查当前模式 + 当前题目包是否有“一半”的进度。
        有则弹窗显示进度摘要，让用户选择【继续】还是【重新开始】；没有则什么都不做，
        等待用户点“开始”。练习与模考进度分文件保存，互不干扰。"""
        saved = self._load_progress()
        if not saved:
            return
        mode_text = "模考" if self.mode == "exam" else "练习"
        desc = self._describe_progress(saved)
        msg = QMessageBox(self)
        msg.setWindowTitle("恢复进度")
        msg.setText(f"检测到未完成的{mode_text}进度：\n\n{desc}\n\n是否继续？")
        btn_continue = msg.addButton("继续" + mode_text, QMessageBox.AcceptRole)
        btn_restart = msg.addButton("重新开始", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_continue)
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked == btn_continue:
            self._restore_progress(saved)
        elif clicked == btn_restart:
            # 重新开始：清除进度文件与已录制音频，回到开考前初始状态
            self._clear_progress(delete_recordings=True)
            self.start_btn.setText("开始" + mode_text)
            self._update_nav_buttons()
        # 若用户直接关闭对话框（未选择），保持当前初始状态，进度文件保留供下次处理

    def _restore_progress(self, data):
        """根据保存的进度数据恢复 session、录音路径与当前检查点，并继续考试。"""
        self.pkg = self.main.current_package
        self.session = PracticeSession(self.pkg.meta.get("name", "练习"))
        recs = data.get("recordings", {})
        self.session.partA_recording = recs.get("partA")
        self.session.partB_slots = recs.get("partB", {}) or {}
        self.session.partC_recording = recs.get("partC")

        self._exam_active = True
        self.start_btn.setEnabled(False)
        self._update_nav_buttons()
        if self.mode == "exam":
            self._enter_fullscreen()
            self._block_windows_key()

        phase = data.get("phase")
        # 试音阶段：如果上次正在录音，直接回到试音录音；否则从头准备
        if phase == "mic":
            self._prepare_text = data.get("prepare_text", "")
            if data.get("mic_test_active"):
                self._mic_test()
            else:
                self._prepare_phase()
            return

        self._b_moments = data.get("b_moments", [])
        self._step_index = data.get("step_index", 0)
        self._b_step = data.get("b_step", 0)
        self._c_step = data.get("c_step", 0)
        self.video_duration = data.get("video_duration", 60)

        if phase == "partA":
            self._go_to_checkpoint(("partA", self._step_index), replay=True)
        elif phase == "partB":
            self._go_to_checkpoint(("partB", self._b_step), replay=True)
        elif phase == "partC":
            self._go_to_checkpoint(("partC", self._c_step), replay=True)
        else:
            # 未知阶段，回到 Part A
            self._run_part_a()

    def _enter_fullscreen(self):
        if sys.platform == "win32":
            self.main.showFullScreen()
        else:
            self.main.showMaximized()
        self._fullscreen = True

    def _restore_window(self):
        if self._fullscreen:
            self.main.showNormal()
            self._fullscreen = False

    def _update_skip_label(self):
        """根据是否正在录音，切换跳过按钮的文案与样式：
        - 录音中：显示「结束录音」（红色停止样式），点它 = 结束当前录音并前进到下一个点；
        - 非录音：显示「跳过」（橙色），点它 = 跳到下一个机读/录音点。
        仍是同一个按钮，不另开一个；模考模式该按钮隐藏，无需切换。"""
        if self.mode == "exam":
            return
        if getattr(self, "_is_recording", False):
            self.skip_btn.setText("⏹  结束录音")
            self.skip_btn.setStyleSheet(
                "QPushButton { background:#e74c3c; color:white; border:1px solid #c0392b; font-weight:bold; }"
                "QPushButton:disabled { background:#bdc3c7; color:#ecf0f1; }"
                "QPushButton:hover:!disabled { background:#c0392b; }"
            )
        else:
            self.skip_btn.setText("⏭  跳过")
            self.skip_btn.setStyleSheet(
                "QPushButton { background:#f39c12; color:white; border:1px solid #e67e22; font-weight:bold; }"
                "QPushButton:disabled { background:#bdc3c7; color:#ecf0f1; }"
                "QPushButton:hover:!disabled { background:#e67e22; }"
            )

    def _update_nav_buttons(self):
        """根据模式与考试状态更新跳过/上一步按钮：
        - 模考模式：直接隐藏（用不了）；
        - 练习模式：始终显示，考试中启用、未考试时禁用。"""
        if self.mode == "exam":
            self.skip_btn.setVisible(False)
            self.prev_btn.setVisible(False)
            return
        self.skip_btn.setVisible(True)
        self.prev_btn.setVisible(True)
        active = getattr(self, "_exam_active", False)
        self.skip_btn.setEnabled(active)
        self.prev_btn.setEnabled(active)
        self._update_skip_label()

    def set_mode(self, mode):
        """设置模式：
        - 'practice' 练习模式：可跳过 / 上一步，开头试音也能跳过；
        - 'exam'    模考模式：跳过 / 上一步 均禁用且隐藏，试音也不能跳过。"""
        self.mode = mode
        if mode == "practice":
            self.start_btn.setText("开始练习")
        else:
            self.start_btn.setText("开始模考")
        self._reset_page_state()

    def start_exam(self):
        """从“开始”按钮触发：以全新状态开考（进度检查/恢复已在进入页面时处理）。"""
        self.pkg = self.main.current_package
        self.session = PracticeSession(self.pkg.meta.get("name", "练习"))
        self.start_btn.setEnabled(False)
        self._exam_active = True
        self._update_nav_buttons()
        if self.mode == "exam":
            self._enter_fullscreen()
            self._block_windows_key()
        self._prepare_phase()

    def _prepare_phase(self):
        self._current_phase = "mic"   # 开头试音阶段（准备 + 麦克风测试）
        author = self.pkg.meta.get("author", "")
        author_text = f"题目作者：{author}" if author else "题目作者：未知"
        self._prepare_text = f"{author_text}\n\n生活就像海洋，只有意志坚强的人才能到达彼岸。\nThis is an apple, I like apples, apples are good for our health."
        self.signal_update_display.emit("准备阶段", self._prepare_text)
        # 准备阶段朗读用原来的系统 SAPI5（中英混合文本，edge-tts 英文语音读不好中文）
        self._start_tts(
            "生活就像海洋，只有意志坚强的人才能到达彼岸。 This is an apple, I like apples, apples are good for our health.",
            self.signal_tts_ready,
            force_fallback=True
        )

    # ---- 麦克风测试 ----
    def _prepare_tts_done(self):
        self._mic_test()

    def _mic_test(self):
        test_text = self._prepare_text + "\n\n请说话，录音10秒..."
        self.signal_update_display.emit("麦克风测试", test_text)
        self._start_recording()
        self._set_timer(10, self._mic_test_playback)

    def _mic_test_playback(self):
        self._stop_recording()
        test_path = self._save_recording("mic_test")
        if os.path.exists(test_path):
            if self.mode == "practice":
                # 练习模式：不弹“麦克风是否正常”询问，直接回放后进入 Part A
                self._player_play_then(test_path, self._run_part_a)
                return
            # 模考模式：保留“麦克风是否正常”询问
            media = self.vlc_instance.media_new(test_path)
            self.player.set_media(media)
            self.player.play()
            reply = QMessageBox.question(self, "麦克风测试", "录音回放中，麦克风是否正常？",
                                         QMessageBox.Yes | QMessageBox.No)
            self.player.stop()
            if reply == QMessageBox.Yes:
                try:
                    os.remove(test_path)
                except:
                    pass
                self.signal_update_display.emit("准备开始", "即将开始模考")
                self._set_timer(3, self._run_part_a)
            else:
                try:
                    os.remove(test_path)
                except:
                    pass
                self._mic_test()
        else:
            QMessageBox.warning(self, "错误", "录音失败，请检查设备")
            self._mic_test()

    def _player_play_then(self, media_path, callback):
        """回放音频，结束后回调（自动清理临时文件）。"""
        self.player.set_media(self.vlc_instance.media_new(media_path))
        self.player.play()
        m = self.player.get_media()
        duration = 0
        try:
            duration = m.get_duration()
        except Exception:
            duration = 0
        # 至少给一个最短回放时间，避免 0 时长直接跳过
        wait_ms = max(1500, int(duration) + 300) if duration and duration > 0 else 3000
        self._set_timer(wait_ms / 1000.0, lambda: self._on_playback_done(media_path, callback))

    def _on_playback_done(self, media_path, callback):
        self.player.stop()
        try:
            os.remove(media_path)
        except Exception:
            pass
        self.signal_update_display.emit("准备开始", "即将开始练习")
        callback()

    # ---- TTS 管理 ----
    def _start_tts(self, text, signal=None, force_fallback=False):
        """开始 TTS 朗读。force_fallback=True 时跳过 edge-tts，直接走系统 SAPI5。"""
        self._cancel_tts()
        self._tts_stop_event = threading.Event()
        self._tts_thread = threading.Thread(target=self._tts_runner, args=(text, signal, force_fallback))
        self._tts_thread.daemon = True
        self._tts_thread.start()

    def _speak_or_play_audio(self, text, audio_path, callback):
        """优先播放音频文件，否则用 TTS 朗读文本；播放/朗读结束后触发 callback。

        Part B / Part C 共用：有音频且文件存在 → VLC 播放（结束回调 _on_vlc_end）；
        否则 → TTS 朗读（结束回调 _on_tts_next）。"""
        if audio_path and os.path.exists(audio_path):
            self._audio_callback = callback
            media = self.vlc_instance.media_new(audio_path)
            self.player.set_media(media)
            self.player.play()
            self.display_stack.setCurrentWidget(self.text_display)  # 只播声音，停在文本区
            QTimer.singleShot(80, lambda: self._delayed_set_volume(100))
        else:
            if audio_path:
                QMessageBox.warning(self, "警告", "音频文件不存在，改用 TTS 朗读")
            self._set_tts_callback(callback)
            self._start_tts(text, self.signal_tts_next)

    def _tts_runner(self, text, signal, force_fallback=False):
        # 优先 edge-tts（神经网络语音）：生成 mp3 后回主线程用 VLC 播放，播完再触发 signal
        if not force_fallback and EDGE_TTS_AVAILABLE and text and text.strip():
            try:
                os.makedirs(TTS_EDGE_DIR, exist_ok=True)
            except Exception:
                pass
            out = os.path.join(TTS_EDGE_DIR, f"tts_{int(time.time()*1000)}_{threading.get_ident()}.mp3")
            if tts_generate_audio(text, out):
                self._tts_done_signal = signal
                self._tts_tmp_file = out
                self.signal_tts_file_ready.emit(out)
                return
        # 回退：系统 SAPI5 逐句朗读
        tts_speak_blocking(text, self._tts_stop_event)
        if signal:
            signal.emit()

    def _on_tts_file_ready(self, path):
        """主线程：播放 edge-tts 生成的音频，播完由 _on_vlc_end 触发 _tts_file_done。"""
        if self._teardown_done:
            return
        if not path or not os.path.exists(path):
            # 文件异常，直接触发原信号继续流程
            sig = getattr(self, "_tts_done_signal", None)
            self._tts_done_signal = None
            if sig:
                sig.emit()
            return
        self._audio_callback = self._tts_file_done
        media = self.vlc_instance.media_new(path)
        self.player.set_media(media)
        self.player.play()
        self.display_stack.setCurrentWidget(self.text_display)  # 只播声音，停在文本区
        QTimer.singleShot(80, lambda: self._delayed_set_volume(100))

    def _tts_file_done(self):
        """edge-tts 音频播放完毕：清理临时文件，触发原 signal 继续流程。"""
        sig = getattr(self, "_tts_done_signal", None)
        self._tts_done_signal = None
        tmp = getattr(self, "_tts_tmp_file", None)
        self._tts_tmp_file = None
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass
        if sig:
            sig.emit()

    def _cancel_tts(self):
        if self._tts_stop_event:
            self._tts_stop_event.set()
            self._tts_stop_event = None

    def _set_tts_callback(self, callback):
        self._tts_next_callback = callback
        try:
            self.signal_tts_next.disconnect()
        except:
            pass
        self.signal_tts_next.connect(self._on_tts_next)

    def _on_tts_next(self):
        if self._teardown_done:
            return
        if self._tts_next_callback:
            cb = self._tts_next_callback
            self._tts_next_callback = None
            cb()

    # ---------- Part A ----------
    def _run_part_a(self):
        self._current_phase = "partA"
        self._step_index = 0
        video_path = self.pkg.partA_video_path
        if not video_path or not os.path.exists(video_path):
            QMessageBox.critical(self, "错误", "Part A 视频文件不存在")
            self.abort_exam()
            return
        # 先给一个兜底时长，避免解析卡住时界面无响应
        self.video_duration = 60
        self._parse_media = self.vlc_instance.media_new(video_path)
        # 异步解析，不阻塞主线程消息循环
        self._parse_media.parse_with_options(vlc.MediaParseFlag.local, 5000)
        self.signal_update_display.emit("加载视频", "正在解析视频时长...")
        self._parse_attempts = 0
        self._parse_timer = QTimer(self)
        self._parse_timer.timeout.connect(self._check_video_duration)
        self._parse_timer.start(200)

    def _check_video_duration(self):
        if self._teardown_done:
            self._stop_parse_timer()
            return
        self._parse_attempts += 1
        dur_ms = -1
        try:
            dur_ms = self._parse_media.get_duration()
        except Exception:
            pass
        if dur_ms > 0 or self._parse_attempts >= 25:  # 解析成功或 5 秒超时
            if dur_ms > 0:
                dur_sec = dur_ms / 1000.0
                self.video_duration = int(dur_sec) if dur_sec == int(dur_sec) else int(dur_sec) + 1
            self._stop_parse_timer()
            self._parse_media = None
            self._exec_partA_step()

    def _stop_parse_timer(self):
        if getattr(self, "_parse_timer", None):
            self._parse_timer.stop()
            try:
                self._parse_timer.deleteLater()
            except Exception:
                pass
            self._parse_timer = None

    def _exec_partA_step(self):
        idx = self._step_index
        # 进入非视频步骤时，确保视频已停并隐藏（修复离开视频步骤视频仍残留的问题）
        if idx not in (1, 6):
            self._stop_video()
        if idx == 0:
            self.signal_update_display.emit("Part A Reading Aloud",
                "In this part, you are required to watch a video clip and read after the speaker.")
            self._set_tts_callback(self._next_partA_step)
            self._start_tts("Part A Reading Aloud. In this part, you are required to watch a video clip and read after the speaker in the video.", self.signal_tts_next)
        elif idx == 1:
            self.signal_update_display.emit("观看视频", "请观看视频并注意发音")
            self._play_video_full()
            self._set_timer(self.video_duration)
        elif idx == 2:
            self.signal_update_display.emit("练习提示", "Now you have one minute to practice reading")
            self._set_tts_callback(self._next_partA_step)
            self._start_tts("Now you have one minute to practice reading", self.signal_tts_next)
        elif idx == 3:
            self.signal_update_display.emit("阅读文本", "请默读准备 (60秒)")
            self._show_text(self.pkg.partA_hidden_text or "(无原文)")
            self._set_timer(60)
        elif idx == 4:
            self.signal_update_display.emit("听录音", "听视频原声，可看文本")
            self._show_text(self.pkg.partA_hidden_text or "(无原文)")
            self._play_video_audio_only()
            self._set_timer(self.video_duration)
        elif idx == 5:
            self.signal_update_display.emit("录音准备", "Now read as the speaker in the video")
            self._set_tts_callback(self._next_partA_step)
            self._start_tts("Now read as the speaker in the video", self.signal_tts_next)
        elif idx == 6:
            self.signal_update_display.emit("模仿朗读", "请看着视频和字幕跟读")
            self._play_video_silent()
            self._start_recording()
            self._set_timer(self.video_duration)
        else:
            self._stop_video()
            self._stop_recording()
            self.session.partA_recording = self._save_recording("PartA")
            self._run_part_b()

    def _next_partA_step(self):
        self._step_index += 1
        self._exec_partA_step()

    # ---------- Part B ----------
    def _run_part_b(self):
        self._current_phase = "partB"
        self._b_step = 0
        self._audio_callback = None
        # 把整个 Part B 拆成一系列「机读 / 录音 / 准备」小点，
        # 每个点都是一次「跳过 / 上一步」的目标，保证只前进/后退一步。
        self._b_moments = []
        self._b_moments.append({"t": "tts", "main": "Part B Role Play",
                                "sub": "In this part, you are required to act as a role and complete the following tasks.",
                                "text": "Part B Role Play. In this part, you are required to act as a role and complete the following tasks."})
        self._b_moments.append({"t": "timer", "sec": 30, "main": "情景介绍", "sub": self.pkg.partB_situation or "(无)"})
        self._b_moments.append({"t": "tts", "main": "听对话", "sub": self.pkg.partB_situation or "(情景)",
                                "text": self.pkg.partB_listening_text,
                                "audio": self.pkg.partB_listening_audio})
        for idx, q in enumerate(self.pkg.partB_three_questions):
            self._b_moments.append({"t": "prep_q", "idx": idx, "main": f"准备提问 {idx+1}", "sub": q['cn_prompt']})
            self._b_moments.append({"t": "record_q", "idx": idx, "main": f"请提问 {idx+1}", "sub": q['cn_prompt']})
            self._b_moments.append({"t": "tts", "main": "电脑回答", "sub": "", "text": q['en_answer'],
                                    "audio": q.get('en_audio')})
        for idx, a in enumerate(self.pkg.partB_five_answers):
            self._b_moments.append({"t": "tts", "main": "请听问题", "sub": "", "text": a['en_question'],
                                    "audio": a.get('q_audio')})
            self._b_moments.append({"t": "tts", "main": "重复提问", "sub": "", "text": a['en_question'],
                                    "audio": a.get('q_audio')})
            self._b_moments.append({"t": "timer", "sec": 10, "main": "准备回答", "sub": ""})
            self._b_moments.append({"t": "record_a", "idx": idx, "main": "请回答", "sub": ""})
        self._exec_partB_step()

    def _exec_partB_step(self, replay=False):
        if self._b_step >= len(self._b_moments):
            self._run_part_c()
            return
        self._exec_b_moment(self._b_moments[self._b_step], replay)

    def _exec_b_moment(self, m, replay=False):
        t = m["t"]
        if t == "tts":
            self.signal_update_display.emit(m["main"], m.get("sub", ""))
            self._speak_or_play_audio(m.get("text", ""), m.get("audio"), self._next_partB_step)
        elif t == "timer":
            self.signal_update_display.emit(m["main"], m.get("sub", ""))
            self._set_timer(m["sec"], callback=self._next_partB_step)
        elif t == "prep_q":
            self.signal_update_display.emit(m["main"], m["sub"])
            self._set_timer(20, callback=self._next_partB_step)
        elif t == "record_q":
            idx = m["idx"]
            if replay:
                # 重进 / 上一步 落到提问录音点：先重播题目（准备提示），再开始录音
                self.signal_update_display.emit(f"准备提问 {idx+1}", m["sub"])
                self._set_timer(20, callback=lambda i=idx: self._begin_b_record(i, "q"))
            else:
                self._begin_b_record(idx, "q")
        elif t == "record_a":
            idx = m["idx"]
            if replay:
                # 先重播问题（请听问题 + 重复提问），再开始录音
                self.signal_update_display.emit("请听问题", "")
                self._speak_or_play_audio(
                    self.pkg.partB_five_answers[idx]['en_question'],
                    self.pkg.partB_five_answers[idx].get('q_audio'),
                    lambda i=idx: self._replay_question_then_record(i))
            else:
                self._begin_b_record(idx, "a")

    def _finish_b_record(self, idx, kind):
        self._stop_recording()
        if kind == "q":
            self.session.partB_slots[f"PartB_Q{idx+1}"] = self._save_recording(f"PartB_Q{idx+1}")
        else:
            self.session.partB_slots[f"PartB_A{idx+1}"] = self._save_recording(f"PartB_A{idx+1}")
        self._next_partB_step()

    def _begin_b_record(self, idx, kind):
        """进入某道 Part B 提问 / 回答的录音（开始录音并计时 8 秒）。"""
        if kind == "q":
            self.signal_update_display.emit(f"请提问 {idx+1}", self.pkg.partB_three_questions[idx]['cn_prompt'])
        else:
            self.signal_update_display.emit("请回答", "")
        self._start_recording()
        self._set_timer(8, callback=lambda i=idx, k=kind: self._finish_b_record(i, k))

    def _replay_question_then_record(self, idx):
        """重播问题第二段（重复提问），结束后开始录音。"""
        self.signal_update_display.emit("重复提问", "")
        self._speak_or_play_audio(
            self.pkg.partB_five_answers[idx]['en_question'],
            self.pkg.partB_five_answers[idx].get('q_audio'),
            lambda i=idx: self._begin_b_record(i, "a"))

    def _next_partB_step(self):
        self._b_step += 1
        self._exec_partB_step()

    # ---------- Part C ----------
    def _run_part_c(self):
        self._current_phase = "partC"
        self._c_step = 0
        self._audio_callback = None
        self._exec_partC_step()

    def _exec_partC_step(self, replay=False):
        s = self._c_step
        # 离开“听独白”步骤时停掉播放器，避免残留视频
        if s not in (2, 4):
            self._stop_video()
        if s == 0:
            self.signal_update_display.emit("Part C Retelling",
                "In this part, you are required to listen to a monologue and retell what you have heard.")
            self._set_tts_callback(self._next_partC_step)
            self._start_tts("Part C Retelling. In this part, you are required to listen to a monologue and retell what you have heard.", self.signal_tts_next)
        elif s == 1:
            text = f"梗概：{self.pkg.partC_summary}\n关键词：{self.pkg.partC_keywords}"
            self.signal_update_display.emit("阅读梗概和关键词", text)
            self._set_timer(15)
        elif s == 2 or s == 4:
            text = f"梗概：{self.pkg.partC_summary}\n关键词：{self.pkg.partC_keywords}"
            self.signal_update_display.emit("听独白", text)
            if self.pkg.partC_source_type == "tts":
                self._set_tts_callback(self._next_partC_step)
                self._start_tts(self.pkg.partC_tts_text, self.signal_tts_next)
            else:
                if self.pkg.partC_audio_path and os.path.exists(self.pkg.partC_audio_path):
                    self._audio_callback = self._next_partC_step
                    media = self.vlc_instance.media_new(self.pkg.partC_audio_path)
                    self.player.set_media(media)
                    self.player.play()
                    self.display_stack.setCurrentWidget(self.text_display)  # 只播声音，停在文本区
                    QTimer.singleShot(80, lambda: self._delayed_set_volume(100))
                else:
                    QMessageBox.warning(self, "警告", "Part C 音频文件不存在")
                    self._next_partC_step()
        elif s == 3:
            self._set_timer(1, self._next_partC_step)
        elif s == 5:
            self.signal_update_display.emit("准备复述 (60秒)", "")
            self._set_timer(60)
        elif s == 6:
            if replay:
                # 重进 / 上一步 落到复述录音点：先重播独白（题目），再开始录音
                text = f"梗概：{self.pkg.partC_summary}\n关键词：{self.pkg.partC_keywords}"
                self.signal_update_display.emit("听独白", text)
                if self.pkg.partC_source_type == "tts":
                    self._set_tts_callback(lambda: self._begin_c_record())
                    self._start_tts(self.pkg.partC_tts_text, self.signal_tts_next)
                else:
                    if self.pkg.partC_audio_path and os.path.exists(self.pkg.partC_audio_path):
                        self._audio_callback = lambda: self._begin_c_record()
                        media = self.vlc_instance.media_new(self.pkg.partC_audio_path)
                        self.player.set_media(media)
                        self.player.play()
                        self.display_stack.setCurrentWidget(self.text_display)
                        QTimer.singleShot(80, lambda: self._delayed_set_volume(100))
                    else:
                        QMessageBox.warning(self, "警告", "Part C 音频文件不存在")
                        self._begin_c_record()
            else:
                self.signal_update_display.emit("请复述故事", "")
                self._start_recording()
                self._set_timer(120)
        else:
            self._stop_recording()
            self.session.partC_recording = self._save_recording("PartC")
            self.signal_finished.emit()

    def _on_vlc_end(self, event):
        if self._teardown_done:
            return
        # 避免在准备阶段等非视频阶段触发异常回调
        if not self._exam_active:
            return
        # Part B 机读音频 / Part C 独白音频 / edge-tts 朗读音频 播放完毕 → 触发下一步
        # （_audio_callback 仅在“只播音频”的路径被设置，视频播放结束不会误触发）
        if self._audio_callback:
            cb = self._audio_callback
            self._audio_callback = None
            QTimer.singleShot(0, cb)

    def _next_partC_step(self):
        self._c_step += 1
        self._exec_partC_step()

    def _begin_c_record(self):
        """进入 Part C 复述录音（开始录音并计时 120 秒）。"""
        self.signal_update_display.emit("请复述故事", "")
        self._start_recording()
        self._set_timer(120)

    # ---------- 计时器 ----------
    def _set_timer(self, seconds, callback=None):
        self._phase_end_time = time.time() + seconds
        self._timer_callback = callback
        self.countdown_label.show()
        self.timer.start(100)

    def _on_tick(self):
        if self._teardown_done:
            return
        remaining = max(0, int(self._phase_end_time - time.time()))
        self.countdown_label.setText(f"剩余时间：{remaining} 秒")
        if time.time() >= self._phase_end_time:
            self.timer.stop()
            self.countdown_label.hide()
            if self._timer_callback:
                cb = self._timer_callback
                self._timer_callback = None
                cb()
            else:
                self._next_step_default()

    def _next_step_default(self):
        if self._current_phase == "partA":
            self._next_partA_step()
        elif self._current_phase == "partB":
            self._next_partB_step()
        elif self._current_phase == "partC":
            self._next_partC_step()

    # ---------- 录音点导航（跳过 / 上一步） ----------
    def _cur_step(self):
        if self._current_phase == "partA":
            return self._step_index
        if self._current_phase == "partB":
            return self._b_step
        if self._current_phase == "partC":
            return self._c_step
        return -1

    def _next_checkpoint(self, phase, step):
        """返回下一个「机读 / 录音 / 环节」点——只前进一步，绝不跳过多段。"""
        if phase == "partA":
            if step < 6:
                return ("partA", step + 1)
            return ("partB", 0)
        if phase == "partB":
            moments = getattr(self, "_b_moments", [])
            if step < len(moments) - 1:
                return ("partB", step + 1)
            return ("partC", 0)
        if phase == "partC":
            if step < 6:
                return ("partC", step + 1)
            return None
        return None

    def _prev_checkpoint(self, phase, step):
        """返回上一个「机读 / 录音 / 环节」点——只后退一步。"""
        if phase == "partA":
            if step > 0:
                return ("partA", step - 1)
            return None
        if phase == "partB":
            if step > 0:
                return ("partB", step - 1)
            return ("partA", 6)
        if phase == "partC":
            if step > 0:
                return ("partC", step - 1)
            moments = getattr(self, "_b_moments", None)
            if moments:
                return ("partB", len(moments) - 1)
            return ("partB", 0)
        return None

    def _go_to_checkpoint(self, cp, replay=False):
        phase, step = cp
        if phase == "partA":
            self._current_phase = "partA"
            self._step_index = step
            self._exec_partA_step()
        elif phase == "partB":
            if step == 0 and self._current_phase != "partB":
                self._run_part_b()
            else:
                self._current_phase = "partB"
                self._b_step = step
                self._exec_partB_step(replay)
        else:
            if step == 0 and self._current_phase != "partC":
                self._run_part_c()
            else:
                self._current_phase = "partC"
                self._c_step = step
                self._exec_partC_step(replay)

    def _finalize_current_recording(self):
        """若当前正在录音，则保存（不丢弃），以便 跳过/上一步 不丢失已录内容。"""
        if not self._is_recording:
            return
        self._stop_recording()
        phase = self._current_phase
        if phase == "partA":
            self.session.partA_recording = self._save_recording("PartA")
        elif phase == "partB":
            moments = getattr(self, "_b_moments", [])
            if 0 <= self._b_step < len(moments):
                m = moments[self._b_step]
                if m["t"] == "record_q":
                    self.session.partB_slots[f"PartB_Q{m['idx']+1}"] = self._save_recording(f"PartB_Q{m['idx']+1}")
                elif m["t"] == "record_a":
                    self.session.partB_slots[f"PartB_A{m['idx']+1}"] = self._save_recording(f"PartB_A{m['idx']+1}")
        elif phase == "partC":
            self.session.partC_recording = self._save_recording("PartC")

    def _cleanup_for_nav(self):
        """跳转换段前，立即清理当前声画/计时/录音（不显示缓冲）。"""
        self.timer.stop()
        self.countdown_label.hide()
        self._timer_callback = None
        self._cancel_tts()
        self._tts_next_callback = None
        try:
            self.signal_tts_next.disconnect()
        except Exception:
            pass
        self._finalize_current_recording()
        self._stop_video()
        self._audio_callback = None
        # 清理未播完的 edge-tts 临时音频
        self._tts_done_signal = None
        tmp = getattr(self, "_tts_tmp_file", None)
        self._tts_tmp_file = None
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _skip_current(self):
        """跳过：跳到下一个录音点 / 跳过开头试音（0.4s 缓冲，不显示）。仅练习模式可用。"""
        if self.mode != "practice":
            return
        if self._current_phase not in ("mic", "partA", "partB", "partC"):
            return
        if self._navigating:
            return
        self._navigating = True
        self._cleanup_for_nav()
        QTimer.singleShot(400, self._skip_advance)

    def _skip_advance(self):
        try:
            if self._teardown_done or not self._exam_active:
                return
            if self._current_phase == "mic":
                # 练习模式：跳过开头试音，直接进入 Part A
                self._stop_recording()
                self.player.stop()
                self.player.audio_set_volume(100)
                self._run_part_a()
                return
            nxt = self._next_checkpoint(self._current_phase, self._cur_step())
            if nxt is None:
                # 已是最后一个录音点，结束考试（当前录音已在清理时保存）
                self.signal_finished.emit()
                return
            self._go_to_checkpoint(nxt)
        finally:
            self._navigating = False

    def _prev_current(self):
        """上一步：跳到上一个录音点重新录音（0.4s 缓冲，不显示）。仅练习模式可用。"""
        if self.mode != "practice":
            return
        if self._current_phase not in ("partA", "partB", "partC"):
            return
        if self._navigating:
            return
        self._navigating = True
        self._cleanup_for_nav()
        QTimer.singleShot(400, self._prev_advance)

    def _prev_advance(self):
        try:
            if self._teardown_done or not self._exam_active:
                return
            prev = self._prev_checkpoint(self._current_phase, self._cur_step())
            if prev is None:
                return          # 已是第一个录音点，无法回退
            self._go_to_checkpoint(prev, replay=True)
        finally:
            self._navigating = False

    @pyqtSlot(str, str)
    def _update_display(self, main, sub):
        self.main_label.setText(main)
        self.sub_label.setText(sub)
        if "观看视频" in main or "模仿朗读" in main:
            self.display_stack.setCurrentWidget(self.video_container)
        else:
            self.display_stack.setCurrentWidget(self.text_display)
            self.text_display.setPlainText(sub if sub else main)

    # ---------- 视频控制（VLC 嵌入 Qt 的 video_frame，禁用点击输入）----------
    def _start_vlc_playback(self, media, silent=False, show_window=True):
        """统一播放：把 VLC 视频嵌入到 Qt 的 video_frame 控件里（set_hwnd）。
        show_window=True 时切换到视频区显示；False 时只播声音，停留在文本区。
        播放后通过 _disable_vlc_input 禁用 VLC 视频子窗口的鼠标/键盘输入，
        这是修复"点击视频导致主线程卡死/未响应"的关键。"""
        if self._teardown_done:
            return
        if show_window and self.display_stack.currentWidget() != self.video_container:
            self.display_stack.setCurrentWidget(self.video_container)
        # 窗口句柄有效性检查：确保 video_frame 已创建且可见
        try:
            if not self.video_frame.isVisible():
                self._vlc_parent_hwnd = None
            else:
                hwnd = int(self.video_frame.winId())
                # 简单验证句柄是否有效（非零且窗口存在）
                if sys.platform == "win32" and hwnd:
                    if ctypes.windll.user32.IsWindow(hwnd):
                        self._vlc_parent_hwnd = hwnd
                        self.player.set_hwnd(hwnd)
                    else:
                        self._vlc_parent_hwnd = None
                else:
                    self._vlc_parent_hwnd = hwnd if hwnd else None
        except Exception:
            self._vlc_parent_hwnd = None
        self.player.set_media(media)
        self.player.play()
        vol = 0 if silent else 100
        QTimer.singleShot(80, lambda v=vol: self._delayed_set_volume(v))
        # VLC 在 play() 之后才创建视频子窗口，稍后禁用其输入
        QTimer.singleShot(150, self._disable_vlc_input)

    def _delayed_set_volume(self, vol):
        """播放后延迟设音量（teardown 守卫）"""
        if self._teardown_done:
            return
        try:
            self.player.audio_set_volume(vol)
        except Exception:
            pass

    def _disable_vlc_input(self):
        """禁用 VLC 视频子窗口的输入（鼠标/键盘），让"点击视频"彻底无效，不再触发卡死。"""
        if self._teardown_done:
            return
        if sys.platform != "win32" or not self._vlc_parent_hwnd:
            return
        # 再次验证父窗口句柄是否仍然有效（阶段切换后可能已销毁）
        if not ctypes.windll.user32.IsWindow(self._vlc_parent_hwnd):
            self._vlc_parent_hwnd = None
            return
        try:
            user32 = ctypes.windll.user32
            GWL_STYLE = -16
            WS_DISABLED = 0x08000000
            cls_buf = ctypes.create_unicode_buffer(128)
            results = []
            EnumChildProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

            @EnumChildProc
            def enum_child(hwnd, lparam):
                user32.GetClassNameW(hwnd, cls_buf, 128)
                name = cls_buf.value
                if name and name.startswith("VLC"):  # VLC video output / VLC video main ...
                    results.append(hwnd)
                return 1

            user32.EnumChildWindows(self._vlc_parent_hwnd, enum_child, 0)
            for hwnd in results:
                style = user32.GetWindowLongW(hwnd, GWL_STYLE)
                if not (style & WS_DISABLED):
                    user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_DISABLED)
        except Exception:
            pass

    def _play_video_full(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self._start_vlc_playback(media, silent=False, show_window=True)

    def _play_video_audio_only(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self._start_vlc_playback(media, silent=False, show_window=False)
        self.display_stack.setCurrentWidget(self.text_display)

    def _play_video_silent(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self._start_vlc_playback(media, silent=True, show_window=True)

    def _stop_video(self):
        # 同步停止，避免 QTimer.singleShot(0, ...) 的停止事件排在新 play() 之后，
        # 导致刚启动的音视频（如 Part A 的"听录音"音频-only 步骤）被立即停掉。
        try:
            self.player.stop()
        except Exception:
            pass
        # 清空媒体，防止旧媒体的 EndReached 事件在阶段切换后仍触发回调
        try:
            self.player.set_media(None)
        except Exception:
            pass
        try:
            self.player.audio_set_volume(100)
        except Exception:
            pass

    def _show_text(self, text):
        self.text_display.setPlainText(text)
        self.display_stack.setCurrentWidget(self.text_display)

    # ---------- 录音控制 ----------
    def _start_recording(self):
        self._audio_frames = []
        self._is_recording = True
        self._update_skip_label()  # 录音开始：跳过按钮变为「结束录音」
        def callback(indata, frames, time_info, status):
            if self._is_recording:
                self._audio_frames.append(indata.copy())
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback)
        self._stream.start()
        if self.display_stack.currentWidget() == self.video_container:
            self.recording_overlay.move((self.video_container.width() - 200) // 2, 20)
            self.recording_overlay.show()
        else:
            self.display_stack.setCurrentWidget(self.text_display)
            self.sub_label.setText("🔴 录音中...")

    def _stop_recording(self):
        self._is_recording = False
        self._update_skip_label()  # 录音结束：按钮恢复为「跳过」
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.recording_overlay.hide()
        if self.display_stack.currentWidget() == self.text_display:
            self.sub_label.setText("")

    def stop_recording(self):
        # 公开的停止入口（用于中途退出收尾），统一复用 _stop_recording 的实现
        self._stop_recording()

    def _save_recording(self, label):
        package_name = self.pkg.meta.get('name', 'unknown')
        safe_name = "".join(c for c in package_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_name:
            safe_name = "untitled"
        base_dir = os.path.join(RECORDINGS_DIR, safe_name)
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{safe_name}_{label}_{ts}.wav"
        path = os.path.join(base_dir, fname)
        if self._audio_frames:
            data = np.concatenate(self._audio_frames)
            wav_write.write(path, SAMPLE_RATE, data)
            return path
        return None

    # ---------- 考试结束与批改 ----------
    def _on_exam_finished(self):
        self._exam_active = False
        self.start_btn.setEnabled(True)
        self._stop_video()
        self._update_nav_buttons()
        # 正常结束也需退出全屏并解除 Windows 键屏蔽
        self._restore_window()
        self._unblock_windows_key()
        # 考试正常完成，清理进度文件
        self._clear_progress()
        if self.mode == "exam":
            QMessageBox.information(self, "模考完成", "模考结束，即将进行离线批改。")
        else:
            QMessageBox.information(self, "练习完成", "练习结束，即将进行离线批改。")
        self.run_evaluation()

    def run_evaluation(self):
        vosk_model = load_vosk_model()
        if not vosk_model:
            QMessageBox.warning(self, "批改", "Vosk 模型不可用，录音已保存，稍后可重新批改。")
            self._save_history()
            return
        self._perform_evaluation(vosk_model)
        self._save_history()
        QMessageBox.information(self, "批改完成", "练习记录与参考批改已保存至历史记录。")

    def _perform_evaluation(self, vosk_model):
        self.session.evaluation = evaluate_recordings({
            "partA": self.session.partA_recording,
            "partB": self.session.partB_slots,
            "partC": self.session.partC_recording,
        }, self.pkg, vosk_model)

    def _save_history(self):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        history = {
            "package": self.pkg.meta.get("name", ""),
            "timestamp": self.session.timestamp.isoformat(),
            "recordings": {
                "partA": self.session.partA_recording,
                "partB": self.session.partB_slots,
                "partC": self.session.partC_recording
            },
            "evaluation": self.session.evaluation,
            "package_data": self.pkg.to_dict()
        }
        fname = f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(HISTORY_DIR, fname), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)

# ------------------ 历史记录页面 ------------------
class HistoryPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.view_detail)
        self.btn_reeval = QPushButton("重新批改选中记录")
        self.btn_reeval.clicked.connect(self.reevaluate_selected)
        self.btn_delete = QPushButton("删除选中记录")
        self.btn_delete.clicked.connect(self.delete_selected)
        btn_back = QPushButton("返回主页")
        btn_back.clicked.connect(lambda: self.main.go_to(self.main.home_page))
        layout.addWidget(QLabel("练习历史记录（双击查看详情）"))
        layout.addWidget(self.list_widget)
        layout.addWidget(self.btn_reeval)
        layout.addWidget(self.btn_delete)
        layout.addWidget(btn_back)
        self.setLayout(layout)

    def showEvent(self, event):
        self.load_history()
        super().showEvent(event)

    def load_history(self):
        self.list_widget.clear()
        if not os.path.exists(HISTORY_DIR):
            return
        files = sorted(Path(HISTORY_DIR).glob("history_*.json"), reverse=True)
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                item_text = f"{data.get('package','?')}  {data.get('timestamp','')}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, str(f))
                self.list_widget.addItem(item)
            except:
                pass

    def delete_selected(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        path = current_item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        reply = QMessageBox.question(self, "确认删除", "确定要删除该历史记录吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                os.remove(path)
                self.load_history()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败：{str(e)}")

    def reevaluate_selected(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        path = current_item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            QMessageBox.critical(self, "错误", "历史记录文件损坏")
            return
        recs = data.get('recordings', {})
        pkg_dict = data.get('package_data', {})
        temp_pkg = SoloPackage()
        temp_pkg.from_dict(pkg_dict)

        # ── 收集该记录实际存在的录音槽位 ──
        b3, b5 = _partB_slots_as_ordered(recs.get('partB', {}))
        n3 = len(temp_pkg.partB_three_questions)
        n5 = len(temp_pkg.partB_five_answers)

        # ── 弹出多选对话框（与详情弹窗风格统一）──
        dlg = QDialog(self)
        dlg.setWindowTitle("选择重新批改的部分")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setMinimumWidth(340)

        dlg_layout = QVBoxLayout()
        dlg_layout.setContentsMargins(0, 0, 0, 0)
        dlg_layout.setSpacing(0)

        # 浅灰外框（与详情弹窗一致）
        outer = QWidget()
        outer.setStyleSheet("background:#f5f6fa")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(20, 18, 20, 14)
        outer_layout.setSpacing(10)

        tip = QLabel("请选择要重新批改的部分：")
        tip.setFont(QFont("Microsoft YaHei", 12))
        tip.setStyleSheet("color:#333;padding:2px 0 8px;")
        outer_layout.addWidget(tip)

        checks = []  # (selection键, QCheckBox)

        if recs.get('partA'):
            cb = QCheckBox("Part A 模仿朗读")
            cb.setFont(QFont("Microsoft YaHei", 12))
            cb.setChecked(True)
            checks.append(('partA', None, cb))
            outer_layout.addWidget(cb)

        has_b3 = any(b3[i] for i in range(min(len(b3), n3)))
        if has_b3:
            cb = QCheckBox("Part B 三问")
            cb.setFont(QFont("Microsoft YaHei", 12))
            cb.setChecked(True)
            checks.append(('partB_three', None, cb))
            outer_layout.addWidget(cb)

        has_b5 = any(b5[i] for i in range(min(len(b5), n5)))
        if has_b5:
            cb = QCheckBox("Part B 五答")
            cb.setFont(QFont("Microsoft YaHei", 12))
            cb.setChecked(True)
            checks.append(('partB_five', None, cb))
            outer_layout.addWidget(cb)

        if recs.get('partC'):
            cb = QCheckBox("Part C 故事复述")
            cb.setFont(QFont("Microsoft YaHei", 12))
            cb.setChecked(True)
            checks.append(('partC', None, cb))
            outer_layout.addWidget(cb)

        if not checks:
            QMessageBox.information(self, "提示", "该记录没有可重新批改的录音")
            return

        # 按钮（紫系，与详情弹窗关闭按钮一致）
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        _btn_style = """
            QPushButton {
                background: #5b4fcf;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                padding: 8px 30px;
                min-height: 40px;
            }
            QPushButton:hover { background: #4a3eb8; }
            QPushButton:pressed { background: #3d32a0; }
        """
        _btn_style_cancel = """
            QPushButton {
                background: #e0e0e0;
                color: #222;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 15px;
                padding: 8px 30px;
                min-height: 40px;
            }
            QPushButton:hover { background: #cfcfcf; }
            QPushButton:pressed { background: #bebebe; }
        """

        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet(_btn_style_cancel)
        btn_ok = QPushButton("确定")
        btn_ok.setStyleSheet(_btn_style)
        btn_ok.setDefault(True)
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        outer_layout.addLayout(btn_row)

        dlg_layout.addWidget(outer)
        dlg.setLayout(dlg_layout)

        if dlg.exec_() != QDialog.Accepted:
            return

        # ── 构建 selection（勾选的 Part → 其下所有子项全选）──
        selection = {}
        for key, idx, cb in checks:
            if cb.isChecked():
                if key in ('partA', 'partC'):
                    selection[key] = True
                elif key == 'partB_three':
                    selection[key] = {i: True for i in range(n3) if i < len(b3) and b3[i]}
                elif key == 'partB_five':
                    selection[key] = {i: True for i in range(n5) if i < len(b5) and b5[i]}
        if not selection:
            QMessageBox.information(self, "提示", "未选择任何题目，已取消重新批改")
            return

        vosk_model = load_vosk_model()
        if not vosk_model:
            QMessageBox.warning(self, "错误", "Vosk 模型不可用，无法重新批改")
            return

        existing_eval = data.get('evaluation') or {}
        eval_result = evaluate_recordings(recs, temp_pkg, vosk_model,
                                          existing_eval=existing_eval, selection=selection)
        data['evaluation'] = eval_result
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        cnt = sum(1 for _, _, cb in checks if cb.isChecked())
        QMessageBox.information(self, "完成", f"已重新批改 {cnt} 个题目")
        self.load_history()

    def view_detail(self, item):
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ---- 构建 HTML 详情 ----
        # ⚠ QTextBrowser (Qt WebKit) 仅支持基础 CSS：
        #   ✅ solid background-color / border / border-radius / padding / margin
        #   ✅ font-size / color / font-weight / text-align
        #   ✅ table 布局
        #   ❌ linear-gradient / display:flex / flex-wrap / rgba / opacity / gap
        h = []

        # ═══════════════ 头部 ═══════════════
        pkg_name = _esc(data.get('package', ''))
        ts = _esc(data.get('timestamp', ''))
        # 去除时间戳中的毫秒等无关内容，仅保留「年-月-日 时:分:秒」
        ts_clean = ts.split('.')[0].replace('T', ' ') if ts else ''
        h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                 ' style="background:#5b4fcf;border-radius:8px;margin-bottom:12px">'
                 '<tr><td style="padding:14px 18px;color:#fff">'
                 f'<span style="font-size:16px;font-weight:bold">📋 题目包：{pkg_name}</span>'
                 f'<span style="font-size:12px;color:#d0c8ff;margin-left:12px"> {ts_clean}</span>'
                 '</td></tr></table>')

        # ═══════════════ 录音文件 — 左右双栏（左=A+3Q+C，右=五答） ═══════════════
        rec = data.get('recordings', {})
        # 分两组：左边 = Part A + 三问(Q1-Q3) + Part C，右边 = 五答(A1-A5)
        left_items = []   # Part A, Q1-Q3, Part C
        right_items = []  # A1-A5

        pa = rec.get('partA', '')
        if pa:
            left_items.append(('Part A', os.path.basename(pa)))

        partB_raw = rec.get('partB', {})
        if isinstance(partB_raw, dict):
            for k in ['PartB_Q1','PartB_Q2','PartB_Q3']:
                v = partB_raw.get(k)
                if v is not None:
                    left_items.append((k.replace('PartB_', ''), os.path.basename(v)))
            for k in ['PartB_A1','PartB_A2','PartB_A3','PartB_A4','PartB_A5']:
                v = partB_raw.get(k)
                if v is not None:
                    right_items.append((k.replace('PartB_', ''), os.path.basename(v)))
        elif isinstance(partB_raw, list):
            # list 格式：前 3 个归左(三问)，后 5 个归右(五答)
            for i, r in enumerate(partB_raw):
                item = (f'B{i+1}', os.path.basename(r) if r else '')
                if i < 3:
                    left_items.append(item)
                else:
                    right_items.append(item)

        pc_rec = rec.get('partC', '')
        if pc_rec:
            left_items.append(('Part C', os.path.basename(pc_rec)))

        all_rec = left_items + right_items
        if all_rec:
            # ═══ 录音文件 — 独立卡片，左右双栏（左=A+Q+C，右=五答） ═══
            h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                     ' style="background:#fff;border:2px solid #bbb;border-radius:8px;margin-bottom:16px">'
                     '<tr><td style="background:#e8eaef;padding:10px 14px;'
                     'border-bottom:2px solid #bbb;border-radius:8px 8px 0 0">'
                     '<span style="font-size:15px;font-weight:bold;color:#222">🎙 录音文件</span>'
                     '</td></tr>'
                     '<tr><td style="padding:12px 14px">')

            # 左右双栏（不强制均分，按内容自适应）
            h.append('<table width="100%" cellpadding="0" cellspacing="0"><tr>')

            # ── 左栏：Part A + Q1-Q3 + Part C ──
            h.append('<td valign="top" style="padding-right:10px">')
            h.append('<table cellpadding="4" cellspacing="3" style="font-size:12px">')
            for lab, fname in left_items:
                display = fname if fname else '(无)'
                h.append(f'<tr><td style="background:#eef2ff;border:1px solid #b0bec5;'
                         f'border-radius:4px;padding:5px 9px;color:#283593;word-break:break-all">'
                         f'<b>{lab}</b>: {_esc(display)}</td></tr>')
            h.append('</table></td>')

            # ── 右栏：A1-A5 ──
            h.append('<td valign="top" style="padding-left:10px">')
            h.append('<table cellpadding="4" cellspacing="3" style="font-size:12px">')
            for lab, fname in right_items:
                display = fname if fname else '(无)'
                h.append(f'<tr><td style="background:#eef2ff;border:1px solid #b0bec5;'
                         f'border-radius:4px;padding:5px 9px;color:#283593;word-break:break-all">'
                         f'<b>{lab}</b>: {_esc(display)}</td></tr>')
            h.append('</table></td>')

            h.append('</tr></table></td></tr></table>')

        # ═══════════════ 批改结果 — 左右双栏布局 ═══════════════
        ev = data.get('evaluation', {})
        if ev:
            # ---- 外层容器 ----
            h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                     ' style="background:#fff;border:2px solid #bbb;border-radius:8px;margin-bottom:16px">'
                     '<tr><td style="background:#e8eaef;padding:10px 14px;'
                     'border-bottom:2px solid #bbb;border-radius:8px 8px 0 0">'
                     '<span style="font-size:15px;font-weight:bold;color:#222">📊 批改结果</span>'
                     '</td></tr>'
                     '<tr><td style="padding:14px 16px">')

            # ── 收集数据（满分制：A=20, B每题=2/B总=16, C=24）──
            PART_A_MAX = 20
            PART_B_PER_ITEM = 2
            PART_C_MAX = 24

            acc_val, wer_val, acc_score, acc_color, acc_bg = None, None, None, '#999', '#f5f5f5'
            if 'partA' in ev:
                a = ev['partA']
                acc_val = a.get('accuracy', 0) * 100          # 百分比
                acc_score = a.get('accuracy', 0) * PART_A_MAX   # 满分制得分
                wer_val = a.get('wer', 0)
                acc_color = '#c0392b' if acc_val < 60 else '#e67e22' if acc_val < 80 else '#27ae60'
                acc_bg = '#fce4ec' if acc_val < 60 else '#fff3e0' if acc_val < 80 else '#e8f5e9'

            # Part B：每项存 (标签, 相似度, 得分, 颜色)
            rows_three = []
            if 'partB_three' in ev:
                for i, t in enumerate(ev['partB_three']):
                    sim = t.get('similarity', 0)
                    pts = sim * PART_B_PER_ITEM
                    c = '#c0392b' if sim < 0.6 else '#e67e22' if sim < 0.8 else '#27ae60'
                    rows_three.append((f'三问{i+1}', sim, pts, c))
            rows_five = []
            if 'partB_five' in ev:
                for i, fv in enumerate(ev['partB_five']):
                    sim = fv.get('similarity', 0)
                    pts = sim * PART_B_PER_ITEM
                    c = '#c0392b' if sim < 0.6 else '#e67e22' if sim < 0.8 else '#27ae60'
                    rows_five.append((f'五答{i+1}', sim, pts, c))
            partB_total_items = len(rows_three) + len(rows_five)
            partB_max = partB_total_items * PART_B_PER_ITEM       # 标准 16
            partB_score = (sum(p for _, _, p, _ in rows_three)
                           + sum(p for _, _, p, _ in rows_five))

            has_partC = 'partC' in ev
            partC_score = None
            partC_sim_pct = None
            partC_color = '#999'
            if has_partC:
                sem_sim = ev["partC"].get("semantic_sim")
                if sem_sim is not None:
                    # 兼容旧记录（可能为字符串）和新格式（float 0~1）
                    try:
                        sem_sim = float(sem_sim)
                    except (TypeError, ValueError):
                        sem_sim = None
                if sem_sim is not None:
                    partC_sim_pct = sem_sim * 100
                    partC_score = sem_sim * PART_C_MAX
                    partC_color = "#4caf50" if sem_sim >= 0.8 else ("#ff9800" if sem_sim >= 0.6 else "#f44336")

            # ── 总分横幅（跨全宽，放在左右双栏上方）──
            TOTAL_MAX = PART_A_MAX + partB_max + PART_C_MAX   # 标准 60
            total_score = 0.0
            if acc_score is not None:
                total_score += acc_score
            total_score += partB_score
            if partC_score is not None:
                total_score += partC_score
            total_ratio = total_score / TOTAL_MAX if TOTAL_MAX else 0
            total_color = '#c0392b' if total_ratio < 0.6 else '#e67e22' if total_ratio < 0.8 else '#27ae60'

            h.append(f'<table width="100%" cellpadding="0" cellspacing="0"'
                     f' style="border:1.5px solid #9FA8DA;border-radius:8px;background:#E8EAF6;margin-bottom:10px"><tr>'
                     f'<td style="padding:12px 18px;font-size:15px;color:#333;font-weight:bold;white-space:nowrap;text-align:center">'
                     f'总分：<b style="font-size:20px;color:{total_color}">{total_score:.1f}/{TOTAL_MAX}</b>'
                     f'</td></tr></table>')

            # ── 左右双栏（强制 50/50）：左=A+B总分+C，右=3Q+5A ──
            h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                     ' style="table-layout:fixed">')

            # ═══ 左栏 (50%)：Part A → Part B 总分 → Part C ═══
            h.append('<tr><td valign="top" style="width:50%;padding-right:10px">')

            # --- Part A 得分（蓝底）---
            if acc_score is not None:
                h.append(f'<table width="100%" cellpadding="0" cellspacing="0"'
                         f' style="border:1.5px solid #ccc;border-radius:6px;background:#e3f2fd;margin-bottom:8px"><tr>'
                         f'<td style="padding:10px 14px;font-size:14px;color:#333;font-weight:bold;vertical-align:middle;white-space:nowrap">'
                         f'Part A 得分：<b style="font-size:18px;color:{acc_color}">{acc_score:.1f}/{PART_A_MAX}</b>'
                         f'</td></tr><tr>'
                         f'<td colspan="2" style="padding:4px 14px 10px;font-size:12px;color:#666;border-top:1px solid rgba(0,0,0,.08)">'
                         f'准确率 {acc_val:.1f}% &nbsp;&nbsp; WER: {wer_val:.2f}'
                         f'</td></tr></table>')

            # --- Part B 总分（固定浅粉底）---
            if rows_three or rows_five:
                partB_accent = '#c06262'
                h.append(f'<table width="100%" cellpadding="0" cellspacing="0"'
                         f' style="border:1.5px solid #e8c4c4;border-radius:6px;background:#F5D5D5;margin-bottom:10px"><tr>'
                         f'<td style="padding:10px 14px;font-size:14px;color:#333;font-weight:bold;vertical-align:middle">'
                         f'Part B 总分：<b style="font-size:18px;color:{partB_accent}">{partB_score:.1f}/{partB_max}</b>'
                         f'</td></tr><tr>'
                         f'<td style="padding:4px 14px 10px;font-size:12px;color:#666;border-top:1px solid rgba(0,0,0,.08)">'
                         f'{len(rows_three)} 问 + {len(rows_five)} 答，每题 {PART_B_PER_ITEM} 分'
                         f'</td></tr></table>')

            # --- Part C（黄底；上面得分，下面匹配度+要点覆盖）---
            if has_partC and partC_score is not None:
                cov = _esc(ev["partC"].get("coverage", ""))
                h.append(f'<table width="100%" cellpadding="0" cellspacing="0"'
                         f' style="border:1.5px solid #ccc;border-radius:6px;background:#fffde7;margin-bottom:8px"><tr>'
                         f'<td style="padding:10px 14px;font-size:14px;color:#333;font-weight:bold;vertical-align:middle">'
                         f'Part C 得分：<b style="font-size:18px;color:{partC_color}">{partC_score:.1f}/{PART_C_MAX}</b>'
                         f'</td></tr><tr>'
                         f'<td style="padding:4px 14px 10px;font-size:12px;color:#666;border-top:1px solid rgba(0,0,0,.08)">'
                         f'匹配度 {partC_sim_pct:.1f}% &nbsp;&nbsp; 要点覆盖: {cov}'
                         f'</td></tr></table>')

            h.append('</td>')  # 关闭左栏

            # ═══ 右栏 (50%)：三问 → 五答 ═══
            h.append('<td valign="top" style="width:50%;padding-left:10px">')

            # --- 三问（匹配度 + 得分）---
            if rows_three:
                h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                         ' style="border:1.5px solid #e1bee7;border-radius:6px;background:#fdf2ff;margin-bottom:8px">')
                for lab, sim, pts, c in rows_three:
                    h.append(f'<tr><td style="background:#fdf2ff;border-left:5px solid #9c27b0;'
                             f'padding:7px 12px;font-size:13px">'
                             f'{lab} 匹配度：<b>{sim*100:.1f}%</b>'
                             f' &nbsp;得分：<b style="color:{c};font-size:15px">{pts:.1f}/{PART_B_PER_ITEM}</b>'
                             f'</td></tr>')
                h.append('</table>')

            # --- 五答 ---
            if rows_five:
                h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                         ' style="border:1.5px solid #a5d6a7;border-radius:6px;background:#e8f5e9">')
                for lab, sim, pts, c in rows_five:
                    h.append(f'<tr><td style="background:#e8f5e9;border-left:5px solid #4caf50;'
                             f'padding:7px 12px;font-size:13px">'
                             f'{lab} 匹配度：<b>{sim*100:.1f}%</b>'
                             f' &nbsp;得分：<b style="color:{c};font-size:15px">{pts:.1f}/{PART_B_PER_ITEM}</b>'
                             f'</td></tr>')
                h.append('</table>')

            h.append('</td></tr>')  # 关闭右栏 + 行
            h.append('</table>')  # 关闭左右双栏

            h.append('</td></tr></table>')  # 关闭外层容器

        # ═══════════════ 题目内容 — 粗边框卡片 ═══════════════
        pkg = data.get('package_data', {})
        if pkg:
            h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                     ' style="background:#fff;border:2px solid #bbb;border-radius:8px;margin-bottom:14px">'
                     '<tr><td style="background:#e8eaef;padding:10px 14px;'
                     'border-bottom:2px solid #bbb;border-radius:8px 8px 0 0">'
                     '<span style="font-size:15px;font-weight:bold;color:#222">📝 题目内容</span>'
                     '</td></tr>'
                     '<tr><td style="padding:14px 16px">')

            pa = pkg.get('partA', {})
            if pa:
                # ═══ Part A 独立卡片 ═══
                h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                         ' style="border:2px solid #3498db;border-radius:6px;margin-bottom:12px">'
                         '<tr><td style="background:#3498db;padding:7px 14px;'
                         'border-radius:4px 4px 0 0">'
                         '<span style="font-size:14px;font-weight:bold;color:#fff">📘 Part A 原文</span>'
                         '</td></tr><tr><td style="padding:10px 14px;background:#f4f8fb;'
                         'font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-all">'
                         f'{_esc(pa.get("hidden_text", "无"))}'
                         '</td></tr></table>')

            pb = pkg.get('partB', {})
            if pb:
                # ═══ Part B 独立卡片 ═══
                h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                         ' style="border:2px solid #f39c12;border-radius:6px;margin-bottom:12px">'
                         '<tr><td style="background:#f39c12;padding:7px 14px;'
                         'border-radius:4px 4px 0 0">'
                         '<span style="font-size:14px;font-weight:bold;color:#fff">📙 Part B 情景对话</span>'
                         '</td></tr><tr><td style="padding:10px 14px">')

                h.append(f'<div style="background:#fef9e7;border-left:4px solid #f1c40f;'
                         f'padding:8px 12px;margin-bottom:8px;font-size:13px;line-height:1.6">'
                         f'{_esc(pb.get("situation", "无"))}</div>')

                lt = pb.get('listening_text', '')
                if lt:
                    h.append(f'<div style="background:#eaf2f8;border-left:4px solid #3498db;'
                             f'padding:8px 12px;margin-bottom:8px;font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-all">'
                             f'{_esc(lt)}</div>')

                three_q = pb.get('three_questions', [])
                if three_q:
                    h.append('<div style="margin-bottom:6px;font-size:13px;font-weight:bold;color:#8e44ad">三问</div>')
                    h.append('<table width="100%" cellpadding="0" cellspacing="3"'
                             ' style="background:#fdf2ff;border:1px solid #e1bee7;border-radius:4px;margin-bottom:8px">')
                    for i, q in enumerate(three_q):
                        prompt = q.get('cn_prompt', '')
                        answer = q.get('hidden_answer', '')
                        h.append(f'<tr>'
                                 f'<td style="padding:5px 10px;white-space:nowrap;color:#8e44ad;font-size:12px;width:35%">'
                                 f'<b>Q{i+1}</b> {_esc(prompt)}</td>'
                                 f'<td style="padding:5px 10px;font-size:12px;line-height:1.6;border-top:1px solid #f3e5f5">'
                                 f'{_esc(answer)}</td></tr>')
                    h.append('</table>')

                five_a = pb.get('five_answers', [])
                if five_a:
                    h.append('<div style="margin-bottom:6px;font-size:13px;font-weight:bold;color:#16a085">五答</div>')
                    h.append('<table width="100%" cellpadding="0" cellspacing="3"'
                             ' style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:4px">')
                    for i, a in enumerate(five_a):
                        question = a.get('en_question', '')
                        answer = a.get('hidden_answer', '')
                        h.append(f'<tr>'
                                 f'<td style="padding:5px 10px;white-space:nowrap;color:#16a085;font-size:12px;width:35%">'
                                 f'<b>A{i+1}</b> {_esc(question)}</td>'
                                 f'<td style="padding:5px 10px;font-size:12px;line-height:1.6;border-top:1px solid #c8e6c9">'
                                 f'{_esc(answer)}</td></tr>')
                    h.append('</table>')

                h.append('</td></tr></table>')

            pc = pkg.get('partC', {})
            if pc:
                # ═══ Part C 独立卡片 ═══
                h.append('<table width="100%" cellpadding="0" cellspacing="0"'
                         ' style="border:2px solid #e74c3c;border-radius:6px;margin-bottom:8px">'
                         '<tr><td style="background:#e74c3c;padding:7px 14px;'
                         'border-radius:4px 4px 0 0">'
                         '<span style="font-size:14px;font-weight:bold;color:#fff">📕 Part C 故事复述</span>'
                         '</td></tr><tr><td style="padding:10px 14px">')
                h.append('<table width="100%" cellpadding="4" cellspacing="2">')
                h.append(f'<tr><td width="70" style="color:#666;font-size:13px">梗概</td>'
                         f'<td style="background:#f8f9fa;border-radius:4px;padding:6px 10px;'
                         f'font-size:13px;line-height:1.5">{_esc(pc.get("summary", "无"))}</td></tr>')
                h.append(f'<tr><td style="color:#666;font-size:13px">关键词</td>'
                         f'<td style="background:#f8f9fa;border-radius:4px;padding:6px 10px;'
                         f'font-size:13px;line-height:1.5">{_esc(pc.get("keywords", "无"))}</td></tr>')
                story = pc.get('tts_text', '')
                if story:
                    h.append(f'<tr><td style="vertical-align:top;color:#666;font-size:13px">故事全文</td>'
                             f'<td style="background:#f8f9fa;border-radius:4px;padding:8px 12px;'
                             f'font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-all">'
                             f'{_esc(story)}</td></tr>')
                kp = pc.get('key_points', '')
                if kp:
                    h.append(f'<tr><td style="vertical-align:top;color:#666;font-size:13px">答案要点</td>'
                             f'<td style="background:#fffde7;border-radius:4px;padding:8px 12px;'
                             f'font-size:13px;line-height:1.6">{_esc(kp)}</td></tr>')
                h.append('</table>')
                h.append('</td></tr></table>')

            h.append('</td></tr></table>')

        h.append('<div style="margin-top:14px;text-align:center;font-size:11px;color:#aaa;padding-bottom:4px">'
                 '⚠ 批改结果基于离线语音识别，仅供参考。')

        self._show_detail_dialog('\n'.join(h), is_html=True)

    def _show_detail_dialog(self, content, is_html=False):
        dialog = QDialog(self)
        dialog.setWindowTitle("练习详情")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.resize(880, 720)
        dialog.setMinimumSize(600, 440)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        # 浅灰外框
        outer = QWidget()
        outer.setStyleSheet("background:#f5f6fa")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(14, 12, 14, 8)

        browser = QTextBrowser(outer)
        browser.setFont(QFont("Microsoft YaHei", 11))
        if is_html:
            browser.setHtml(content)
        else:
            browser.setPlainText(content)
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet("QTextBrowser { background:transparent; border:none; }")

        outer_layout.addWidget(browser)

        btn_close = QPushButton("  关闭  ", outer)
        btn_close.setFont(QFont("Microsoft YaHei", 11))
        btn_close.setMinimumHeight(36)
        btn_close.clicked.connect(dialog.close)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #5b4fcf;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                padding: 6px 28px;
            }
            QPushButton:hover { background: #4a3eb8; }
            QPushButton:pressed { background: #3d32a0; }
        """)
        outer_layout.addWidget(btn_close, alignment=Qt.AlignCenter)

        layout.addWidget(outer)
        dialog.setLayout(layout)
        dialog.exec_()


def _esc(s):
    """HTML 转义，防止 XSS 和排版破坏"""
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')\
                  .replace('"', '&quot;').replace('\n', '<br>')


def load_vosk_model():
    if not VOSK_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return Model(MODEL_PATH)

if __name__ == "__main__":
    # 启动时打印资源路径（已自动写入诊断日志）
    _lines = []
    _lines.append("=" * 60)
    _lines.append(f"冻结模式: {getattr(sys, 'frozen', False)}  |  _BASE_DIR: {_BASE_DIR}")
    _lines.append(f"_INTERNAL_DIR: {_INTERNAL_DIR}")
    _vlc_status = os.environ.get("VLC_PLUGIN_PATH", "")
    _lines.append(f"VLC  : {'本地 ' + _vlc_status if _vlc_status else '未找到本地, 注册表也未找到'}")
    _lines.append(f"Vosk : {MODEL_PATH} {'√' if os.path.isdir(MODEL_PATH) else '✗ 未找到'}")
    _lines.append(f"BGE  : {_BGE_MODEL_DIR}")
    _lines.append(f"      model_optimized.onnx: {'√' if os.path.isfile(os.path.join(_BGE_MODEL_DIR, 'model_optimized.onnx')) else '✗ 未找到'}")
    _lines.append(f"      fastembed 可用: {_FASTEMBED_AVAILABLE}")
    _lines.append(f"TTS  : edge-tts 神经语音 {'√ 可用' if EDGE_TTS_AVAILABLE else '✗ 未安装/不可用(将用系统SAPI5, 音质较差)'} ({TTS_EDGE_VOICE})")
    if not os.path.isdir(_BGE_MODEL_DIR):
        _lines.append(f"      ⚠ 模型目录不存在，Part B/C 将回退简单匹配")
    elif not _FASTEMBED_AVAILABLE:
        _lines.append(f"      ⚠ fastembed 不可用，Part B/C 将回退简单匹配")
    _lines.append("=" * 60)
    for _line in _lines:
        print(_line)

    app = QApplication(sys.argv)
    # 应用级图标：主窗口、所有对话框统一使用
    _app_icon = _resolve_resource_file("xixi.ico")
    if _app_icon:
        app.setWindowIcon(QIcon(_app_icon))
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background: #f5f6fa; }
        QLabel { color: #2c3e50; }
        QPushButton { font-size:16px; padding:10px 20px; border-radius:6px; background:#ecf0f1; border:1px solid #bdc3c7; }
        QPushButton:hover { background:#dcdde1; }
        QTextEdit, QLineEdit { border:1px solid #bdc3c7; border-radius:4px; padding:6px; }
    """)
    window = MainWindow()
    sys.exit(app.exec_())
