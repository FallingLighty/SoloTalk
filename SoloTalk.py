#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoloTalk 1.0 — 单机版听说模考编辑器
依赖安装：pip install PyQt5 pyttsx3 sounddevice vosk python-vlc numpy scipy
Vosk 英语模型请下载并解压到本脚本同级目录，或修改 MODEL_PATH 变量。
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
from pathlib import Path

import pyttsx3
import sounddevice as sd
import numpy as np
from scipy.io import wavfile as wav_write
import vlc

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ------------------ 全局配置 ------------------
MODEL_PATH = "vosk-model-small-en-us-0.15"
RECORDINGS_DIR = "recordings"
HISTORY_DIR = "history"
SAMPLE_RATE = 16000
TTS_TIMEOUT = 30

# ------------------ TTS 工具（每次创建独立引擎）------------------
def tts_speak_blocking(text, stop_event=None):
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        if stop_event:
            def timeout_monitor():
                if stop_event.wait(TTS_TIMEOUT):
                    try:
                        engine.stop()
                    except:
                        pass
            threading.Thread(target=timeout_monitor, daemon=True).start()
        engine.runAndWait()
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

def keyword_coverage(text, keywords_str):
    keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return len(found), len(keywords), found

def recognize_audio_file(filepath, model):
    if not model or not os.path.exists(filepath):
        return ""
    wf = wave.open(filepath, 'rb')
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

# ------------------ 数据模型 ------------------
class SoloPackage:
    def __init__(self):
        self.meta = {
            "name": "Untitled",
            "version": "1.0",
            "created": datetime.datetime.now().isoformat(),
            "author": "",
            "anonymous": False
        }
        self.partA_video_path = None
        self.partA_hidden_text = ""
        self.partB_situation = ""
        self.partB_listening_text = ""
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
                "three_questions": self.partB_three_questions,
                "five_answers": self.partB_five_answers
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
        self.partB_three_questions = pb.get("three_questions", [])
        self.partB_five_answers = pb.get("five_answers", [])
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
        self.partB_recordings = []
        self.partC_recording = None
        self.evaluation = {}

# ------------------ 主窗口 ------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoloTalk 1.0 - 单机版听说模考编辑器")
        self.setMinimumSize(1024, 700)

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
        # 如果当前在练习页面，先执行放弃考试
        if self.central.currentWidget() == self.practice_page:
            self.practice_page.abort_exam()
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
        btn_practice.clicked.connect(self.load_and_practice)
        btn_history = QPushButton("📋  历史记录")
        btn_history.setStyleSheet(btn_style)
        btn_history.clicked.connect(lambda: self.main.go_to(self.main.history_page))
        layout.addWidget(btn_editor, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_practice, alignment=Qt.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(btn_history, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def load_and_practice(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 .solo 文件", "", "SoloTalk 文件 (*.solo)")
        if path:
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    with zf.open('data.json') as f:
                        data = json.load(f)
                    temp_dir = tempfile.mkdtemp(prefix="solotalk_")
                    zf.extractall(temp_dir)
                    self.main.current_package.from_dict(data, base_dir=temp_dir)
                    self.main.current_package.meta['temp_dir'] = temp_dir
                self.main.go_to(self.main.practice_page)
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
                if self.pkg.partC_source_type == "audio" and self.pkg.partC_audio_path and os.path.exists(self.pkg.partC_audio_path):
                    zf.write(self.pkg.partC_audio_path, os.path.basename(self.pkg.partC_audio_path))
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
        self.setup_ui()

    def setup_ui(self):
        scroll = QScrollArea()
        widget = QWidget()
        form = QFormLayout()
        self.situation_edit = QTextEdit()
        self.situation_edit.setPlaceholderText("情景介绍（学生自读）")
        form.addRow("情景介绍：", self.situation_edit)
        self.listen_edit = QTextEdit()
        self.listen_edit.setPlaceholderText("三问前听力文本（电脑 TTS 朗读）")
        form.addRow("听力文本：", self.listen_edit)

        self.three_group = QGroupBox("三问设置（每题）")
        three_layout = QVBoxLayout()
        self.three_widgets = []
        for i in range(3):
            g = QGroupBox(f"第{i+1}问")
            inner = QFormLayout()
            cn = QLineEdit(); cn.setPlaceholderText("中文提示")
            en = QLineEdit(); en.setPlaceholderText("英文机答 (TTS)")
            hid = QLineEdit(); hid.setPlaceholderText("标准答案(隐藏)")
            inner.addRow("中文提示：", cn)
            inner.addRow("英文机答：", en)
            inner.addRow("标准答案：", hid)
            g.setLayout(inner)
            three_layout.addWidget(g)
            self.three_widgets.append((cn, en, hid))
        self.three_group.setLayout(three_layout)
        form.addRow(self.three_group)

        self.five_group = QGroupBox("五答设置（每题）")
        five_layout = QVBoxLayout()
        self.five_widgets = []
        for i in range(5):
            g = QGroupBox(f"第{i+1}答")
            inner = QFormLayout()
            q = QLineEdit(); q.setPlaceholderText("英文问题 (TTS)")
            hid = QLineEdit(); hid.setPlaceholderText("标准答案(隐藏)")
            inner.addRow("英文问题：", q)
            inner.addRow("标准答案：", hid)
            g.setLayout(inner)
            five_layout.addWidget(g)
            self.five_widgets.append((q, hid))
        self.five_group.setLayout(five_layout)
        form.addRow(self.five_group)
        widget.setLayout(form)
        scroll.setWidget(widget)
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def save_to_pkg(self):
        self.pkg.partB_situation = self.situation_edit.toPlainText()
        self.pkg.partB_listening_text = self.listen_edit.toPlainText()
        self.pkg.partB_three_questions = []
        for cn, en, hid in self.three_widgets:
            self.pkg.partB_three_questions.append({
                "cn_prompt": cn.text(), "en_answer": en.text(), "hidden_answer": hid.text()
            })
        self.pkg.partB_five_answers = []
        for q, hid in self.five_widgets:
            self.pkg.partB_five_answers.append({
                "en_question": q.text(), "hidden_answer": hid.text()
            })

    def refresh(self):
        self.situation_edit.setPlainText(self.pkg.partB_situation)
        self.listen_edit.setPlainText(self.pkg.partB_listening_text)
        for i, (cn, en, hid) in enumerate(self.three_widgets):
            if i < len(self.pkg.partB_three_questions):
                q = self.pkg.partB_three_questions[i]
                cn.setText(q.get("cn_prompt","")); en.setText(q.get("en_answer","")); hid.setText(q.get("hidden_answer",""))
            else:
                cn.setText(""); en.setText(""); hid.setText("")
        for i, (q, hid) in enumerate(self.five_widgets):
            if i < len(self.pkg.partB_five_answers):
                a = self.pkg.partB_five_answers[i]
                q.setText(a.get("en_question","")); hid.setText(a.get("hidden_answer",""))
            else:
                q.setText(""); hid.setText("")

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
        self.tts_edit.setPlaceholderText("故事全文（电脑朗读）")
        layout.addRow("故事全文：", self.tts_edit)

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
        self.tts_edit.setVisible(is_tts)
        self.audio_label.setVisible(not is_tts)
        self.audio_btn.setVisible(not is_tts)

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

# ------------------ 练习模式（增强）------------------
class PracticePage(QWidget):
    signal_update_display = pyqtSignal(str, str)
    signal_tts_ready = pyqtSignal()
    signal_tts_next = pyqtSignal()
    signal_finished = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.pkg = None
        self.session = None
        self.vlc_instance = vlc.Instance("--no-video-title-show")
        self.player = self.vlc_instance.media_player_new()
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_tick)
        self._current_phase = None
        self._phase_end_time = 0
        self._timer_callback = None
        self._audio_frames = []
        self._stream = None
        self._is_recording = False
        self._tts_stop_event = None
        self._tts_thread = None
        self._tts_next_callback = None
        self.player.event_manager().event_attach(vlc.EventType.MediaPlayerEndReached, self._on_vlc_end)

        self.setup_ui()
        self.signal_update_display.connect(self._update_display)
        self.signal_tts_ready.connect(self._prepare_tts_done)
        self.signal_tts_next.connect(self._on_tts_next)
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

        self.video_container = QFrame()
        self.video_container.setStyleSheet("background:black;")
        self.video_frame = QFrame(self.video_container)
        self.video_frame.setStyleSheet("background:black;")
        self.video_frame.setGeometry(0, 0, self.video_container.width(), self.video_container.height())
        self.recording_overlay = QLabel("🔴 录音中...", self.video_container)
        self.recording_overlay.setAlignment(Qt.AlignCenter)
        self.recording_overlay.setStyleSheet("color:white; font-size:28px; background:rgba(0,0,0,150); border-radius:10px;")
        self.recording_overlay.setFixedSize(200, 60)
        self.recording_overlay.move((self.video_container.width()-200)//2, 20)
        self.recording_overlay.hide()
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("font-size:18px;")
        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.video_container)
        self.display_stack.addWidget(self.text_display)

        layout.addWidget(self.main_label)
        layout.addWidget(self.sub_label)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.display_stack)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始模考")
        self.start_btn.clicked.connect(self.start_exam)
        btn_back = QPushButton("返回主页")
        btn_back.clicked.connect(self.abort_exam)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(btn_back)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.video_container.resizeEvent = self._resize_video

    def _resize_video(self, event):
        self.video_frame.resize(self.video_container.size())
        self.recording_overlay.move((self.video_container.width()-200)//2, 20)

    def abort_exam(self):
        self.timer.stop()
        # 强制停止 TTS 线程
        self._cancel_tts()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=0.5)
        # 同步停止 VLC
        self.player.stop()
        self.player.audio_set_volume(100)
        self.stop_recording()
        self.countdown_label.hide()
        self.main.go_to(self.main.home_page)

    def start_exam(self):
        self.pkg = self.main.current_package
        self.session = PracticeSession(self.pkg.meta.get("name", "练习"))
        self.start_btn.setEnabled(False)
        self._prepare_phase()

    def _prepare_phase(self):
        author = self.pkg.meta.get("author", "")
        author_text = f"题目作者：{author}" if author else "题目作者：未知"
        self._prepare_text = f"{author_text}\n\n生活就像海洋，只有意志坚强的人才能到达彼岸。\nThis is an apple, I like apples, apples are good for our health."
        self.signal_update_display.emit("准备阶段", self._prepare_text)
        self._start_tts(
            "生活就像海洋，只有意志坚强的人才能到达彼岸。 This is an apple, I like apples, apples are good for our health.",
            self.signal_tts_ready
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

    # ---- TTS 管理 ----
    def _start_tts(self, text, signal=None):
        self._cancel_tts()
        self._tts_stop_event = threading.Event()
        self._tts_thread = threading.Thread(target=self._tts_runner, args=(text, signal))
        self._tts_thread.daemon = True
        self._tts_thread.start()

    def _tts_runner(self, text, signal):
        tts_speak_blocking(text, self._tts_stop_event)
        if signal:
            signal.emit()

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
        media = self.vlc_instance.media_new(video_path)
        media.parse()
        dur_ms = media.get_duration()
        if dur_ms < 0:
            dur_ms = 60000
        dur_sec = dur_ms / 1000.0
        self.video_duration = int(dur_sec) if dur_sec == int(dur_sec) else int(dur_sec) + 1
        self._exec_partA_step()

    def _exec_partA_step(self):
        idx = self._step_index
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
        self._exec_partB_step()

    def _exec_partB_step(self):
        s = self._b_step
        if s == 0:
            self.signal_update_display.emit("Part B Role Play",
                "In this part, you are required to act as a role and complete the following tasks.")
            self._set_tts_callback(self._next_partB_step)
            self._start_tts("Part B Role Play. In this part, you are required to act as a role and complete the following tasks.", self.signal_tts_next)
        elif s == 1:
            self.signal_update_display.emit("情景介绍", self.pkg.partB_situation or "(无)")
            self._set_timer(20)
        elif s == 2:
            self.signal_update_display.emit("听对话", self.pkg.partB_situation or "(情景)")
            self._show_text(self.pkg.partB_situation or "")
            self._set_tts_callback(self._next_partB_step)
            self._start_tts(self.pkg.partB_listening_text, self.signal_tts_next)
        elif 3 <= s <= 5:
            idx = s - 3
            if idx < len(self.pkg.partB_three_questions):
                self._do_three_question(idx)
            else:
                self._next_partB_step()
        elif 6 <= s <= 10:
            idx = s - 6
            if idx < len(self.pkg.partB_five_answers):
                self._do_five_answer(idx)
            else:
                self._next_partB_step()
        else:
            self._run_part_c()

    def _do_three_question(self, idx):
        q = self.pkg.partB_three_questions[idx]
        self.signal_update_display.emit(f"准备提问 {idx+1}", q['cn_prompt'])
        self._set_timer(15, callback=lambda: self._record_three_q(idx))

    def _record_three_q(self, idx):
        q = self.pkg.partB_three_questions[idx]
        self.signal_update_display.emit(f"请提问 {idx+1}", q['cn_prompt'])
        self._start_recording()
        self._set_timer(8, callback=lambda: self._finish_three_q(idx))

    def _finish_three_q(self, idx):
        self._stop_recording()
        self.session.partB_recordings.append(self._save_recording(f"PartB_Q{idx+1}"))
        q = self.pkg.partB_three_questions[idx]
        self.signal_update_display.emit("电脑回答", "")
        self._set_tts_callback(self._next_partB_step)
        self._start_tts(q['en_answer'], self.signal_tts_next)

    def _do_five_answer(self, idx):
        a = self.pkg.partB_five_answers[idx]
        self.signal_update_display.emit("Now answer the questions.", "")
        self._set_tts_callback(lambda id=idx: self._five_second_time(id))
        self._start_tts(a['en_question'], self.signal_tts_next)

    def _five_second_time(self, idx):
        a = self.pkg.partB_five_answers[idx]
        self.signal_update_display.emit("重复提问", "")
        self._set_tts_callback(lambda id=idx: self._prep_five_answer(id))
        self._start_tts(a['en_question'], self.signal_tts_next)

    def _prep_five_answer(self, idx):
        self.signal_update_display.emit("准备回答", "")
        self._set_timer(1, callback=lambda: self._record_five_a(idx))

    def _record_five_a(self, idx):
        self.signal_update_display.emit("请回答", "")
        self._start_recording()
        self._set_timer(8, callback=lambda: self._finish_five_a(idx))

    def _finish_five_a(self, idx):
        self._stop_recording()
        self.session.partB_recordings.append(self._save_recording(f"PartB_A{idx+1}"))
        self._set_timer(1, callback=self._next_partB_step)

    def _next_partB_step(self):
        self._b_step += 1
        self._exec_partB_step()

    # ---------- Part C ----------
    def _run_part_c(self):
        self._current_phase = "partC"
        self._c_step = 0
        self._audio_callback = None
        self._exec_partC_step()

    def _exec_partC_step(self):
        s = self._c_step
        if s == 0:
            self.signal_update_display.emit("Part C Retelling",
                "In this part, you are required to listen to a monologue and retell what you have heard.")
            self._set_tts_callback(self._next_partC_step)
            self._start_tts("Part C Retelling. In this part, you are required to listen to a monologue and retell what you have heard.", self.signal_tts_next)
        elif s == 1:
            text = f"梗概：{self.pkg.partC_summary}\n关键词：{self.pkg.partC_keywords}"
            self.signal_update_display.emit("阅读梗概和关键词", text)
            self._set_timer(20)
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
                    self.player.audio_set_volume(100)
                    self.player.play()
                else:
                    QMessageBox.warning(self, "警告", "Part C 音频文件不存在")
                    self._next_partC_step()
        elif s == 3:
            self._set_timer(1, self._next_partC_step)
        elif s == 5:
            self.signal_update_display.emit("准备复述 (60秒)", "")
            self._set_timer(60)
        elif s == 6:
            self.signal_update_display.emit("请复述故事", "")
            self._start_recording()
            self._set_timer(120)
        else:
            self._stop_recording()
            self.session.partC_recording = self._save_recording("PartC")
            self.signal_finished.emit()

    def _on_vlc_end(self, event):
        if self._current_phase == "partC" and self._audio_callback:
            cb = self._audio_callback
            self._audio_callback = None
            QTimer.singleShot(0, cb)

    def _next_partC_step(self):
        self._c_step += 1
        self._exec_partC_step()

    # ---------- 计时器 ----------
    def _set_timer(self, seconds, callback=None):
        self._phase_end_time = time.time() + seconds
        self._timer_callback = callback
        self.countdown_label.show()
        self.timer.start(100)

    def _on_tick(self):
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

    @pyqtSlot(str, str)
    def _update_display(self, main, sub):
        self.main_label.setText(main)
        self.sub_label.setText(sub)
        if "观看视频" in main or "模仿朗读" in main:
            self.display_stack.setCurrentWidget(self.video_container)
        else:
            self.display_stack.setCurrentWidget(self.text_display)
            self.text_display.setPlainText(sub if sub else main)

    # ---------- 视频控制 ----------
    def _play_video_full(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self.player.set_media(media)
        if sys.platform == "win32":
            self.player.set_hwnd(int(self.video_frame.winId()))
        else:
            self.player.set_nsobject(int(self.video_frame.winId()))
        self.player.audio_set_volume(100)
        self.player.play()
        self.display_stack.setCurrentWidget(self.video_container)

    def _play_video_audio_only(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self.player.set_media(media)
        self.player.audio_set_volume(100)
        self.player.play()
        self.display_stack.setCurrentWidget(self.text_display)

    def _play_video_silent(self):
        media = self.vlc_instance.media_new(self.pkg.partA_video_path)
        self.player.set_media(media)
        self.player.audio_set_volume(0)
        self.player.play()
        self.display_stack.setCurrentWidget(self.video_container)

    def _stop_video(self):
        QTimer.singleShot(0, self.player.stop)
        self.player.audio_set_volume(100)

    def _show_text(self, text):
        self.text_display.setPlainText(text)
        self.display_stack.setCurrentWidget(self.text_display)

    # ---------- 录音控制 ----------
    def _start_recording(self):
        self._audio_frames = []
        self._is_recording = True
        def callback(indata, frames, time_info, status):
            if self._is_recording:
                self._audio_frames.append(indata.copy())
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback)
        self._stream.start()
        if self.display_stack.currentWidget() == self.video_container:
            self.recording_overlay.show()
        else:
            self.display_stack.setCurrentWidget(self.text_display)
            self.sub_label.setText("🔴 录音中...")

    def _stop_recording(self):
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.recording_overlay.hide()
        if self.display_stack.currentWidget() == self.text_display:
            self.sub_label.setText("")

    def stop_recording(self):
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.recording_overlay.hide()

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

    # ---------- 考试结束与批改 ----------
    def _on_exam_finished(self):
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, "模考完成", "练习结束，即将进行离线批改。")
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
        eval_result = {}
        if self.session.partA_recording:
            hyp = recognize_audio_file(self.session.partA_recording, vosk_model)
            ref = self.pkg.partA_hidden_text
            wer, acc = levenshtein_wer(ref, hyp)
            eval_result['partA'] = {'recognized': hyp, 'wer': wer, 'accuracy': acc}
        b3 = self.session.partB_recordings[:3]
        b5 = self.session.partB_recordings[3:]
        eval_result['partB_three'] = []
        for i, rec in enumerate(b3):
            if i < len(self.pkg.partB_three_questions):
                ref = self.pkg.partB_three_questions[i].get('hidden_answer', '')
                hyp = recognize_audio_file(rec, vosk_model)
                sim = jaccard_similarity(ref, hyp)
                eval_result['partB_three'].append({'recognized': hyp, 'similarity': sim})
        eval_result['partB_five'] = []
        for i, rec in enumerate(b5):
            if i < len(self.pkg.partB_five_answers):
                ref = self.pkg.partB_five_answers[i].get('hidden_answer', '')
                hyp = recognize_audio_file(rec, vosk_model)
                sim = jaccard_similarity(ref, hyp)
                eval_result['partB_five'].append({'recognized': hyp, 'similarity': sim})
        if self.session.partC_recording:
            hyp = recognize_audio_file(self.session.partC_recording, vosk_model)
            found, total, _ = keyword_coverage(hyp, self.pkg.partC_key_points)
            eval_result['partC'] = {'recognized': hyp, 'coverage': f"{found}/{total}"}
        self.session.evaluation = eval_result

    def _save_history(self):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        history = {
            "package": self.pkg.meta.get("name", ""),
            "timestamp": self.session.timestamp.isoformat(),
            "recordings": {
                "partA": self.session.partA_recording,
                "partB": self.session.partB_recordings,
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
        vosk_model = load_vosk_model()
        if not vosk_model:
            QMessageBox.warning(self, "错误", "Vosk 模型不可用，无法重新批改")
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
        eval_result = {}
        if recs.get('partA'):
            hyp = recognize_audio_file(recs['partA'], vosk_model)
            ref = temp_pkg.partA_hidden_text
            wer, acc = levenshtein_wer(ref, hyp)
            eval_result['partA'] = {'recognized': hyp, 'wer': wer, 'accuracy': acc}
        b3 = recs.get('partB', [])[:3]
        b5 = recs.get('partB', [])[3:]
        eval_result['partB_three'] = []
        for i, rec in enumerate(b3):
            if i < len(temp_pkg.partB_three_questions):
                ref = temp_pkg.partB_three_questions[i].get('hidden_answer', '')
                hyp = recognize_audio_file(rec, vosk_model)
                sim = jaccard_similarity(ref, hyp)
                eval_result['partB_three'].append({'recognized': hyp, 'similarity': sim})
        eval_result['partB_five'] = []
        for i, rec in enumerate(b5):
            if i < len(temp_pkg.partB_five_answers):
                ref = temp_pkg.partB_five_answers[i].get('hidden_answer', '')
                hyp = recognize_audio_file(rec, vosk_model)
                sim = jaccard_similarity(ref, hyp)
                eval_result['partB_five'].append({'recognized': hyp, 'similarity': sim})
        if recs.get('partC'):
            hyp = recognize_audio_file(recs['partC'], vosk_model)
            found, total, _ = keyword_coverage(hyp, temp_pkg.partC_key_points)
            eval_result['partC'] = {'recognized': hyp, 'coverage': f"{found}/{total}"}
        data['evaluation'] = eval_result
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        QMessageBox.information(self, "完成", "重新批改已完成")
        self.load_history()

    def view_detail(self, item):
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        detail = f"题目包：{data.get('package','')}\n时间：{data.get('timestamp','')}\n\n录音文件：\n"
        rec = data.get('recordings', {})
        detail += f"Part A: {rec.get('partA','')}\n"
        for i, r in enumerate(rec.get('partB', [])):
            detail += f"Part B 录音{i+1}: {r}\n"
        detail += f"Part C: {rec.get('partC','')}\n\n批改结果：\n"
        ev = data.get('evaluation', {})
        if 'partA' in ev:
            detail += f"Part A 准确率: {ev['partA'].get('accuracy',0)*100:.1f}%  WER: {ev['partA'].get('wer',0):.2f}\n"
        if 'partB_three' in ev:
            for i, t in enumerate(ev['partB_three']):
                detail += f"三问{i+1} 相似度: {t.get('similarity',0)*100:.1f}%\n"
        if 'partB_five' in ev:
            for i, fv in enumerate(ev['partB_five']):
                detail += f"五答{i+1} 相似度: {fv.get('similarity',0)*100:.1f}%\n"
        if 'partC' in ev:
            detail += f"Part C 要点覆盖: {ev['partC'].get('coverage','')}\n"
        # 显示题目完整内容
        pkg = data.get('package_data', {})
        if pkg:
            detail += "\n【题目内容】\n"
            pa = pkg.get('partA', {})
            detail += f"Part A 原文: {pa.get('hidden_text', '无')}\n"
            pb = pkg.get('partB', {})
            detail += f"Part B 情景: {pb.get('situation', '无')}\n"
            detail += f"Part B 听力文本: {pb.get('listening_text', '无')}\n"
            for i, q in enumerate(pb.get('three_questions', [])):
                detail += f"  三问{i+1}: {q.get('cn_prompt','')} | 答案: {q.get('hidden_answer','')}\n"
            for i, a in enumerate(pb.get('five_answers', [])):
                detail += f"  五答{i+1}: {a.get('en_question','')} | 答案: {a.get('hidden_answer','')}\n"
            pc = pkg.get('partC', {})
            detail += f"Part C 梗概: {pc.get('summary', '无')}\n"
            detail += f"Part C 关键词: {pc.get('keywords', '无')}\n"
            if pc.get('source_type') == 'tts':
                detail += f"Part C 故事全文: {pc.get('tts_text', '无')}\n"
            else:
                detail += f"Part C 音频文件: {pc.get('audio', '无')}\n"
            detail += f"Part C 答案要点: {pc.get('key_points', '无')}\n"
        detail += "\n批改结果基于离线语音识别，仅供参考。"
        QMessageBox.information(self, "练习详情", detail)

def load_vosk_model():
    if not VOSK_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return Model(MODEL_PATH)

if __name__ == "__main__":
    app = QApplication(sys.argv)
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