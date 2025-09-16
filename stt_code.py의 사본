import toga
from toga.style import Pack
from toga.style.pack import COLUMN, CENTER
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import os
import time

class STTApp(toga.App):
    def startup(self):
        self.fs = 16000  # 음성인식용 샘플링 주파수
        self.recording = False
        self.audio_buffer = []
        self.stop_event = threading.Event()

        # 상태 표시용 라벨
        self.status_label = toga.Label(
            "버튼을 눌러 편지를 작성해주세요을 시작하세요.", 
            style=Pack(margin=10, text_align='center', align_items=CENTER)
        )

        # 녹음 시작 버튼
        self.start_button = toga.Button(
            "🎙️ 녹음 시작", 
            on_press=self.start_recording, 
            style=Pack(margin=10)
        )
        # 녹음 완료 / 중지 버튼 (처음엔 비활성화)
        self.stop_button = toga.Button(
            "🛑 녹음 완료", 
            on_press=self.stop_recording, 
            style=Pack(margin=10),
            enabled=False
        )

        # 메인 박스
        box = toga.Box(style=Pack(direction=COLUMN, align_items=CENTER, margin=20))
        box.add(self.status_label)
        box.add(self.start_button)
        box.add(self.stop_button)

        self.main_window = toga.MainWindow(title="STT 음성인식 + TXT 저장")
        self.main_window.content = box
        self.main_window.show()

    def start_recording(self, widget):
        if self.recording:
            self.status_label.text = "이미 녹음 중입니다. '녹음 완료' 버튼을 눌러 중지하세요."
            return
        self.recording = True
        self.audio_buffer = []
        self.stop_event.clear()
        self.start_button.enabled = False
        self.stop_button.enabled = True
        self.status_label.text = "녹음 중... 말을 해주세요."

        threading.Thread(target=self._record_recognize_thread, daemon=True).start()

    def stop_recording(self, widget):
        if not self.recording:
            return
        self.status_label.text = "녹음 중지 요청 수신... 처리 중입니다."
        self.stop_event.set()  # 녹음 스레드에 종료 신호

    def _record_recognize_thread(self):
        try:
            # 콜백으로 들어오는 오디오를 버퍼에 쌓음
            def callback(indata, frames, time_info, status):
                if status:
                    pass
                self.audio_buffer.append(indata.copy())
                if self.stop_event.is_set():
                    raise sd.CallbackStop()

            # InputStream을 사용해 사용자가 중지할 때까지 녹음
            with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
                while not self.stop_event.is_set():
                    time.sleep(0.1)  # 이벤트 체크

            # 버퍼 합치기
            if not self.audio_buffer:
                self.status_label.text = "[오류] 녹음된 데이터가 없습니다."
                return
            audio = np.concatenate(self.audio_buffer, axis=0)

            # float -> int16 변환 (클리핑)
            audio_clipped = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)

            # 임시 WAV 파일로 저장
            wav_filename = "temp_record.wav"
            write(wav_filename, self.fs, audio_int16)

            # STT
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_filename) as source:
                audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data, language="ko-KR")
            except sr.UnknownValueError:
                text = "[인식 실패: 음성을 이해하지 못했습니다]"
            except sr.RequestError as e:
                text = f"[인식 실패: 서비스 오류 {e}]"

            # 결과 텍스트 파일로 저장
            self._save_to_txt(text)

            # 상태 업데이트
            self.status_label.text = f"인식 결과:\n{text}"

            # 임시 파일 삭제
            if os.path.exists(wav_filename):
                os.remove(wav_filename)

        except Exception as e:
            if isinstance(e, sd.CallbackStop):
                pass
            else:
                self.status_label.text = f"오류 발생: {e}"
        finally:
            self.recording = False
            self.start_button.enabled = True
            self.stop_button.enabled = False

    def _save_to_txt(self, text):
        txt_dir = "stt_results"
        os.makedirs(txt_dir, exist_ok=True)
        safe_ts = time.strftime("%Y-%m-%d_%H%M%S")  # 콜론 제거하고 언더스코어로 구분
        filename = f"results_{safe_ts}.txt"
        txt_file = os.path.join(txt_dir, filename)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(txt_file, mode='w', encoding='utf-8') as f:
            f.write(f"timestamp: {timestamp}\n")
            f.write("recognized_text:\n")
            f.write(text + "\n")

def main():
    return STTApp("STT 음성인식 + TXT 저장", "org.example.sttapp")

if __name__ == "__main__":
    app = main()
    app.main_loop()
