from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TextClassificationPipeline

from pathlib import Path

# 모델이 실제로 저장된 깔끔한 로컬 경로 (예: 한글/공백 때문에 문제 생기면 다른 ASCII-safe 폴더로 복사)
save_dir = Path("C:/Users/user/Desktop/ai_filter/model_save_5")

# 존재 확인
if not save_dir.is_dir():
    raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {save_dir}")

# 로드
tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
model = TFAutoModelForSequenceClassification.from_pretrained(save_dir, local_files_only=True)
from transformers import TextClassificationPipeline
import re

label_map = {"LABEL_0": "도덕", "LABEL_1": "비도덕"}

#문장 구분 함수
def load_and_preprocess(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        text = f.read().replace("\n", " ").strip()
    sentences = [s for s in re.split(r'(?<=[\.!?])\s+', text) if s]
    return sentences


def classify_and_filter(sentences, classifier, threshold=0.6, max_length=128):
    results = classifier(
        sentences,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_all_scores=True
    )
    immoral = []
    for sent, res in zip(sentences, results):
        top = max(res, key=lambda x: x['score'])
        label = label_map.get(top['label'], top['label'])
        if label == "비도덕" and top['score'] >= threshold:
            immoral.append((top['score'], sent))
    immoral.sort(key=lambda x: x[0], reverse=True)
    return immoral

classifier = TextClassificationPipeline(tokenizer=tokenizer, model=model, framework='tf', return_all_scores=True)




# 텍스트 불러오기 및 전처리
# 모델이 실제로 저장된 깔끔한 로컬 경로 (예: 한글/공백 때문에 문제 생기면 다른 ASCII-safe 폴더로 복사)
txt_dir = Path("C:/Users/user/Desktop/ai_filter/txt_data/negative.txt")



sentences = load_and_preprocess(txt_dir)

# 문장 분류 및 필터링
immoral_sentences = classify_and_filter(sentences, classifier, threshold=0.6)


# 결과 출력
if not immoral_sentences:
    print("기준 이상의 비도덕 문구가 발견되지 않았습니다.")
else:
    for i, (score, sent) in enumerate(immoral_sentences[:5], 1):
        print(f"{i}. {sent} ▶▶ [{score*100:.1f}%] 비도덕성 문구가 발견됐습니다.")

