import re
import io
from pathlib import Path

import easyocr
import fitz
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Распознавание договоров", page_icon="📄", layout="wide")

st.title("📄 Распознавание данных из договора")
st.write(
    "Загрузите скан договора в PDF, JPG или PNG. Программа распознает текст "
    "с помощью OCR и извлечет имя, фамилию, номер договора и телефон."
)

@st.cache_resource
def get_reader():
    # При первом запуске EasyOCR загрузит необходимые веса нейросети.
    return easyocr.Reader(["ru", "en"], gpu=False)

def file_to_images(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        pdf = fitz.open(stream=data, filetype="pdf")
        images = []
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return images

    return [Image.open(io.BytesIO(data)).convert("RGB")]

def run_ocr(images, reader):
    page_texts = []
    for image in images:
        result = reader.readtext(image, detail=0, paragraph=True)
        page_texts.append("\n".join(result))
    return "\n".join(page_texts)

def clean_value(value):
    value = re.sub(r"\s+", " ", value).strip(" \t\n:№,;.")
    return value if value else None

def extract_after_labels(text, labels, stop_labels=None):
    stop_labels = stop_labels or (
        r"номер договора|договор №|телефон|контактный телефон|"
        r"фамилия|имя|паспорт|адрес|дата рождения|"
        r"заказчик|исполнитель|подпись"
    )
    label_pattern = "|".join(labels)
    pattern = (
        rf"(?:{label_pattern})\s*[:№\-]?\s*"
        rf"([А-ЯЁA-Z0-9][А-ЯЁа-яёA-Za-z0-9 №/\-+.()]{1,70}?)"
        rf"(?=\s+(?:{stop_labels})\b|[\n\r]|$)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return clean_value(match.group(1)) if match else None

def extract_fields(text):
    normalized = re.sub(r"[ \t]+", " ", text)

    contract_number = extract_after_labels(
        normalized,
        [r"номер\s+договора", r"договор\s*№", r"№\s*договора", r"контракт\s*№"]
    )

    surname = extract_after_labels(
        normalized,
        [r"фамилия", r"ф\.?\s*и\.?\s*о\.?\s*фамилия"]
    )
    first_name = extract_after_labels(
        normalized,
        [r"имя", r"ф\.?\s*и\.?\s*о\.?\s*имя"]
    )

    # Если поля ФИО подписаны одной меткой, пробуем разобрать ФИО целиком.
    if not surname or not first_name:
        fio = extract_after_labels(normalized, [r"фио", r"ф\.?\s*и\.?\s*о\.?"])
        if fio:
            parts = fio.split()
            if len(parts) >= 2:
                surname = surname or parts[0]
                first_name = first_name or parts[1]

    phone_match = re.search(
        r"(?<!\w)(?:\+7|8|7)\s*[\-(]?\s*\d{3}\s*[\)-]?\s*\d{3}"
        r"[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)",
        normalized
    )
    phone = clean_value(phone_match.group(0)) if phone_match else None

    return {
        "Имя": first_name,
        "Фамилия": surname,
        "Номер договора": contract_number,
        "Контактный телефон": phone,
    }

uploaded_files = st.file_uploader(
    "Выберите один или несколько договоров",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Распознать документы", type="primary"):
    reader = get_reader()
    rows = []
    with st.spinner("Распознаём документы. При первом запуске загрузка OCR-модели может занять некоторое время..."):
        for uploaded in uploaded_files:
            try:
                images = file_to_images(uploaded)
                text = run_ocr(images, reader)
                fields = extract_fields(text)
                rows.append({"Файл": uploaded.name, **fields, "Статус": "Обработан"})
                with st.expander(f"Распознанный текст: {uploaded.name}"):
                    st.text(text if text else "Текст не распознан. Попробуйте скан более высокого качества.")
            except Exception as exc:
                rows.append({
                    "Файл": uploaded.name,
                    "Имя": None,
                    "Фамилия": None,
                    "Номер договора": None,
                    "Контактный телефон": None,
                    "Статус": f"Ошибка обработки: {exc}"
                })

    result_df = pd.DataFrame(rows)
    st.session_state["result_df"] = result_df

if "result_df" in st.session_state:
    st.subheader("Результаты распознавания")
    st.caption("Проверьте извлечённые сведения: OCR может ошибаться, особенно на некачественных сканах.")
    edited = st.data_editor(st.session_state["result_df"], use_container_width=True, num_rows="dynamic")
    csv_data = edited.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Скачать таблицу CSV",
        data=csv_data,
        file_name="contract_data.csv",
        mime="text/csv"
    )

with st.expander("О программе и ограничениях"):
    st.write(
        "Для распознавания текста используется EasyOCR — OCR-библиотека на основе "
        "нейросетевых моделей. После распознавания программа ищет значения по "
        "текстовым меткам и шаблону телефонного номера. Формы договоров могут "
        "отличаться, поэтому результат необходимо проверять вручную. "
        "Программа не является юридическим средством проверки договора."
    )
