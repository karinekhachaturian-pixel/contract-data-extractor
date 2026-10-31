import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import easyocr
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(page_title="Распознавание паспортов РФ", page_icon="🪪", layout="wide")
st.title("Распознавание паспортов граждан РФ")
st.write("Загрузите скан или фотографию разворота паспорта. Проверьте извлечённые значения перед использованием.")

st.warning(
    "Учебный прототип. OCR может ошибаться и не подтверждает подлинность документа. "
    "Используйте только вымышленные или обезличенные образцы. Не загружайте реальные паспорта "
    "в публичное приложение без надлежащей защиты и законного основания."
)

@st.cache_resource
def get_reader():
    return easyocr.Reader(["ru", "en"], gpu=False)

def to_images(uploaded):
    raw = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".pdf":
        pdf = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return pages
    return [Image.open(io.BytesIO(raw)).convert("RGB")]

def enhance(img, scale=2):
    img = img.convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = img.filter(ImageFilter.SHARPEN)
    return img.convert("RGB")

def ocr_image(reader, img, rotations=(0,)):
    outputs = []
    for angle in rotations:
        current = img.rotate(angle, expand=True) if angle else img
        arr = np.array(enhance(current, 2))
        result = reader.readtext(arr, detail=1, paragraph=False)
        for item in result:
            if len(item) >= 3 and str(item[1]).strip():
                outputs.append((str(item[1]).strip(), float(item[2])))
    return outputs

def normalize_text(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def best_text(items):
    return "\n".join(t for t, _ in items)

def crop_box(img, box):
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))

def parse_date(text):
    m = re.search(r"(?<!\d)(\d{2})[.\-/ ](\d{2})[.\-/ ](\d{4})(?!\d)", text)
    if not m:
        return None
    try:
        return datetime.strptime(".".join(m.groups()), "%d.%m.%Y").strftime("%d.%m.%Y")
    except ValueError:
        return None

def field_value(items):
    return normalize_text(" ".join(t for t, _ in items)) or None

def parse_domestic_spread(img, reader):
    # Coordinates are approximate zones for a photographed open internal passport.
    # Each zone is OCR'd independently to reduce interference from stamps and background.
    issuer_zone = crop_box(img, (0.04, 0.07, 0.90, 0.24))
    issue_date_zone = crop_box(img, (0.08, 0.22, 0.48, 0.31))
    serial_zone = crop_box(img, (0.89, 0.08, 0.995, 0.91))
    surname_zone = crop_box(img, (0.32, 0.53, 0.90, 0.62))
    name_zone = crop_box(img, (0.32, 0.62, 0.90, 0.69))
    patronymic_zone = crop_box(img, (0.32, 0.67, 0.90, 0.74))
    gender_birth_zone = crop_box(img, (0.32, 0.70, 0.91, 0.79))

    issuer_items = ocr_image(reader, issuer_zone)
    issue_items = ocr_image(reader, issue_date_zone)
    # Series/number are printed vertically along the right edge on many internal passports.
    serial_items = ocr_image(reader, serial_zone, rotations=(0, 90, 270))
    surname_items = ocr_image(reader, surname_zone)
    name_items = ocr_image(reader, name_zone)
    patronymic_items = ocr_image(reader, patronymic_zone)
    gender_birth_items = ocr_image(reader, gender_birth_zone)

    issuer_text = best_text(issuer_items)
    issue_text = best_text(issue_items)
    serial_text = best_text(serial_items)
    surname_text = best_text(surname_items)
    name_text = best_text(name_items)
    patronymic_text = best_text(patronymic_items)
    gender_birth_text = best_text(gender_birth_items)

    # Extract dates only from their expected regions.
    issue_date = parse_date(issue_text)
    birth_date = parse_date(gender_birth_text)

    # Passport series and number: digits often appear as two groups of 2 + 6.
    serial_digits = re.sub(r"\D", "", serial_text)
    series = number = None
    if len(serial_digits) >= 10:
        series = serial_digits[:4]
        number = serial_digits[4:10]
    elif len(serial_digits) >= 4:
        series = serial_digits[:4]
        if len(serial_digits) >= 6:
            number = serial_digits[4:10] or None

    # Try remove printed field labels and keep only the most plausible name text.
    def clean_name(s, labels):
        s = re.sub(r"(?i)\b(" + "|".join(labels) + r")\b", " ", s)
        s = re.sub(r"[^А-ЯЁа-яёA-Za-z\-\s]", " ", s)
        words = [x for x in normalize_text(s).split() if len(x) > 1]
        return " ".join(words[:3]) if words else None

    surname = clean_name(surname_text, ["фамилия"])
    first_name = clean_name(name_text, ["имя"])
    patronymic = clean_name(patronymic_text, ["отчество"])

    # Gender may be printed as МУЖ. or ЖЕН.
    g = gender_birth_text.upper()
    if re.search(r"\bМУЖ\b|МУЖСК", g):
        gender = "Мужской"
    elif re.search(r"\bЖЕН\b|ЖЕНСК", g):
        gender = "Женский"
    else:
        gender = None

    issuer = re.sub(r"(?i)паспорт выдан|кем выдан|орган,? выдавший", " ", issuer_text)
    issuer = normalize_text(re.sub(r"[^А-ЯЁа-яёA-Za-z0-9№.,\- ]", " ", issuer))
    issuer = issuer or None

    # No printed expiry date on the internal Russian passport.
    return {
        "Фамилия": surname,
        "Имя": first_name,
        "Отчество": patronymic,
        "Пол": gender,
        "Серия документа": series,
        "Номер документа": number,
        "Кем выдан (государственный орган)": issuer,
        "Дата выдачи": issue_date,
        "Дата рождения": birth_date,
        "Дата окончания действия": "Не указана в документе",
        "Источник": "OCR по областям; требуется проверка",
        "_Текст OCR": {
            "Орган выдачи": issuer_text,
            "Дата выдачи": issue_text,
            "Серия/номер": serial_text,
            "Фамилия": surname_text,
            "Имя": name_text,
            "Отчество": patronymic_text,
            "Пол/дата рождения": gender_birth_text,
        }
    }

def parse_foreign(lines):
    # Fallback OCR-based extraction for foreign passport. MRZ parser can be extended
    # separately; this basic version keeps uncertain fields blank rather than guessing.
    text = "\n".join(lines)
    mrz_lines = [re.sub(r"[^A-Z0-9<]", "", x.upper()) for x in lines]
    mrz = None
    for i in range(len(mrz_lines)-1):
        a, b = mrz_lines[i], mrz_lines[i+1]
        if a.startswith("P") and len(b) >= 30:
            b = (b + "<"*44)[:44]
            names = a[5:].split("<<", 1)
            surname = names[0].replace("<", " ").strip() if names else None
            given = names[1].replace("<", " ").strip() if len(names) > 1 else ""
            parts = given.split()
            def mrz_date(s, expiry=False):
                if not re.fullmatch(r"\d{6}", s): return None
                yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:])
                year = (2000 + yy)
                if expiry and year < date.today().year - 10: year += 100
                if not expiry and year > date.today().year: year -= 100
                try: return date(year, mm, dd).strftime("%d.%m.%Y")
                except ValueError: return None
            mrz = {
                "Фамилия": surname or None,
                "Имя": parts[0] if parts else None,
                "Отчество": " ".join(parts[1:]) if len(parts)>1 else None,
                "Пол": "Мужской" if b[20:21] == "M" else ("Женский" if b[20:21] == "F" else None),
                "Серия документа": None,
                "Номер документа": b[:9].replace("<", "") or None,
                "Кем выдан (государственный орган)": None,
                "Дата выдачи": None,
                "Дата рождения": mrz_date(b[13:19]),
                "Дата окончания действия": mrz_date(b[21:27], True),
                "Источник": "MRZ; требуется проверка"
            }
            break
    if mrz:
        return mrz
    return {
        "Фамилия": None, "Имя": None, "Отчество": None, "Пол": None,
        "Серия документа": None, "Номер документа": None,
        "Кем выдан (государственный орган)": None, "Дата выдачи": None,
        "Дата рождения": None, "Дата окончания действия": None,
        "Источник": "OCR не смог уверенно извлечь поля"
    }

passport_type = st.radio(
    "Выберите тип паспорта",
    ["Внутренний паспорт РФ", "Заграничный паспорт РФ"],
    horizontal=True
)

uploads = st.file_uploader(
    "Загрузите скан или фотографию паспорта",
    type=None,
    accept_multiple_files=True
)
allowed = {".png", ".jpg", ".jpeg", ".jfif", ".jpe", ".pdf"}
if uploads:
    bad = [f.name for f in uploads if Path(f.name).suffix.lower() not in allowed]
    if bad:
        st.error("Неподдерживаемый формат: " + ", ".join(bad))
        uploads = [f for f in uploads if Path(f.name).suffix.lower() in allowed]

if uploads and st.button("Распознать паспортные данные", type="primary"):
    reader = get_reader()
    rows = []
    for uploaded in uploads:
        try:
            with st.spinner(f"Обрабатываем {uploaded.name}..."):
                images = to_images(uploaded)
                if passport_type == "Внутренний паспорт РФ":
                    # The field-zone layout expects an open spread image.
                    data = parse_domestic_spread(images[0], reader)
                else:
                    all_lines = []
                    for im in images:
                        all_lines.extend(t for t, _ in ocr_image(reader, im))
                    data = parse_foreign(all_lines)
                debug = data.pop("_Текст OCR", None)
                rows.append({"Файл": uploaded.name, **data})
                if debug:
                    with st.expander(f"Промежуточное распознавание — {uploaded.name}"):
                        for label, value in debug.items():
                            st.write(f"**{label}:** {value or 'Не распознано'}")
        except Exception as exc:
            st.error(f"Не удалось обработать {uploaded.name}: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Извлечённые данные")
        st.caption("Проверьте и исправьте значения вручную. Для внутреннего паспорта дата окончания не печатается в документе.")
        edited = st.data_editor(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать результаты CSV",
            edited.to_csv(index=False).encode("utf-8-sig"),
            file_name="passport_data.csv",
            mime="text/csv"
        )

with st.expander("Ограничения и безопасность"):
    st.write(
        "Это учебный прототип, а не средство проверки подлинности документа. "
        "Распознавание по областям рассчитано на фотографию открытого разворота внутреннего паспорта "
        "с расположением страниц, похожим на тестовый шаблон. Для других ракурсов и шаблонов точность "
        "может снизиться. Загранпаспорт обрабатывается отдельно через MRZ, если зона распознана. "
        "Внутренний паспорт РФ не содержит напечатанной даты окончания действия. "
        "Не используйте результаты без ручной проверки и не загружайте реальные паспорта в публичное приложение."
    )
