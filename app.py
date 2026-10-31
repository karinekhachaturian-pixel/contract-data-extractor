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


def read_digits_zone(reader, zone):
    """
    Читает красную вертикальную серию/номер на правом поле внутреннего паспорта.
    Возвращает найденные цифровые последовательности с оценкой уверенности.
    """
    candidates = []
    rgb = zone.convert("RGB")
    # Выделяем красную печать: у красных символов R заметно выше G и B.
    arr = np.array(rgb)
    red_mask = (
        (arr[:, :, 0] > 85)
        & (arr[:, :, 0] > arr[:, :, 1] * 1.18)
        & (arr[:, :, 0] > arr[:, :, 2] * 1.12)
    )
    if red_mask.any():
        # Белый фон, чёрные символы для OCR.
        masked = np.where(red_mask, 0, 255).astype(np.uint8)
        red_img = Image.fromarray(masked).convert("RGB")
        variants = [red_img, red_img.rotate(90, expand=True), red_img.rotate(270, expand=True)]
    else:
        variants = [rgb, rgb.rotate(90, expand=True), rgb.rotate(270, expand=True)]

    for variant in variants:
        enlarged = variant.resize((variant.width * 4, variant.height * 4), Image.Resampling.LANCZOS)
        result = reader.readtext(
            np.array(enlarged),
            detail=1,
            paragraph=False,
            allowlist="0123456789"
        )
        for item in result:
            if len(item) >= 3:
                digits = re.sub(r"\\D", "", str(item[1]))
                if len(digits) >= 4:
                    candidates.append((digits, float(item[2])))
    return candidates


def extract_code_subdivision(reader, img):
    # Код подразделения печатается справа от даты выдачи (формат XXX-XXX).
    zone = crop_box(img, (0.54, 0.195, 0.82, 0.245))
    items = ocr_image(reader, zone)
    text = " ".join(t for t, _ in items)
    match = re.search(r"(?<!\\d)(\\d{3})\\s*[-–—]?\\s*(\\d{3})(?!\\d)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}", text

    # Второй проход только по цифрам помогает, если дефис не распознан.
    arr = np.array(enhance(zone, 4))
    numeric = reader.readtext(arr, detail=1, paragraph=False, allowlist="0123456789")
    digits = "".join(re.sub(r"\\D", "", str(x[1])) for x in numeric if len(x) >= 2)
    if len(digits) >= 6:
        return f"{digits[:3]}-{digits[3:6]}", text
    return None, text


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
    """
    Распознавание открытого разворота внутреннего паспорта по узким зонам.
    Координаты рассчитаны под тестовый образец 960x1328; для других кадров
    потребуется выравнивание/настройка зон.
    """
    # Верхняя страница: только строки наименования органа, без даты и кода.
    issuer_zone = crop_box(img, (0.27, 0.085, 0.84, 0.205))
    issue_date_zone = crop_box(img, (0.13, 0.195, 0.39, 0.245))

    # Нижняя страница: вырезаем именно значения справа от подписей.
    surname_zone = crop_box(img, (0.47, 0.545, 0.78, 0.585))
    name_zone = crop_box(img, (0.49, 0.625, 0.79, 0.665))
    patronymic_zone = crop_box(img, (0.42, 0.665, 0.80, 0.705))
    gender_zone = crop_box(img, (0.35, 0.705, 0.49, 0.745))
    birth_date_zone = crop_box(img, (0.56, 0.705, 0.80, 0.75))

    # Серия и номер печатаются вертикально на правом краю разворота.
    # Обрабатываем верхний и нижний сегменты отдельно, чтобы не склеивать повторы.
    serial_top_zone = crop_box(img, (0.905, 0.105, 0.995, 0.405))
    serial_bottom_zone = crop_box(img, (0.905, 0.665, 0.995, 0.91))

    zones = {
        "Орган выдачи": issuer_zone,
        "Дата выдачи": issue_date_zone,
        "Серия/номер верх": serial_top_zone,
        "Серия/номер низ": serial_bottom_zone,
        "Фамилия": surname_zone,
        "Имя": name_zone,
        "Отчество": patronymic_zone,
        "Пол": gender_zone,
        "Дата рождения": birth_date_zone,
    }
    texts = {}
    for label, zone in zones.items():
        # Узкие области читаем с увеличением; вертикальные цифры пробуем
        # в нескольких ориентациях.
        rotations = (0, 90, 270) if "Серия/номер" in label else (0,)
        items = ocr_image(reader, zone, rotations=rotations)
        texts[label] = best_text(items)

    def clean_name(value):
        value = re.sub(r"[^А-ЯЁа-яёA-Za-z\- ]", " ", value or "")
        words = [w for w in normalize_text(value).split() if len(w) > 1]
        # Отбрасываем возможные OCR-остатки подписей, оставляя максимум 2 слова.
        ignored = {"ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОЛ", "МУЖ", "ЖЕН"}
        words = [w for w in words if w.upper() not in ignored]
        return " ".join(words[:2]) or None

    surname = clean_name(texts["Фамилия"])
    first_name = clean_name(texts["Имя"])
    patronymic = clean_name(texts["Отчество"])

    issue_date = parse_date(texts["Дата выдачи"])
    birth_date = parse_date(texts["Дата рождения"])

    g = texts["Пол"].upper()
    if re.search(r"МУЖ|Mуж", g):
        gender = "Мужской"
    elif re.search(r"ЖЕН", g):
        gender = "Женский"
    else:
        gender = None

    issuer = normalize_text(re.sub(
        r"[^А-ЯЁа-яёA-Za-z0-9№.,\- ]", " ", texts["Орган выдачи"]
    )) or None

    # Серию и номер считываем только с верхней вертикальной зоны,
    # выделяя красную печать и разрешая OCR только цифры.
    digit_candidates = read_digits_zone(reader, serial_top_zone)
    # Принимаем только полную последовательность из 10 цифр.
    # Если OCR не уверен/не нашёл 10 цифр, оставляем поля пустыми для ручной проверки.
    full_candidates = [(d[:10], conf) for d, conf in digit_candidates if len(d) >= 10]
    series = number = None
    if full_candidates:
        candidate, _confidence = max(full_candidates, key=lambda x: x[1])
        series, number = candidate[:4], candidate[4:10]

    subdivision, subdivision_debug = extract_code_subdivision(reader, img)

    return {
        "Фамилия": surname,
        "Имя": first_name,
        "Отчество": patronymic,
        "Пол": gender,
        "Серия документа": series,
        "Номер документа": number,
        "Кем выдан (государственный орган)": issuer,
        "Код подразделения": subdivision,
        "Дата выдачи": issue_date,
        "Дата рождения": birth_date,
        "Источник": "OCR по отдельным областям; требуется проверка",
        "_Текст OCR": texts
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
                "Код подразделения": None,
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
        "Кем выдан (государственный орган)": None, "Код подразделения": None, "Дата выдачи": None,
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
        st.caption("Проверьте и исправьте значения вручную. Для внутреннего паспорта отображается код подразделения; дата окончания действия не включается, поскольку она не печатается в документе.")
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
        "Для внутреннего паспорта РФ выводится код подразделения; дата окончания действия не включается, поскольку она не печатается в документе. "
        "Не используйте результаты без ручной проверки и не загружайте реальные паспорта в публичное приложение."
    )
