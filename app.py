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

# Визуальное оформление интерфейса. Логика распознавания и обработки файлов не меняется.
st.markdown("""
<style>
    /* Shared layout */
    .block-container { max-width:1450px; padding-top:2rem; padding-bottom:3rem; }
    .app-hero { position:relative; overflow:hidden; padding:1.7rem 1.9rem; border-radius:20px; margin:0 0 1.35rem; }
    .app-hero .eyebrow { font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.45rem; }
    .app-hero h1 { font-size:clamp(1.65rem,3vw,2.35rem); line-height:1.18; margin:0 0 .55rem; font-weight:750; }
    .app-hero p { margin:0; font-size:1rem; }
    .stButton>button, .stDownloadButton>button { border-radius:11px; padding:.62rem 1.15rem; font-weight:700; transition:all .15s ease; }
    .stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-1px); }

    /* Default/light appearance: warm pink and cream */
    .stApp {
        background:linear-gradient(180deg,#fffaf6 0%,#f8eef0 100%) !important;
        color:#49363d !important;
    }
    [data-testid="stHeader"] { background:rgba(255,250,246,.94) !important; }
    .stApp p, .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"] { color:#49363d; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color:#593d47 !important; }
    .app-hero {
        background:linear-gradient(120deg,#9f5d75 0%,#bd7b8e 55%,#d6a2a8 100%);
        border:1px solid #e4c4cc; box-shadow:0 10px 26px rgba(135,76,96,.15);
        color:#fffaf6 !important;
    }
    .app-hero * { color:#fffaf6 !important; }
    .app-hero .eyebrow { color:#fbecef !important; }
    .app-hero p { color:#fff5f1 !important; }
    .app-hero:after {
        content:"✦   ·   ✧   ·   ✦";
        position:absolute; right:1.5rem; top:.9rem; color:rgba(255,245,248,.8);
        font-size:1.1rem; letter-spacing:.4rem; pointer-events:none;
    }
    div[data-testid="stAlert"] { border-radius:12px; border:1px solid #e8c9cf; background:#fff1ed; color:#68464f !important; }
    div[data-testid="stAlert"] * { color:#68464f !important; }
    div[data-testid="stRadio"]>label, div[data-testid="stFileUploader"] label { color:#593d47 !important; font-weight:700; }
    div[data-testid="stFileUploader"] section { background:#fffdfb; border:1.5px dashed #d6aab5; border-radius:14px; padding:1rem; }
    div[data-testid="stFileUploader"] section * { color:#59434a !important; }
    .stButton>button[kind="primary"] { background:#ad687f !important; color:#fffaf6 !important; border:1px solid #ad687f !important; }
    .stButton>button[kind="primary"] * { color:#fffaf6 !important; }
    .stDownloadButton>button { background:#f4e4e6 !important; color:#684653 !important; border:1px solid #e4c6ce !important; }
    .stDownloadButton>button * { color:#684653 !important; }
    div[data-testid="stExpander"] { background:#fffdfb; border:1px solid #ead9dc; border-radius:12px; }
    div[data-testid="stExpander"] summary * { color:#593d47 !important; }
    [data-testid="stCaptionContainer"] { color:#806b72 !important; }
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] { border-radius:12px; padding:.35rem; }

    /*
      Streamlit may attach its active theme marker to different root containers
      depending on version. Check the document, body, app root, and descendants.
      Crucially, there is no "no data-theme" light selector that can override Dark.
    */
    :root[data-theme="dark"] .stApp,
    body[data-theme="dark"] .stApp,
    .stApp[data-theme="dark"],
    .stApp:has([data-theme="dark"]),
    :root:has([data-theme="dark"]) .stApp,
    body:has([data-theme="dark"]) .stApp {
        background-color:#100d1d !important;
        background-image:
            radial-gradient(circle at 4% 8%,#e5d9ff 0 1.2px,transparent 2px),
            radial-gradient(circle at 11% 24%,rgba(208,190,255,.85) 0 1px,transparent 2px),
            radial-gradient(circle at 18% 11%,rgba(255,255,255,.9) 0 1.3px,transparent 2px),
            radial-gradient(circle at 24% 42%,rgba(190,166,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 31% 17%,rgba(255,255,255,.8) 0 1.2px,transparent 2px),
            radial-gradient(circle at 38% 7%,rgba(208,190,255,.8) 0 1px,transparent 2px),
            radial-gradient(circle at 45% 31%,rgba(255,255,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 52% 13%,rgba(208,190,255,.85) 0 1.2px,transparent 2px),
            radial-gradient(circle at 59% 40%,rgba(255,255,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 66% 9%,rgba(208,190,255,.8) 0 1.2px,transparent 2px),
            radial-gradient(circle at 73% 23%,rgba(255,255,255,.9) 0 1px,transparent 2px),
            radial-gradient(circle at 80% 8%,rgba(208,190,255,.85) 0 1.2px,transparent 2px),
            radial-gradient(circle at 87% 34%,rgba(255,255,255,.8) 0 1px,transparent 2px),
            radial-gradient(circle at 95% 15%,rgba(208,190,255,.9) 0 1.2px,transparent 2px),
            radial-gradient(circle at 7% 59%,rgba(255,255,255,.65) 0 1px,transparent 2px),
            radial-gradient(circle at 16% 76%,rgba(208,190,255,.7) 0 1px,transparent 2px),
            radial-gradient(circle at 27% 65%,rgba(255,255,255,.7) 0 1.1px,transparent 2px),
            radial-gradient(circle at 35% 88%,rgba(208,190,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 47% 72%,rgba(255,255,255,.65) 0 1px,transparent 2px),
            radial-gradient(circle at 57% 91%,rgba(208,190,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 68% 65%,rgba(255,255,255,.7) 0 1.1px,transparent 2px),
            radial-gradient(circle at 77% 82%,rgba(208,190,255,.75) 0 1px,transparent 2px),
            radial-gradient(circle at 89% 68%,rgba(255,255,255,.7) 0 1px,transparent 2px),
            radial-gradient(circle at 97% 92%,rgba(208,190,255,.75) 0 1px,transparent 2px),
            linear-gradient(145deg,#100d1d 0%,#171127 48%,#211638 100%) !important;
        background-attachment:fixed !important;
        color:#eee9ff !important;
    }

    :root[data-theme="dark"] [data-testid="stHeader"],
    body[data-theme="dark"] [data-testid="stHeader"],
    :root:has([data-theme="dark"]) [data-testid="stHeader"],
    body:has([data-theme="dark"]) [data-testid="stHeader"] {
        background:rgba(16,13,29,.92) !important;
    }
    :root[data-theme="dark"] .stApp p,
    :root[data-theme="dark"] .stApp label,
    :root[data-theme="dark"] .stApp li,
    :root[data-theme="dark"] .stApp [data-testid="stMarkdownContainer"],
    :root:has([data-theme="dark"]) .stApp p,
    :root:has([data-theme="dark"]) .stApp label,
    :root:has([data-theme="dark"]) .stApp li,
    :root:has([data-theme="dark"]) .stApp [data-testid="stMarkdownContainer"],
    body:has([data-theme="dark"]) .stApp p,
    body:has([data-theme="dark"]) .stApp label,
    body:has([data-theme="dark"]) .stApp li,
    body:has([data-theme="dark"]) .stApp [data-testid="stMarkdownContainer"] {
        color:#eee9ff !important;
    }
    :root[data-theme="dark"] .stApp h1,
    :root[data-theme="dark"] .stApp h2,
    :root[data-theme="dark"] .stApp h3,
    :root[data-theme="dark"] .stApp h4,
    :root:has([data-theme="dark"]) .stApp h1,
    :root:has([data-theme="dark"]) .stApp h2,
    :root:has([data-theme="dark"]) .stApp h3,
    :root:has([data-theme="dark"]) .stApp h4 { color:#f4efff !important; }

    :root[data-theme="dark"] .app-hero,
    :root:has([data-theme="dark"]) .app-hero,
    body:has([data-theme="dark"]) .app-hero {
        background:linear-gradient(120deg,rgba(48,34,78,.98),rgba(79,54,126,.97) 58%,rgba(111,78,159,.96)) !important;
        border:1px solid rgba(194,170,255,.32);
        box-shadow:0 14px 38px rgba(5,3,15,.35),0 0 26px rgba(151,117,232,.16);
        color:#fff !important;
    }
    :root[data-theme="dark"] .app-hero *,
    :root:has([data-theme="dark"]) .app-hero *,
    body:has([data-theme="dark"]) .app-hero * { color:#fff !important; }
    :root[data-theme="dark"] .app-hero:after,
    :root:has([data-theme="dark"]) .app-hero:after,
    body:has([data-theme="dark"]) .app-hero:after {
        content:"✦   ·   ✧   ·   ✦   ·   ⋆   ·   ✧   ·   ✦   ·   ⋆   ·   ✧";
        position:absolute; right:1.2rem; top:.8rem; color:rgba(245,237,255,.9);
        font-size:1.25rem; letter-spacing:.32rem; pointer-events:none;
    }
    :root[data-theme="dark"] div[data-testid="stAlert"],
    :root:has([data-theme="dark"]) div[data-testid="stAlert"],
    body:has([data-theme="dark"]) div[data-testid="stAlert"] {
        background:rgba(52,39,76,.97) !important; border:1px solid #745b9d !important;
        color:#f0eaff !important; border-radius:12px;
    }
    :root[data-theme="dark"] div[data-testid="stAlert"] *,
    :root:has([data-theme="dark"]) div[data-testid="stAlert"] *,
    body:has([data-theme="dark"]) div[data-testid="stAlert"] * { color:#f0eaff !important; }

    :root[data-theme="dark"] div[data-testid="stRadio"]>label,
    :root:has([data-theme="dark"]) div[data-testid="stRadio"]>label,
    body:has([data-theme="dark"]) div[data-testid="stRadio"]>label,
    :root[data-theme="dark"] div[data-testid="stFileUploader"] label,
    :root:has([data-theme="dark"]) div[data-testid="stFileUploader"] label,
    body:has([data-theme="dark"]) div[data-testid="stFileUploader"] label {
        color:#e9e0ff !important; font-weight:700;
    }
    :root[data-theme="dark"] div[data-testid="stFileUploader"] section,
    :root:has([data-theme="dark"]) div[data-testid="stFileUploader"] section,
    body:has([data-theme="dark"]) div[data-testid="stFileUploader"] section {
        background:rgba(35,27,54,.97) !important; border:1.5px dashed #8068b2 !important;
    }
    :root[data-theme="dark"] div[data-testid="stFileUploader"] section *,
    :root:has([data-theme="dark"]) div[data-testid="stFileUploader"] section *,
    body:has([data-theme="dark"]) div[data-testid="stFileUploader"] section * { color:#eee9ff !important; }

    :root[data-theme="dark"] .stButton>button[kind="primary"],
    :root:has([data-theme="dark"]) .stButton>button[kind="primary"],
    body:has([data-theme="dark"]) .stButton>button[kind="primary"] {
        background:linear-gradient(120deg,#8b6bd1,#a184ee) !important;
        color:#fff !important; border:1px solid #b69aff !important;
    }
    :root[data-theme="dark"] .stButton>button[kind="primary"] *,
    :root:has([data-theme="dark"]) .stButton>button[kind="primary"] *,
    body:has([data-theme="dark"]) .stButton>button[kind="primary"] * { color:#fff !important; }
    :root[data-theme="dark"] .stDownloadButton>button,
    :root:has([data-theme="dark"]) .stDownloadButton>button,
    body:has([data-theme="dark"]) .stDownloadButton>button {
        background:#302448 !important; color:#eee6ff !important; border:1px solid #69558f !important;
    }
    :root[data-theme="dark"] .stDownloadButton>button *,
    :root:has([data-theme="dark"]) .stDownloadButton>button *,
    body:has([data-theme="dark"]) .stDownloadButton>button * { color:#eee6ff !important; }

    :root[data-theme="dark"] div[data-testid="stExpander"],
    :root:has([data-theme="dark"]) div[data-testid="stExpander"],
    body:has([data-theme="dark"]) div[data-testid="stExpander"] {
        background:rgba(32,25,49,.97) !important; border:1px solid #493967 !important;
    }
    :root[data-theme="dark"] div[data-testid="stExpander"] summary *,
    :root:has([data-theme="dark"]) div[data-testid="stExpander"] summary *,
    body:has([data-theme="dark"]) div[data-testid="stExpander"] summary * { color:#eee9ff !important; }
    :root[data-theme="dark"] [data-testid="stCaptionContainer"],
    :root:has([data-theme="dark"]) [data-testid="stCaptionContainer"],
    body:has([data-theme="dark"]) [data-testid="stCaptionContainer"] { color:#c9bddf !important; }

    :root[data-theme="dark"] input,
    :root[data-theme="dark"] textarea,
    :root[data-theme="dark"] [data-baseweb="select"]>div,
    :root:has([data-theme="dark"]) input,
    :root:has([data-theme="dark"]) textarea,
    :root:has([data-theme="dark"]) [data-baseweb="select"]>div,
    body:has([data-theme="dark"]) input,
    body:has([data-theme="dark"]) textarea,
    body:has([data-theme="dark"]) [data-baseweb="select"]>div {
        background-color:#2b2144 !important; color:#f3edff !important; border-color:#66528b !important;
    }
    @media(max-width:700px) {
        .block-container { padding:1rem .8rem 2rem; }
        .app-hero { padding:1.25rem; border-radius:15px; }
        .app-hero:after { right:.6rem; letter-spacing:.15rem; font-size:.95rem; }
    }
</style>
<div class="app-hero">
  <div class="eyebrow">Документы · распознавание данных</div>
  <h1>Распознавание паспортов граждан РФ</h1>
  <p>Загрузите скан или фотографию паспорта, проверьте извлечённые сведения и сохраните результат в Excel.</p>
</div>
""", unsafe_allow_html=True)

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
    Сохраняет отдельные варианты OCR и объединённую последовательность цифр.
    """
    candidates = []
    rgb = zone.convert("RGB")
    arr = np.array(rgb)

    # Выделяем красные цифры, ослабляя влияние красного фона и линий бланка.
    red_mask = (
        (arr[:, :, 0] > 75)
        & (arr[:, :, 0] > arr[:, :, 1] * 1.12)
        & (arr[:, :, 0] > arr[:, :, 2] * 1.08)
    )
    if red_mask.any():
        masked = np.where(red_mask, 0, 255).astype(np.uint8)
        base_img = Image.fromarray(masked).convert("RGB")
    else:
        base_img = rgb

    variants = [
        base_img,
        base_img.rotate(90, expand=True),
        base_img.rotate(270, expand=True),
    ]

    for variant in variants:
        enlarged = variant.resize(
            (variant.width * 5, variant.height * 5),
            Image.Resampling.LANCZOS
        )
        result = reader.readtext(
            np.array(enlarged),
            detail=1,
            paragraph=False,
            allowlist="0123456789"
        )

        # Отдельные распознанные фрагменты.
        fragments = []
        for item in result:
            if len(item) >= 3:
                digits = re.sub(r"\D", "", str(item[1]))
                if digits:
                    fragments.append((digits, float(item[2])))
                    candidates.append((digits, float(item[2])))

        # EasyOCR иногда разбивает вертикальную серию и номер на несколько
        # фрагментов. Объединяем их в порядке расположения сверху вниз/слева направо.
        if fragments:
            ordered = sorted(
                result,
                key=lambda item: (
                    (min(pt[1] for pt in item[0]) + max(pt[1] for pt in item[0])) / 2,
                    (min(pt[0] for pt in item[0]) + max(pt[0] for pt in item[0])) / 2
                )
            )
            merged = "".join(
                re.sub(r"\D", "", str(item[1]))
                for item in ordered
                if len(item) >= 3
            )
            merged = re.sub(r"\D", "", merged)
            if len(merged) >= 8:
                avg_conf = sum(conf for _, conf in fragments) / len(fragments)
                candidates.append((merged, avg_conf))

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
    digits = "".join(re.sub(r"\D", "", str(x[1])) for x in numeric if len(x) >= 2)
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



def parse_series_number_from_text(text):
    """
    На правом поле паспорта цифры обычно идут как 2 + 2 + 6:
    например, '11 04 000000'. Извлекаем только такую полную группу.
    """
    if not text:
        return None, None
    normalized = str(text).replace("О", "0").replace("O", "0")
    # Keep OCR token boundaries: require 2-digit, 2-digit, 6-digit groups.
    match = re.search(r"(?<!\d)(\d{2})[\s:.-]+(\d{2})[\s:.-]+(\d{6})(?!\d)", normalized)
    if match:
        return match.group(1) + match.group(2), match.group(3)

    # Some OCR engines return a single uninterrupted 10-digit token.
    match = re.search(r"(?<!\d)(\d{10})(?!\d)", re.sub(r"\s+", "", normalized))
    if match:
        return match.group(1)[:4], match.group(1)[4:]
    return None, None



def clean_issuer_text(value):
    """Нормализует частые ошибки OCR в наименовании органа выдачи."""
    if not value:
        return None
    value = normalize_text(value).upper()
    # Исправляем типичные лишние/подменённые символы в слове ДЕЛ.
    value = re.sub(r"\bДЕЛЛ\b", "ДЕЛ", value)
    value = re.sub(r"\bДЕПЛ\b", "ДЕЛ", value)
    value = re.sub(r"\bДЕ1Л\b", "ДЕЛ", value)
    value = re.sub(r"\bВНУТРЕННИХ\s+ДЕЛ{2,}\b", "ВНУТРЕННИХ ДЕЛ", value)
    # Убираем мусорные символы по краям и повторные пробелы.
    value = re.sub(r"[^А-ЯЁ0-9№.,\- ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -.,")
    return value or None


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

    # Убираем OCR-пунктуацию в конце значения пола (например, «МУЖ,» -> «МУЖ»).
    texts["Пол"] = re.sub(r"[,;:.]+\\s*$", "", normalize_text(texts["Пол"]))
    g = texts["Пол"].upper()
    if re.search(r"МУЖ|Mуж", g):
        gender = "Мужской"
    elif re.search(r"ЖЕН", g):
        gender = "Женский"
    else:
        gender = None

    issuer = clean_issuer_text(texts["Орган выдачи"])
    # В диагностике показываем только нормализованную дату, без хвостов OCR
    # вроде "ачн", если дата распознана.
    if issue_date:
        texts["Дата выдачи"] = issue_date
    else:
        texts["Дата выдачи"] = normalize_text(
            re.sub(r"(?i)\b[а-яё]{1,5}\b", " ", texts["Дата выдачи"])
        )
    if issuer:
        texts["Орган выдачи"] = issuer

    # Сначала используем обычный OCR текста верхней вертикальной зоны:
    # на тестовом образце он уже видит последовательность вида 11 04 000000.
    serial_items = ocr_image(reader, serial_top_zone, rotations=(0, 90, 270))
    serial_text = best_text(serial_items)
    series, number = parse_series_number_from_text(serial_text)

    # Если обычный OCR не собрал полную группу, пробуем цифровой allowlist.
    if not series or not number:
        digit_candidates = read_digits_zone(reader, serial_top_zone)
        full_candidates = []
        for digits, confidence in digit_candidates:
            digits = re.sub(r"\D", "", digits)
            if len(digits) == 10:
                full_candidates.append((digits, confidence))
            elif len(digits) > 10:
                # Не берём произвольные 10 цифр из длинной строки:
                # ищем последовательность, совпадающую с шаблоном 2+2+6.
                for k in range(len(digits) - 9):
                    candidate = digits[k:k+10]
                    full_candidates.append((candidate, confidence))
        if full_candidates:
            candidate, _ = max(full_candidates, key=lambda x: x[1])
            series, number = candidate[:4], candidate[4:10]

    # Чистый диагностический вывод вместо сырых OCR-строк с шумом.
    texts["Серия документа"] = series or "Не распознана — проверьте вручную"
    texts["Номер документа"] = number or "Не распознан — проверьте вручную"

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


def make_excel_file(dataframe):
    """Создаёт аккуратно оформленную Excel-таблицу с паспортными данными."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Паспортные данные"

    last_col = max(1, len(dataframe.columns))
    last_letter = get_column_letter(last_col)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    title = ws.cell(1, 1, "РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ПАСПОРТНЫХ ДАННЫХ")
    title.font = Font(name="Aptos Display", size=16, bold=True, color="FFFFFF")
    title.fill = PatternFill("solid", fgColor="183B5B")
    title.alignment = Alignment(vertical="center", horizontal="left", indent=1)

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=last_col)
    subtitle = ws.cell(2, 1, "Проверьте корректность распознанных сведений перед использованием.")
    subtitle.font = Font(name="Aptos", size=10, italic=True, color="5B6B7D")
    subtitle.fill = PatternFill("solid", fgColor="F1F5F9")
    subtitle.alignment = Alignment(vertical="center", horizontal="left", indent=1)

    header_row = 4
    for col_idx, header in enumerate(dataframe.columns, start=1):
        ws.cell(header_row, col_idx, str(header))

    for row_idx, row in enumerate(dataframe.itertuples(index=False, name=None), start=header_row + 1):
        for col_idx, value in enumerate(row, start=1):
            cell = ws.cell(row_idx, col_idx, "" if pd.isna(value) else str(value))
            cell.font = Font(name="Aptos", size=10, color="243447")
            cell.alignment = Alignment(vertical="center", horizontal="left", wrap_text=True)
            cell.fill = PatternFill("solid", fgColor="F7FAFD" if row_idx % 2 else "FFFFFF")
            cell.border = Border(bottom=Side(style="hair", color="DCE5EF"))

    for cell in ws[header_row]:
        cell.font = Font(name="Aptos", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E5F8A")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color="183B5B"),
                             right=Side(style="thin", color="7896B3"))

    preferred_widths = {
        "Файл": 24, "Фамилия": 19, "Имя": 17, "Отчество": 21, "Пол": 13,
        "Серия документа": 18, "Номер документа": 19,
        "Кем выдан (государственный орган)": 42, "Код подразделения": 20,
        "Дата выдачи": 17, "Дата рождения": 17, "Источник": 30,
    }
    for col_idx, column_name in enumerate(dataframe.columns, start=1):
        letter = get_column_letter(col_idx)
        if str(column_name) in preferred_widths:
            width = preferred_widths[str(column_name)]
        else:
            values = [len(str(column_name))] + [
                len(str(v)) for v in dataframe.iloc[:, col_idx - 1].fillna("").tolist()
            ]
            width = min(max(max(values, default=14) + 3, 15), 32)
        ws.column_dimensions[letter].width = width

    ws.row_dimensions[1].height = 34
    ws.row_dimensions[2].height = 24
    ws.row_dimensions[3].height = 8
    ws.row_dimensions[header_row].height = 38
    for row_idx in range(header_row + 1, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 34

    ws.freeze_panes = "A5"
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 90
    ws.auto_filter.ref = f"A{header_row}:{last_letter}{ws.max_row}"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = f"1:{header_row}"

    if ws.max_row > header_row and ws.max_column > 0:
        table_ref = f"A{header_row}:{last_letter}{ws.max_row}"
        table = Table(displayName="PassportData", ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False
        )
        ws.add_table(table)

    wb.save(output)
    output.seek(0)
    return output.getvalue()


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
        excel_bytes = make_excel_file(edited)
        st.download_button(
            "Скачать таблицу Excel",
            data=excel_bytes,
            file_name="passport_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
