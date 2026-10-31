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

# Визуальное оформление.
# Цвет шапки зависит от primaryColor выбранной темы Streamlit.
st.html("""
<style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        position: relative;
        z-index: 1;
    }

    /* Звёздные точки по экрану. На светлом фоне белые точки почти не видны,
       на тёмном фиолетовом фоне становятся заметными. */
    .stApp {
        position: relative;
        isolation: isolate;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        opacity: .78;
        mix-blend-mode: screen;
        background-image:
            radial-gradient(circle, rgba(255,255,255,.95) 0 1px, transparent 1.7px),
            radial-gradient(circle, rgba(224,205,255,.9) 0 1.2px, transparent 2px),
            radial-gradient(circle, rgba(255,255,255,.8) 0 .8px, transparent 1.6px);
        background-size: 173px 151px, 257px 223px, 311px 277px;
        background-position: 13px 21px, 71px 43px, 127px 89px;
        background-repeat: repeat;
    }

    .passport-hero {
        box-sizing: border-box;
        width: 100%;
        position: relative;
        overflow: hidden;
        padding: 1.7rem 1.9rem 1.6rem;
        margin: 0 0 1.35rem 0;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(35, 20, 45, .18);
    }
    .passport-hero,
    .passport-hero * {
        color: #fff !important;
        opacity: 1 !important;
    }
    .passport-hero .eyebrow {
        font-size: .95rem;
        font-weight: 700;
        margin: 0 0 1.4rem 0;
        letter-spacing: .02em;
    }
    .passport-hero h1 {
        font-size: clamp(1.65rem, 3vw, 2.35rem);
        line-height: 1.18;
        margin: 0 0 .7rem 0;
        font-weight: 750;
    }
    .passport-hero .subtitle {
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }
    .passport-hero .hero-stars {
        position: absolute;
        top: .65rem;
        right: 1.2rem;
        font-size: 1.05rem;
        letter-spacing: .3rem;
        color: rgba(255,255,255,.92) !important;
        pointer-events: none;
        white-space: nowrap;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;
        padding: .62rem 1.15rem;
        font-weight: 700;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .passport-hero { padding: 1.25rem; border-radius: 15px; }
        .passport-hero .hero-stars {
            right: .6rem;
            letter-spacing: .12rem;
            font-size: .9rem;
        }
    }
</style>
""")

# У шапки задан фон непосредственно в HTML-элементе (inline style),
# поэтому он не зависит от CSS-классов и DOM-структуры Streamlit.
# --primary-color берётся из текущей темы: розовый для Light, фиолетовый для Dark.
st.html("""
<div class="passport-hero" style="
    background: linear-gradient(
        120deg,
        color-mix(in srgb, var(--primary-color, #b96f89) 82%, #30204f) 0%,
        var(--primary-color, #d49aaa) 58%,
        color-mix(in srgb, var(--primary-color, #eed0d4) 68%, #ffffff) 100%
    );
    border: 1px solid var(--primary-color, #d8a8b7);
">
    <div class="hero-stars">✦ · ✧ · ✦ · ⋆ · ✧ · ✦</div>
    <div class="eyebrow">Документы · распознавание данных</div>
    <h1>Распознавание паспортов граждан РФ</h1>
    <p class="subtitle">Загрузите скан или фотографию паспорта, проверьте извлечённые сведения и сохраните результат в Excel.</p>
</div>
""")

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
