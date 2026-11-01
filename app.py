import io
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pymupdf as fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(page_title="Распознавание паспортов РФ", page_icon="🪪", layout="wide")

# Базовый интерфейс без пользовательского HTML/CSS.
# Сначала восстанавливаем гарантированное отображение штатных компонентов Streamlit.
st.title("Распознавание паспортов граждан РФ")
st.caption("Загрузите скан или фотографию паспорта, проверьте извлечённые сведения и сохраните результат в Excel.")

# Декоративные звёздные точки включаются только при выбранной тёмной теме.
# Основной фон и все цвета интерфейса остаются под управлением Streamlit.
try:
    active_theme = st.context.theme.type
except Exception:
    active_theme = "light"

if active_theme == "dark":
    st.markdown(
        """<style>
        .stApp {
            background-image:
                radial-gradient(circle, rgba(255,255,255,.92) 0 1px, transparent 1.6px),
                radial-gradient(circle, rgba(204,180,255,.88) 0 1.2px, transparent 2px),
                radial-gradient(circle, rgba(255,255,255,.72) 0 .8px, transparent 1.5px);
            background-size: 173px 151px, 257px 223px, 311px 277px;
            background-position: 13px 21px, 71px 43px, 127px 89px;
            background-repeat: repeat;
        }
        </style>""",
        unsafe_allow_html=True,
    )

st.warning(
    "Учебный прототип. OCR может ошибаться и не подтверждает подлинность документа. "
    "Используйте только вымышленные или обезличенные образцы. Не загружайте реальные паспорта "
    "в публичное приложение без надлежащей защиты и законного основания."
)

@st.cache_resource
def get_reader():
    # Загружаем EasyOCR и PyTorch только по нажатию кнопки распознавания.
    # Благодаря этому интерфейс отображается сразу, без ожидания тяжёлого импорта.
    import easyocr
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

def resize_for_ocr(img, max_side=1800):
    """Ограничивает размер изображения, сохраняя пропорции."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    longest = max(img.size)
    if longest > max_side:
        ratio = max_side / longest
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
    return img


def enhance(img, scale=1):
    img = resize_for_ocr(img)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img.convert("RGB")


def ocr_image(reader, img, rotations=(0,)):
    """Один быстрый OCR-проход; дополнительные повороты не запускаются автоматически."""
    current = resize_for_ocr(img)
    if rotations and rotations[0] != 0:
        current = current.rotate(rotations[0], expand=True)
        current = resize_for_ocr(current)
    result = reader.readtext(np.array(current), detail=1, paragraph=False)
    return [(str(item[1]).strip(), float(item[2])) for item in result if len(item) >= 3 and str(item[1]).strip()]


def read_digits_zone(reader, zone):
    """Один проход распознавания цифровой зоны без трёх поворотов и увеличения x5."""
    rgb = resize_for_ocr(zone, max_side=900)
    arr = np.array(rgb)
    red_mask = ((arr[:, :, 0] > 75) & (arr[:, :, 0] > arr[:, :, 1] * 1.12) & (arr[:, :, 0] > arr[:, :, 2] * 1.08))
    if red_mask.any():
        arr = np.where(red_mask[:, :, None], 0, 255).astype(np.uint8)
    result = reader.readtext(arr, detail=1, paragraph=False, allowlist="0123456789")
    fragments = []
    for item in result:
        if len(item) >= 3:
            digits = re.sub(r"\D", "", str(item[1]))
            if digits:
                fragments.append((digits, float(item[2])))
    candidates = list(fragments)
    if fragments:
        ordered = sorted(result, key=lambda item: (
            (min(pt[1] for pt in item[0]) + max(pt[1] for pt in item[0])) / 2,
            (min(pt[0] for pt in item[0]) + max(pt[0] for pt in item[0])) / 2))
        merged = "".join(re.sub(r"\D", "", str(item[1])) for item in ordered if len(item) >= 3)
        if len(merged) >= 8:
            candidates.append((merged, sum(x[1] for x in fragments) / len(fragments)))
    return candidates


def extract_code_subdivision(reader, img):
    """Считывает код подразделения одним проходом по его области."""
    zone = crop_box(img, (0.48, 0.16, 0.88, 0.31))
    items = ocr_image(reader, zone)
    text = " ".join(t for t, _ in items)
    match = re.search(r"(?<!\d)(\d{3})\s*[-–—]?\s*(\d{3})(?!\d)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}", text
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


def ocr_multi(reader, img, rotations=(0,), scales=(1,)):
    """Совместимая обёртка: ровно один OCR-проход на изображении."""
    current = resize_for_ocr(img, max_side=1800)
    result = reader.readtext(np.array(current), detail=1, paragraph=False)
    found = []
    for item in result:
        if len(item) >= 3:
            value = normalize_text(item[1])
            if value:
                found.append((value, float(item[2])))
    return found

def parse_mrz_domestic(reader, img):
    """
    Читает машиночитаемую зону внутреннего паспорта.
    Возвращает поля только если найдены характерные строки MRZ.
    """
    zone = crop_box(img, (0.02, 0.76, 0.99, 0.995))
    zone = ImageOps.autocontrast(ImageOps.grayscale(zone))
    zone = zone.resize((zone.width * 2, zone.height * 2), Image.Resampling.LANCZOS)
    raw = reader.readtext(
        np.array(zone.convert("RGB")),
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    )

    # Собираем фрагменты, расположенные на одной горизонтальной строке.
    fragments = []
    for item in raw:
        if len(item) < 3:
            continue
        pts = item[0]
        y = sum(point[1] for point in pts) / len(pts)
        x = min(point[0] for point in pts)
        value = re.sub(r"[^A-Z0-9<]", "", str(item[1]).upper())
        if value:
            fragments.append((y, x, value, float(item[2])))

    rows = []
    for y, x, value, confidence in sorted(fragments):
        target = next((row for row in rows if abs(row["y"] - y) < zone.height * 0.075), None)
        if target is None:
            rows.append({"y": y, "parts": [(x, value)], "scores": [confidence]})
        else:
            target["parts"].append((x, value))
            target["scores"].append(confidence)
            target["y"] = sum([target["y"], y]) / 2

    lines = []
    for row in sorted(rows, key=lambda item: item["y"]):
        line = "".join(value for _, value in sorted(row["parts"]))
        if len(line) >= 15:
            lines.append(line)

    # В некоторых версиях EasyOCR уже возвращает каждую строку целиком.
    first = next((line for line in lines if line.startswith(("PNRUS", "P<RUS", "P0RUS")) and "<<" in line), None)
    second = next((line for line in lines if re.match(r"^\d{6,}", line) and "RUS" in line), None)

    # Если OCR склеил строки/ошибся в первом символе, ищем по общим признакам.
    if not first:
        first = next((line for line in lines if line.startswith("P") and "<<" in line and "RUS" in line[:10]), None)
    if not second:
        second = next((line for line in lines if "RUS" in line and re.search(r"\d{6}[MF]", line)), None)

    result = {"Фамилия": None, "Имя": None, "Пол": None, "Дата рождения": None}
    if first:
        # Формат: PNRUSФАМИЛИЯ<<ИМЯ...
        name_part = first
        if "RUS" in name_part:
            name_part = name_part[name_part.find("RUS") + 3:]
        name_part = name_part.lstrip("<")
        if "<<" in name_part:
            surname_raw, given_raw = name_part.split("<<", 1)
            def mrz_name(value):
                value = value.replace("<", " ")
                value = re.sub(r"[^A-ZА-ЯЁ ]", "", value)
                return normalize_text(value) or None
            result["Фамилия"] = mrz_name(surname_raw)
            result["Имя"] = mrz_name(given_raw.split("<", 1)[0])

    if second:
        # TD3-подобная строка: номер документа, код страны, дата рождения, пол.
        # Дата рождения занимает позиции 13–18, пол — позицию 20.
        second = second.ljust(44, "<")
        birth = second[13:19]
        if re.fullmatch(r"\d{6}", birth):
            yy, mm, dd = int(birth[:2]), int(birth[2:4]), int(birth[4:6])
            year = 2000 + yy if yy <= (date.today().year % 100) else 1900 + yy
            try:
                result["Дата рождения"] = date(year, mm, dd).strftime("%d.%m.%Y")
            except ValueError:
                pass
        sex = second[20:21]
        if sex == "M":
            result["Пол"] = "Мужской"
        elif sex == "F":
            result["Пол"] = "Женский"

    return result, lines


def parse_vertical_document_number(reader, img):
    """Пробует прочитать вертикальную красную серию/номер на правом краю разворота."""
    zone = crop_box(img, (0.88, 0.02, 0.995, 0.96))
    candidates = []
    for rotation in (90, 270):
        rotated = zone.rotate(rotation, expand=True)
        rotated = ImageOps.autocontrast(ImageOps.grayscale(rotated))
        rotated = rotated.resize((rotated.width * 2, rotated.height * 2), Image.Resampling.LANCZOS)
        found = reader.readtext(
            np.array(rotated.convert("RGB")),
            detail=1,
            paragraph=False,
            allowlist="0123456789"
        )
        digits = "".join(re.sub(r"\D", "", str(item[1])) for item in found if len(item) >= 3)
        if len(digits) >= 10:
            for i in range(len(digits) - 9):
                candidate = digits[i:i + 10]
                candidates.append(candidate)
    if not candidates:
        return None, None
    # Для серии/номера принимаем только десятизначную последовательность.
    candidate = max(candidates, key=lambda value: sum(ch.isdigit() for ch in value))
    return candidate[:4], candidate[4:]


def parse_domestic_spread(img, reader):
    """
    Разбирает разворот российского внутреннего паспорта.
    Использует один общий OCR-проход, отдельную проверку MRZ и отдельное
    чтение вертикального номера. Ненадёжные поля остаются пустыми.
    """
    img = resize_for_ocr(img, max_side=1800)
    w, h = img.size

    # Один общий проход сохраняет координаты, чтобы не смешивать соседние поля.
    raw = reader.readtext(np.array(img), detail=1, paragraph=False)
    full_items = []
    positioned = []
    for item in raw:
        if len(item) < 3:
            continue
        value = normalize_text(item[1])
        if not value:
            continue
        pts = item[0]
        cx = sum(point[0] for point in pts) / len(pts)
        cy = sum(point[1] for point in pts) / len(pts)
        full_items.append((value, float(item[2])))
        positioned.append((value, cx / w, cy / h, float(item[2])))

    full_text = "\n".join(value for value, _ in full_items)
    upper_text = "\n".join(value for value, x, y, score in positioned if y < 0.48)
    lower_text = "\n".join(value for value, x, y, score in positioned if y >= 0.45)

    # MRZ — наиболее надёжный источник для имени, пола и даты рождения.
    mrz_data, mrz_lines = parse_mrz_domestic(reader, img)

    def clean_name(value):
        if not value:
            return None
        value = re.sub(r"[^А-ЯЁа-яёA-Za-z\- ]", " ", str(value))
        words = [word for word in normalize_text(value).split() if len(word) > 1]
        ignored = {
            "ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОЛ", "МУЖ", "ЖЕН",
            "РОССИЙСКАЯ", "ФЕДЕРАЦИЯ", "ПАСПОРТ", "КРАЙ", "ГУ",
            "ДАТА", "РОЖДЕНИЯ", "ВЫДАЧИ"
        }
        words = [word for word in words if word.upper() not in ignored]
        return " ".join(words[:2]) or None

    # Если MRZ прочитана, используем её. Иначе ищем значения рядом с подписями,
    # но не подставляем произвольный текст из соседних строк.
    def value_after_label(label, source):
        lines = [normalize_text(line) for line in source.splitlines() if normalize_text(line)]
        for index, line in enumerate(lines):
            match = re.search(label + r"\s*[:\-]?\s*(.*)$", line, re.I)
            if match:
                value = match.group(1).strip(" :.-")
                if len(value) > 1:
                    return value
                if index + 1 < len(lines):
                    next_line = lines[index + 1]
                    if len(next_line) > 1 and not re.search(r"ФАМИЛИЯ|ИМЯ|ОТЧЕСТВО|ПОЛ|ДАТА", next_line, re.I):
                        return next_line
        return None

    surname = mrz_data["Фамилия"] or clean_name(value_after_label(r"ФАМИЛИЯ", lower_text))
    first_name = mrz_data["Имя"] or clean_name(value_after_label(r"ИМЯ", lower_text))
    patronymic = clean_name(value_after_label(r"ОТЧЕСТВО", lower_text))

    gender = mrz_data["Пол"]
    if not gender:
        gender_text = lower_text.upper()
        if re.search(r"\bМУЖ", gender_text):
            gender = "Мужской"
        elif re.search(r"\bЖЕН", gender_text):
            gender = "Женский"

    # Даты верхней и нижней частей не смешиваем: дата выдачи берётся сверху,
    # дата рождения — прежде всего из MRZ.
    date_pattern = r"(?<!\d)(\d{2}[.\-/ ]\d{2}[.\-/ ]\d{4})(?!\d)"
    upper_dates = re.findall(date_pattern, upper_text)
    issue_date = parse_date(upper_text)
    if not issue_date and upper_dates:
        issue_date = parse_date(upper_dates[0])

    birth_date = mrz_data["Дата рождения"]
    if not birth_date:
        lower_dates = re.findall(date_pattern, lower_text)
        parsed_lower = [parse_date(value) for value in lower_dates]
        parsed_lower = [value for value in parsed_lower if value]
        if parsed_lower:
            birth_date = min(parsed_lower, key=lambda value: datetime.strptime(value, "%d.%m.%Y"))

    issuer = clean_issuer_text(upper_text)
    if issuer:
        # Убираем дату и короткие цифровые фрагменты из названия органа.
        issuer = re.sub(date_pattern, " ", issuer)
        issuer = re.sub(r"\b\d{3}[- ]?\d{3}\b", " ", issuer)
        issuer = normalize_text(issuer)
        if len(issuer) < 8:
            issuer = None
    if not issuer:
        issuer = clean_issuer_text(value_after_label(
            r"(?:ГУ\s*МВД|УМВД|ОВД|ОРГАН\s+ВЫДАЧИ)", upper_text
        ))

    series, number = parse_vertical_document_number(reader, img)
    if not series or not number:
        # Резерв: ищем последовательность 4+6 цифр в общем OCR.
        series_text = " ".join(value for value, x, y, score in positioned if x > 0.82)
        series, number = parse_series_number_from_text(series_text)

    # Код подразделения обычно имеет формат XXX-XXX и находится на верхней странице.
    subdivision = None
    sub_match = re.search(r"(?<!\d)(\d{3})\s*[-–—]\s*(\d{3})(?!\d)", upper_text)
    if sub_match:
        subdivision = f"{sub_match.group(1)}-{sub_match.group(2)}"

    debug = {
        "Орган выдачи": issuer,
        "Дата выдачи": issue_date,
        "Фамилия": surname,
        "Имя": first_name,
        "Отчество": patronymic,
        "Пол": gender,
        "Дата рождения": birth_date,
        "Серия документа": series or "Не распознана — проверьте вручную",
        "Номер документа": number or "Не распознан — проверьте вручную",
        "MRZ": "\n".join(mrz_lines) if mrz_lines else "Не распознана",
        "Текст верхней страницы": upper_text,
        "Текст нижней страницы": lower_text,
    }

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
        "Источник": "OCR + проверка MRZ; требуется ручная проверка",
        "_Текст OCR": debug,
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
    with st.spinner("Загружаем модель распознавания при первом запуске…"):
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
