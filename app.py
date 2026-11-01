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


def _ocr_geometry(reader, img):
    """
    Распознаёт разворот целиком и отдельно увеличенные области с полями.
    Координаты всех найденных фрагментов возвращаются к масштабу исходного фото.
    """
    work = resize_for_ocr(img, max_side=2200)
    w, h = work.size
    items = []

    def add_detections(raw, source_w, source_h, x_offset=0, y_offset=0,
                       crop_w=None, crop_h=None, scale=1.0):
        crop_w = crop_w or source_w
        crop_h = crop_h or source_h
        for det in raw:
            if len(det) < 3:
                continue
            text = normalize_text(det[1])
            if not text:
                continue
            pts = det[0]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            # Переводим координаты увеличенного фрагмента обратно в общий кадр.
            x1 = (x_offset + min(xs) / scale) / w
            x2 = (x_offset + max(xs) / scale) / w
            y1 = (y_offset + min(ys) / scale) / h
            y2 = (y_offset + max(ys) / scale) / h
            items.append({
                "text": text, "conf": float(det[2]),
                "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "height": max(0.002, y2 - y1),
            })

    raw_full = reader.readtext(np.array(work), detail=1, paragraph=False)
    add_detections(raw_full, w, h)

    # Два дополнительных прохода по содержательным зонам. Они помогают,
    # когда мелкие подписи и значения теряются при распознавании всего фото.
    for box in ((0.035, 0.035, 0.97, 0.46), (0.035, 0.48, 0.97, 0.84)):
        x1, y1, x2, y2 = box
        left, top = int(w * x1), int(h * y1)
        right, bottom = int(w * x2), int(h * y2)
        crop = work.crop((left, top, right, bottom))
        crop = ImageOps.autocontrast(ImageOps.grayscale(crop))
        crop = ImageEnhance.Contrast(crop).enhance(1.2)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        scale = 1.5
        crop = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
            Image.Resampling.LANCZOS
        )
        raw_crop = reader.readtext(np.array(crop.convert("RGB")), detail=1, paragraph=False)
        add_detections(
            raw_crop, crop.width, crop.height,
            x_offset=left, y_offset=top,
            crop_w=right-left, crop_h=bottom-top, scale=scale
        )

    # Убираем дубли одного и того же слова, оставляя более уверенное обнаружение.
    unique = []
    for item in sorted(items, key=lambda x: x["conf"], reverse=True):
        duplicate = any(
            item["text"].casefold() == old["text"].casefold()
            and abs(item["cx"] - old["cx"]) < 0.012
            and abs(item["cy"] - old["cy"]) < 0.012
            for old in unique
        )
        if not duplicate:
            unique.append(item)
    return work, unique


def _compact_label(text):
    return re.sub(r"[^А-ЯЁA-Z]", "", str(text).upper().replace("Ё", "Е"))


def _group_ocr_rows(items):
    """Группирует детекции по фактической вертикальной близости, не по координатам-шаблонам."""
    rows = []
    for item in sorted(items, key=lambda x: (x["cy"], x["x1"])):
        best = None
        best_dist = 1e9
        for row in rows[-5:]:
            avg_y = sum(x["cy"] for x in row["items"]) / len(row["items"])
            avg_h = sum(x["height"] for x in row["items"]) / len(row["items"])
            limit = max(0.008, avg_h * 0.85, item["height"] * 0.75)
            dist = abs(item["cy"] - avg_y)
            if dist <= limit and dist < best_dist:
                best, best_dist = row, dist
        if best is None:
            rows.append({"items": [item]})
        else:
            best["items"].append(item)
    for row in rows:
        row["items"].sort(key=lambda x: x["x1"])
        row["y"] = sum(x["cy"] for x in row["items"]) / len(row["items"])
        row["text"] = normalize_text(" ".join(x["text"] for x in row["items"]))
        row["x1"] = min(x["x1"] for x in row["items"])
        row["x2"] = max(x["x2"] for x in row["items"])
    return sorted(rows, key=lambda x: x["y"])


def _label_kind(text):
    value = _compact_label(text)
    aliases = {
        "surname": ("ФАМИЛИЯ",),
        "given": ("ИМЯ",),
        "patronymic": ("ОТЧЕСТВО",),
        "gender": ("ПОЛ",),
        "birth": ("ДАТАРОЖДЕНИЯ", "ДАТАРОЖД"),
        "issue": ("ДАТАВЫДАЧИ",),
        "issuer": ("ПАСПОРТВЫДАН", "ВЫДАН", "КЕМВЫДАН"),
        "subdivision": ("КОДПОДРАЗДЕЛЕНИЯ", "КОДПОДРАЗД"),
    }
    for kind, variants in aliases.items():
        if any(alias in value for alias in variants):
            return kind
    return None


def _text_after_label(text, kind):
    patterns = {
        "surname": r"ФАМИЛИЯ", "given": r"ИМЯ", "patronymic": r"ОТЧЕСТВО",
        "gender": r"ПОЛ", "birth": r"ДАТА\s*РОЖДЕНИЯ", "issue": r"ДАТА\s*ВЫДАЧИ",
        "issuer": r"(?:ПАСПОРТ\s*)?ВЫДАН", "subdivision": r"КОД\s*ПОДРАЗДЕЛЕНИЯ",
    }
    return re.sub(patterns[kind], "", text, count=1, flags=re.I).strip(" :.-")


def _near_label_value(items, rows, kind, y_min=0.0, y_max=1.0):
    """Берёт ближайшую OCR-детекцию справа/рядом с найденной подписью поля."""
    labels = [x for x in items if _label_kind(x["text"]) == kind and y_min <= x["cy"] <= y_max]
    candidates = []
    for label in labels:
        inline = _text_after_label(label["text"], kind)
        if len(re.sub(r"\W", "", inline)) >= 2:
            candidates.append((0.0, inline, label["conf"]))
        for item in items:
            if item is label or not (y_min <= item["cy"] <= y_max):
                continue
            dy = abs(item["cy"] - label["cy"])
            # Текст значения обычно справа от подписи на той же строке.
            right = item["x1"] >= label["x2"] - 0.012
            if right and dy <= max(0.025, label["height"] * 2.0):
                dx = max(0.0, item["x1"] - label["x2"])
                score = dy * 4 + dx * 0.35 - min(item["conf"], 1.0) * 0.006
                candidates.append((score, item["text"], item["conf"]))
        # Если значение стоит чуть выше подписи, учитываем соседнюю строку
        # только при близком горизонтальном положении.
        for item in items:
            if item is label or not (y_min <= item["cy"] <= y_max):
                continue
            dy = abs(item["cy"] - label["cy"])
            x_overlap = min(item["x2"], label["x2"] + 0.32) - max(item["x1"], label["x1"])
            if dy <= 0.022 and x_overlap > 0:
                score = 0.018 + dy * 4 - min(item["conf"], 1.0) * 0.004
                candidates.append((score, item["text"], item["conf"]))
    if not candidates:
        # Резерв: строка OCR могла содержать подпись и значение одним блоком.
        for row in rows:
            if not (y_min <= row["y"] <= y_max):
                continue
            if _label_kind(row["text"]) == kind:
                value = _text_after_label(row["text"], kind)
                if value:
                    candidates.append((0.03, value, 0.5))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _clean_person_value(value, max_words=2):
    if not value:
        return None
    value = str(value).upper().replace("Ё", "Е")
    value = re.sub(r"[^А-ЯA-Z\- ]", " ", value)
    ignored = {
        "ФАМИЛИЯ", "ИМЯ", "ОТЧЕСТВО", "ПОЛ", "МУЖ", "ЖЕН", "МУЖСКОЙ", "ЖЕНСКИЙ",
        "РОССИЙСКАЯ", "ФЕДЕРАЦИЯ", "ПАСПОРТ", "ДАТА", "РОЖДЕНИЯ", "ВЫДАЧИ",
        "МЕСТО", "РОЖДЕНИЯ", "ГРУЗИЯ", "КРАЙ", "ГОРОД", "ГУ", "МВД"
    }
    words = [w for w in normalize_text(value).split() if len(w) >= 2 and w not in ignored]
    if not words:
        return None
    # В полях российского паспорта обычно одно слово; второе оставляем лишь
    # для двойных имён/фамилий, если оно не похоже на соседнюю подпись.
    return " ".join(words[:max_words])


def _date_candidates(items, y_min, y_max):
    found = []
    for item in items:
        if not (y_min <= item["cy"] <= y_max):
            continue
        text = item["text"].replace("O", "0").replace("О", "0")
        parsed = parse_date(text)
        if parsed:
            found.append((parsed, item["cx"], item["cy"], item["conf"]))
    return found


def parse_domestic_spread(img, reader):
    """Распознаёт разворот по координатам фактически найденных OCR-строк и связям с подписями."""
    img, items = _ocr_geometry(reader, img)
    rows = _group_ocr_rows(items)
    full_text = "\n".join(row["text"] for row in rows)
    upper_rows = [r for r in rows if r["y"] < 0.48]
    lower_rows = [r for r in rows if r["y"] >= 0.43]
    upper_text = "\n".join(r["text"] for r in upper_rows)
    lower_text = "\n".join(r["text"] for r in lower_rows)

    # MRZ запускаем только если общий OCR заметил характерные угловые скобки
    # или длинную латинско-цифровую строку в нижней части.
    bottom_hint = any(
        r["y"] > 0.72 and ("<<" in r["text"] or len(re.sub(r"[^A-Z0-9<]", "", r["text"].upper())) >= 22)
        for r in rows
    )
    if bottom_hint:
        mrz_data, mrz_lines = parse_mrz_domestic(reader, img)
    else:
        mrz_data = {"Фамилия": None, "Имя": None, "Пол": None, "Дата рождения": None}
        mrz_lines = []

    surname_raw = _near_label_value(items, rows, "surname", 0.42, 0.88)
    given_raw = _near_label_value(items, rows, "given", 0.42, 0.90)
    patronymic_raw = _near_label_value(items, rows, "patronymic", 0.42, 0.90)
    surname = mrz_data.get("Фамилия") or _clean_person_value(surname_raw, 1)
    first_name = mrz_data.get("Имя") or _clean_person_value(given_raw, 1)
    patronymic = _clean_person_value(patronymic_raw, 1)

    gender_raw = _near_label_value(items, rows, "gender", 0.48, 0.92)
    gender_text = (gender_raw or "").upper()
    gender = mrz_data.get("Пол")
    if not gender:
        if re.search(r"МУЖ|MALE|\bM\b", gender_text):
            gender = "Мужской"
        elif re.search(r"ЖЕН|FEMALE|\bF\b", gender_text):
            gender = "Женский"

    birth_raw = _near_label_value(items, rows, "birth", 0.48, 0.95)
    issue_raw = _near_label_value(items, rows, "issue", 0.02, 0.48)
    birth_date = mrz_data.get("Дата рождения") or parse_date(birth_raw or "")
    issue_date = parse_date(issue_raw or "")

    # Если подпись даты OCR не распознал, используем дату из соответствующей
    # половины документа, не смешивая дату рождения и выдачи.
    if not issue_date:
        dates = _date_candidates(items, 0.02, 0.46)
        if dates:
            issue_date = max(dates, key=lambda x: (x[3], -abs(x[2] - 0.27)))[0]
    if not birth_date:
        dates = _date_candidates(items, 0.48, 0.94)
        if dates:
            birth_date = max(dates, key=lambda x: (x[3], -abs(x[2] - 0.78)))[0]

    # Орган выдачи: ищем текстовые строки рядом с меткой «Паспорт выдан».
    issuer_label_items = [x for x in items if _label_kind(x["text"]) == "issuer" and x["cy"] < 0.46]
    issuer_parts = []
    if issuer_label_items:
        label = min(issuer_label_items, key=lambda x: x["cy"])
        for item in items:
            if item is label or item["cy"] >= 0.44:
                continue
            dy = item["cy"] - label["cy"]
            if -0.015 <= dy <= 0.13 and item["x2"] > label["x1"] + 0.08:
                candidate = item["text"]
                if parse_date(candidate) or re.search(r"\b\d{3}\s*[-–—]\s*\d{3}\b", candidate):
                    continue
                if _label_kind(candidate) in {"issue", "subdivision"}:
                    continue
                issuer_parts.append((item["cy"], item["x1"], candidate, item["conf"]))
    if issuer_parts:
        issuer_raw = " ".join(x[2] for x in sorted(issuer_parts, key=lambda x: (x[0], x[1])))
    else:
        # Fallback: только длинные текстовые строки верхней страницы;
        # дату, номер кода и заголовки исключаем.
        candidates = []
        for row in upper_rows:
            text = row["text"]
            compact = _compact_label(text)
            if any(k in compact for k in ("РОССИЙСКАЯФЕДЕРАЦИЯ", "ПАСПОРТВЫДАН", "ДАТАВЫДАЧИ", "КОДПОДРАЗДЕЛЕНИЯ")):
                continue
            text_wo_digits = re.sub(r"\d|[./,:-]", "", text)
            letters = re.findall(r"[А-ЯЁA-Z]{2,}", text_wo_digits.upper())
            if sum(map(len, letters)) >= 12:
                candidates.append((sum(map(len, letters)), text))
        issuer_raw = max(candidates, default=(0, None))[1]

    issuer = clean_issuer_text(issuer_raw)
    if issuer:
        issuer = re.sub(r"\b\d{2}[.\-/ ]\d{2}[.\-/ ]\d{4}\b", " ", issuer)
        issuer = re.sub(r"\b\d{3}\s*[-–—]\s*\d{3}\b", " ", issuer)
        issuer = normalize_text(issuer)
        if len(re.sub(r"[^А-ЯЁ]", "", issuer)) < 8:
            issuer = None

    subdivision_raw = _near_label_value(items, rows, "subdivision", 0.02, 0.48)
    subdivision = None
    sub_text = subdivision_raw or upper_text
    sub_match = re.search(r"(?<!\d)(\d{3})\s*[-–— ]\s*(\d{3})(?!\d)", sub_text)
    if sub_match:
        subdivision = f"{sub_match.group(1)}-{sub_match.group(2)}"

    # Серия/номер: сначала цифры, обнаруженные общим OCR у правого края,
    # затем один отдельный вертикальный проход как резерв.
    right_digits = []
    for item in items:
        if item["cx"] > 0.82:
            digits = re.sub(r"\D", "", item["text"].replace("О", "0").replace("O", "0"))
            if digits:
                right_digits.append((item["cy"], digits, item["conf"]))
    right_digits.sort(key=lambda x: x[0])
    digit_stream = "".join(x[1] for x in right_digits)
    series, number = parse_series_number_from_text(" ".join(x[1] for x in right_digits))
    if not series or not number:
        series, number = parse_vertical_document_number(reader, img)

    debug = {
        "Орган выдачи (по строкам)": issuer_raw or "Не распознано",
        "Дата выдачи (рядом с подписью)": issue_raw or "Не распознано",
        "Фамилия (рядом с подписью)": surname_raw or "Не распознано",
        "Имя (рядом с подписью)": given_raw or "Не распознано",
        "Отчество (рядом с подписью)": patronymic_raw or "Не распознано",
        "Пол (рядом с подписью)": gender_raw or "Не распознано",
        "Дата рождения (рядом с подписью)": birth_raw or "Не распознано",
        "Серия документа": series or "Не распознана — проверьте вручную",
        "Номер документа": number or "Не распознан — проверьте вручную",
        "MRZ": "\n".join(mrz_lines) if mrz_lines else "Не обнаружена или отсутствует",
        "Строки верхней страницы": upper_text,
        "Строки нижней страницы": lower_text,
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
        "Источник": "OCR по координатам строк + MRZ при наличии; требуется ручная проверка",
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
        "Внутренний паспорт обрабатывается по координатам найденных OCR-строк и подписям полей; "
        "MRZ используется дополнительно, если присутствует. Разные ракурсы, качество снимка и старые "
        "образцы могут влиять на точность. Загранпаспорт обрабатывается отдельно через MRZ, если зона распознана. "
        "Для внутреннего паспорта РФ выводится код подразделения; дата окончания действия не включается, поскольку она не печатается в документе. "
        "Не используйте результаты без ручной проверки и не загружайте реальные паспорта в публичное приложение."
    )
