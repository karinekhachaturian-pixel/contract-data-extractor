import io
import re
from datetime import date, datetime, timedelta
from typing import Optional

import easyocr
import fitz
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Распознавание паспортов РФ", page_icon="🪪", layout="wide")

st.title("🪪 Распознавание паспортов граждан РФ")
st.write("Загрузите изображение или PDF. Приложение распознает текст и извлечёт паспортные данные для проверки.")

st.warning(
    "Учебный прототип. Результаты OCR могут содержать ошибки и не подтверждают подлинность "
    "или действительность паспорта. Используйте только тестовые/обезличенные образцы. "
    "Не загружайте реальные паспорта в публичное демонстрационное приложение."
)

@st.cache_resource
def get_reader():
    return easyocr.Reader(["ru", "en"], gpu=False)

def uploaded_to_images(uploaded):
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return pages
    return [Image.open(io.BytesIO(raw)).convert("RGB")]

def ocr_document(images, reader):
    all_lines = []
    for image in images:
        lines = reader.readtext(image, detail=0, paragraph=False)
        all_lines.extend(str(x).strip() for x in lines if str(x).strip())
    return all_lines

def norm(s):
    return re.sub(r"\s+", " ", s.replace("\u00a0", " ")).strip()

def find_labeled_value(lines, labels, max_len=90):
    label_re = re.compile(r"^\s*(?:" + "|".join(labels) + r")\s*[:№\-]?\s*(.*)$", re.I)
    for i, line in enumerate(lines):
        m = label_re.search(line)
        if m:
            value = norm(m.group(1)).strip(" :№-")
            if not value and i + 1 < len(lines):
                value = norm(lines[i + 1])
            if value and len(value) <= max_len:
                return value
    return None

def find_date_after_label(lines, labels):
    val = find_labeled_value(lines, labels, 60)
    if val:
        m = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\b", val)
        if m:
            return normalize_date(m.group(1))
    return None

def normalize_date(s):
    s = s.replace("/", ".").replace("-", ".")
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            d = datetime.strptime(s, fmt).date()
            return d.strftime("%d.%m.%Y")
        except ValueError:
            pass
    return s

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except ValueError:
        return None

def date_from_mrz(yymmdd, is_expiry=False):
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    today = date.today()
    if is_expiry:
        year = 2000 + yy
        if year < today.year - 10:
            year += 100
    else:
        year = 2000 + yy
        if year > today.year:
            year -= 100
    try:
        return date(year, mm, dd).strftime("%d.%m.%Y")
    except ValueError:
        return None

def find_mrz(lines):
    candidates = [re.sub(r"[^A-Z0-9<]", "<", x.upper().replace(" ", "")) for x in lines]
    # TD3 passport MRZ: two lines, normally 44 characters each.
    for i in range(len(candidates) - 1):
        a, b = candidates[i], candidates[i + 1]
        if a.startswith("P") and len(b) >= 35:
            b = (b + "<" * 44)[:44]
            number = b[0:9].replace("<", "")
            nationality = b[10:13]
            birth = date_from_mrz(b[13:19], False)
            sex_code = b[20:21]
            expiry = date_from_mrz(b[21:27], True)
            names = a[5:].split("<<", 1)
            surname = names[0].replace("<", " ").strip() if names else None
            given = names[1].replace("<", " ").strip() if len(names) > 1 else None
            given_parts = given.split() if given else []
            return {
                "Фамилия": surname,
                "Имя": given_parts[0] if given_parts else None,
                "Отчество": " ".join(given_parts[1:]) if len(given_parts) > 1 else None,
                "Пол": "Мужской" if sex_code == "M" else ("Женский" if sex_code == "F" else None),
                "Серия документа": None,
                "Номер документа": number or None,
                "Дата выдачи": None,
                "Дата рождения": birth,
                "Дата окончания действия": expiry,
                "Источник": "Машиночитаемая зона (MRZ)"
            }
    return None

def parse_foreign(lines):
    mrz = find_mrz(lines)
    text = "\n".join(lines)
    result = mrz or {}
    result.setdefault("Фамилия", find_labeled_value(lines, [r"фамилия", r"surname"]))
    result.setdefault("Имя", find_labeled_value(lines, [r"имя", r"given names?", r"given name"]))
    result.setdefault("Отчество", find_labeled_value(lines, [r"отчество", r"middle name"]))
    result.setdefault("Пол", find_labeled_value(lines, [r"пол", r"sex"]))
    result.setdefault("Серия документа", None)
    result.setdefault("Номер документа", find_labeled_value(lines, [r"номер паспорта", r"passport no", r"passport number"]))
    result.setdefault("Дата выдачи", find_date_after_label(lines, [r"дата выдачи", r"date of issue"]))
    result.setdefault("Дата рождения", find_date_after_label(lines, [r"дата рождения", r"date of birth"]))
    result.setdefault("Дата окончания действия", find_date_after_label(lines, [r"дата окончания", r"действителен до", r"date of expiry", r"date of expiration"]))
    result.setdefault("Источник", "OCR / MRZ")
    if result.get("Пол"):
        p = result["Пол"].lower()
        if p in ("m", "м", "муж", "мужской"):
            result["Пол"] = "Мужской"
        elif p in ("f", "ж", "жен", "женский"):
            result["Пол"] = "Женский"
    return result

def parse_domestic(lines):
    result = {
        "Фамилия": find_labeled_value(lines, [r"фамилия"]),
        "Имя": find_labeled_value(lines, [r"имя"]),
        "Отчество": find_labeled_value(lines, [r"отчество"]),
        "Пол": find_labeled_value(lines, [r"пол"]),
        "Серия документа": None,
        "Номер документа": None,
        "Кем выдан (государственный орган)": find_labeled_value(
            lines,
            [r"кем выдан", r"орган,? выдавший паспорт", r"выдавший орган"],
            max_len=180
        ),
        "Дата выдачи": find_date_after_label(lines, [r"дата выдачи"]),
        "Дата рождения": find_date_after_label(lines, [r"дата рождения"]),
        "Дата окончания действия": None,
        "Источник": "OCR (поля внутреннего паспорта)"
    }

    joined = "\n".join(lines)
    # Common printed internal passport format: 4-digit series + 6-digit number.
    serial_match = re.search(r"(?<!\d)(\d{2})\s?(\d{2})\s+(\d{6})(?!\d)", joined)
    if serial_match:
        result["Серия документа"] = serial_match.group(1) + " " + serial_match.group(2)
        result["Номер документа"] = serial_match.group(3)
    else:
        # Fallback for digits separated by OCR line breaks/spaces.
        digits = re.findall(r"(?<!\d)(\d{4})\s?(\d{6})(?!\d)", joined)
        if digits:
            result["Серия документа"] = digits[0][0]
            result["Номер документа"] = digits[0][1]

    if result["Пол"]:
        p = result["Пол"].lower()
        if "муж" in p or p == "м":
            result["Пол"] = "Мужской"
        elif "жен" in p or p == "ж":
            result["Пол"] = "Женский"

    dob = parse_date(result["Дата рождения"])
    if dob:
        age = (date.today() - dob).days // 365
        if age < 20:
            due = dob.replace(year=dob.year + 20)
            result["Дата окончания действия"] = (
                (due + timedelta(days=90)).strftime("%d.%m.%Y")
                + " (расчётный предел после 20-летия)"
            )
        elif age < 45:
            due = dob.replace(year=dob.year + 45)
            result["Дата окончания действия"] = (
                (due + timedelta(days=90)).strftime("%d.%m.%Y")
                + " (расчётный предел после 45-летия)"
            )
        else:
            result["Дата окончания действия"] = "Бессрочно (после 45 лет)"
    return result

passport_type = st.radio(
    "Выберите тип паспорта",
    ["Внутренний паспорт РФ", "Заграничный паспорт РФ"],
    horizontal=True
)

uploads = st.file_uploader(
    "Загрузите скан или фотографию паспорта",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploads and st.button("Распознать паспортные данные", type="primary"):
    reader = get_reader()
    rows = []
    for file in uploads:
        try:
            with st.spinner(f"Обрабатываем {file.name}..."):
                images = uploaded_to_images(file)
                lines = ocr_document(images, reader)
                data = parse_domestic(lines) if passport_type == "Внутренний паспорт РФ" else parse_foreign(lines)
                rows.append({"Файл": file.name, **data})
                with st.expander(f"Распознанный текст — {file.name}"):
                    st.text("\n".join(lines))
        except Exception as exc:
            st.error(f"Не удалось обработать {file.name}: {exc}")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Извлечённые данные")
        st.caption("Проверьте все поля вручную. Для внутреннего паспорта дата окончания рассчитывается ориентировочно по возрасту; она не напечатана в документе.")
        edited = st.data_editor(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать результаты CSV",
            edited.to_csv(index=False).encode("utf-8-sig"),
            file_name="passport_data.csv",
            mime="text/csv"
        )

with st.expander("Ограничения и безопасность"):
    st.write(
        "Это демонстрационный OCR-прототип, а не система проверки подлинности паспорта. "
        "Распознавание зависит от качества изображения и расположения полей. "
        "У внутреннего паспорта РФ дата окончания не печатается: срок действия связан с достижением "
        "20 и 45 лет, после 45 лет паспорт действует бессрочно. Расчёт в приложении является ориентировочным "
        "и не учитывает все основания для замены паспорта. Не используйте приложение для юридически значимых "
        "решений и не загружайте реальные документы в публичное приложение без необходимых мер защиты."
    )
