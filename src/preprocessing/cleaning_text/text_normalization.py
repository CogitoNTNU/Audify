import os
import re
import json
from num2words import num2words
from datetime import datetime


class TextNormalizer:
    def __init__(self, language="en"):
        self.language = language

        # symbols.json have symbols and abbreviations that need 'translations'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        symbols_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "data", "symbols", "symbols.json"))
        with open(symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.symbols = data.get("symbols", {})
        self.units = data.get("units", {})
        self.months = data.get("months", {})
        self.weekdays = data.get("weekdays", {})

    def normalize(self, text):
        text = normalize_whitespace(text)
        text = remove_unwanted_signs(text)
        text = normalize_dates(text, self.language)
        text = normalize_time(text, self.language)
        text = normalize_symbols(text, self.symbols, self.language)
        text = normalize_units(text,self.units)
        text = normalize_numbers(text, self.language)
        text = normalize_references(text, self.language)
        text = normalize_ordinals(text, self.language)
        text = normalize_abbreviations(text,self.months, self.weekdays)
        
        return text

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace:
    - Removes leading and trailing spaces
    - Collapses multiple spaces or tabs into one
    - Collapses multiple line breaks into a single one
    """
    text = re.sub(r"[ \t]+", " ", text)   #multiple spaces/tabs -> single space 
    text = re.sub(r"\s*\n\s*", "\n", text) #clean up around line breaks
    text = re.sub(r"\n+", "\n", text) #multiple newlines -> one newline
    text = text.replace("\u00A0", " ")
    return text.strip() #removes leading/trailing spaces


def remove_unwanted_signs(text: str) -> str:
    """
    Remove unwanted signs like [1], (kilde), {note}, and stray symbols.
    """
    text = re.sub(r"\[\d+\]", "", text)    #remove footnote markers [1]
    text = re.sub(r"\([^)]*\)", "", text)  #remove text inside parentheses
    text = re.sub(r"\{[^}]*\}", "", text)  #remove curly brace notes
    text = re.sub(r"[*_#><`]", "", text)   #remove markdown-style signs
    text = re.sub(r"\s*\n\s*", "\n", text) #remove spaces/tabs around \n

    #remove links, have already removed #
    text = re.sub(r"\[.*?\]\(.*?\)", "", text) #remove markdown-style links: [text](url)
    text = re.sub(r"\S+@\S+", "", text) #remove email addresses
    return text

def normalize_numbers(text, language):
    text = re.sub(r'\b(\d{3,4})s\b', lambda m: replace_decade(m, language), text)
    text = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)*\b', lambda m: replace_number(m, language), text)
    text = re.sub(r'\b([A-Za-z])(\d+)\b', lambda m: f"{m.group(1).upper()} {num2words(int(m.group(2)), lang=language)}", text)
    return text

def replace_number(m, language):
    raw = m.group()
    cleaned = raw.replace(',', '')

    if re.fullmatch(r'pi|Pi|PI|phi|Phi|e', raw):
        return raw

    # decimal numbers
    if "." in cleaned:
        parts = cleaned.split(".")
        try:
            left = num2words(int(parts[0]), lang=language)
        except Exception:
            return raw

        right_parts = []
        for p in parts[1:]:
            if p.isdigit():
                right_parts.append(num2words(int(p), lang=language))
            else:
                right_parts.append(p)

        right_text = " point ".join(right_parts)
        return f"{left} point {right_text}"

    if not cleaned.isdigit():
        return raw

    val = int(cleaned)
    context_before = m.string[max(0, m.start() - 10):m.start()].lower()
    if 1000 <= val <= 2099 and re.search(r'\b(year|in|since|during|by|before|after)\b', context_before):
        return year_to_words(val, language)

    return num2words(val, lang=language)


def year_to_words(num, language):
    if num == 2000:
        return num2words(num) + "s"

    if 2001 <= num <= 2009:
        # the right spoken way: 
        return f"{num2words(2000)} and {num2words(num % 100, lang=language)}"

    if 1000 <= num <= 2099:
        first = num // 100  
        last = num % 100
        first_part = num2words(first, lang=language)
        if last == 0:
            return f"{first_part} hundred"
        last_part = num2words(last, lang=language)
        return f"{first_part} {last_part}"

    return num2words(num, lang=language)


def replace_decade(m, language):
    num = int(m.group(1))

    # years lke: 1500s, 1900s etc
    if num % 100 == 0:
        century = num // 100
        return f"{num2words(century, lang=language)} hundreds"
    # years like: 1960, 1980 etc 
    if 1000 <= num <= 2099 and num % 10 == 0:
        century = num // 100
        decade = num % 100
        century_word = num2words(century, lang=language)
        decade_word = num2words(decade, lang=language)

        if decade_word.endswith("y"):
            decade_word = decade_word[:-1] + "ies"

        elif not decade_word.endswith("s"):
            decade_word += "s"

        if num == 2000:
            return num2words(num)
        
        decade_word = decade_word.rstrip("s") + "s"
        return f"{century_word} {decade_word}"

    return num2words(num, lang=language)


def normalize_dates(text, language):
    # if the date is formated: 2/04/20 or 2-04-20
    date_pattern = r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b'
    text = re.sub(date_pattern, lambda m: replace_date_if_valid(text, m, language), text)
    # if the date is formated: 2 apr 2020
    name_date_pattern = r'\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{2,4})\b'
    text = re.sub(name_date_pattern, lambda m: replace_date(m, language), text)
    return text


def replace_date_if_valid(full_text, m, language):
    start, _ = m.span()
    before = full_text[max(0, start - 25):start]
    
    # if its not a date: 
    if should_skip_date(before):
        return m.group()
    return replace_date(m, language)


def should_skip_date(before_text):
    skip_words = ["section", "sections", "chapter", "version", "part","article", "page", "figure", "ref", "reference"]
    for w in skip_words:
        if re.search(rf'\b{w}\s*$', before_text, re.IGNORECASE):
            return True
        
    if re.search(r'[\(\[\{]\s*$', before_text):
        return True
    
    if re.search(r'\b(section|sections)\b', before_text, re.IGNORECASE):
        return True
    
    return False


def replace_date(m, language):
    parts = m.groups()
    d, mth, y = parts[0], parts[1], parts[2]

    # Handle month as text: 2 April 2005
    if mth.isalpha():
        month_str = mth.strip().rstrip(".,").capitalize()

        try:
            day = int(d)
            year = int(y)
        except ValueError:
            return m.group()

        day_word = num2words(day, to="ordinal", lang=language)
        year_word = year_to_words(year, language)
        return f"{day_word} of {month_str} {year_word}"

    # Handle number months: 02/04/2005 or 2-4-05
    try:
        first, second = int(d), int(mth)
    except ValueError:
        return m.group()

    if first > 12:
        day, month = first, second
    else:
        day, month = second, first

    if len(y) == 2:
        y = "20" + y if int(y) < 50 else "19" + y

    try:
        date_obj = datetime.strptime(f"{day}-{month}-{y}", "%d-%m-%Y")
    except ValueError:
        return m.group()

    day_word = num2words(date_obj.day, to="ordinal", lang=language)
    month_word = date_obj.strftime("%B")
    year_word = year_to_words(date_obj.year, language)
    return f"{day_word} of {month_word} {year_word}"

def normalize_units(text, units):
    # TODO: capture degrees and compound units
    unit_pattern = (r'(\d+(?:\.\d+)?)' r'(?:\s*)' r'([°º]?\s*[a-zA-ZμΩ²³]+(?:\s*/\s*[a-zA-ZμΩ²³]+)*)')
    text = re.sub(unit_pattern, lambda m: replace_unit(text, m, units), text)

    # replace standalone units
    for unit, word in units.items():
        text = re.sub(rf'\b{re.escape(unit)}\b', word, text, flags=re.IGNORECASE)

    return text
 

def replace_unit(full_text, m, units):
    number, unit = m.groups()
    unit_clean = unit.replace(" ", "").lower()
    start, end = m.span()

    unit_word = unit_clean
    context_before = full_text[max(0, start - 30):start].lower()
    context_after = full_text[end:end + 30].lower()

    # TODO: fix units so that "per" also comes. 
    if '/' in unit_clean:
        parts = [p for p in unit_clean.split('/') if p]
        spoken_parts = []
        for i, p in enumerate(parts):
            word = units.get(p, p)
            if i == len(parts) - 1 and word.endswith('s'):
                word = word[:-1]
            spoken_parts.append(word)
        unit_word = " per ".join(spoken_parts)


    # unit 'm' can be meters or minutes
    if unit_clean == "m":
        if re.search(r'(h|hour|day|d|min|sec|s|am|pm|duration|after|within)',
                     context_before + context_after):
            unit_word = "minutes"
        elif re.search(r'(cm|mm|km|long|tall|wide|deep|distance|height|length|road)',
                       context_before + context_after):
            unit_word = "meters"
        # defult meter
        else:
            unit_word = "meters"

    try:
        number_text = normalize_numbers(number, "en")
    except Exception:
        number_text = number

    return f"{number_text} {unit_word}".strip()


def normalize_abbreviations(text, months, weekdays):
    # Replace month abbreviations
    for abbr, full in months.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)

    # Replace weekday abbreviations
    for abbr, full in weekdays.items():
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)
    return text


def normalize_time(text, language):
    time_pattern = r'\b(\d{1,2})[:\.](\d{2})(\s?[ap]\.?m\.?)?\b'
    return re.sub(time_pattern, lambda m: replace_time(m, language), text)


def replace_time(m, language):
    hour = int(m.group(1))
    minute = int(m.group(2))
    suffix = m.group(3).strip().replace(".", "").lower() if m.group(3) else ""

    if suffix in ("am", "a"):
        period = "a m"
    elif suffix in ("pm", "p"):
        period = "p m"
    else:
        period = "a m" if hour < 12 else "p m"

    if hour > 12:
        hour -= 12
    hour_word = num2words(hour, lang=language)
    if minute == 0:
        return f"{hour_word} {period}"
    
    minute_word = num2words(minute, lang=language)
    return f"{hour_word} {minute_word} {period}"

def normalize_currency(text,symbols, language):
    curreny_pattern = r'([£$€])\s?(\d+(?:,\d{3})*(?:\.\d+)?)'
    return re.sub(curreny_pattern, lambda m: replace_currency(m, symbols, language), text)


def replace_currency(m, symbols, language):
    symbol, amount = m.groups()
    currency_word = symbols.get(symbol, "")
    amount_text = normalize_numbers(amount, language)
    return f"{amount_text} {currency_word}".strip()


def normalize_ordinals(text, language):
    ordinals_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    return re.sub(ordinals_pattern, lambda m: replace_ordinal(m, language), text)


def replace_ordinal(m, language):
    num = m.group(1)
    try:
        return num2words(int(num), to='ordinal', lang=language)
    except Exception:
        return m.group()


def normalize_references(text, language):
    ref_pattern = r'(?i)\b(section|sections|chapter|article|part|version|ref)\s+((?:\d+(?:\.\d+)+(?:\s*(?:,|and)\s*\d+(?:\.\d+)+)*))'
    text = re.sub(ref_pattern, lambda m: replace_reference_group(m, language), text)
    text = re.sub(r'(\()(\d+(?:\.\d+)+)(\))', lambda m: f"({speak_reference_chain(m.group(2), language)})", text)
    return text


def replace_reference_group(m, language):
    keyword = m.group(1)
    refs_str = m.group(2)
    refs = re.split(r'\s*(?:,|and)\s*', refs_str)
    spoken_refs = []
    for r in refs:
        spoken_ref = speak_reference_chain(r, language)
        spoken_refs.append(spoken_ref)

    if len(spoken_refs) > 1:
        spoken = ", ".join(spoken_refs[:-1]) + " and " + spoken_refs[-1]
    else:
        spoken = spoken_refs[0]
    return f"{keyword} {spoken}"


def speak_reference_chain(ref, language):
    parts = ref.split('.')
    spoken = []
    for p in parts:
        try:
            spoken.append(num2words(int(p), lang=language))
        except Exception:
            spoken.append(p)
    return ".".join(spoken)

def remove_links(text):
    # TODO: remove or find a solution for links, right now they just say link
    text = re.sub(r'(!?\[[^\]]*\]\()[^)]+(\))', r'\1: link.\2', text)
    text = re.sub(r'\b(?:https?://|http://|www\.)\S+\b', ': link.', text, flags=re.IGNORECASE)
    text = re.sub(r'https?\s*[:\-]?\s*(slash\s*){1,5}[a-z0-9\-\.]+(\s*(dot|slash)\s*[a-z0-9\-\.]+)*',': link.',text,flags=re.IGNORECASE)
    text = re.sub(r'www\s*(dot\s*[a-z0-9\-]+)+',': link.',text,flags=re.IGNORECASE)
    text = re.sub(r'\blink\s*(slash\s*link)+',': link.',text,flags=re.IGNORECASE)
    return text.strip()

def normalize_emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return re.sub(email_pattern, lambda m: f"{m.group(0).split('@', 1)[0]} at {m.group(0).split('@', 1)[1].replace('.', ' dot ')}", text)

def clean_text(text):
    # Remove characters that are not normally spoken
    text = re.sub(r"[_\(\)\[\]\{\}\"\'<>]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def replace_math(match, language):
    math_problem = match.group(0)
    math_problem = math_problem.replace("+", " plus ").replace("-", " minus ").replace("=", " equals ")
    math_problem = math_problem.replace("*", " times ").replace("/", " divided by ").replace("÷", " divided by ")
    math_problem = re.sub(r'\s+', ' ', math_problem).strip()
    math_problem = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group()), lang=language), math_problem)
    return math_problem.strip()

def normalize_symbols(text, symbols, language):
    text = remove_links(text)
    text = clean_text(text)
    text = normalize_emails(text)
    text = normalize_currency(text, symbols, language)

    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*%',
        lambda m: f"{normalize_numbers(m.group(1), language)} percent",
        text
    )
    # also number-words already normalized but still followed by %
    text = re.sub(r'\s*%', " percent", text)

    # Replace math expressions
    text = re.sub(
        r'(?<!\w)(\d+(?:\s*[\+\-\*/÷=]\s*\d+)+)(?!\w)',
        lambda m: replace_math(m, language),
        text,
    )
    text = re.sub(r'(?<!\d)([*\/÷])(?!\d)', ' ', text)

    # Symbol replacements from symbols.json
    for symbol, word in symbols.items():
        if symbol in ['+', '-', '*', '/', '=']:
            continue
        text = re.sub(re.escape(symbol), f" {word} ", text, flags=re.IGNORECASE)

    text = clean_text(text)
    return text




if __name__ == "__main__":
    normalizer = TextNormalizer(language="en")
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file = os.path.abspath(os.path.join(script_dir, "../../../data/markdown/Gx5qb1uHss4.md"))
    
    with open(file,"r",encoding="utf-8") as f: 
        text = f.read()
    # text = "1+2=3"
    result = normalizer.normalize(text)
    print(result)
