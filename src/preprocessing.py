import asyncio
import re
import pandas as pd
from googletrans import Translator, LANGUAGES

TRANSLATOR = Translator()
DELIMITER = "|"
RELEVANT_COLS = [
    "Year",
    "DonorName",
    "RecipientName",
    "RegionName",
    "LongDescription",
    "PurposeCode",
    "USD_Received",
    "Biodiversity",
    "ClimateMitigation",
    "ClimateAdaptation",
    "Desertification",
]
RAW_DATA_PATH = "data/raw"
PREPROCESSED_DATA_PATH = "data/preprocessed"


async def translate(text: str, target_language: str = "en", source_language: str = "auto") -> str:
    translation = await TRANSLATOR.translate(text, dest=target_language, src=source_language)
    return translation.text

def scrub_words(text: str) -> str:
    """Basic cleaning of texts."""

    # remove html markup
    text = re.sub("(<.*?>)", "", text)

    # remove other weird letters
    text = re.sub(r"\w*[□,©,$]\w*", "", text)

    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)

    # remove whitespace
    text = text.strip()
    return text

async def preprocess_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=DELIMITER, encoding="utf-8", low_memory=False, dtype={"LongDescription": str})
    # Only leave relevant columns
    df = df[RELEVANT_COLS]

    # Remove rows with missing LongDescription
    df = df.dropna(subset=["LongDescription"])

    # Scrub LongDescription
    df["LongDescription"] = df["LongDescription"].apply(scrub_words)

    # Detect language for each row
    for i, row in df.iterrows():
        detected = await TRANSLATOR.detect(row["LongDescription"])
        # If the language is not English, translate it
        if detected.lang != "en":
            # if language has region code e.g pt-PT, we want to use pt (language code only)
            detected.lang = detected.lang.split("-")[0]
            if detected.lang not in LANGUAGES:
                print(f"Language {detected.lang} not supported, skipping translation.")
                continue
            confidence = round(detected.confidence, 2)
            print(f"Row {i} is in {detected.lang} with {confidence} confidence, translating to English...")
            df.at[i, "LongDescription"] = await translate(row["LongDescription"], source_language=detected.lang)
            await asyncio.sleep(0.1)  # To avoid hitting the API too fast

    return df

async def main():
    # Preprocess example file
    filename = "CRS 2023 data"
    df = await preprocess_data(f"{RAW_DATA_PATH}/{filename}.txt")
    print(df.head())
    # Save the preprocessed data
    df.to_csv(f"{PREPROCESSED_DATA_PATH}/{filename}.csv", index=False, sep=DELIMITER, encoding="utf-8")


asyncio.run(main())
