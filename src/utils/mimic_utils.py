import pandas as pd
import re

def filter_notes(notes_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # strip texts from leading and trailing and white spaces
    notes_df["text"] = notes_df["text"].str.strip()

    # remove entries without subject id or text
    notes_df = notes_df.dropna(subset=["subject_id", "text"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "chief_complaint": "chief complaint:",
        "present_illness": "present illness:",
        "medical_history": "medical history:",
        "medication_adm": "medications on admission:",
        "allergies": "allergies:",
        "physical_exam": "physical exam:",
        "family_history": "family history:",
        "social_history": "social history:"
    }

    # replace linebreak indicators
    # notes_df['text'] = notes_df['text'].str.replace("\n", "\\n")
    notes_df['text'] = notes_df['text'].str.replace("___\nFamily History:", "___\n\nFamily History:", flags=re.IGNORECASE)

    # extract each section by regex
    for key, section in admission_sections.items():
        notes_df[key] = notes_df.text.str.extract('{}([\\s\\S]+?)\n\\s*?\n[^(\\\\|\\d|\\.)]+?:'.format(section), flags=re.IGNORECASE)

        notes_df[key] = notes_df[key].str.replace('\n', ' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df.loc[notes_df[key].str.startswith("[]"), key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.chief_complaint != "") | (notes_df.present_illness != "") |
                        (notes_df.medical_history != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(text="CHIEF COMPLAINT: " + notes_df.chief_complaint.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.present_illness.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.medical_history.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.medication_adm.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.allergies.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.physical_exam.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.family_history.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.social_history.astype(str))['text']

    return notes_df