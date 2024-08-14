import streamlit as st
import sqlite3
import os
import json
import pyaudio
import boto3
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from fpdf import FPDF
import openai
import tempfile
import base64
import time
import pandas as pd
import numpy as np
import random
import openpyxl
from datetime import datetime
import wave
import threading
import hashlib
from cryptography.fernet import Fernet
from streamlit_mic_recorder import mic_recorder
import tempfile
import translators as ts

st.set_page_config(page_title="AuraSync", page_icon="🧑‍⚕️")

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

if "recording" not in st.session_state:
    st.session_state.recording = False

if "stop_button_clicked" not in st.session_state:
    st.session_state.stop_button_clicked = False

conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT
    )
"""
)
conn.commit()


def create_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash),
    )
    conn.commit()


def verify_login(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, password_hash),
    )
    return cursor.fetchone() is not None


class AudioDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS audio_files (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                patient_name TEXT,
                                patient_id TEXT,
                                audio_file_s3_uri TEXT,
                                transcription TEXT,
                                generated_ehr TEXT)"""
        )
        self.conn.commit()

    def insert_audio_file(
        self,
        patient_name,
        patient_id,
        audio_file_s3_uri,
        transcription,
        generated_ehr,
    ):
        self.cursor.execute(
            "INSERT INTO audio_files (patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr) VALUES (?, ?, ?, ?, ?)",
            (
                patient_name,
                patient_id,
                audio_file_s3_uri,
                transcription,
                generated_ehr,
            ),
        )
        self.conn.commit()

    def fetch_records(self, patient_name=None, patient_id=None):
        if patient_name and patient_id:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_name=? AND patient_id=?",
                (patient_name, patient_id),
            )
        elif patient_name:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_name=?",
                (patient_name,),
            )
        elif patient_id:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_id=?",
                (patient_id,),
            )
        else:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files"
            )
        return self.cursor.fetchall()

    def update_record(
        self,
        record_id,
        patient_name,
        patient_id,
        audio_file_s3_uri,
        transcription,
        generated_ehr,
    ):
        self.cursor.execute(
            "UPDATE audio_files SET patient_name=?, patient_id=?, audio_file_s3_uri=?, transcription=?, generated_ehr=? WHERE id=?",
            (
                patient_name,
                patient_id,
                audio_file_s3_uri,
                transcription,
                generated_ehr,
                record_id,
            ),
        )
        self.conn.commit()

    def close_connection(self):
        self.conn.close()


class RecordAudioWindow:
    def __init__(self):
        self.audio_db = AudioDatabase("patient_database.db")
        self.stream = None  # Placeholder for audio stream

        st.title(":blue[AuraSync] - AI based Clinical Documentation")
        st.subheader(
            "_Revolutionizing HealthCare with Artificial Intelligence_", divider="blue"
        )
        st.sidebar.title("Menu")
        self.choice = st.sidebar.radio(
            "Select an option:", ("Record Audio", "Upload Audio File", "View Data")
        )

        if self.choice == "Record Audio":
            self.record_audio()
        elif self.choice == "Upload Audio File":
            self.upload_audio_file()
        elif self.choice == "View Data":
            self.view_data()

    def record_audio(self):
        st.header("Record Doctor-Patient Conversation")
        patient_name = st.text_input("Patient Name")
        patient_id = st.text_input("Patient ID")

        audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key="recorder")
        if audio:
            st.audio(audio["bytes"])
            audio_data = audio["bytes"]
            self.process_audio(audio_data, patient_name, patient_id)

    def upload_audio_file(self):
        st.header("Upload Audio File")
        patient_name = st.text_input("Patient Name")
        patient_id = st.text_input("Patient ID")
        uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

        if uploaded_file:
            audio_data = uploaded_file.read()
            self.process_audio(audio_data, patient_name, patient_id)

    def process_audio(self, audio_data, patient_name, patient_id):
        progress_text = "Processing audio..."
        my_progress = st.progress(0, text=progress_text)

        if audio_data:
            my_progress.progress(0, text="Uploading audio to S3...")
            s3_key = self.upload_to_s3(audio_data, patient_name, patient_id)
            st.success("Audio uploaded to S3 successfully!")

            my_progress.progress(33, text="Transcribing audio...")
            _, transcribed_text = self.transcribe_audio(audio_data)
            st.success("Transcription complete!")
            my_progress.progress(100, text="Generating clinical notes...")
            generated_text = self.generate_text(
                transcribed_text, patient_name, patient_id
            )
            self.audio_db.insert_audio_file(
                patient_name, patient_id, s3_key, transcribed_text, generated_text
            )
            my_progress.progress(100, text="Generating EHR")
            st.success("Clinical notes generated successfully!")
            my_progress.empty()
        else:
            st.error("No audio data provided.")

    def upload_to_s3(self, audio_data, patient_name, patient_id):
        s3_key = f"{patient_name}/{patient_id}_audio.wav"
        print("Uploading")

        try:
            return s3_key  # Return the S3 key of the uploaded file
        except Exception as e:
            st.error(f"Error uploading audio to S3: {e}")
            return None

    def transcribe_audio(self, audio_data):
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_audio_file:
                temp_audio_file.write(audio_data)
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, "rb") as audio_file:
                payload = {"buffer": audio_file}
                print("Start")
                options = PrerecordedOptions(
                    smart_format=True,
                    model="nova-2",
                    language="en-IN",
                    diarize=True,
                )
                response = deepgram.listen.prerecorded.v("1").transcribe_file(
                    payload, options
                )
                print("Done")
                output_json = json.loads(response.to_json(indent=4))
                transcribed_text = output_json["results"]["channels"][0][
                    "alternatives"
                ][0]["transcript"]
            os.remove(temp_audio_file_path)

            return output_json, transcribed_text
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None, None

    def generate_text(self, transcribed_text, patient_name, patient_id):
        if not transcribed_text:
            st.error("Transcribed text is empty.")
            return None
        template_summary = """Given the provided patient-doctor dialog, write the corresponding patient information summary in JSON format.
                    Make sure to extract all the information from the dialog into the template, but do not add any new information.  If a field is not mentioned, simply write "None".
                    {
                    "visit motivation": "",
                    "admission": [
                        {
                        "reason": "",
                        "date": "",
                        "duration": "",
                        "care center details": ""
                        }
                    ],
                    "patient information": {
                        "age": "",
                        "sex": "",
                        "ethnicity": "",
                        "weight": "",
                        "height": "",
                        "family medical history": "",
                        "recent travels": "",
                        "socio economic context": "",
                        "occupation": ""
                    },
                    "patient medical history": {
                        "physiological context": "",
                        "psychological context": "",
                        "vaccination history": "",
                        "allergies": "",
                        "exercise frequency": "",
                        "nutrition": "",
                        "sexual history": "",
                        "alcohol consumption": "",
                        "drug usage": "",
                        "smoking status": ""
                    },
                    "surgeries": [{
                        "reason": "",
                        "Type": "",
                        "time": "",
                        "outcome": "",
                        "details": ""
                    }],
                    "symptoms": [
                        {
                        "name of symptom": "",
                        "intensity of symptom": "",
                        "location": "",
                        "time": "",
                        "temporalisation": "",
                        "behaviours affecting the symptom": "",
                        "details": ""
                        }
                    ],
                    "medical examinations": [{
                        "name" : "",
                        "result": "",
                        "details": ""
                    }],
                    "diagnosis tests": [
                        {
                        "test": "",
                        "severity": "",
                        "result": "",
                        "condition": "",
                        "time": "",
                        "details": ""
                        }
                    ],
                    "treatments": [
                        {
                        "name": "Treatment or medication prescribed to the patient",
                        "related condition": "Medical condition that the treatment is prescribed for",
                        "dosage": "Amount or strength of the treatment",
                        "time": "Any temporal details about when the treatment was performed",
                        "frequency": "How often the treatment is taken",
                        "duration": "",
                        "reason for taking": "The medical reason for taking the treatment",
                        "reaction to treatment": "Patient's reaction or response to the prescribed treatment",
                        "details": "All additional details about the treatment"
                        }
                    ]
                    """
        try:
            query = (
                template_summary
                + "give some recommendations to improve the patient's health for patient name: "
                + patient_name
                + "patient id: "
                + patient_id
                + ". Neglect the small conversations."
            )
            response1 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": transcribed_text},
                    {"role": "user", "content": query},
                ],
            )
            print("Generated")
            generated_summary = response1["choices"][0]["message"]["content"]

            return generated_summary
        except Exception as e:
            st.error(f"Error generating EHR: {e}")
            return None

    def display_download_button(self, text, filename):
        txt_b64 = base64.b64encode(text.encode()).decode()
        txt_href = f'<a href="data:file/txt;base64,{txt_b64}" download="{filename}.txt">Download as TXT</a>'
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        lines = text.split("\n")
        for line in lines:
            line = self.wrap_text(line, pdf)
            pdf.multi_cell(0, 10, txt=line)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        pdf_href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="{filename}.pdf">Download as PDF</a>'

        return f"{txt_href} | {pdf_href}"

    def wrap_text(self, text, pdf, max_width=200):
        """
        Wrap text to fit within the specified maximum width.
        """
        lines = []
        words = text.split()
        line = ""
        for word in words:
            if pdf.get_string_width(line + " " + word) <= max_width:
                line += " " + word
            else:
                lines.append(line.strip())
                line = word
        lines.append(line.strip())
        return "\n".join(lines)

    def display_translated_text(original_text, dest):
        translations = {
            "Tamil": "ta",
            "Gujarati": "gu",
            "Telugu": "te",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Japanese": "ja",
            "Korean": "ko",
            "Hindi": "hi",
            "Bengali": "bn",
        }
        translated = ts.translate_text(
            query_text=original_text,
            translator="google",
            from_language="en",
            to_language=translations[dest],
            limit_of_length=2000000,
        )
        if translated:
            return translated
        else:
            st.warning("Error with translation")

    def view_data(self):
        st.header("View Data")
        patient_name = st.text_input("Search by Patient Name")
        patient_id = st.text_input("Search by Patient ID")
        selected_translation = st.selectbox(
            "Select translation language:",
            [
                "Tamil",
                "Gujarati",
                "Telugu",
                "Spanish",
                "French",
                "German",
                "Japanese",
                "Korean",
                "Hindi",
                "Bengali",
            ],
            index=None,
            placeholder="Select Language...",
        )
        search_button = st.button("Search")

        if search_button:
            records = self.audio_db.fetch_records(patient_name, patient_id)
            if records:
                st.write("Found records:")
                for record in records:
                    st.write(f"Patient Name: {record[1]}")
                    st.write(f"Patient ID: {record[2]}")
                    st.write(f"Audio File S3 URI: {record[3]}")
                    st.write(f"Transcription: {record[4]}")
                    st.write(f"EHR: {record[5]}")
                    if selected_translation:
                        translated = RecordAudioWindow.display_translated_text(
                            str(record[5]), selected_translation
                        )
                        st.write(translated)
                    download_button = self.display_download_button(
                        record[5], f"{record[1]}_{record[2]}_EHR.txt"
                    )
                    st.markdown(download_button, unsafe_allow_html=True)

                    edit_button = st.button(f"Edit EHR for Record {record[0]}")
                    if edit_button:
                        edited_text = self.edit_generated_text(record[0], record[5])
                        # Update the database with the edited text
                        self.audio_db.update_record(
                            record[0],  # Assuming the first element is the record ID
                            record[1],  # Patient name
                            record[2],  # Patient ID
                            record[3],  # Audio file S3 URI
                            record[4],  # Output JSON data
                            edited_text,  # Edited generated text
                        )
                st.subheader("Edit Data")
                df_records = pd.DataFrame(
                    records,
                    columns=[
                        "ID",
                        "Patient Name",
                        "Patient ID",
                        "Audio File S3 URI",
                        "transcription",
                        "Generated EHR",
                    ],
                )
                edited_data = st.data_editor(
                    df_records, width=1000, use_container_width=True, hide_index=True
                )
                if edited_data is not None:

                    for index, row in edited_data.iterrows():
                        self.audio_db.update_record(
                            row["ID"],
                            row["Patient Name"],
                            row["Patient ID"],
                            row["Audio File S3 URI"],
                            row["transcription"],
                            row["Generated EHR"],
                        )
                    st.success("Database updated successfully!")
                self.display_pulse_spo2_temp_graph(df_records)

    def display_pulse_spo2_temp_graph(self, df_records):
        st.subheader(" Vital Signs")
        df = pd.read_excel("Wifilogs.xlsx")

        df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%m-%Y")
        df["TIME"] = pd.to_datetime(df["TIME"], format="%H:%M:%S").dt.time
        df["DATETIME"] = df.apply(
            lambda row: datetime.combine(row["DATE"].date(), row["TIME"]), axis=1
        )
        df.drop(columns=["DATE", "TIME"], inplace=True)

        st.subheader("Vital Signs Overview")

        st.line_chart(
            df.set_index("DATETIME")[["TEMPRATURE", "HEARTBEAT", "RESPIRATION"]],
            use_container_width=True,
        )

    def edit_generated_text(self, record_id, current_text):
        st.info("Editing generated text...")
        if self.session_state.edited_data is None:

            self.session_state.edited_data = self.audio_db.get_generated_text(record_id)

        edited_data = st.data_editor(
            self.session_state.edited_data, height=500, num_rows="dynamic"
        )

        if edited_data is not None:

            self.session_state.edited_data = edited_data
            print("Edited data:")
            print(self.session_state.edited_data)  # Print edited data
            print("Updating database...")
            self.update_database()
            st.success("Generated text updated successfully!")

    def update_database(self):
        for index, row in self.session_state.edited_data.iterrows():
            self.audio_db.update_record(
                row["ID"],
                row["Patient Name"],
                row["Patient ID"],
                row["Audio File S3 URI"],
                row["transcription"],
                row["Generated EHR"],
            )


def main():
    if not st.session_state.get("logged_in"):
        page = st.sidebar.radio("Select an option:", ["Login", "Signup"])

        if page == "Login":
            st.title("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_login(username, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password.")
        elif page == "Signup":
            st.title("Signup")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Signup"):
                create_user(new_username, new_password)
                st.success("Signup successful! You can now login.")

    if st.session_state.get("logged_in"):
        RecordAudioWindow()
    else:
        st.warning("Please login or signup to access the application.")


if __name__ == "__main__":
    main()
