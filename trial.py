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
import base64

# Set page config
st.set_page_config(page_title="AuraSync", page_icon="🧑‍⚕️")

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Audio database class

class AudioDatabase:
    def __init__(self, db_name, key):
        self.db_name = db_name
        self.key = key
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

    def _encrypt(self, data):
        f = Fernet(self.key)
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data

    def _decrypt(self, encrypted_data):
        f = Fernet(self.key)
        decrypted_data = f.decrypt(encrypted_data).decode()
        return decrypted_data

    def insert_audio_file(
        self,
        patient_name,
        patient_id,
        audio_file_s3_uri,
        transcription,
        generated_ehr,
    ):
        patient_name_encrypted = self._encrypt(patient_name)
        patient_id_encrypted = self._encrypt(patient_id)
        audio_file_s3_uri_encrypted = self._encrypt(audio_file_s3_uri)
        transcription_encrypted = self._encrypt(transcription)
        generated_ehr_encrypted = self._encrypt(generated_ehr)
        
        self.cursor.execute(
            "INSERT INTO audio_files (patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr) VALUES (?, ?, ?, ?, ?)",
            (
                patient_name_encrypted,
                patient_id_encrypted,
                audio_file_s3_uri_encrypted,
                transcription_encrypted,
                generated_ehr_encrypted,
            ),
        )
        self.conn.commit()

    def fetch_records(self, patient_name=None, patient_id=None):
        if patient_name:
            patient_name_encrypted = self._encrypt(patient_name)
        if patient_id:
            patient_id_encrypted = self._encrypt(patient_id)

        if patient_name and patient_id:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_name=? AND patient_id=?",
                (patient_name_encrypted, patient_id_encrypted),
            )
        elif patient_name:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_name=?",
                (patient_name_encrypted,),
            )
        elif patient_id:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files WHERE patient_id=?",
                (patient_id_encrypted,),
            )
        else:
            self.cursor.execute(
                "SELECT id, patient_name, patient_id, audio_file_s3_uri, transcription, generated_ehr FROM audio_files"
            )

        records = self.cursor.fetchall()
        decrypted_records = []
        for record in records:
            decrypted_record = (
                self._decrypt(record[0]),
                self._decrypt(record[1]),
                self._decrypt(record[2]),
                self._decrypt(record[3]),
                self._decrypt(record[4]),
                self._decrypt(record[5]),
            )
            decrypted_records.append(decrypted_record)
        return decrypted_records

    def update_record(
        self,
        record_id,
        patient_name,
        patient_id,
        audio_file_s3_uri,
        transcription,
        generated_ehr,
    ):
        patient_name_encrypted = self._encrypt(patient_name)
        patient_id_encrypted = self._encrypt(patient_id)
        audio_file_s3_uri_encrypted = self._encrypt(audio_file_s3_uri)
        transcription_encrypted = self._encrypt(transcription)
        generated_ehr_encrypted = self._encrypt(generated_ehr)
        
        self.cursor.execute(
            "UPDATE audio_files SET patient_name=?, patient_id=?, audio_file_s3_uri=?, transcription=?, generated_ehr=? WHERE id=?",
            (
                patient_name_encrypted,
                patient_id_encrypted,
                audio_file_s3_uri_encrypted,
                transcription_encrypted,
                generated_ehr_encrypted,
                record_id,
            ),
        )
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

# Record audio window
class RecordAudioWindow:
    def __init__(self):
        new_key = Fernet.generate_key()
        encoded_key = base64.urlsafe_b64encode(new_key)

        # Store the encoded key securely, for example, in an environment variable
        os.environ["DATABASE_KEY"] = encoded_key.decode()

        # Retrieve the key from the environment variable
        encoded_key = os.getenv("DATABASE_KEY")

        # Decode the key from base64
        key = base64.urlsafe_b64decode(encoded_key)

        # Initialize the AudioDatabase class with the retrieved key
        self.audio_db = AudioDatabase("audio_data.db", key)
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

        # st.write("Record  Conversation")
        audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key="recorder")
        if audio:
            st.audio(audio["bytes"])
            # st.write("Temporary WAV file saved at:", temp_wav_file.name)
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
        # st.info("Processing audio...")
        progress_text = "Processing audio..."
        my_progress = st.progress(0, text=progress_text)

        if audio_data:
            # Start uploading audio to S3
            my_progress.progress(0, text="Uploading audio to S3...")
            s3_key = self.upload_to_s3(audio_data, patient_name, patient_id)
            st.success("Audio uploaded to S3 successfully!")

            # Start transcription
            my_progress.progress(33, text="Transcribing audio...")
            _, transcribed_text = self.transcribe_audio(audio_data)
            st.success("Transcription complete!")

            # Start generating clinical notes
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
        # Initialize the S3 client
        # s3_client = boto3.client("s3")

        # Define the S3 key where the audio file will be stored
        s3_key = f"{patient_name}/{patient_id}_audio.wav"
        print("Uploading")

        try:
            # Upload the audio data to S3
            # s3_client.put_object(Body=audio_data, Bucket=S3_BUCKET_NAME, Key=s3_key)
            return s3_key  # Return the S3 key of the uploaded file
        except Exception as e:
            st.error(f"Error uploading audio to S3: {e}")
            return None

    def transcribe_audio(self, audio_data):
        # Initialize the Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        try:
            # Transcribe the audio data using Deepgram
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

            # Remove the temporary audio file
            os.remove(temp_audio_file_path)

            return output_json, transcribed_text
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return None, None

    def generate_text(self, transcribed_text, patient_name, patient_id):
        if not transcribed_text:
            st.error("Transcribed text is empty.")
            return None

        try:
            # Define the query to be used for text generation
            query = (
                "You are an AI trained to generate detailed Clinical Notes in perfect EHR format from the provided Doctor-Patient Conversation and give some recommendations to improve the patient's health for patient name: "
                + patient_name
                + "patient id: "
                + patient_id
                + ". Neglect the small conversations."
            )

            # Create a chat completion using OpenAI's GPT-3.5 model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": transcribed_text},
                    {"role": "user", "content": query},
                ],
            )
            print("Generated")

            # Extract the generated text from the response
            generated_text = response["choices"][0]["message"]["content"]

            return generated_text
        except Exception as e:
            st.error(f"Error generating EHR: {e}")
            return None

    # Inside the RecordAudioWindow class
    def display_download_button(self, text, filename):
        # Generate a download link for the text content in TXT format
        txt_b64 = base64.b64encode(text.encode()).decode()
        txt_href = f'<a href="data:file/txt;base64,{txt_b64}" download="{filename}.txt">Download as TXT</a>'

        # Generate a download link for the text content in PDF format
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Split text into lines and add each line to the PDF
        lines = text.split("\n")
        for line in lines:
            # Ensure that the line does not exceed the width of the PDF
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

    def view_data(self):
        st.header("View Data")
        patient_name = st.text_input("Search by Patient Name")
        patient_id = st.text_input("Search by Patient ID")
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

                    # Add download button for generated text
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

                # Display data editor for editing and updating the database
                st.subheader("Edit Data")
                df_records = pd.DataFrame(
                    records,
                    columns=[
                        "ID",
                        "Patient Name",
                        "Patient ID",
                        "Audio File S3 URI",
                        "Transcription",
                        "Generated EHR",
                    ],
                )
                edited_data = st.data_editor(
                    df_records, width=1000, use_container_width=True, hide_index=True
                )
                if edited_data is not None:
                    # If data is edited, update the database
                    for index, row in edited_data.iterrows():
                        self.audio_db.update_record(
                            row["ID"],
                            row["Patient Name"],
                            row["Patient ID"],
                            row["Audio File S3 URI"],
                            row["Transcription"],
                            row["Generated EHR"],
                        )
                    st.success("Database updated successfully!")

                # Display graph of Pulse, sPO2, and Temperature
                self.display_pulse_spo2_temp_graph(df_records)

    def display_pulse_spo2_temp_graph(self, df_records):
        st.subheader(" Vital Signs")

        # Read data from Excel file into a DataFrame
        df = pd.read_excel("Wifilogs.xlsx")

        df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%m-%Y")
        df["TIME"] = pd.to_datetime(df["TIME"], format="%H:%M:%S").dt.time

        # Combine DATE and TIME columns into a single datetime column
        df["DATETIME"] = df.apply(
            lambda row: datetime.combine(row["DATE"].date(), row["TIME"]), axis=1
        )

        # Drop the original DATE and TIME columns
        df.drop(columns=["DATE", "TIME"], inplace=True)

        st.subheader("Temperature(°C)")
        st.area_chart(df.set_index("DATETIME")["TEMPRATURE"], color=["#ed6868"])

        # Plot the area chart for HEARTBEAT
        st.subheader("Heartbeat")
        st.area_chart(df.set_index("DATETIME")["HEARTBEAT"], color=["#5d5dc9"])

        # Plot the area chart for RESPIRATION
        st.subheader("Respiration")
        st.area_chart(df.set_index("DATETIME")["RESPIRATION"], color=["#9fe09f"])

    def edit_generated_text(self, record_id, current_text):
        st.info("Editing generated text...")
        if self.session_state.edited_data is None:
            # Retrieve the current generated text from the database based on record_id
            self.session_state.edited_data = self.audio_db.get_generated_text(record_id)

        # Display the current generated text in a data editor for editing
        edited_data = st.data_editor(
            self.session_state.edited_data, height=500, num_rows="dynamic"
        )

        if edited_data is not None:
            # Update the edited data in session state
            self.session_state.edited_data = edited_data
            print("Edited data:")
            print(self.session_state.edited_data)  # Print edited data
            # Update the database with the edited data
            print("Updating database...")
            self.update_database()
            st.success("Generated text updated successfully!")

    def update_database(self):
        # Iterate over the edited data and update the database
        for index, row in self.session_state.edited_data.iterrows():
            self.audio_db.update_record(
                row["ID"],
                row["Patient Name"],
                row["Patient ID"],
                row["Audio File S3 URI"],
                row["Transcription"],
                row["Generated EHR"],
            )


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


# Function to create new user
def create_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash),
    )
    conn.commit()


# Function to verify login
def verify_login(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, password_hash),
    )
    return cursor.fetchone() is not None


# Main Streamlit app
def main():
    if not st.session_state.get("logged_in"):
        # Sidebar navigation
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

    # Main content
    if st.session_state.get("logged_in"):
        st.title("Welcome to AuraSync")
        RecordAudioWindow()
    else:
        st.warning("Please login or signup to access the application.")


if __name__ == "__main__":
    main()
