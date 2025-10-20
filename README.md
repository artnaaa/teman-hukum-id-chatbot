## Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use `miniconda` or `conda` for environment management.

### Installation

1.  **Install Miniconda (if not already installed)**

    Download and install Miniconda from the official website: <mcurl name="Miniconda Installer" url="https://docs.conda.io/en/latest/miniconda.html"></mcurl>

2.  **Create a Conda Environment**

    Open your terminal or Anaconda Prompt and create a new environment:

    ```bash
    conda create -n chatbot-env python=3.9
    conda activate chatbot-env
    ```

3.  **Install Requirements**

    Navigate to the project directory and install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application**

    1) Jalankan aplikasi utama:

    ```bash
    streamlit run streamlit_hukum_id_app.py
    ```

    2) Buka sidebar di aplikasi dan masukkan Google AI API Key Anda pada field "Google AI API Key" untuk mulai menggunakan chatbot.

    3) (Opsional) Untuk fitur RAG PDF, unggah PDF Anda di sidebar, klik "Bangun/Perbarui Indeks PDF", lalu centang "Aktifkan RAG dari PDF".

### Running with Docker (Optional)

1.  **Build the Docker Image**

    Navigate to the project directory and build the Docker image:

    ```bash
    docker build -t chatbot-streamlit-demo .
    ```

2.  **Run the Docker Container**

    Run the Docker container, mapping port 8501:

    ```bash
    docker run -p 8501:8501 chatbot-streamlit-demo
    ```

    The application will be accessible in your web browser at `http://localhost:8501`.

## Code Structure

- streamlit_hukum_id_app.py: Main Streamlit app untuk chatbot Teman Hukum ID (sidebar parameter, RAG PDF, chat UI).
- database_hukum_tools.py: Utilitas database untuk menyimpan histori percakapan dan potongan PDF.
- streamlit_chat_app.py: Varian sederhana/eksperimental untuk antarmuka chat.
- streamlit_app_basic.py: Contoh aplikasi Streamlit dasar (opsional).
- streamlit_react_app.py: Contoh integrasi (opsional).
- streamlit_react_tools_app.py: Contoh integrasi tools (opsional).
- requirements.txt: Daftar dependensi Python.
