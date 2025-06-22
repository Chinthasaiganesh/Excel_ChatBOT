# Excel ChatBot

A Streamlit-based web application that allows users to upload Excel files, browse their contents, and interact with the data through a conversational AI interface.

## Features

- **Excel File Management**: Upload and manage Excel files with multiple sheets
- **Data Preview**: Browse and preview uploaded Excel files and their sheets
- **AI-Powered Chat**: Chat with your Excel data using natural language
- **Data Analysis**: Get insights and analysis of your tabular data
- **Persistent Storage**: All uploaded files and their metadata are stored in a database

## Project Structure

- `app.py`: Main Streamlit application with UI components and navigation
- `database.py`: Database connection and session management
- `models.py`: SQLAlchemy models for Excel files and tables
- `excel_parser.py`: Handles Excel file parsing and data extraction
- `serializers.py`: Data serialization utilities
- `ai_utils.py`: AI and natural language processing utilities
- `requirements.txt`: Python dependencies

## Setup

1. **Prerequisites**
   - Python 3.8+
   - PostgreSQL (or SQLite for development)

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the project root with the following variables:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/excel_chatbot
   Groq_API_KEY=your_Groq_api_key
   ```

4. **Initialize Database**
   The application will automatically create the required database tables on first run.

## Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Upload Excel Files**
   - Navigate to the "Upload" page
   - Select one or more Excel files to upload
   - The system will process and store the files

2. **Browse Files**
   - View all uploaded Excel files
   - Select a file to see its sheets and data

3. **Chat with Data**
   - Go to the "Chat" page
   - Select a table to analyze
   - Ask questions about your data in natural language

## Dependencies

- Streamlit: Web application framework
- Pandas: Data manipulation
- SQLAlchemy: ORM and database management
- LangChain: AI and language model integration
- FAISS: Vector similarity search
- Transformers: NLP models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
