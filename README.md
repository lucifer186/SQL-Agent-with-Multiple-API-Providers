# SQL Agent with Multiple API Providers

## Overview
A comprehensive SQL learning and analytics platform that combines natural language database querying, AI-powered explanations, visual analytics, and complete learning resources. Built with Streamlit, LangChain, and supporting both OpenAI and Groq APIs.

# Features
##  Core Database Functionality
•	Multiple Database Support: SQLite, MySQL, and custom database creation \
•	Natural Language Querying: Ask questions about your data in plain English \
•	AI-Powered SQL Generation: Automatic SQL query generation and execution \
•	Sample Database: Pre-loaded Chinook music store database for testing \
•	Custom Database Upload: Support for SQLite file uploads

## Advanced Visualization
•	Entity-Relationship Diagrams (ERD): Professional Power BI-style database visualizations \
o	Full-screen viewing mode 

## AI-Powered Database Creation
•	Custom Database Generator: Describe any business domain and AI creates appropriate schema \
•	Realistic Dummy Data: Auto-generation of sample data with proper relationships \
•	Instant Visualization: Immediate ERD generation for created databases

## SQL Learning Hub
•	SQL Q&A Assistant: Ask any SQL/RDBMS question and get detailed explanations \
•	Curated Tutorials: Links to W3Schools, GeeksforGeeks, SQLBolt, Mode Analytics \
•	Interview Preparation: Resources from DataCamp, LeetCode, HackerRank \
•	Practice Challenges: Coding platforms and skill assessments \
•	Structured Learning Paths: Beginner to advanced progression

# Technical Stack
## Core Dependencies
streamlit \
langchain-community \
langchain-openai \
langchain-groq \
sqlite3 \
pymysql \
requests \
faker \
Optional Dependencies (for advanced features) \
graphviz  # For ERD generation \
sqlalchemy  # For database introspection 

# Setup Instructions
## 1. Clone Repository 
git clone <repository-url> \
cd sql-agent-app 

## 2. Install Dependencies
`pip install streamlit langchain-community langchain-openai langchain-groq` \
`pip install plotly pandas pymysql faker requests` \
`pip install graphviz sqlalchemy  # Optional for ERD features`

## 3. Run Application
`streamlit run app10.py`

## 4. Configuration
1.	Choose API provider (OpenAI or Groq) 
2.	Enter your API key: 
   
o	OpenAI: Get from platform.openai.com/api-keys \
o	Groq: Get from console.groq.com/keys  \
4.	Select your database option \
5.	Start querying! 

## Database Options
1. Sample Chinook Database 
•	Pre-loaded music store database \
•	Ideal for learning and testing \
•	Contains customers, tracks, albums, artists, invoices 
2. Upload Custom SQLite Database 
•	Upload your own .db files \
•	Automatic validation and schema analysis \
•	Instant querying capabilities 
3. Connect to MySQL Database 
•	Remote MySQL server connection \
•	Full credential management \
•	Connection testing and validation 
4. Create Custom Database 
•	Describe any business domain \
•	AI generates appropriate schema \
•	Auto-populated with realistic data \
•	Example domains: e-commerce, hospital, school, restaurant 

## 5. SQL Learning & Practice
•	Interactive tutorials and resources \
•	Interview preparation materials \
•	Practice challenges and assessments \

## Usage Examples
Natural Language Queries 
- "How many customers are from each country?" 
- "Show me the top 10 best-selling tracks" 
- "What are the total sales by year?" 
- "Which genres are most popular?" 
Database Creation Examples 
- "Create a database for an online store with customers, products, and orders" 
- "Create a hospital management system with patients and doctors" 
- "Create a library system with books, authors, and borrowings" 

# Architecture
## Core Components 
1.	Database Manager: Handles connections to SQLite/MySQL databases 
2.	AI Agent: LangChain-powered SQL agent for natural language processing 
3.	Visualization Engine: Automatic chart generation and ERD creation 
4.	Learning Hub: Comprehensive SQL education resources
## Data Flow
1.	User inputs natural language question 
2.	LangChain agent generates appropriate SQL query 
3.	Query executed against database 
4.	Results processed and visualized 
5.	Natural language explanation provided 

## File Structure 
sql-agent-app/ \
├── app.py                 # Main Streamlit application \
├── requirements.txt       # Python dependencies \
├── README.md             # This file \
└── temp/                 # Temporary files (auto-created) 

## API Support
OpenAI Models \
•	GPT-3.5 Turbo \
•	GPT-4 \
•	GPT-4 Turbo \
•	GPT-4o \
•	GPT-4o Mini \
Groq Models \
•	Llama 3.1 8B Instant \
•	Llama 3.3 70B Versatile \
•	Qwen 3 32B \
•	Gemma2 9B IT 

## Security Considerations
•	API keys handled securely through Streamlit secrets \
•	No persistent storage of sensitive data \
•	Temporary database files auto-cleanup \
•	Connection validation and error handling 

## Performance Features
•	Connection pooling and caching \
•	Optimized query execution \
•	Progressive loading for large datasets \
•	Memory-efficient chart generation

## Educational Value
•	Perfect for SQL learning and practice \
•	Database design understanding \
•	Data visualization principles \
•	Interview preparation \
•	Real-world data analysis scenarios

# Troubleshooting
## Common Issues
1.	API Key Validation Failed: Verify your API key is correct and has sufficient credits 
2.	Database Connection Error: Check file permissions and database format 
3.	ERD Generation Failed: Install graphviz and sqlalchemy packages 
4.	MySQL Connection Issues: Verify host, port, credentials, and network access

## Error Handling
•	Comprehensive error messages with suggestions \
•	Graceful fallbacks for missing dependencies \
•	Connection retry mechanisms \ 
•	User-friendly error displays

## License
MIT License - See LICENSE file for details
