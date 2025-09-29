import streamlit as st
import sqlite3
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import urllib.parse
import requests
import tempfile
import pymysql  # For MySQL connection
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import base64
from io import BytesIO
from faker import Faker
# Try importing optional libraries for ERD generation
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    
try:
    from sqlalchemy import create_engine, MetaData, inspect
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="SQL Agent with Multiple Providers",
    page_icon="🗃️",
    layout="wide"
)

# Title and description
st.title("🗃️ SQL Agent with Multiple API Providers")
st.markdown("Ask natural language questions about your database and get SQL-powered answers!")

# Sidebar for configuration
st.sidebar.header("Configuration")

# API Provider Selection
api_provider = st.sidebar.selectbox(
    "Choose API Provider:",
    ["OpenAI", "Groq"],
    index=0
)

# Dynamic API Key and Model Selection based on provider
if api_provider == "OpenAI":
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
    selected_model = st.sidebar.selectbox(
        "Select OpenAI Model:",
        models,
        index=0
    )
else:  # Groq
    api_key = st.sidebar.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your API key from https://console.groq.com/keys"
    )
    
    models = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "gemma2-9b-it" ]
    selected_model = st.sidebar.selectbox(
        "Select Groq Model:",
        models,
        index=0
    )

# Debug info
if api_key:
    st.sidebar.success(f"✅ {api_provider} API Key entered ({len(api_key)} characters)")
    # Test the API key
    try:
        if api_provider == "OpenAI":
            test_llm = ChatOpenAI(
                openai_api_key=api_key, 
                model=selected_model, 
                temperature=0
            )
        else:  # Groq
            test_llm = ChatGroq(
                groq_api_key=api_key,
                model=selected_model,
                temperature=0
            )
        # Simple test to validate key
        response = test_llm.invoke("Hello")
        st.sidebar.success(f"✅ {api_provider} API Key validated!")
    except Exception as e:
        st.sidebar.error(f"❌ {api_provider} API Key validation failed: {str(e)}")
else:
    st.sidebar.info(f"🔑 Please enter your {api_provider} API key above")

# Database options
db_option = st.sidebar.radio(
    "Choose Database or SQL Learning Option:",
    ["Use Sample Chinook Database", "Upload Custom SQLite Database", "Connect to MySQL Database", "Create Custom Database", "SQL Learning & Practice"]
)

if 'selected_db_option' not in st.session_state:
    st.session_state['selected_db_option'] = db_option

# If user changed DB option
if st.session_state['selected_db_option'] != db_option:
    # Clear old DB/agent/chat session states
    for key in ['db_path', 'db_uri', 'db', 'agent', 'db_uploaded', 'chat_history', 'mysql_details', 'created_db_path', 'erd_diagram', 'erd_explanations']:
        if key in st.session_state:
            del st.session_state[key]
    # Update to new selection
    st.session_state['selected_db_option'] = db_option


def download_chinook_db():
    """Download the Chinook sample database"""
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    
    with st.spinner("Downloading Chinook sample database..."):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            st.error(f"Failed to download database: {str(e)}")
            return None

def test_mysql_connection(host, port, user, password, database):
    """Test MySQL database connection"""
    try:
        connection = pymysql.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database,
            connect_timeout=10
        )
        
        # Test query
        cursor = connection.cursor()
        cursor.execute(f"USE `{database}`;")
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return True, len(tables), [table[0] for table in tables]
    except Exception as e:
        return False, 0, str(e)

def create_database_schema_and_data(description, api_key, selected_model, api_provider):
    """Create a database schema and populate it with dummy data based on description"""
    try:
        # Initialize LLM
        if api_provider == "OpenAI":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model=selected_model,
                temperature=0.3
            )
        else:  # Groq
            llm = ChatGroq(
                groq_api_key=api_key,
                model=selected_model,
                temperature=0.3
            )
        
        # Create schema generation prompt
        schema_prompt = f"""
        Based on the following description, create a SQLite database schema with CREATE TABLE statements.
        Make the schema realistic and include appropriate data types, primary keys, and foreign keys where needed.
        
        Description: {description}
        
        Requirements:
        1. Create 3-7 tables maximum
        2. Include primary keys and foreign keys where appropriate
        3. Use realistic column names and data types
        4. Add constraints where needed
        5. Return ONLY the SQL CREATE TABLE statements, no explanations
        
        Format your response as clean SQL statements separated by semicolons.
        """
        
        with st.spinner("🤖 Generating database schema..."):
            schema_response = llm.invoke(schema_prompt)
            schema_sql = schema_response.content.strip()
        
        # Clean up the schema SQL
        import re
        schema_sql = re.sub(r'```sql\s*', '', schema_sql)
        schema_sql = re.sub(r'\s*```', '', schema_sql)
        
        st.success("✅ Database schema generated!")
        
        # Show the generated schema
        with st.expander("📋 Generated Database Schema"):
            st.code(schema_sql, language="sql")
        
        # Create temporary database file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        # Create database and tables
        conn = sqlite3.connect(temp_file.name)
        cursor = conn.cursor()
        
        # Execute schema creation
        try:
            # Split statements and execute them
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            for statement in statements:
                if statement.upper().startswith('CREATE'):
                    cursor.execute(statement)
            
            conn.commit()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out system tables like sqlite_sequence
            tables = [table for table in all_tables if not table.startswith('sqlite_')]
            
            if not tables:
                st.error("❌ No user tables were created. Please try a different description.")
                conn.close()
                os.unlink(temp_file.name)
                return None
            
            st.success(f"✅ Created {len(tables)} tables: {', '.join(tables)}")
            if len(all_tables) > len(tables):
                st.info(f"📝 Note: {len(all_tables) - len(tables)} system tables were also created automatically")
            
        except Exception as schema_error:
            st.error(f"❌ Error creating schema: {str(schema_error)}")
            conn.close()
            os.unlink(temp_file.name)
            return None
        
        # Generate dummy data for each table
        with st.spinner("🎲 Generating dummy data..."):
            for table in tables:
                try:
                    # Get table structure
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns_info = cursor.fetchall()
                    
                    if not columns_info:
                        st.warning(f"⚠️ Could not get column info for table '{table}', skipping...")
                        continue
                    
                    # Create data generation prompt
                    columns_desc = []
                    for col_info in columns_info:
                        # Handle different column info formats
                        if len(col_info) >= 6:
                            cid, col_name, col_type, not_null, default_val, pk = col_info[:6]
                        elif len(col_info) >= 3:
                            col_name, col_type, pk = col_info[1], col_info[2], col_info[-1]
                        else:
                            continue  # Skip malformed column info
                            
                        columns_desc.append(f"{col_name} ({col_type})" + (" PRIMARY KEY" if pk else ""))
                    
                    if not columns_desc:
                        st.warning(f"⚠️ No valid columns found for table '{table}', skipping...")
                        continue
                    
                    data_prompt = f"""
                    Generate realistic dummy data for a SQLite table named '{table}' with the following columns:
                    {', '.join(columns_desc)}
                    
                    Context: {description}
                    
                    Requirements:
                    1. Generate INSERT statements for 15-25 rows of data
                    2. Make the data realistic and related to the context
                    3. Respect foreign key relationships if any exist
                    4. Use appropriate data formats (dates, numbers, text)
                    5. Return ONLY the SQL INSERT statements, no explanations
                    6. Ensure primary key values are unique and sequential (1, 2, 3, etc.)
                    7. For foreign keys, use values that likely exist in referenced tables
                    
                    Format: INSERT INTO {table} (columns) VALUES (values);
                    """
                    
                    try:
                        data_response = llm.invoke(data_prompt)
                        insert_sql = data_response.content.strip()
                        
                        # Clean up the insert SQL
                        insert_sql = re.sub(r'```sql\s*', '', insert_sql)
                        insert_sql = re.sub(r'\s*```', '', insert_sql)
                        
                        # Execute insert statements
                        insert_statements = [stmt.strip() for stmt in insert_sql.split(';') if stmt.strip()]
                        successful_inserts = 0
                        
                        for statement in insert_statements:
                            if statement.upper().startswith('INSERT'):
                                try:
                                    cursor.execute(statement)
                                    successful_inserts += 1
                                except sqlite3.Error as insert_error:
                                    # Skip problematic inserts
                                    continue
                        
                        if successful_inserts > 0:
                            st.info(f"📊 Inserted {successful_inserts} rows into table '{table}'")
                            conn.commit()
                        else:
                            st.warning(f"⚠️ Could not insert AI-generated data into table '{table}' - using fallback data")
                            # Generate simple fallback data
                            generate_fallback_data(cursor, table, columns_info, 15)
                            conn.commit()
                            
                    except Exception as data_error:
                        st.warning(f"⚠️ Error generating data for '{table}': {str(data_error)} - using fallback")
                        generate_fallback_data(cursor, table, columns_info, 15)
                        conn.commit()
                        
                except Exception as table_error:
                    st.error(f"❌ Error processing table '{table}': {str(table_error)}")
                    continue
        
        # Verify data was inserted
        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_rows += count
            
        cursor.close()
        conn.close()
        
        if total_rows > 0:
            st.success(f"✅ Database created successfully with {total_rows} total rows across {len(tables)} tables!")
            return temp_file.name
        else:
            st.error("❌ Database created but no data was inserted.")
            os.unlink(temp_file.name)
            return None
            
    except Exception as e:
        st.error(f"❌ Error creating database: {str(e)}")
        return None

def generate_erd_diagram(db_path_or_uri, db_type, api_key, selected_model, api_provider):
    """Generate ERD diagram for any database type"""
    if not GRAPHVIZ_AVAILABLE or not SQLALCHEMY_AVAILABLE:
        st.warning("⚠️ ERD generation requires 'graphviz' and 'sqlalchemy' packages. Please install them.")
        st.code("pip install graphviz sqlalchemy", language="bash")
        return None, None
    
    try:
        # Create SQLAlchemy engine based on database type
        if db_type == "sqlite":
            engine = create_engine(f'sqlite:///{db_path_or_uri}')
        else:  # MySQL
            engine = create_engine(db_path_or_uri)
        
        inspector = inspect(engine)
        
        # Get tables and their relationships
        tables = inspector.get_table_names()
        relationships = []
        table_columns = {}
        
        # Filter out system tables
        user_tables = [t for t in tables if not t.startswith('sqlite_')]
        
        # Analyze each table
        for table in user_tables:
            try:
                columns = inspector.get_columns(table)
                foreign_keys = inspector.get_foreign_keys(table)
                primary_keys = inspector.get_pk_constraint(table)['constrained_columns']
                
                table_columns[table] = {
                    'columns': columns,
                    'primary_keys': primary_keys,
                    'foreign_keys': foreign_keys
                }
                
                # Extract relationships
                for fk in foreign_keys:
                    if fk['constrained_columns'] and fk['referred_columns']:
                        relationships.append({
                            'from_table': table,
                            'from_column': fk['constrained_columns'][0],
                            'to_table': fk['referred_table'],
                            'to_column': fk['referred_columns'][0]
                        })
            except Exception as e:
                st.warning(f"⚠️ Could not analyze table '{table}': {str(e)}")
                continue
        
        if not table_columns:
            st.warning("⚠️ No tables found or accessible for ERD generation.")
            return None, None
        
        # Generate compact ERD using Graphviz (Power BI style)
        dot = graphviz.Digraph(comment='Database ERD', format='svg')
        
        # Very compact layout settings for better fit
        dot.attr(
            rankdir='TB',      # Top to Bottom
            size='8,10',       # Smaller size for better fit
            dpi='120',         # Lower DPI for smaller file size
            bgcolor='white',
            fontname='Arial',
            splines='ortho',   # Straight lines like Power BI
            concentrate='true', # Reduce edge crossings
            overlap='false',
            sep='+5',          # Smaller separation between nodes
            esep='+2',         # Smaller separation between edges
            nodesep='0.3',     # Minimum space between nodes
            ranksep='0.4',     # Minimum space between ranks
            margin='0.2'       # Smaller margins
        )
        
        # Compact node styling (Power BI-like)
        dot.attr('node', 
                 shape='record',
                 style='filled,rounded',
                 fillcolor='#f8f9fa',
                 color='#2c3e50',
                 fontname='Arial',
                 fontsize='8',      # Smaller font
                 margin='0.05,0.02', # Tighter margins
                 height='0.3',      # Fixed height
                 width='1.5'        # Fixed width for consistency
        )
        
        # Compact edge styling (Power BI-like)
        dot.attr('edge',
                 color='#3498db',
                 fontname='Arial',
                 fontsize='7',      # Smaller font for edges
                 fontcolor='#2c3e50',
                 penwidth='1.2',    # Thinner lines
                 arrowsize='0.6',   # Smaller arrows
                 len='1.0'          # Shorter edge length preference
        )
        
        # Add tables as nodes with compact layout
        for table, info in table_columns.items():
            # Create compact table node label
            label_parts = [f"<B>{table}</B>"]
            
            # Show fewer columns for compactness (max 6 instead of 8)
            columns_to_show = info['columns'][:6]
            remaining_cols = len(info['columns']) - 6
            
            for col in columns_to_show:
                col_name = col['name']
                col_type = str(col['type'])
                
                # Shorten data types more aggressively for compactness
                if 'VARCHAR' in col_type.upper():
                    col_type = 'VARCHAR'
                elif 'INTEGER' in col_type.upper() or 'INT' in col_type.upper():
                    col_type = 'INT'
                elif 'DECIMAL' in col_type.upper() or 'REAL' in col_type.upper() or 'FLOAT' in col_type.upper():
                    col_type = 'NUM'
                elif 'TEXT' in col_type.upper():
                    col_type = 'TEXT'
                elif 'DATE' in col_type.upper() or 'TIME' in col_type.upper():
                    col_type = 'DATE'
                elif 'BOOL' in col_type.upper():
                    col_type = 'BOOL'
                else:
                    col_type = col_type[:6]  # Truncate long type names more
                
                # Create very compact column representation
                col_label = f"{col_name}:{col_type}"
                
                # Mark primary keys with key icon
                if col_name in info['primary_keys']:
                    col_label = f"🔑{col_label}"
                
                # Mark foreign keys with link icon
                for fk in info['foreign_keys']:
                    if col_name in fk['constrained_columns']:
                        col_label = f"🔗{col_label}"
                        break
                
                label_parts.append(col_label)
            
            # Add "..." if there are more columns
            if remaining_cols > 0:
                label_parts.append(f"+{remaining_cols} more")
            
            # Create the final label
            label = "{" + "|".join(label_parts) + "}"
            
            # Color coding based on table type/role (lighter colors for compactness)
            fillcolor = '#f8f9fa'  # Default light gray
            if any(keyword in table.lower() for keyword in ['customer', 'user', 'client', 'member']):
                fillcolor = '#e8f5e8'  # Light green for customer-related
            elif any(keyword in table.lower() for keyword in ['order', 'purchase', 'transaction', 'payment', 'invoice']):
                fillcolor = '#fff2e8'  # Light orange for order-related
            elif any(keyword in table.lower() for keyword in ['product', 'item', 'inventory', 'stock', 'goods']):
                fillcolor = '#e8f0ff'  # Light blue for product-related
            
            dot.node(table, label=label, fillcolor=fillcolor)
        
        # Add relationships as edges with simpler labeling
        relationship_counts = {}
        for rel in relationships:
            # Simplify edge labels for compactness
            rel_key = f"{rel['from_table']}-{rel['to_table']}"
            if rel_key not in relationship_counts:
                relationship_counts[rel_key] = 0
            relationship_counts[rel_key] += 1
            
            # Create very simple edge label
            edge_label = f"{rel['from_column']}→{rel['to_column']}"
            
            # Determine relationship cardinality (simplified)
            cardinality = "1:N"  # Default to one-to-many
            if 'id' in rel['to_column'].lower():
                cardinality = "N:1"
            
            dot.edge(
                rel['from_table'], 
                rel['to_table'],
                label=f"{cardinality}",  # Only show cardinality for compactness
                color='#3498db',
                fontcolor='#2c3e50'
            )
        
        # Generate relationship explanations using AI
        if api_provider == "OpenAI":
            llm = ChatOpenAI(openai_api_key=api_key, model=selected_model, temperature=0.3)
        else:
            llm = ChatGroq(groq_api_key=api_key, model=selected_model, temperature=0.3)
        
        explanation_prompt = f"""
        Based on the following database schema relationships, explain why each relationship type makes sense from a business logic perspective.
        Tables: {list(table_columns.keys())}
        Relationships found:
        {json.dumps(relationships, indent=2)}
        Table structures (showing primary keys and foreign keys):
        {json.dumps({k: {
            'columns': [col['name'] for col in v['columns']],
            'primary_keys': v['primary_keys'],
            'foreign_keys': [fk['constrained_columns'] for fk in v['foreign_keys']]
        } for k, v in table_columns.items()}, indent=2)}
        
        For each relationship, provide a concise explanation covering:
        1. What type of relationship it represents (One-to-Many, Many-to-One, Ont-to-One etc.)
        2. Why this relationship makes business sense
        3. A brief real-world example
        
        Keep explanations concise and educational. Format as bullet points.
        """
        
        try:
            explanation_response = llm.invoke(explanation_prompt)
            explanations = explanation_response.content
        except Exception as e:
            explanations = f"Unable to generate relationship explanations: {str(e)}"
        
        return dot, explanations
        
    except Exception as e:
        st.error(f"❌ Error generating ERD: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None, None

def display_erd_with_controls(dot_diagram, explanations):
    """Display ERD with zoom and scroll controls"""
    if not dot_diagram:
        return
    
    st.markdown("### 🗺️ Entity-Relationship Diagram")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        zoom_level = st.selectbox("🔍 Zoom Level", ["50%", "75%", "100%", "125%", "150%"], index=2)
    with col2:
        show_full_screen = st.button("🔍 Full Screen View")
    with col3:
        show_raw_dot = st.button("🔧 View DOT Source")
    
    # Convert zoom level to scale
    zoom_scale = int(zoom_level.replace('%', '')) / 100.0
    
    try:
        # Generate SVG with optimized size
        svg_data = dot_diagram.pipe(format='svg')
        if isinstance(svg_data, bytes):
            svg_content = svg_data.decode('utf-8')
        else:
            svg_content = svg_data
        
        # Create custom HTML container with proper scrolling
        container_height = 500
        svg_container = f"""
        <div style="
            width: 100%;
            height: {container_height}px;
            overflow: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: white;
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                transform: scale({zoom_scale});
                transform-origin: top left;
                padding: 20px;
                display: inline-block;
            ">
                {svg_content}
            </div>
        </div>
        <div style="
            margin-top: 8px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 12px;
            color: #6c757d;
            text-align: center;
        ">
            💡 Use mouse wheel to scroll. Drag scrollbars to navigate the diagram.
        </div>
        """
        
        # Display the container
        st.components.v1.html(svg_container, height=620, scrolling=False)
        
        # Full screen view
        if show_full_screen:
            st.markdown("---")
            st.markdown("### 🔍 Full Screen ERD View")
            
            col_close, col_zoom_full = st.columns([1, 3])
            with col_close:
                if st.button("❌ Close Full Screen"):
                    st.rerun()
            with col_zoom_full:
                full_zoom = st.selectbox("Full Screen Zoom", ["50%", "75%", "100%", "125%", "150%", "200%"], index=2, key="full_zoom")
            
            full_zoom_scale = int(full_zoom.replace('%', '')) / 100.0
            
            # Full screen container with larger height
            full_screen_container = f"""
            <div style="
                width: 100%;
                height: 700px;
                overflow: auto;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                background-color: white;
                position: relative;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            ">
                <div style="
                    transform: scale({full_zoom_scale});
                    transform-origin: top left;
                    padding: 20px;
                    display: inline-block;
                    min-width: 100%;
                    min-height: 100%;
                ">
                    {svg_content}
                </div>
            </div>
            <div style="
                margin-top: 8px;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
                font-size: 12px;
                color: #6c757d;
                text-align: center;
            ">
                💡 Full screen mode - Use scrollbars or mouse wheel to navigate the entire diagram
            </div>
            """
            st.components.v1.html(full_screen_container, height=750, scrolling=False)
        
        # Raw DOT source
        if show_raw_dot:
            with st.expander("🔧 Raw DOT Source Code", expanded=False):
                st.code(dot_diagram.source, language="dot")
        
    except Exception as e:
        st.error(f"Error displaying ERD: {str(e)}")
        # Fallback to standard Streamlit display
        try:
            st.graphviz_chart(dot_diagram.source, width='stretch')
        except Exception as fallback_error:
            st.error(f"Fallback display also failed: {str(fallback_error)}")
            st.code(dot_diagram.source, language="dot")
    
    # Show explanations
    if explanations:
        st.markdown("---")
        with st.expander("🧠 Relationship Explanations", expanded=False):
            st.markdown(explanations)
            
        # Legend
        with st.expander("🔍 ERD Legend", expanded=False):
            st.markdown("""
            **Symbols:**
            - 🔑 **Primary Key** - Uniquely identifies each record in the table
            - 🔗 **Foreign Key** - References a primary key in another table
            - **Bold text** - Primary key columns
            - *Italic text* - Foreign key columns
            
            **Relationship Types:**
            - **1:N (One-to-Many)**: One record in the first table can relate to many records in the second
            - **N:1 (Many-to-One)**: Many records in the first table relate to one record in the second  
            - **1:1 (One-to-One)**: One record in each table relates to exactly one record in the other
            
            **Color Coding:**
            - 🟢 **Green**: Customer/User related tables
            - 🟠 **Orange**: Order/Transaction related tables  
            - 🔵 **Blue**: Product/Item related tables
            - ⚪ **Gray**: General/Other tables
            
            **Navigation:**
            - Use zoom controls to resize the diagram
            - Scroll within the diagram container using mouse wheel or scrollbars
            - Click "Full Screen View" for larger viewing area
            """)
    
    return True

def generate_fallback_data(cursor, table_name, columns_info, num_rows=10):
    """Generate simple fallback data when AI generation fails"""
    fake = Faker()
    
    for i in range(num_rows):
        values = []
        columns = []
        
        for col_info in columns_info:
            # SQLite PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
            if len(col_info) == 6:
                cid, col_name, col_type, not_null, default_val, pk = col_info
            else:
                # Fallback for unexpected format
                col_name = col_info[1] if len(col_info) > 1 else f"col_{i}"
                col_type = col_info[2] if len(col_info) > 2 else "TEXT"
                pk = col_info[-1] if len(col_info) > 0 else 0
            
            columns.append(col_name)
            
            # Generate data based on column type and name
            if pk:  # Primary key
                values.append(str(i + 1))
            elif 'id' in col_name.lower() and col_name.lower() != 'id':
                values.append(str(random.randint(1, 5)))  # Foreign key
            elif col_type.upper().startswith('TEXT') or col_type.upper().startswith('VARCHAR'):
                if 'name' in col_name.lower():
                    values.append(f"'{fake.name()}'")
                elif 'email' in col_name.lower():
                    values.append(f"'{fake.email()}'")
                elif 'address' in col_name.lower():
                    values.append(f"'{fake.address().replace('\n', ', ')}'")
                elif 'phone' in col_name.lower():
                    values.append(f"'{fake.phone_number()}'")
                elif 'description' in col_name.lower():
                    values.append(f"'{fake.text(max_nb_chars=100)}'")
                else:
                    values.append(f"'{fake.word()} {i+1}'")
            elif col_type.upper().startswith('INTEGER') or col_type.upper().startswith('INT'):
                if 'price' in col_name.lower() or 'amount' in col_name.lower():
                    values.append(str(random.randint(10, 1000)))
                elif 'quantity' in col_name.lower():
                    values.append(str(random.randint(1, 100)))
                else:
                    values.append(str(random.randint(1, 100)))
            elif col_type.upper().startswith('REAL') or col_type.upper().startswith('DECIMAL'):
                values.append(str(round(random.uniform(10.0, 1000.0), 2)))
            elif col_type.upper().startswith('DATE'):
                random_date = fake.date_between(start_date='-1y', end_date='today')
                values.append(f"'{random_date}'")
            else:
                values.append(f"'Sample {i+1}'")
        
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)})"
        try:
            cursor.execute(insert_sql)
        except sqlite3.Error as e:
            # More specific error handling
            st.warning(f"⚠️ Could not insert row {i+1} into {table_name}: {str(e)}")
            continue  # Skip problematic inserts
    
# agent, db = initialize_sql_agent(connection_path, api_key, selected_model, api_provider, db_type)
def initialize_sql_agent(db_path_or_uri, api_key, model_name, api_provider, db_type="sqlite"):
    """Initialize the SQL agent with selected LLM"""
    try:
        if db_type == "sqlite":
            # Test database connection first
            st.info("🔍 Testing database connection...")
            
            # Try to connect to the database and get basic info
            test_conn = sqlite3.connect(db_path_or_uri)
            cursor = test_conn.cursor()
            
            # Check if database is accessible and get table count
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            table_count = len(tables)
            
            cursor.close()
            test_conn.close()
            
            if table_count == 0:
                st.warning("⚠️ The database appears to be empty or has no tables.")
                return None, None
            else:
                st.success(f"✅ Database connection successful! Found {table_count} tables.")
            
            db_uri = f"sqlite:///{db_path_or_uri}"
        else:  # MySQL
            st.info("🔍 Testing MySQL connection...")
            db_uri = db_path_or_uri  # Already formatted MySQL URI
        
        # Initialize LLM based on provider
        st.info(f"🤖 Initializing {api_provider} LLM...")
        if api_provider == "OpenAI":
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model=model_name,
                temperature=0
            )
        else:  # Groq
            llm = ChatGroq(
                groq_api_key=api_key,
                model=model_name,
                temperature=0
            )
        
        # Create SQLDatabase instance
        st.info("🗃️ Creating database connection...")
        db = SQLDatabase.from_uri(db_uri)
        
        # Test the LangChain database connection
        try:
            test_tables = db.get_usable_table_names()
            st.success(f"✅ LangChain database connection successful! Accessible tables: {len(test_tables)}")
        except Exception as db_error:
            st.error(f"❌ LangChain database connection failed: {str(db_error)}")
            return None, None
        
        # Create SQL agent
        st.info("🔧 Creating SQL agent...")
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
            max_iterations=10,
            max_execution_time=60
        )
        
        st.success(f"✅ SQL Agent initialized successfully with {api_provider}!")
        return agent_executor, db
        
    except Exception as e:
        st.error(f"❌ Failed to initialize SQL agent: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None, None

def display_database_info(db):
    """Display database schema information"""
    try:
        with st.spinner("Loading database schema..."):
            # Get table names
            tables = db.get_usable_table_names()
            
            if not tables:
                st.warning("⚠️ No accessible tables found in the database.")
                return
            
            st.subheader("📋 Database Schema")
            st.success(f"Found {len(tables)} tables: {', '.join(tables)}")
            
            # Display tables in expandable sections
            for table in tables:
                with st.expander(f"Table: {table}"):
                    try:
                        # Get table info with timeout handling
                        with st.spinner(f"Loading schema for table '{table}'..."):
                            table_info = db.get_table_info([table])
                            st.code(table_info, language="sql")
                            
                        # Try to get row count
                        try:
                            row_count = db.run(f"SELECT COUNT(*) FROM {table}")
                            st.info(f"📊 Row count: {row_count}")
                        except Exception as count_error:
                            st.warning(f"Could not get row count: {str(count_error)}")
                            
                    except Exception as e:
                        st.error(f"Error getting info for table {table}: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error displaying database info: {str(e)}")

# Main application logic
if api_key and len(api_key.strip()) > 0:
    st.write(f"🔑 **Status:** {api_provider} API Key provided, ready to proceed!")
    
    # Set the API key as environment variable
    if api_provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["GROQ_API_KEY"] = api_key
    
    # Handle database selection
    db_path = None
    db_uri = None
    db_type = "sqlite"
    
    if db_option == "Use Sample Chinook Database":
        st.write("📋 **Next Step:** Load the sample database below")
        if st.sidebar.button("Load Chinook Database", type="primary"):
            db_path = download_chinook_db()
            if db_path:
                st.session_state['db_path'] = db_path
                st.session_state['db_type'] = "sqlite"
                st.success("✅ Chinook database loaded successfully!")
                st.rerun()
                
    elif db_option == "Upload Custom SQLite Database":
        st.write("📋 **Next Step:** Upload your SQLite database file below")
        uploaded_file = st.sidebar.file_uploader(
            "Upload SQLite Database",
            type=['db', 'sqlite', 'sqlite3']
        )
        if uploaded_file:
            for key in ['db_path', 'db', 'agent', 'db_uploaded', 'chat_history', 'erd_diagram', 'erd_explanations']:
                if key in st.session_state:
                    del st.session_state[key]
            with st.spinner("Processing uploaded database..."):
                try:
                    # Save uploaded file temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
                    file_content = uploaded_file.read()
                    temp_file.write(file_content)
                    temp_file.close()
                    
                    # Validate the database file
                    st.info(f"📁 File size: {len(file_content) / (1024*1024):.2f} MB")
                    
                    # Test database connection
                    try:
                        test_conn = sqlite3.connect(temp_file.name)
                        cursor = test_conn.cursor()
                        
                        # Test basic query
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        table_names = [row[0] for row in tables]
                        
                        cursor.close()
                        test_conn.close()
                        
                        if table_names:
                            st.success(f"✅ Database validation successful!")
                            st.info(f"📊 Found {len(table_names)} tables: {', '.join(table_names[:5])}{'...' if len(table_names) > 5 else ''}")
                            
                            st.session_state['db_path'] = temp_file.name
                            st.session_state['db_type'] = "sqlite"
                            st.success(f"✅ Database '{uploaded_file.name}' uploaded successfully!")
                            # st.rerun()
                        else:
                            st.error("❌ Database appears to be empty (no tables found)")
                            os.unlink(temp_file.name)
                            
                    except sqlite3.Error as db_error:
                        st.error(f"❌ Database validation failed: {str(db_error)}")
                        st.error("The uploaded file might not be a valid SQLite database.")
                        os.unlink(temp_file.name)
                        
                except Exception as upload_error:
                    st.error(f"❌ Error processing uploaded file: {str(upload_error)}")
                    if 'temp_file' in locals():
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                            
    elif db_option == "Connect to MySQL Database":
        st.write("📋 **Next Step:** Enter your MySQL database credentials below")
        
        with st.sidebar.form("mysql_connection"):
            st.subheader("MySQL Connection Details")
            mysql_host = st.text_input("Host:", value="127.0.0.1", help="Database server host")
            mysql_port = st.text_input("Port:", value="3306", help="Database server port")
            mysql_user = st.text_input("Username:", help="Database username")
            mysql_password = st.text_input("Password:", type="password", help="Database password")
            mysql_database = st.text_input("Database Name:", help="Name of the database to connect to")
            
            mysql_connect = st.form_submit_button("Connect to MySQL", type="primary")
        
        if mysql_connect:
            if all([mysql_host, mysql_port, mysql_user, mysql_password, mysql_database]):
                with st.spinner("Testing MySQL connection..."):
                    try:
                        encoded_password = urllib.parse.quote_plus(mysql_password)

                        success, table_count, tables_or_error = test_mysql_connection(
                            mysql_host, mysql_port, mysql_user, mysql_password, mysql_database
                        )
                        
                        if success:
                            st.success(f"✅ MySQL connection successful!")
                            st.info(f"📊 Found {table_count} tables: {', '.join(tables_or_error[:5])}{'...' if len(tables_or_error) > 5 else ''}")
                            
                            # Create MySQL URI
                            mysql_uri = f"mysql+pymysql://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_database}"
                            if 'db' in st.session_state:
                               del st.session_state['db']
                            if 'agent' in st.session_state:
                               del st.session_state['agent'] 
                            st.session_state['db_uri'] = mysql_uri
                            st.session_state['db_type'] = "mysql"
                            st.session_state['mysql_details'] = {
                                'host': mysql_host,
                                'port': mysql_port,
                                'user': mysql_user,
                                'database': mysql_database
                            }
                            st.success("✅ MySQL database connected successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ MySQL connection failed: {tables_or_error}")
                            st.error("Please check your credentials and ensure:")
                            st.error("• MySQL server is running")
                            st.error("• Host and port are correct")
                            st.error("• Username and password are valid")
                            st.error("• Database exists and user has access")
                    except Exception as mysql_error:
                        st.error(f"❌ MySQL connection error: {str(mysql_error)}")
            else:
                st.warning("Please fill in all MySQL connection fields!")
    
    elif db_option=="Create Custom Database":  # Create Custom Database
        st.write("🎯 **Create Your Own Database:** Describe what kind of database you want and AI will create it with dummy data!")
        
        # Database creation form
        with st.form("create_database_form"):
            st.subheader("🛠️ Database Creation")
            
            # Pre-filled examples
            example_descriptions = {
                "🏪 E-commerce Store": "Create a database for an online store with customers, products, orders, categories, and order items",
                "🏥 Hospital Management": "Create a hospital database with patients, doctors, appointments, departments, and medical records",
                "📚 Library System": "Create a library database with books, authors, members, borrowings, and categories",
                "🏫 School Management": "Create a school database with students, teachers, classes, subjects, and grades",
                "🍕 Restaurant": "Create a restaurant database with menu items, customers, orders, staff, and tables",
                "📦 Inventory Management": "Create an inventory database with products, suppliers, warehouses, stock levels, and purchase orders",
                "🚗 Car Rental": "Create a car rental database with vehicles, customers, rentals, locations, and maintenance records"
            }
            
            # Example selection
            selected_example = st.selectbox(
                "Choose an example or create your own:",
                ["Custom Description"] + list(example_descriptions.keys())
            )
            
            # Description input
            if selected_example == "Custom Description":
                description = st.text_area(
                    "Describe your database:",
                    placeholder="Example: Create a database for a fitness gym with members, trainers, classes, memberships, and attendance records",
                    height=100,
                    help="Be as specific as possible. Mention the type of business/domain and what kind of data you want to track."
                )
            else:
                description = st.text_area(
                    "Database Description (you can modify this):",
                    value=example_descriptions[selected_example],
                    height=100
                )
            
            # Additional options
            col1, col2 = st.columns(2)
            with col1:
                data_complexity = st.selectbox(
                    "Data Complexity:",
                    ["Simple", "Moderate", "Complex"],
                    index=1,
                    help="Simple: Basic data types | Moderate: Mixed data with relationships | Complex: Rich data with multiple constraints"
                )
            
            with col2:
                num_records = st.selectbox(
                    "Approximate Records per Table:",
                    ["10-15", "20-30", "30-50"],
                    index=1,
                    help="Choose how much sample data to generate"
                )
            
            create_db_button = st.form_submit_button("🚀 Create Database", type="primary")
        
        if create_db_button:
            if description.strip():
                # Clear any existing database states
                for key in ['db_path', 'db_uri', 'db', 'agent', 'chat_history', 'created_db_path', 'erd_diagram', 'erd_explanations']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Create the database
                created_db_path = create_database_schema_and_data(
                    description, api_key, selected_model, api_provider
                )
                
                if created_db_path:
                    st.session_state['created_db_path'] = created_db_path
                    st.session_state['db_path'] = created_db_path
                    st.session_state['db_type'] = "sqlite"
                    st.balloons()  # Celebration effect
                    st.rerun()
                else:
                    st.error("❌ Failed to create database. Please try again with a different description.")
            else:
                st.warning("⚠️ Please provide a description for your database!")

    else:  # SQL Learning & Practice
        st.write("📚 **SQL Learning & Practice Section**")
        st.markdown("---")
        
        # Main tabs for different learning sections
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 SQL Q&A Assistant", "📖 SQL Tutorials", "💼 Interview Prep", "🎯 Practice Challenges"])
        
        with tab1:
            st.header("🤖 SQL Q&A Assistant")
            st.markdown("Ask any question about SQL, RDBMS, database concepts, and get detailed explanations!")
            
            # Example SQL questions
            st.markdown("**💡 Example questions you can ask:**")
            sql_examples = [
                "What is SQL and why is it important?",
                "Explain window functions in SQL",
                "What's the difference between INNER and LEFT JOIN?",
                "How do indexes work in databases?",
                "What are database normalization forms?"
            ]
            
            cols = st.columns(len(sql_examples))
            for i, example in enumerate(sql_examples):
                if cols[i].button(f"📝 {example}", key=f"sql_example_{i}"):
                    st.session_state['sql_question'] = example
            
            # SQL Question input
            sql_question = st.text_input(
                "Ask your SQL question:",
                value=st.session_state.get('sql_question', ''),
                placeholder="e.g., What are stored procedures and how do they work?",
                key="sql_question_input"
            )
            
            # Clear example question after use
            if 'sql_question' in st.session_state:
                del st.session_state['sql_question']
            
            if st.button("🔍 Get Answer", type="primary"):
                if sql_question and api_key:
                    with st.spinner("🤖 Generating detailed SQL explanation..."):
                        try:
                            # Initialize LLM for SQL Q&A
                            if api_provider == "OpenAI":
                                llm = ChatOpenAI(
                                    openai_api_key=api_key,
                                    model=selected_model,
                                    temperature=0.3
                                )
                            else:
                                llm = ChatGroq(
                                    groq_api_key=api_key,
                                    model=selected_model,
                                    temperature=0.3
                                )
                            
                            sql_qa_prompt = f"""
                            You are an expert SQL and database instructor. Answer the following question about SQL/RDBMS in a clear, educational way.
                            
                            Question: {sql_question}
                            
                            Please provide:
                            1. A clear, concise explanation
                            2. Practical examples with SQL code when relevant
                            3. Use cases or scenarios where this applies
                            4. Any best practices or tips
                            5. Common pitfalls to avoid (if applicable)
                            
                            Format your response in a structured, easy-to-read manner with examples in SQL code blocks.
                            Keep the explanation beginner-friendly but comprehensive.
                            """
                            
                            response = llm.invoke(sql_qa_prompt)
                            
                            st.subheader("💬 Answer")
                            st.markdown(response.content)
                            
                            # Add to SQL Q&A history
                            if 'sql_qa_history' not in st.session_state:
                                st.session_state['sql_qa_history'] = []
                            
                            st.session_state['sql_qa_history'].append({
                                'question': sql_question,
                                'answer': response.content,
                                'provider': api_provider,
                                'model': selected_model
                            })
                            
                        except Exception as e:
                            st.error(f"Error getting SQL answer: {str(e)}")
                elif not api_key:
                    st.warning("Please enter your API key to use the SQL Q&A Assistant!")
                else:
                    st.warning("Please enter a SQL question!")
            
            # Display recent SQL Q&A
            if 'sql_qa_history' in st.session_state and st.session_state['sql_qa_history']:
                st.markdown("---")
                st.subheader("📜 Recent SQL Q&A")
                for i, entry in enumerate(reversed(st.session_state['sql_qa_history'][-5:])):
                    with st.expander(f"Q: {entry['question'][:60]}..."):
                        st.write(f"**Question:** {entry['question']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
        
        with tab2:
            st.header("📖 SQL Tutorials & Learning Resources")
            st.markdown("Access the best SQL learning resources from top educational platforms!")
            
            # Tutorial Categories
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎓 Comprehensive Tutorials")
                
                # W3Schools SQL
                st.markdown("### 🌐 W3Schools SQL Tutorial")
                st.markdown("Perfect for beginners - step by step SQL learning")
                tutorial_options = {
                    "SQL Introduction": "https://www.w3schools.com/sql/sql_intro.asp",
                    "SQL Syntax": "https://www.w3schools.com/sql/sql_syntax.asp",
                    "SQL SELECT": "https://www.w3schools.com/sql/sql_select.asp",
                    "SQL WHERE": "https://www.w3schools.com/sql/sql_where.asp",
                    "SQL Joins": "https://www.w3schools.com/sql/sql_join.asp",
                    "SQL Functions": "https://www.w3schools.com/sql/sql_functions.asp",
                    "Complete W3Schools SQL": "https://www.w3schools.com/sql/"
                }
                
                selected_w3 = st.selectbox("Choose W3Schools Topic:", list(tutorial_options.keys()))
                if st.button("🔗 Open W3Schools Tutorial", key="w3schools"):
                    st.markdown(f"**Opening:** [W3Schools - {selected_w3}]({tutorial_options[selected_w3]})")
                    st.markdown(f"**Direct Link:** {tutorial_options[selected_w3]}")
                
                st.markdown("---")
                
                # GeeksforGeeks SQL
                st.markdown("### 🚀 GeeksforGeeks SQL")
                st.markdown("In-depth tutorials with examples and explanations")
                gfg_options = {
                    "SQL Basics": "https://www.geeksforgeeks.org/sql-tutorial/",
                    "SQL Commands": "https://www.geeksforgeeks.org/sql-ddl-dql-dml-dcl-tcl-commands/",
                    "SQL Joins": "https://www.geeksforgeeks.org/sql-join-set-1-inner-left-right-and-full-joins/",
                    "SQL Subqueries": "https://www.geeksforgeeks.org/sql-subquery/",
                    "SQL Window Functions": "https://www.geeksforgeeks.org/window-functions-in-sql/",
                    "SQL Indexes": "https://www.geeksforgeeks.org/indexing-in-databases-set-1/",
                    "Complete GFG SQL": "https://www.geeksforgeeks.org/sql-tutorial/"
                }
                
                selected_gfg = st.selectbox("Choose GeeksforGeeks Topic:", list(gfg_options.keys()))
                if st.button("🔗 Open GeeksforGeeks Tutorial", key="geeksforgeeks"):
                    st.markdown(f"**Opening:** [GeeksforGeeks - {selected_gfg}]({gfg_options[selected_gfg]})")
                    st.markdown(f"**Direct Link:** {gfg_options[selected_gfg]}")
            
            with col2:
                st.subheader("🎯 Advanced Learning")
                
                # SQLBolt
                st.markdown("### ⚡ SQLBolt")
                st.markdown("Interactive SQL lessons and exercises")
                sqlbolt_options = {
                    "Interactive SQL Lessons": "https://sqlbolt.com/",
                    "SQL Lesson 1: SELECT queries 101": "https://sqlbolt.com/lesson/select_queries_introduction",
                    "SQL Lesson 6: Multi-table queries with JOINs": "https://sqlbolt.com/lesson/select_queries_with_joins",
                    "SQL Lesson 12: Order of execution of a Query": "https://sqlbolt.com/lesson/select_queries_order_of_execution"
                }
                
                selected_sqlbolt = st.selectbox("Choose SQLBolt Lesson:", list(sqlbolt_options.keys()))
                if st.button("🔗 Open SQLBolt", key="sqlbolt"):
                    st.markdown(f"**Opening:** [SQLBolt - {selected_sqlbolt}]({sqlbolt_options[selected_sqlbolt]})")
                    st.markdown(f"**Direct Link:** {sqlbolt_options[selected_sqlbolt]}")
                
                st.markdown("---")
                
                # Mode Analytics SQL Tutorial
                st.markdown("### 📊 Mode Analytics SQL Tutorial")
                st.markdown("SQL for data analysis and business intelligence")
                mode_options = {
                    "SQL Tutorial": "https://mode.com/sql-tutorial/",
                    "Basic SQL": "https://mode.com/sql-tutorial/sql-basics/",
                    "Intermediate SQL": "https://mode.com/sql-tutorial/sql-intermediate/",
                    "Advanced SQL": "https://mode.com/sql-tutorial/sql-advanced/",
                    "SQL Analytics Training": "https://mode.com/sql-tutorial/sql-analytics-training/"
                }
                
                selected_mode = st.selectbox("Choose Mode Analytics Topic:", list(mode_options.keys()))
                if st.button("🔗 Open Mode Analytics", key="mode"):
                    st.markdown(f"**Opening:** [Mode Analytics - {selected_mode}]({mode_options[selected_mode]})")
                    st.markdown(f"**Direct Link:** {mode_options[selected_mode]}")
            
            # Additional Resources
            st.markdown("---")
            st.subheader("📚 Additional Learning Resources")
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.markdown("**🎥 Video Tutorials**")
                st.markdown("- [freeCodeCamp SQL Course](https://www.youtube.com/watch?v=HXV3zeQKqGY)")
                st.markdown("- [SQL Tutorial - Full Database Course for Beginners](https://www.youtube.com/watch?v=HXV3zeQKqGY)")
            
            with col4:
                st.markdown("**📖 Documentation**")
                st.markdown("- [PostgreSQL Docs](https://www.postgresql.org/docs/)")
                st.markdown("- [MySQL Docs](https://dev.mysql.com/doc/)")
                st.markdown("- [SQLite Docs](https://sqlite.org/docs.html)")
            
            with col5:
                st.markdown("**🛠️ Online Tools**")
                st.markdown("- [DB Fiddle](https://www.db-fiddle.com/)")
                st.markdown("- [SQLiteOnline](https://sqliteonline.com/)")
                st.markdown("- [SQL Formatter](https://www.freeformatter.com/sql-formatter.html)")
        
        with tab3:
            st.header("💼 SQL Interview Preparation")
            st.markdown("Prepare for SQL interviews with questions, challenges, and resources!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Interview Question Banks")
                
                # Interview resources
                interview_resources = {
                    "DataCamp SQL Interview Questions": "https://www.datacamp.com/blog/sql-interview-questions",
                    "GeeksforGeeks SQL Interview Questions": "https://www.geeksforgeeks.org/sql-interview-questions/",
                    "InterviewBit SQL Questions": "https://www.interviewbit.com/sql-interview-questions/",
                    "LeetCode Database Problems": "https://leetcode.com/problemset/database/",
                    "HackerRank SQL Challenges": "https://www.hackerrank.com/domains/sql",
                    "Stratascratch SQL Questions": "https://www.stratascratch.com/"
                }
                
                for resource, url in interview_resources.items():
                    if st.button(f"🔗 {resource}", key=f"interview_{resource}"):
                        st.markdown(f"**Opening:** [{resource}]({url})")
                        st.markdown(f"**Direct Link:** {url}")
                
                st.markdown("---")
                
                st.subheader("📋 Common Interview Topics")
                topics = [
                    "SQL Joins (INNER, LEFT, RIGHT, FULL)",
                    "Subqueries vs CTEs",
                    "Window Functions (ROW_NUMBER, RANK, LAG, LEAD)",
                    "Indexing and Performance Optimization",
                    "Database Normalization (1NF, 2NF, 3NF)",
                    "ACID Properties",
                    "Stored Procedures vs Functions",
                    "Triggers and their uses"
                ]
                
                for i, topic in enumerate(topics, 1):
                    st.markdown(f"{i}. **{topic}**")
            
            with col2:
                st.subheader("🤖 Practice Interview Q&A")
                st.markdown("Get instant answers to common SQL interview questions!")
                
                common_questions = [
                    "What's the difference between DELETE and TRUNCATE?",
                    "Explain the difference between UNION and UNION ALL",
                    "What are window functions and when to use them?",
                    "How do you optimize a slow SQL query?",
                    "What's the difference between clustered and non-clustered indexes?"
                ]
                
                st.markdown("**🔥 Popular Interview Questions:**")
                for i, question in enumerate(common_questions):
                    if st.button(f"❓ {question}", key=f"interview_q_{i}"):
                        st.session_state['sql_question'] = question
                        st.rerun()
                
                st.markdown("---")
                st.subheader("💡 Interview Tips")
                st.markdown("""
                **🎯 Key Tips for SQL Interviews:**
                - Always ask about the data size and expected performance
                - Explain your thought process while writing queries
                - Consider edge cases (NULL values, duplicates)
                - Know when to use indexes and their trade-offs
                - Practice explaining complex concepts simply
                - Be ready to optimize queries on the spot
                - Understand database design principles
                """)
        
        with tab4:
            st.header("🎯 SQL Practice Challenges")
            st.markdown("Test your SQL skills with practice platforms and coding challenges!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Coding Challenge Platforms")
                
                challenge_platforms = {
                    "LeetCode Database": {
                        "url": "https://leetcode.com/problemset/database/",
                        "description": "200+ SQL problems from easy to hard difficulty",
                        "best_for": "Interview preparation, algorithm practice"
                    },
                    "HackerRank SQL": {
                        "url": "https://www.hackerrank.com/domains/sql",
                        "description": "Structured SQL challenges with instant feedback",
                        "best_for": "Learning progressively, skill certification"
                    },
                    "Codewars SQL": {
                        "url": "https://www.codewars.com/kata/search/sql",
                        "description": "Community-driven SQL challenges",
                        "best_for": "Creative problem solving"
                    },
                    "StrataScratch": {
                        "url": "https://www.stratascratch.com/",
                        "description": "Real company interview questions",
                        "best_for": "Job interview preparation"
                    },
                    "SQLZoo": {
                        "url": "https://sqlzoo.net/",
                        "description": "Interactive SQL tutorials and quizzes",
                        "best_for": "Beginners learning SQL step by step"
                    }
                }
                
                for platform, info in challenge_platforms.items():
                    with st.expander(f"🚀 {platform}"):
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Best for:** {info['best_for']}")
                        if st.button(f"🔗 Visit {platform}", key=f"platform_{platform}"):
                            st.markdown(f"**Opening:** [{platform}]({info['url']})")
                            st.markdown(f"**Direct Link:** {info['url']}")
            
            with col2:
                st.subheader("📊 Skill Assessment Tools")
                
                assessment_tools = {
                    "W3Schools SQL Quiz": "https://www.w3schools.com/sql/sql_quiz.asp",
                    "TestDome SQL Assessment": "https://www.testdome.com/tests/sql-online-test/82",
                    "Pluralsight Skill IQ": "https://www.pluralsight.com/product/skill-iq",
                    "LinkedIn SQL Assessment": "https://www.linkedin.com/skill-assessments/",
                }
                
                st.markdown("**🎯 Test Your SQL Knowledge:**")
                for tool, url in assessment_tools.items():
                    if st.button(f"📝 {tool}", key=f"assessment_{tool}"):
                        st.markdown(f"**Opening:** [{tool}]({url})")
                        st.markdown(f"**Direct Link:** {url}")
                
                st.markdown("---")
                
                st.subheader("📈 Learning Path Suggestions")
                
                learning_paths = {
                    "Beginner": [
                        "Complete W3Schools SQL Tutorial",
                        "Practice basic queries on SQLBolt",
                        "Take W3Schools SQL Quiz",
                        "Try easy problems on HackerRank"
                    ],
                    "Intermediate": [
                        "Study joins and subqueries on GeeksforGeeks",
                        "Practice window functions",
                        "Solve LeetCode easy/medium problems",
                        "Learn about indexes and optimization"
                    ],
                    "Advanced": [
                        "Master complex queries and CTEs",
                        "Practice hard LeetCode problems",
                        "Study database design principles",
                        "Prepare for technical interviews"
                    ]
                }
                
                selected_level = st.selectbox("Choose your level:", list(learning_paths.keys()))
                st.markdown(f"**📚 Recommended path for {selected_level}:**")
                for step in learning_paths[selected_level]:
                    st.markdown(f"• {step}")
        
        # Quick Links Section
        st.markdown("---")
        st.subheader("🔗 Quick Access Links")
        
        quick_links_col1, quick_links_col2, quick_links_col3, quick_links_col4 = st.columns(4)
        
        with quick_links_col1:
            st.markdown("**📖 Tutorials**")
            st.markdown("- [W3Schools SQL](https://www.w3schools.com/sql/)")
            st.markdown("- [GeeksforGeeks SQL](https://www.geeksforgeeks.org/sql-tutorial/)")
            
        with quick_links_col2:
            st.markdown("**🎯 Practice**")
            st.markdown("- [LeetCode DB](https://leetcode.com/problemset/database/)")
            st.markdown("- [HackerRank SQL](https://www.hackerrank.com/domains/sql)")
            
        with quick_links_col3:
            st.markdown("**💼 Interview**")
            st.markdown("- [DataCamp Questions](https://www.datacamp.com/blog/sql-interview-questions)")
            st.markdown("- [InterviewBit SQL](https://www.interviewbit.com/sql-interview-questions/)")
            
        with quick_links_col4:
            st.markdown("**🛠️ Tools**")
            st.markdown("- [DB Fiddle](https://www.db-fiddle.com/)")
            st.markdown("- [SQLite Online](https://sqliteonline.com/)")
    
    # Handle database connection only for database-related options
    if db_option != "SQL Learning & Practice":
        # Use database path/URI from session state if available
        if 'db_path' in st.session_state:
            db_path = st.session_state['db_path']
            db_type = st.session_state.get('db_type', 'sqlite')
        elif 'db_uri' in st.session_state:
            db_uri = st.session_state['db_uri']
            db_type = st.session_state.get('db_type', 'mysql')
    
    # Initialize agent if database is loaded
    if db_path or db_uri:
        # Show database connection info
        if db_type == "sqlite" and db_path and os.path.exists(db_path):
            file_size = os.path.getsize(db_path) / (1024*1024)
            st.info(f"📁 SQLite Database file size: {file_size:.2f} MB")
            
            # Show special message for created databases
            if 'created_db_path' in st.session_state and st.session_state['created_db_path'] == db_path:
                st.success("🎉 Using your custom created database!")
                
        elif db_type == "mysql" and 'mysql_details' in st.session_state:
            details = st.session_state['mysql_details']
            st.info(f"🗄️ MySQL Database: {details['user']}@{details['host']}:{details['port']}/{details['database']}")
        
        if ('agent' not in st.session_state or 
            'current_model' not in st.session_state or 
            st.session_state.get('current_model') != selected_model or
            st.session_state.get('current_api_key') != api_key or
            st.session_state.get('current_provider') != api_provider):
            
            # Clear any existing database info when reinitializing
            if 'db' in st.session_state:
                del st.session_state['db']
            
            connection_path = db_path if db_type == "sqlite" else db_uri
            agent, db = initialize_sql_agent(connection_path, api_key, selected_model, api_provider, db_type)
            if agent and db:
                st.session_state['agent'] = agent
                st.session_state['db'] = db
                st.session_state['current_model'] = selected_model
                st.session_state['current_api_key'] = api_key
                st.session_state['current_provider'] = api_provider
                st.success(f"✅ SQL Agent initialized with {api_provider} {selected_model}!")
            else:
                st.error("❌ Failed to initialize SQL agent. Please check the database connection.")
                # Clean up session state
                for key in ['db_path', 'db_uri', 'agent', 'db', 'current_model', 'current_api_key', 'current_provider', 'erd_diagram', 'erd_explanations']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop()
        
        # Display database info only if we have a valid db connection
        if 'db' in st.session_state and st.session_state['db']:
            display_database_info(st.session_state['db'])
            
            # Show ERD option for ALL database types
            st.markdown("---")
            st.subheader("📊 Database Visualization")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🗺️ Generate ERD Diagram", type="secondary", use_container_width=True):
                    with st.spinner("🎨 Generating Entity-Relationship Diagram..."):
                        connection_path = db_path if db_type == "sqlite" else db_uri
                        dot_diagram, explanations = generate_erd_diagram(
                            connection_path, db_type, api_key, selected_model, api_provider
                        )
                        
                        if dot_diagram:
                            st.session_state['erd_diagram'] = dot_diagram
                            st.session_state['erd_explanations'] = explanations
                            st.success("✅ ERD generated successfully!")
                        else:
                            st.error("❌ Failed to generate ERD diagram")
            
            with col2:
                if GRAPHVIZ_AVAILABLE and SQLALCHEMY_AVAILABLE:
                    if 'created_db_path' in st.session_state and st.session_state.get('created_db_path') == db_path:
                        st.info("💡 Generate an ERD to visualize your custom database structure and relationships")
                    elif db_type == "sqlite":
                        st.info("💡 Generate an ERD to visualize this SQLite database structure and relationships")
                    elif db_type == "mysql":
                        st.info("💡 Generate an ERD to visualize this MySQL database structure and relationships")
                else:
                    st.warning("⚠️ ERD generation requires additional packages:")
                    st.code("pip install graphviz sqlalchemy", language="bash")
            
            # Display ERD if generated with improved controls
            if 'erd_diagram' in st.session_state:
                st.markdown("---")
                display_erd_with_controls(
                    st.session_state['erd_diagram'],
                    st.session_state.get('erd_explanations', '')
                )
        else:
            st.error("❌ Database connection not available.")
            st.stop()
        
        # Full screen ERD view
        if st.session_state.get('show_erd_fullscreen', False):
            st.markdown("---")
            st.subheader("🔍 Full Screen ERD View")
            
            if st.button("❌ Close Full Screen"):
                st.session_state['show_erd_fullscreen'] = False
                st.rerun()
            
            if 'erd_diagram' in st.session_state:
                st.graphviz_chart(st.session_state['erd_diagram'].source, use_container_width=True)
                
                # Show explanations in full screen too
                if 'erd_explanations' in st.session_state:
                    st.markdown("### 🧠 Relationship Explanations")
                    st.markdown(st.session_state['erd_explanations'])
        
        # Chat interface
        st.subheader("💬 Ask Questions About Your Database")
        
        # Example questions (dynamic based on database type and source)
        st.markdown("**Example questions you can ask:**")
        
        if 'created_db_path' in st.session_state and st.session_state.get('created_db_path') == db_path:
            # Custom questions for created databases
            example_questions = [
                "What tables are in this database?",
                "How many records are in each table?",
                "Show me sample data from each table",
                "What are the relationships between tables?",
                "Give me insights about the data structure"
            ]
        elif db_type == "sqlite":
            example_questions = [
                "How many customers are there?",
                "What are the top 5 best-selling tracks?",
                "Which country has the most customers?",
                "Show me the total sales by year",
                "What are the different music genres available?"
            ]
        else:  # MySQL
            example_questions = [
                "How many records are in each table?",
                "Show me the table with the most data",
                "What columns exist in the main tables?",
                "Give me a summary of the database structure",
                "Show me sample data from the largest table"
            ]
        
        cols = st.columns(len(example_questions))
        for i, question in enumerate(example_questions):
            if cols[i].button(f"📝 {question}", key=f"example_{i}"):
                st.session_state['example_question'] = question
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get('example_question', ''),
            placeholder=f"e.g., How many tables are in this {db_type} database?",
            key="question_input"
        )
        
        # Clear example question after use
        if 'example_question' in st.session_state:
            del st.session_state['example_question']
        
        if st.button("🔍 Ask Question", type="primary"):
            if question and 'agent' in st.session_state:
                with st.spinner("Processing your question..."):
                    try:
                        # Execute the query with explicit return_only_outputs=False to get intermediate steps
                        response = st.session_state['agent'].invoke(
                            {"input": question},
                            return_only_outputs=False
                        )
                        
                        # Debug: Show the full response structure
                        with st.expander("🐛 Debug: Full Response Structure"):
                            st.json(response)
                        
                        # Extract SQL queries and results from intermediate steps
                        sql_queries = []
                        query_results = []
                        
                        
                        # Display SQL Queries Section
                        st.subheader("🗄️ Generated SQL Queries")
                            # Try to execute a direct query if we can infer one
                        if 'db' in st.session_state:
                            st.info("💡 Let me try to generate a SQL query for you...")
                            try:
                                    # Use the LLM to generate SQL directly
                                from langchain.prompts import PromptTemplate
                                    
                                prompt_template = """Given the following database schema and question, write a SQL query to answer it.
                                    
                                    Database schema:
                                    {schema}

                                    Question: {question}

                                    Please provide only the SQL query without any explanation:"""

                                    # Get schema information
                                schema_info = st.session_state['db'].get_table_info()
                                    
                                    # Create prompt
                                prompt = PromptTemplate(
                                        template=prompt_template,
                                        input_variables=["schema", "question"]
                                    )
                                    
                                    # Get LLM response
                                if api_provider == "OpenAI":
                                    llm = ChatOpenAI(
                                            openai_api_key=api_key,
                                            model=selected_model,
                                            temperature=0
                                    )
                                else:
                                    llm = ChatGroq(
                                            groq_api_key=api_key,
                                            model=selected_model,
                                            temperature=0
                                    )
                                
                                formatted_prompt = prompt.format(schema=schema_info, question=question)
                                sql_response = llm.invoke(formatted_prompt)
                                    
                                    # Clean up the SQL
                                generated_sql = sql_response.content.strip()
                                    # Remove markdown formatting if present
                                generated_sql = re.sub(r'```sql\s*', '', generated_sql)
                                generated_sql = re.sub(r'\s*```', '', generated_sql)
                                    
                                st.markdown("**Generated SQL Query:**")
                                st.code(generated_sql, language="sql")
                                    
                                    # Try to execute it
                                try:
                                    result = st.session_state['db'].run(generated_sql)
                                    with st.expander("📋 Query Results"):
                                        st.text(str(result))
                                except Exception as exec_error:
                                    st.warning(f"Could not execute generated query: {str(exec_error)}")
                                        
                            except Exception as gen_error:
                                st.error(f"Could not generate SQL query: {str(gen_error)}")
                        
                        # Display Natural Language Answer
                        st.subheader("💬 Natural Language Answer")
                        st.write(response.get('output', 'No output available'))
                        
                        # Add to chat history with both SQL and answer
                        if 'chat_history' not in st.session_state:
                            st.session_state['chat_history'] = []
                        
                        chat_entry = {
                            'question': question,
                            'sql_queries': sql_queries,
                            'answer': response.get('output', 'No output available'),
                            'raw_results': query_results,
                            'provider': api_provider,
                            'model': selected_model,
                        }
                        st.session_state['chat_history'].append(chat_entry)
                    
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        # Show more detailed error information
                        import traceback
                        with st.expander("🐛 Error Details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("Please enter a question!")
        
        # Display recent queries
        if 'chat_history' in st.session_state and st.session_state['chat_history']:
            st.subheader("📜 Recent Queries")
            for i, entry in enumerate(reversed(st.session_state['chat_history'][-5:])):
                provider_info = f" ({entry.get('provider', 'Unknown')} {entry.get('model', '')})" if 'provider' in entry else ""
                with st.expander(f"Q: {entry['question'][:50]}...{provider_info}"):
                    st.write(f"**Question:** {entry['question']}")
                    
                    # Show SQL queries
                    if entry['sql_queries']:
                        st.write("**SQL Queries:**")
                        for j, query in enumerate(entry['sql_queries'], 1):
                            st.code(query, language="sql")
                    
                    st.write(f"**Answer:** {entry['answer']}")
                    
                    
                    # Show raw results in collapsible section
                    if entry.get('raw_results'):
                        with st.expander(f"Raw Query Results"):
                            for j, result in enumerate(entry['raw_results'], 1):
                                provider_info = f" ({entry.get('provider', 'Unknown')} {entry.get('model', '')})" if 'provider' in entry else ""
                                st.text(f"Query {j} Result{provider_info}: {result}")
                with st.expander(f"Q: {entry['question'][:50]}...{provider_info}"):
                    st.write(f"**Question:** {entry['question']}")
                    
                    # Show SQL queries
                    if entry['sql_queries']:
                        st.write("**SQL Queries:**")
                        for j, query in enumerate(entry['sql_queries'], 1):
                            st.code(query, language="sql")
                    
                    st.write(f"**Answer:** {entry['answer']}")

else:
    st.warning(f"🔑 Please enter your {api_provider} API key in the sidebar to get started!")
    st.markdown(f"""
    ### How to get started:
    1. Select your preferred API provider ({api_provider})
    2. Get your API key:
       - **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
       - **Groq**: [https://console.groq.com/keys](https://console.groq.com/keys)
    3. Choose your option:
       - **Sample Chinook database**: Pre-loaded music store database
       - **Upload custom SQLite database**: Upload your own .db file
       - **Connect to MySQL database**: Connect to remote MySQL server
       - **🆕 Create custom database**: Let AI create a database with dummy data based on your description!
       - **📚 SQL Learning & Practice**: Learn SQL with tutorials, Q&A, and interview prep
    4. Start asking natural language questions about your data or learning SQL!
    
    ### ✨ All Features:
    
    #### 📊 **ERD Diagrams for ALL Databases**
    - Generate Entity-Relationship Diagrams for any database type (SQLite, MySQL, Custom, Chinook)
    - Professional Power BI-style layouts with clean, compact design
    - Interactive zoom controls (50% to 150%) and scrollable view
    - Full-screen mode for detailed exploration
    - AI-powered relationship explanations with business logic insights
    - Color-coded tables by functional area (customers, orders, products)
    
    #### 📈 **Automatic Chart Generation**
    - Educational insights explaining why each chart type was chosen
    - Works with all database types (SQLite, MySQL, custom, Chinook)
    
    #### 🎯 **Create Custom Database**
    - Describe any type of business or domain (e-commerce, hospital, school, etc.)
    - AI automatically creates appropriate tables with foreign key relationships
    - Generate realistic dummy data (15-40 records per table)
    - Instant ERD visualization of your created schema
    - Start querying immediately with natural language
    
    #### 📚 **SQL Learning & Practice Hub**
    - **🤖 SQL Q&A Assistant**: Ask any SQL/RDBMS question and get detailed explanations
    - **📖 SQL Tutorials**: Curated links to W3Schools, GeeksforGeeks, SQLBolt, Mode Analytics
    - **💼 Interview Preparation**: Resources from DataCamp, LeetCode, HackerRank, InterviewBit
    - **🎯 Practice Challenges**: Coding platforms and skill assessments
    - **📋 Learning Paths**: Structured progression from beginner to advanced
    
    #### 💡 **Chart-Friendly Query Examples:**
    - **Count queries**: "How many customers per country?"  
    - **Aggregations**: "Total sales by year/month/category"
    - **Rankings**: "Top 10 products by sales"
    - **Comparisons**: "Compare performance between regions"
    - **Trends**: "Sales growth over time" or "Monthly patterns"
    
    Perfect for learning, testing, prototyping, interview preparation, and data exploration!
    
    ### 📋 **Requirements for Advanced Features:**
    ```bash
    # For ERD diagrams:
    pip install graphviz sqlalchemy
        
    # For realistic dummy data:
    pip install faker
    ```
    ### About this app:
    A comprehensive SQL platform combining database querying, AI-powered explanations, visual analytics, and complete learning resources - all powered by LangChain and OpenAI/Groq APIs.
    """)

# Footer
st.markdown("---")

st.markdown("Built with LangChain, OpenAI/Groq APIs, Graphviz and Streamlit | 🗃️ Complete SQL Learning & Analytics Platform")
