from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
def get_llm():
    """Initialize and return the LLM with Groq."""
    return ChatGroq(
        temperature=0.1,
        model_name="llama3-70b-8192",  # Using LLaMA 3 70B through Groq
        api_key=os.getenv("GROQ_API_KEY"),
    )

def analyze_table(data, question: str) -> str:
    """
    Analyze table data and answer questions about it using LLaMA 3.
    
    Args:
        data: List of dictionaries containing table data
        question: User's question about the data
        
    Returns:
        str: Generated response
    """
    print(f"Debug - analyze_table called with question: {question}")
    
    try:
        if not data or not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            error_msg = "Error: Invalid data format. Expected a list of dictionaries with table data."
            print(error_msg)
            return error_msg
        
        print(f"Debug - Processing {len(data)} tables")
        context_parts = ["The Excel file contains the following tables:"]
        
        # Build context with table summaries
        for table in data:
            sheet = table.get('sheet', 'Unknown Sheet')
            table_name = table.get('table', 'Unnamed Table')
            columns = table.get('columns', [])
            sample_data = table.get('sample_data', [])
            
            context_parts.append(f"\nSheet: {sheet}, Table: {table_name}")
            context_parts.append(f"Columns: {', '.join(columns)}")
            
            if sample_data and len(sample_data) > 0:
                try:
                    sample_df = pd.DataFrame(sample_data)
                    sample_str = sample_df.head(3).to_markdown(index=False)
                    context_parts.append(f"Sample data (first {min(3, len(sample_data))} rows):\n{sample_str}")
                except Exception as e:
                    print(f"Warning: Could not process sample data: {str(e)}")
                    context_parts.append("Sample data: [Could not display]")
        
        table_context = "\n".join(context_parts)
        print(f"Debug - Table context length: {len(table_context)} characters")
        
        # Create a prompt template with table context
        template = """You are a helpful data analyst assistant. Analyze the following Excel file data and answer the question.
        
{context}

Question: {question}

Instructions:
1. Provide a clear and concise answer based on the available data
2. If the question requires specific data that isn't available, mention what's missing
3. If the question is about relationships between tables, analyze the relevant tables
4. If the data is insufficient to answer, say so
5. Be specific and include relevant numbers and details from the data

Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        print("Debug - Initializing LLM...")
        llm = get_llm()
        print("Debug - LLM initialized successfully")
        
        # Initialize the LLM chain
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True
        )
        
        print("Debug - Sending request to LLM...")
        # Get response from the model
        response = llm_chain.run({
            "context": table_context,
            "question": question
        })
        
        print("Debug - Received response from LLM")
        return response.strip()
        
    except Exception as e:
        error_msg = f"Error in analyze_table: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

def generate_chat_response(chat_history: List[Dict[str, str]], current_question: str, table_context: pd.DataFrame = None) -> str:
    """
    Generate a response for the chat interface.
    
    Args:
        chat_history: List of previous messages in format [{"role": "user/assistant", "content": "message"}]
        current_question: The latest user question
        table_context: Optional DataFrame for table-specific questions
        
    Returns:
        str: Generated response
    """
    if table_context is not None and not table_context.empty:
        return analyze_table(table_context, current_question)
    
    # For general questions without table context
    template = """You are a helpful assistant. Continue the conversation naturally.
    
    Previous conversation:
    {history}
    
    User: {question}
    Assistant:"""
    
    # Format chat history
    history_str = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in chat_history[-5:]  # Use last 5 messages for context
    )
    
    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template=template
    )
    
    llm_chain = LLMChain(
        llm=get_llm(),
        prompt=prompt,
        verbose=True
    )
    
    try:
        response = llm_chain.run({
            "history": history_str,
            "question": current_question
        })
        return response.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
