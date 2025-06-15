import os
import pandas as pd
from typing import Union

# LangChain and LangGraph imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Data loader utility ---

def load_data(data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Accepts either a parquet file path or a DataFrame; returns a DataFrame.
    """
    if isinstance(data_source, str):
        return pd.read_parquet(data_source)
    elif isinstance(data_source, pd.DataFrame):
        return data_source.copy()
    else:
        raise ValueError("data_source must be a parquet filepath or a pandas DataFrame.")

# --- Tool 1: Correlation inspector ---
@tool
def correlation_inspector(data_source: str, target: str, top_n: int = 10) -> str:
    """
    Compute top N features most correlated with the target variable.
    """
    df = load_data(data_source)
    corrs = df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
    top_feats = corrs.drop(labels=[target]).head(top_n).index.tolist()
    return f"Top {top_n} features correlated with '{target}': {top_feats}"

# --- Tool 2: Feature generator ---
@tool
def feature_generator(data_source: str, target: str, suggestions: int = 5) -> str:
    """
    Generate Python code for new features based on existing data columns.
    """
    df = load_data(data_source)
    cols = df.columns.tolist()

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a quant developer. Given the columns: {columns} and target: {target},
propose {n} new pandas feature expressions.

Return only the raw Python code lines (no markdown or explanations).
Provide one expression per line, for example: df['new_feature'] = df['col1'] / df['col2']
"""
    )

    # Initialize the modern chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the modern chain using LangChain Expression Language (LCEL)
    chain = prompt_template | llm | StrOutputParser()

    # Invoke the chain with a dictionary of inputs
    raw_output = chain.invoke({
        "columns": cols,
        "target": target,
        "n": suggestions
    })

    # Clean the output to ensure it's valid code
    code_lines = [
        line.strip() for line in raw_output.splitlines()
        if line.strip().startswith("df[")
    ]
    code = "\n".join(code_lines).strip() + "\n"

    # Write to a file
    file_path = os.path.join(os.getcwd(), 'generated_features.py')
    with open(file_path, 'w') as f:
        f.write("# Auto-generated feature code\n")
        f.write("import pandas as pd\n") # Add common imports
        f.write("import numpy as np\n\n")
        f.write(code)

    return f"Generated {len(code_lines)} feature lines and saved to '{file_path}'"

# --- Agent Setup ---

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# We recommend using ChatOpenAI for better performance with modern models
agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# The create_react_agent function is a modern part of langgraph
agent_executor = create_react_agent(
    agent_llm,
    tools=[correlation_inspector, feature_generator]
)

# --- Main Execution ---

if __name__ == "__main__":
    # Ensure you have a parquet file at this path or create a dummy one
    # For example:
    # dummy_df = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'reg_target_lookahead6'])
    # dummy_df.to_parquet('labeled_data.parquet')
    
    data_input = f"parquet/labeled_data_6NQ.parquet"
    target_col = "reg_target_lookahead6"
    
    # Check if the data file exists
    if not os.path.exists(data_input):
        print(f"Error: Data file not found at '{data_input}'.")
        print("Please create a dummy parquet file to run this example.")
    else:
        # Define the messages for the agent
        messages = [
            ("user", f"My data is in '{data_input}' and the target is '{target_col}'."),
            ("user", "First, find the top 5 most correlated features."),
            ("user", "Next, generate 5 new feature ideas and save the code."),
            ("user", "Finally, recommend a combined feature list of 8-10 features (both existing and new) with brief reasoning.")
        ]

        # Invoke the agent with a list of messages
        # The stream() method provides real-time output of the agent's steps
        for chunk in agent_executor.stream({"messages": messages}):
            print(chunk)
            print("----")