# requirements: pip install openai langchain pandas pyarrow

import os
import pandas as pd
from typing import Union
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType

# --- Data loader utility ---

def load_data(data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Accepts either a parquet file path or a DataFrame; returns a DataFrame.
    """
    if isinstance(data_source, str):
        if data_source.endswith('.parquet'):
            return pd.read_parquet(data_source)
        raise ValueError("Unsupported file type: only .parquet files are allowed when passing a path.")
    elif isinstance(data_source, pd.DataFrame):
        return data_source.copy()
    else:
        raise ValueError("data_source must be a parquet filepath or a pandas DataFrame.")

# --- Tool 1: Correlation inspector ---

def top_correlated_features(
    data_source: Union[str, pd.DataFrame],
    target: str,
    top_n: int = 10
) -> str:
    df = load_data(data_source)
    corrs = df.corr()[target].abs().sort_values(ascending=False)
    top_feats = corrs.drop(labels=[target]).head(top_n).index.tolist()
    return f"Top {top_n} features by absolute correlation with '{target}': {top_feats}"

corr_tool = Tool(
    name="correlation_inspector",
    func=top_correlated_features,
    description=(
        "Compute and return top correlated features with the target column."
    )
)

# --- Tool 2: Feature generator ---

def generate_feature_code(
    data_source: Union[str, pd.DataFrame],
    target: str,
    suggestions: int = 5
) -> str:
    """
    Invokes the LLM to propose new trading feature formulas,
    writes the generated Python code into 'generated_features.py',
    and returns a summary list.
    """
    df = load_data(data_source)
    # Gather existing columns for context
    cols = df.columns.tolist()

    prompt = (
        f"You are a trading quant. Given existing columns: {cols}\n"
        f"and target '{target}', propose {suggestions} new feature formulas. "
        "For each, provide a one-line pandas expression."
    )
    template = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    )
    llm = OpenAI(model="gpt-4o-mini")
    chain = LLMChain(llm=llm, prompt=template)
    response = chain.run(prompt=prompt)

    # Write to file
    file_path = 'generated_features.py'
    with open(file_path, 'w') as f:
        f.write("# Auto-generated feature code\n")
        f.write(response)

    # Parse feature names from response for summary
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    summary = [l for l in lines if l.startswith('#') or '=' in l]
    return f"Generated {len(summary)} feature lines and saved to '{file_path}'."

feat_tool = Tool(
    name="feature_generator",
    func=generate_feature_code,
    description=(
        "Generate and write new trading feature code based on existing data."
    )
)

# --- Agent setup ---

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "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")
llm = OpenAI(model="gpt-4o-mini")

agent = initialize_agent(
    tools=[corr_tool, feat_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    data_input = "../../notebooks/parquet/labeled_data_6NQ.parquet"  # or a pandas DataFrame
    target_col = "clf_target_numba_6"
    prompt = (
        f"Step 1: Call correlation_inspector to find top correlated features.\n"
        f"Step 2: Use feature_generator to propose 5 new trading features, generate code, and save to file.\n"
        f"Finally, recommend a combined feature list of 8–12 features (existing + new) with reasoning."
    )
    output = agent.run(prompt)
    print(output)
