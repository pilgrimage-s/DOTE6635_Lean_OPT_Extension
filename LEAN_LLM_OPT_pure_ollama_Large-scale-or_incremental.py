#!/usr/bin/env python
# coding: utf-8

# Auto-generated from LEAN_LLM_OPT_pure_ollama_Large-scale-or.ipynb.
# Run-only version: stops after Large-Scale-OR orchestrated CSV output.
# The CSV is updated after every completed test example.


# In[1]:

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_classic.chains import LLMChain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from typing import Dict, List, Union
from pathlib import Path
from io import StringIO
import ast
import contextlib
import csv
import io
import json
import os
import re
import requests
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# # Pure Ollama Version
# 
# This notebook removes OpenAI API usage from the gpt-oss Large-Scale-OR workflow. Both generation and embeddings use the local Ollama server.


# ## Main Models


# ### Preperations


# In[5]:

# Pure Ollama configuration.
# Before running this notebook, start Ollama and pull the models, for example:
#   ollama serve
#   ollama pull gpt-oss:20b
#   ollama pull embeddinggemma
# You can override these with environment variables if needed.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")


def build_llm(model: str = OLLAMA_LLM_MODEL, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=OLLAMA_BASE_URL,
        timeout=500,
    )


embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)
llm1 = build_llm()
llm2 = build_llm()
llm_code = build_llm()
llm_orchestrator = build_llm()

print(f"Using Ollama LLM model: {OLLAMA_LLM_MODEL} @ {OLLAMA_BASE_URL}")
print(f"Using Ollama embedding model: {OLLAMA_EMBED_MODEL}")

# In[6]:

def get_code(output,selected_problem):

    prompt = f"""
    You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

    Mathematical Optimization Model:
    {output}
    """

    if selected_problem == "Network Revenue Management" or selected_problem == "NRM" or selected_problem == "Network Revenue Management Problem":

        prompt += r"""
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\quad \quad \max \quad \sum_i A_i \cdot x_i$
Constraints
1. Inventory Constraints:
$\quad \quad x_i \leq I_i, \quad \forall i$
2. Demand Constraints:
$x_i \leq d_i, \quad \forall i$
3. Startup Constraint:
$\sum_i x_i \geq s$
Retrieved Information
$\small I = [7550, 6244]$
$\small A = [149, 389]$
$\small d = [15057, 12474]$
$\small s = 100$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Product_Optimization")

# Decision variables for the number of units of each product
x_1 = m.addVar(vtype=GRB.INTEGER, name="x_1") # Number of units of product 1
x_2 = m.addVar(vtype=GRB.INTEGER, name="x_2") # Number of units of product 2

# Objective function: Maximize 149 x_1 + 389 x_2
m.setObjective(149 * x_1 + 389 * x_2, GRB.MAXIMIZE)

# Constraints
m.addConstr(x_1 <= 7550, name="inventory_constraint_1")
m.addConstr(x_2 <= 6244, name="inventory_constraint_2")
m.addConstr(x_1 <= 15057, name="demand_constraint_1")
m.addConstr(x_2 <= 12474, name="demand_constraint_2")

# Non-negativity constraints are implicitly handled by the integer constraints (x_1, x_2 >= 0)

# Solve the model
m.optimize()

        """

    elif selected_problem == "Facility Location Problem" or selected_problem == "FLP" or selected_problem == "Facility Location":
        prompt += r"""
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\quad \quad \min \quad \sum_{i} \sum_{j} A_{ij} \cdot x_{ij} + \sum_{i} c_i \cdot y_i$

Constraints
1. Demand Constraint:
$\quad \quad \sum_i x_{ij} = d_j, \quad \forall j$
2. Capacity Constraint:
$\quad \quad \sum_j x_{ij} \leq M \cdot y_i, \quad \forall i$
3. Non-negativity:
$\quad \quad x_{ij} \geq 0, \quad \forall i,j$
4. Binary Requirement:
$\quad \quad y_i \in \{0,1\}, \quad \forall i$

Retrieved Information
$\small d = [1083, 776, 16214, 553, 17106, 594, 732]$
$\small c = [102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16]$
$\small A = \begin{bmatrix}
1506.22 & 70.90 & 8.44 & 260.27 & 197.47 & 71.71 & 61.19 \\  
1732.65 & 1780.72 & 567.44 & 448.68 & 29.00 & 1484.91 & 963.92 \\  
115.66 & 100.76 & 64.68 & 1324.53 & 64.99 & 134.88 & 2102.83 \\  
1254.78 & 1115.63 & 52.31 & 1036.16 & 892.63 & 1464.04 & 1383.41 \\  
42.90 & 891.01 & 1013.94 & 1128.72 & 58.91 & 42.89 & 1570.31 \\  
0.70 & 139.46 & 70.03 & 79.15 & 1482.00 & 0.91 & 110.46 \\  
1732.30 & 1780.44 & 486.50 & 523.74 & 522.08 & 82.48 & 826.41
\end{bmatrix}$
$\small M = \sum_j d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 38058 $


The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
d = np.array([1083, 776, 16214, 553, 17106, 594, 732])
c = np.array([102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16])
A = np.array([[1506.22, 70.90, 8.44, 260.27, 197.47, 71.71, 61.19],  
[1732.65, 1780.72, 567.44, 448.68, 29.00, 1484.91, 963.92],  
[115.66, 100.76, 64.68, 1324.53, 64.99, 134.88, 2102.83],  
[1254.78, 1115.63, 52.31, 1036.16, 892.63, 1464.04, 1383.41],  
[42.90, 891.01, 1013.94, 1128.72, 58.91, 42.89, 1570.31],  
[0.70, 139.46, 70.03, 79.15, 1482.00, 0.91, 110.46],  
[1732.30, 1780.44, 486.50, 523.74, 522.08, 82.48, 826.41]])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(A.shape[0], A.shape[1], lb=0, name="x")
y = m.addVars(A.shape[0], vtype=GRB.BINARY, name="y")

# Objective function
m.setObjective(gp.quicksum(A[i, j]*x[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])) + gp.quicksum(c[i]*y[i] for i in range(A.shape[0])), GRB.MINIMIZE)

# Constraints
for j in range(A.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(A.shape[0])) == d[j], name=f"demand_constraint_{j}")

M = 1000000  # large number
for i in range(A.shape[0]):
    m.addConstr(-M*y[i] + gp.quicksum(x[i, j] for j in range(A.shape[1])) <= 0, name=f"M_constraint_{i}")

# Solve the model
m.optimize()
        """

    elif selected_problem == "Assignment Problem" or selected_problem == "AP" or selected_problem == "Assignment":
        prompt += r"""
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\quad \quad \min \quad \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} \cdot x_{ij}$

Constraints
1. Row Assignment Constraint:
$\quad \quad \sum_{j=1}^3 x_{ij} = 1, \quad \forall i \in \{1,2,3\}$
2. Column Assignment Constraint:
$\quad \quad \sum_{i=1}^3 x_{ij} = 1, \quad \forall j \in \{1,2,3\}$
3. Binary Constraint:
$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i,j$

Retrieved Information
$\small c = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 
\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [3000, 3200, 3100],
    [2800, 3300, 2900],
    [2900, 3100, 3000]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize()
"""

    
    elif selected_problem == "Transportation Problem" or selected_problem == "TP" or selected_problem == "Transportation":
        prompt += r"""
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\quad \quad \min \quad \sum_i \sum_j c_{ij} \cdot x_{ij}$

Constraints
1. Demand Constraint:
$\quad \quad \sum_i x_{ij} \geq d_j, \quad \forall j$
2. Capacity Constraint:
$\quad \quad \sum_j x_{ij} \leq s_i, \quad \forall i$

Retrieved Information
$\small d = [94, 39, 65, 435]$
$\small s = [2531, 20, 210, 241]$
$\small c = \begin{bmatrix}
883.91 & 0.04 & 0.03 & 44.45 \\
543.75 & 23.68 & 23.67 & 447.75 \\
537.34 & 23.76 & 498.95 & 440.60 \\
1791.49 & 68.21 & 1432.48 & 1527.76
\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization")

# Decision variables
x_S1_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C1")
x_S1_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C2")
x_S1_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C3")
x_S1_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C4")
x_S2_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C1")
x_S2_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C2")
x_S2_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C3")
x_S2_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C4")
x_S3_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C1")
x_S3_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C2")
x_S3_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C3")
x_S3_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C4")
x_S4_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C1")
x_S4_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C2")
x_S4_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C3")
x_S4_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C4")

# Objective function
m.setObjective(883.91 * x_S2_C1 + 0.04 * x_S2_C2 + 0.03 * x_S2_C3 + 44.45 * x_S2_C4 + 543.75 * x_S1_C1 + 23.68 * x_S1_C2 + 23.67 * x_S1_C3 + 447.75 * x_S1_C4 + 537.34 * x_S3_C1 + 23.76 * x_S3_C2 + 498.95 * x_S3_C3 + 440.60 * x_S3_C4 + 1791.49 * x_S4_C1 + 68.21 * x_S4_C2 + 1432.48 * x_S4_C3 + 1527.76 * x_S4_C4, GRB.MINIMIZE)

# Constraints
m.addConstr(x_S1_C1 + x_S2_C1 + x_S3_C1 + x_S4_C1 >= 94, name="demand_constraint1")
m.addConstr(x_S1_C2 + x_S2_C2 + x_S3_C2 + x_S4_C2 >= 39, name="demand_constraint2")
m.addConstr(x_S1_C3 + x_S2_C3 + x_S3_C3 + x_S4_C3 >= 65, name="demand_constraint3")
m.addConstr(x_S1_C4 + x_S2_C4 + x_S3_C4 + x_S4_C4 >= 435, name="demand_constraint4")
m.addConstr(x_S1_C1 + x_S1_C2 + x_S1_C3 + x_S1_C4 <= 2531, name="capacity_constraint1")
m.addConstr(x_S2_C1 + x_S2_C2 + x_S2_C3 + x_S2_C4 <= 20, name="capacity_constraint2")
m.addConstr(x_S3_C1 + x_S3_C2 + x_S3_C3 + x_S3_C4 <= 210, name="capacity_constraint3")
m.addConstr(x_S4_C1 + x_S4_C2 + x_S4_C3 + x_S4_C4 <= 241, name="capacity_constraint4")

# Solve the model
m.optimize()
        """
    
    elif selected_problem == "Resource Allocation" or selected_problem == "RA" or selected_problem == "Resource Allocation Problem":
        prompt += r"""
For example, here is a simple instance for reference:

Always remember: If not specified. All the variables are non-negative interger.

Mathematical Optimization Model:

Objective Function:
$\quad \quad \max \quad \sum_i \sum_j p_i \cdot x_{ij}$

Constraints
1. Capacity Constraint:
$\quad \quad \sum_i a_i \cdot x_{ij} \leq c_j, \quad \forall j$
2. Non-negativity Constraint:
$\quad \quad x_{ij} \geq 0, \quad \forall i,j$

Retrieved Information
$\small p = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]$
$\small a = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]$
$\small c = [4466]$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(20, vtype=GRB.INTEGER, name="x")

# Objective function
m.setObjective(sum(x[i]*c[i] for i in range(20)), GRB.MAXIMIZE)

# Constraints
m.addConstr(sum(x[i]*w[i] for i in range(20)) <= 4466, name="capacity_constraint")

# Coefficients for the objective function
c = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]

# Coefficients for the capacity constraint
w = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]

# Solve the model
m.optimize()
```

-----
Here is another simple instance for reference:

Objective Function:
$\quad \quad \max \quad \sum_i p_i \cdot x_i$

Constraints
1. Capacity Constraint:
$\quad \quad \sum_i a_i \cdot x_i \leq 180$
2. Dependency Constraint:
$\quad \quad x_1 \leq x_3$
3. Non-negativity Constraint:
$\quad \quad x_i \geq 0, \quad \forall i$

Retrieved Information
$\small p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]$
$\small a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]$

The corresponding Python code for this instance is as follows:

import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(10, vtype=GRB.INTEGER, name="x")

# Objective function
p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]
m.setObjective(sum(x[i]*p[i] for i in range(10)), GRB.MAXIMIZE)

# Constraints
a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]
m.addConstr(sum(x[i]*a[i] for i in range(10)) <= 180, name="capacity_constraint")
m.addConstr(x[0] <= x[2], name="dependency_constraint")

# Solve the model
m.optimize()
        
        """
    else:
        prompt += r"""
For example, here is a simple instance for reference:

Mathematical Optimization Model:
Maximize 5x_S + 8x_F
Subject to
    2x_S + 5x_F <= 200
    x_S <= 0.3(x_S + x_F)
    x_F >= 10
    x_S, x_F _ Z+

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Worker_Optimization")

# Decision variables for the number of seasonal (x_S) and full-time (x_F) workers
x_S = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_S")  # Number of seasonal workers
x_F = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_F")  # Number of full-time workers

# Objective function: Maximize Z = 5x_S + 8x_F
m.setObjective(5 * x_S + 8 * x_F, GRB.MAXIMIZE)

# Constraints
m.addConstr(2 * x_S + 5 * x_F <= 200, name="resource_constraint")
m.addConstr(x_S <= 0.3 * (x_S + x_F), name="seasonal_ratio_constraint")
m.addConstr(x_F >= 10, name="full_time_minimum_constraint")

# Non-negativity constraints are implicitly handled by the integer constraints (x_S, x_F >= 0)

# Solve the model
m.optimize()
```
The another example is:

Mathematical Optimization Model:
Minimize 919x_11 + 556x_12 + 951x_13 + 21x_21 + 640x_22 + 409x_23 + 59x_31 + 786x_32 + 304x_33
Subject to
    x_11 + x_12 + x_13 = 1
    x_21 + x_22 + x_23 = 1
    x_31 + x_32 + x_33 = 1
    x_11 + x_21 + x_31 = 1
    x_12 + x_22 + x_32 = 1
    x_13 + x_23 + x_33 = 1
    x_11, x_12, x_13, x_21, x_22, x_23, x_31, x_32, x_33 ∈ {{0,1}}


The corresponding Python code for this instance is as follows:

```python

import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [919, 556, 951],
    [21, 640, 409],
    [59, 786, 304]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize() 
```
"""
    messages = [
        HumanMessage(content=prompt) 
    ]

    response = llm_code.invoke(messages)


    print(response.content)

    return response.content


# In[7]:

def extract_python_code(text):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_gurobi_code(code):
    try:
        
        local_vars = {'gp': gp, 'GRB': GRB, '__builtins__': __builtins__}
        
        exec(code, local_vars)
        
        models = []
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, gp.Model):
                models.append(var_value)
        
        if models:
            model = models[-1] 
            if model.status == GRB.OPTIMAL:
                return model.objVal
            else:
                print(f"Model status is not optimal: {model.status}")
                return None
        
        return None
    except Exception as e:
        print(f"Error running Gurobi code: {e}")
        return None

# In[8]:

def escape_braces(text: str) -> str:
    """
    Replace every { with {{  and every } with }}
    so that ChatPromptTemplate won't treat them as variables.
    """
    return text.replace("{", "{{").replace("}", "}}")

def load_documents(file_path: str) -> List[Document]:
    """Load a CSV or Excel-like table into LangChain Documents using CSVLoader-compatible text."""
    path = Path(file_path)
    try:
        with path.open("rb") as f:
            is_excel_zip = f.read(4) == b"PK\x03\x04"
    except OSError:
        is_excel_zip = False

    if is_excel_zip:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")

    docs = []
    for row_idx, row in df.iterrows():
        lines = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            lines.append(f"{col}: {value}")
        docs.append(Document(page_content="\n".join(lines), metadata={"source": str(path), "row": int(row_idx)}))
    return docs


REF_CSV_PATH = "Large_Scale_Or_Files/RefData.csv"
ref_docs = load_documents(REF_CSV_PATH)
ref_store: FAISS = FAISS.from_documents(ref_docs, embeddings)


def split_by_newline(docs: List[Document], min_len: int = 1) -> List[Document]:
    """Split CSVLoader documents into non-empty line-level documents for finer retrieval."""
    out = []
    for doc in docs:
        for line in doc.page_content.splitlines():
            line = line.strip()
            if len(line) >= min_len:
                out.append(Document(page_content=line, metadata=doc.metadata))
    return out
#②
# def split_by_newline(docs, min_len=1):
#     out = []
#     for d in docs:
#         lines = d.page_content.splitlines()
#         for line in lines:
#             line = line.strip()
#             if len(line) >= min_len:
#                 out.append(Document(page_content=line, metadata=d.metadata))
#     return out

# ref_docs_split = split_by_newline(ref_docs)
# ref_store = FAISS.from_documents(ref_docs_split, embeddings)
#③
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,      # 先用 800~1500 试
#     chunk_overlap=100
# )

# ref_docs_split = splitter.split_documents(ref_docs)
# ref_store = FAISS.from_documents(ref_docs_split, embeddings)

def retrieve_ref_examples(query: str, k: int = 5) -> List[Dict]:
    retriever = ref_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)   # ✅ LangChain v1
    return docs

def build_dynamic_few_shot(query: str, k: int = 5) -> str:
    examples = []
    for doc in retrieve_ref_examples(query, k=k):
        text = doc.page_content
        prompt_part = text.split("prompt:", 1)[1].split("Data_address:", 1)[0].strip()
        data_addr   = text.split("Data_address:", 1)[1].split("Label:", 1)[0].strip()
        label_part = text.split("Label:", 1)[1].split("Related:", 1)[0].strip()

        data_blocks = []
        for fp in map(str.strip, data_addr.splitlines()):
            if not fp:
                continue
            try:
                df = pd.read_csv(fp)
                header = f"[Data from {os.path.basename(fp)}]"
                rows = "\n".join(
                    ", ".join(f"{col}={row[col]}" for col in df.columns)
                    for _, row in df.iterrows()
                )
                data_blocks.append(header + "\n" + rows)
            except Exception as e:
                data_blocks.append(f"[Could not read {fp}: {e}]")

        data_section = "\n".join(data_blocks) if data_blocks else "[No data found]"

        example_str=(
            f"<EXAMPLE>\n"
            f"Query: {prompt_part}\n"
            f"Data:\n{data_section}\n"
            f"Answer: {label_part}\n"
            f"</EXAMPLE>"
        )
        example_str = escape_braces(example_str)
        examples.append(example_str)


    return "\n\n".join(examples)

ALLOWED_CATS = [
    "Network Revenue Management",
    "Resource Allocation",
    "Transportation",
    "Facility Location Problem",
    "Assignment Problem",
    "Others with CSV",
]

FEW_SHOT_FALLBACK_NO_CSV = """
<EXAMPLE>
Query: A book distributor moves books between two warehouses and two libraries (no csv mentioned).
Answer: Others without CSV
</EXAMPLE>
""".strip()

CLASSIFY_SYS_MSG = (
    "You are an assistant that classifies operations-research problems. "
    "Return **exactly one** category from the following list:\n"
    f"{', '.join(ALLOWED_CATS)}\n\n"
    "Rules:\n"
    "1. Follow the provided few-shot examples strictly.\n"
    "2. Output only the category name, without explanation."
    "3. For mixed-type problems such as including some additional requirements in resource allocation problem, return the label 'Others with CSV' instead of 'resource allocation'."

).strip()


def classify_problem(user_query: str) -> str:
    few_shot_dynamic = build_dynamic_few_shot(user_query, k=5)
    few_shot_section = few_shot_dynamic + "\n\n" + FEW_SHOT_FALLBACK_NO_CSV
    few_shot_section = escape_braces(few_shot_section)

    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFY_SYS_MSG),
            ("assistant", few_shot_section),
            ("human", "{input}"),
        ]
    )
    msgs = prompt_tmpl.format_messages(input=user_query)
    resp = llm1.invoke(msgs)           
    category = resp.content.strip()   
    return category if category in ALLOWED_CATS else "Others with CSV"


def process_dataset_address(dataset_address: str) -> List[Document]:

    documents = []
    file_addresses = dataset_address.strip().split('\n')  
    for file_idx, file_address in enumerate(file_addresses, start=1):
        try:
            df = pd.read_csv(file_address.strip())  
            file_name = file_address.strip().split('/')[-1]  
            for row_idx, row in df.iterrows():
                page_content = ", ".join([f"{col} = {row[col]}" for col in df.columns])
                documents.append(Document(page_content=page_content))
                
        except Exception as e:
            print(f"Error processing file {file_address}: {e}")
            continue
    
    return documents

NRM_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_NRM2_MD.csv"
nrm_docs = load_documents(NRM_RAG_PATH)
nrm_store: FAISS = FAISS.from_documents(nrm_docs, embeddings)

RA_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_RA2_MD.csv"
ra_docs = load_documents(RA_RAG_PATH)
ra_store: FAISS = FAISS.from_documents(ra_docs, embeddings)

TP_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_TP2_MD.csv"
tp_docs = load_documents(TP_RAG_PATH)
tp_store: FAISS = FAISS.from_documents(tp_docs, embeddings)

FLP_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_FLP2_MD.csv"
flp_docs = load_documents(FLP_RAG_PATH)
flp_store: FAISS = FAISS.from_documents(flp_docs, embeddings)

AP_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_AP2_MD.csv"
ap_docs = load_documents(AP_RAG_PATH)
ap_store: FAISS = FAISS.from_documents(ap_docs, embeddings)

OTHERS_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_Others.csv"
Others_docs = load_documents(OTHERS_RAG_PATH)
# ①
# Others_store: FAISS = FAISS.from_documents(Others_docs, embeddings)
# ②
Others_docs_split = split_by_newline(Others_docs)
Others_store = FAISS.from_documents(Others_docs_split, embeddings)


OW_RAG_PATH = "Large_Scale_Or_Files/RAG_Example_Others_Without_CSV.csv"
OW_docs = load_documents(OW_RAG_PATH)
OW_store: FAISS = FAISS.from_documents(OW_docs, embeddings)


def retrieve_examples(store, query: str, k: int = 1):
    retriever = store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)   # ✅ LangChain v1


def build_few_shot(store, user_query: str, k: int = 1):
    """
    Build few-shot block.  
    Each example includes:
      • Question (prompt)
      • A 'Thought' line reminding to look for <Related> rows
      • Final Answer (Label)
    """
    examples = []
    label_part = ""

    # retrieve k most-similar reference examples
    for doc in retrieve_examples(store, user_query, k=k):
        txt = doc.page_content
        
        split_at_formulation = txt.split("Data_address:", 1)
        prompt_part = split_at_formulation[0].replace("prompt:", "").strip()  
        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_addr = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label_part = split_at_label[0].strip()  
        related_part = split_at_label[1].strip()
        
        # ---------- read CSVs listed in Data_address ----------
        data_blocks = []
        for fp in map(str.strip, data_addr.splitlines()):
            if not fp:
                continue
            try:
                df = pd.read_csv(fp)
                df_show = df.head()
                header = f"[{os.path.basename(fp)} | showing {len(df_show)}/{len(df)} rows]"
                rows   = "\n".join(
                    ", ".join(f"{c}={row[c]}" for c in df_show.columns)
                    for _, row in df_show.iterrows()
                )
                data_blocks.append(header + "\n" + rows)
            except Exception as e:
                data_blocks.append(f"[Could not read {fp}: {e}]")

        data_section = "\n".join(data_blocks) if data_blocks else "[No data found]"

        # ---------- compose few-shot example ----------
        ex = (
"<EXAMPLE>\n"
f"Question: {prompt_part}\n\n"
f'Thought: Retrieve rows related to "{related_part}"; then use the retrieved CSV to define sets and coefficients; choose variable domains (binary/integer for selections or lots or assignments, nonnegative reals for flows or amounts); build the linear objective; add capacity or supply bounds, demand/target constraints, flow conservation or assignment equalities, and any linking, batch, fixed-charge, or ratio/share constraints; then state nonnegativity and variable domains; output only the LP.'
"from the user's CSV file(s) and then formulate the optimisation model.\n\n"
f"Here is the data from retrieval in this demo example:\n{data_section}\n\n"
"Final Answer:\n"
f"{label_part}\n"
"</EXAMPLE>"
        )
        examples.append(escape_braces(ex))


    return "\n\n".join(examples),label_part


def build_few_shot_Other(store, user_query: str, k: int = 1,t='Model'):
    """
    Build few-shot block.  
    Each example includes:
      • Question (prompt)
      • A 'Thought' line reminding to look for <Related> rows
      • Final Answer (Label)
    """
    examples = []
    label_part = ""

    # retrieve k most-similar reference examples
    for doc in retrieve_examples(store, user_query, k=k):
        txt = doc.page_content
        
        split_at_formulation = txt.split("Data_address:", 1)
        prompt_part = split_at_formulation[0].replace("prompt:", "").strip()  
        if t =='Model':
            split_at_address = split_at_formulation[1].split("Label:", 1)
        else:
            split_at_address = split_at_formulation[1].split("Label_Code:", 1)
        data_addr = split_at_address[0].strip()

        data_blocks = []
        for fp in map(str.strip, data_addr.splitlines()):
            if not fp:
                continue
            try:
                df = pd.read_csv(fp)
                df_show = df.head()
                header = f"[{os.path.basename(fp)} | showing {len(df_show)}/{len(df)} rows]"
                rows   = "\n".join(
                    ", ".join(f"{c}={row[c]}" for c in df_show.columns)
                    for _, row in df_show.iterrows()
                )
                data_blocks.append(header + "\n" + rows)
            except Exception as e:
                data_blocks.append(f"[Could not read {fp}: {e}]")

        data_section = "\n".join(data_blocks) if data_blocks else "[No data found]"

        if t == 'Model':
            # ---------- compose few-shot example ----------
            ex = (
    "<EXAMPLE>\n"
    f"Question: {prompt_part}\n"
    f'Thought: Load the referenced CSV tables and extract needed columns for coefficients and limits; define index sets from table rows and columns; choose variable types (counts and on-off as integer or binary, flows and amounts as nonnegative reals); then build the model using only the retrieved data.'
    "Action: CSVQA\n"
    "Action Input: Retrieve all necesscary data\n"
    f"Observation: Here is the data from retrieval in this demo example:\n{data_section}\n"
    "Final Answer:\n"
    f"{label_part}\n"
    "</EXAMPLE>"
            )
            examples.append(escape_braces(ex))
        else:
            ex = (
    "<EXAMPLE>\n"
    f"Question: {prompt_part}\n"
    "Thought: I need to inspect the CSV file's structure (columns and data types) before writing the code."
    "Action: GetCSVSchema\n"
    f"Action Input: {data_addr}\n"
    f"Observation: \n{data_section}\n"
    "Final Answer:\n"
    f"{label_part}\n"
    "</EXAMPLE>"
            )
            examples.append(escape_braces(ex))

    return "\n\n".join(examples),label_part


def build_few_shot_TP(store, user_query: str, k: int = 1):
    """
    Build few-shot block.  
    Each example includes:
      • Question (prompt)
      • A 'Thought' line reminding to look for <Related> rows
      • Final Answer (Label)
    """
    examples = []
    label_part = ""

    # retrieve k most-similar reference examples
    for doc in retrieve_examples(store, user_query, k=k):
        txt = doc.page_content
        
        split_at_formulation = txt.split("Data_address:", 1)
        prompt_part = split_at_formulation[0].replace("prompt:", "").strip()  
        split_at_address = split_at_formulation[1].split("Label:", 1)
        data_addr = split_at_address[0].strip()

        split_at_label = split_at_address[1].split("Related:", 1)
        label_part = split_at_label[0].strip()  
        related_part = split_at_label[1].strip()
        
        # ---------- read CSVs listed in Data_address ----------
        data_blocks = []
        for fp in map(str.strip, data_addr.splitlines()):
            if not fp:
                continue
            try:
                df = pd.read_csv(fp)
                df_show = df.head()
                header = f"[{os.path.basename(fp)} | showing {len(df_show)}/{len(df)} rows]"
                rows   = "\n".join(
                    ", ".join(f"{c}={row[c]}" for c in df_show.columns)
                    for _, row in df_show.iterrows()
                )
                data_blocks.append(header + "\n" + rows)
            except Exception as e:
                data_blocks.append(f"[Could not read {fp}: {e}]")

        data_section = "\n".join(data_blocks) if data_blocks else "[No data found]"
        ex = (
            f"Question: {prompt_part}\n\n"

            f"Thought: The user wants me to formulate an optimization model. First, I must use the `csvqa` tool to get the data from the file paths. \n"
            "Action: csvqa \n"
            f"Action Input: {data_addr}\n"
            "Observation: d = [11, 1148, 54, 833], s = [4, 575, 1504], c = \\begin{{bmatrix}} 0.63 & 49.71 & 33.75 & 1570.67 \\605.47 & 64.53 & 478.47 & 887.04 \\1139.04 & 4.78 & 1805.62 & 1302.89\\end{{bmatrix}}$\n"
            "Thought: I have successfully received the data for demand (d), supply (s), and costs (c). Now I will formulate the complete mathematical model in Markdown as my final answer.\n"
            "Final Answer: FINAL_MODEL_OUTPUT: \n"
            f"{label_part}\n"
            "</EXAMPLE>"
        )
        examples.append(escape_braces(ex))


    return "\n\n".join(examples),label_part



DISALLOWED = {
    "Revenue",
    "sales",
    "demand",
    "inventory",
    "initial inventory",
    "product",
    "product name",
    "column",
    "row",
    "none",
}

def _post_process_kw(raw_kw: str) -> str:
    """Clean quotes & punctuation, lower-case check against disallowed set."""
    kw = re.sub(r"[\"'“”‘’]", "", raw_kw).strip()
    return "" if kw.lower() in DISALLOWED or kw == "" else kw

def extract_retrieval_keyword(user_query: str) -> str:
    """
    Return ONE product / item keyword; empty string means 'no keyword'.
    Filters out obvious column names via DISALLOWED set.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Read the USER QUESTION and output ONE product name or product type "
                "to look up in the CSV files. "
                "If the question doesn't mention a specific product or type, output NONE.\n\n"
                "❌  Do NOT output column names like 'Revenue', 'Inventory', 'Demand', etc.\n"
                "✅  Example:\n"
                "   Question: The store wants to maximise revenue for the Nike x Olivia Kim shoes …\n"
                "   Answer: Nike x Olivia Kim\n"
            ),
            ("human", "{q}"),
        ]
    )

    # Get raw LLM output
    raw_kw = (prompt | llm2).invoke({"q": user_query}).content.strip()
    return _post_process_kw(raw_kw)

def extract_retrieval_keyword_other(user_query: str) -> tuple[str, str]:
    """
    Extract a specific product/item/category keyword ONLY if the user explicitly
    states that the modeling should focus on a subset of the data.
    Otherwise, return NONE.
    
    Also returns the English-translated version of the original query.

    Returns:
        (keyword: str, query_en: str)
    """

    # Step 2: Keyword extraction prompt
    keyword_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "Read the USER QUESTION and check if it explicitly states that the modeling "
                "should only consider a specific subset of the data, such as one product, "
                "one factory, or one category.\n"
                "- If the query does NOT clearly restrict to a subset, output NONE.\n"
                "- If it does, output ONLY the concise English name of that product/item/category.\n\n"
                "Rules:\n"
                "1. NEVER output column names like Revenue, Inventory, Demand, Sales, etc.\n"
                "2. NEVER output vague or generic terms like 'market', 'production', 'data', 'factory', etc.\n"
                "3. Keyword must be specific.\n"
                "4. Output should ONLY be the keyword or NONE."
            )
        ),
        ("human", "{q}")
    ])
    messages = keyword_prompt.format_messages(q=user_query)   
    resp = llm2.invoke(messages)                    
    raw_kw = resp.content.strip()

    # Step 3: Post-processing filter
    def _post_process_kw_en(s: str) -> str:
        disallowed = {
            'revenue', 'inventory', 'demand', 'sales', 'profit',
            'market', 'production', 'data', 'factory', 'plant', 'type',
            'cost', 'price', 'amount', 'value', 'total', 'quantity'
        }
        kw = s.strip().lower()
        # If NONE or empty, return NONE
        if not kw or kw == 'none':
            return 'NONE'
        # Filter out disallowed and overly generic terms
        if kw in disallowed or any(word in disallowed for word in kw.split()):
            return 'NONE'
        # Keep keyword in original English formatting (capitalize if needed)
        return s.strip()

    keyword_final = _post_process_kw_en(raw_kw)

    return keyword_final


def llm_extract_keyword(llm: ChatOllama, user_query: str) -> str:
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a precise text analyzer. Your task is to identify the specific **product ID, category, or classification** that the user's request is focusing on for data filtering. "
         "If no specific filtering condition is clearly mentioned, return **ONLY** the phrase: 'all relevant data'."),
        ("human", "Analyze the following request: {query}")
    ])
    
    extraction_chain = (
        extraction_prompt 
        | llm  
        | StrOutputParser()
    )
    
    try:
        keyword = extraction_chain.invoke({"query": user_query}).strip().strip("'\"").lower()
        if not keyword or keyword in ["the", "a", "of", "products", "items"]:
            return "all relevant data"
        return keyword
        
    except Exception as e:
        print(f"[ERROR] LLM Keyword Extraction Failed: {e}")
        return "all relevant data"

# In[9]:

def extract_all_lists_from_docs_no_mapping(documents: List[Document]):
    if not documents:
        return {"ERROR": "[FATAL_EXTRACTION_FAILURE: No documents retrieved.]"}
    first_content = documents[0].page_content
    key_value_pairs = first_content.split(', ')
    column_names = []
    
    for pair in key_value_pairs:
        if '=' in pair:
            key = pair.split('=', 1)[0].strip()
            column_names.append(key)

    if not column_names:
        return {"ERROR": "[FATAL_MAPPING_FAILURE: Could not identify any column names in the documents.]"}
    extracted_lists: Dict[str, List[str]] = {name: [] for name in column_names}

    for doc in documents:
        content = doc.page_content
        
        for col_name in column_names:
            tag = f"{col_name} ="
            value_str = 'N/A' 
            
            try:
                start_index = content.index(tag) + len(tag)
                end_index = content.find(',', start_index)
                
                if end_index == -1:
                    value_str = content[start_index:].strip()
                else:
                    value_str = content[start_index:end_index].strip()
                
                value_str = value_str.replace('{', '').replace('}', '').replace('"', '').strip()
                
            except ValueError:
                pass 
            
            extracted_lists[col_name].append(value_str)
            
    return extracted_lists


def process_dataset_address(dataset_address: str) -> List[Document]:
    documents = []
    file_addresses = dataset_address.strip().split('\n')
    for file_address in file_addresses:
        file_address = file_address.strip()
        if not file_address: continue

        try:
            df = pd.read_csv(file_address)
            for row_idx, row in df.iterrows():
                page_content = ", ".join([f"{col} = {row[col]}" for col in df.columns])
                documents.append(Document(page_content=page_content))
        except Exception as e:
            print(f"Error processing file {file_address}: {e}")
            continue
    return documents

def process_dataset_address_RA(dataset_address: str) -> List[Document]:
    """Reads CSV files and creates a list of Documents for FAISS indexing.
    
    MODIFIED: This version adds the source filename to each document's metadata.
    """
    documents = []
    file_addresses = dataset_address.strip().split('\n')
    for file_address in file_addresses:
        file_address = file_address.strip()
        if not file_address: continue

        try:
            df = pd.read_csv(file_address)
            
            # --- 1. Get the filename from the full path ---
            file_name = os.path.basename(file_address)
            
            for row_idx, row in df.iterrows():
                page_content = ", ".join([f"{col} = {row[col]}" for col in df.columns])
                
                # --- 2. Create the metadata dictionary ---
                metadata = {"source": file_name}
                
                # --- 3. Pass the metadata when creating the Document ---
                documents.append(Document(page_content=page_content, metadata=metadata))
                
        except Exception as e:
            print(f"Error processing file {file_address}: {e}")
            continue
            
    return documents



def process_dataset_address_vector_FLP(dataset_address: str) -> List[Document]:
    documents = []
    file_addresses = dataset_address.strip().split('\n')
    df_index = 0
    data_description = " "
    for file_address in file_addresses:
        try:
            df = pd.read_csv(file_address) 
            file_name = file_address.split('/')[-1] 
            if 'demand' in df.columns:
                result = df['demand'].values.tolist()
                data_description += "d=" + str(result) + "\n"
            elif 'fixed_costs' in df.columns:
                result = df['fixed_costs'].values.tolist()
                data_description +="c=" + str(result) + "\n"
            elif df_index == 2:
                matrix = df.iloc[:,1:].values
                data_description +="A=" + np.array_str(matrix)+ "."
            else:
                for row_idx, row in df.iterrows():
                    data_description += ", ".join([f"{col} = {row[col]}" for col in df.columns])
            df_index += 1
        except Exception as e:
            print(f"Error reading file {file_address}: {e}")
    documents = [data_description]
    return documents


def process_dataset_address_vector_AP(dataset_address: str) -> List[Document]:
    dfs=[]
    file_addresses = dataset_address.strip().split('\n')
    df_index = 0
    data_description = " "
    for file_address in file_addresses:
        try:
            df = pd.read_csv(file_address) 
            file_name = file_address.split('/')[-1] 
            matrix = df.iloc[:,1:].values
            data_description +="C=" + np.array_str(matrix)+ "."
            dfs.append((file_name, df))
            df_index += 1
            dfs.append((file_name, df))
        except Exception as e:
            print(f"Error reading file {file_address}: {e}")
    documents = [data_description]
    return documents

def process_dataset_address_vector_TP(dataset_address: str) -> List[Document]:
    data = []
    dfs=[]

    file_addresses = dataset_address.strip().split('\n')
    for file_address in file_addresses:
        try:
            df = pd.read_csv(file_address) 
            file_name = file_address.split('/')[-1] 
            dfs.append((file_name, df))
        except Exception as e:
            print(f"Error reading file {file_address}: {e}")

    for df_index, (file_name, df) in enumerate(dfs):
        data.append(f"\nDataFrame {df_index + 1} - {file_name}:\n")

        for i, r in df.iterrows():
            description = ""
            description += ", ".join([f"{col} = {r[col]}" for col in df.columns])
            data.append(description + "\n")

    documents = [content for content in data]
    return documents

# In[10]:

import pandas as pd
import json
from typing import List, Dict

def identify_data_roles(dfs: List[pd.DataFrame], llm) -> Dict:
    """
    Analyzes the structure of a list of DataFrames using an LLM and returns a JSON object
    mapping each data role to its corresponding column name and DataFrame index.
    """
    if not dfs:
        return {}

    # Create a string describing the structure of all DataFrames to serve as context for the LLM.
    context_str = ""
    for i, df in enumerate(dfs):
        context_str += f"DataFrame {i} columns: {df.columns.tolist()}\n"
        context_str += f"First two rows of DataFrame {i}:\n{df.head(2).to_string()}\n\n"

    prompt = f"""
    You are an expert data analyst. Your task is to analyze the structure of the DataFrames below and identify the role each one plays.
    You need to identify three roles: 'demand' (customer needs), 'supply' (supplier capacity), and 'costs' (a cost matrix).

    Here is the structural information for all DataFrames:
    ---
    {context_str}
    ---

    Based on this information, please return your analysis in a strict JSON object format. The JSON object must contain three keys: 'demand', 'supply', and 'costs'.
    - For 'demand' and 'supply', provide their DataFrame index (`df_index`) and the specific column name (`column_name`).
    - For 'costs', provide its DataFrame index (`df_index`) and the name of the column used to identify the supplier ID (`supplier_id_column`).

    JSON format example:
    {{
      "demand": {{ "df_index": 0, "column_name": "requirement" }},
      "supply": {{ "df_index": 1, "column_name": "capacity" }},
      "costs": {{ "df_index": 2, "supplier_id_column": "supplier_id" }}
    }}
    
    Output the JSON object directly, without any other explanations or markdown.
    """
    
    try:
        response = llm.invoke(prompt)
        # Attempt to parse the string returned by the LLM into JSON.
        # For newer versions of LangChain, you can use .with_structured_output(MyPydanticModel) for more stable JSON.
        roles = json.loads(response.content)
        return roles
    except Exception as e:
        print(f"❌ Failed to identify data roles: {e}")
        return {}


def load_dataframes_from_files(file_paths_str: str) -> List[pd.DataFrame]:
    """
    A general-purpose function to read one or more CSV file paths
    and load them into a list of DataFrames. This version is enhanced
    to handle malformed input strings where paths might be joined together.
    """
    list_of_dfs = []

    # --- Robust Parsing Logic ---
    # 1. Normalize the input by replacing common incorrect separators (like spaces) with a known marker.
    #    We'll use the '.csv' extension as an anchor.
    # 2. This line finds every '.csv' and inserts a newline character right after it,
    #    effectively splitting mashed-together paths.
    normalized_str = file_paths_str.replace('.csv ', '.csv\n')

    # 3. Now, split the normalized string by newlines and filter out any empty strings.
    file_paths = [path.strip() for path in normalized_str.strip().split('\n') if path.strip()]
    
    print(f"Preparing to load files (after parsing): {file_paths}")

    for file_path in file_paths:
        if not file_path:
            continue
        try:
            df = pd.read_csv(file_path)
            list_of_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File '{file_path}' not found, skipping.")
        except Exception as e:
            print(f"Warning: An error occurred while loading '{file_path}': {e}, skipping.")

    if not list_of_dfs:
        print("Warning: No data was loaded, returning an empty list.")
    else:
        print(f"Successfully loaded {len(list_of_dfs)} DataFrame(s).")
        
    return list_of_dfs

# ### NRM


# In[12]:


def get_NRM_response(user_query: str, dataset_address: str) -> str:
    
    keyword = llm_extract_keyword(llm1, user_query)
    action_input_kw = keyword if keyword else "all relevant data"
    few_shot_block, _ = build_few_shot(nrm_store, user_query, k=1)
    print("[INFO] Loading all documents from CSVs...")
    user_docs = process_dataset_address(dataset_address)
    if not user_docs:
        return "Final Answer: [Could not process dataset. Please check file paths.]"
        
    print("[INFO] Creating a complete FAISS index...")
    user_store = FAISS.from_documents(user_docs, embeddings)
    retriever = user_store.as_retriever(search_kwargs={"k": 300}) 
    print("[INFO] Retriever is ready.")

    def csvqa_list_extractor_tool_func(query: str) -> str:
        print(f"[INFO] Tool received query: '{query}'")
        is_keyword_query = ' ' not in query.strip() and len(query.strip()) < 10

        docs_to_process = []

        if "all relevant data" in query:
            print(f"[INFO] Semantic query detected. Using vector retriever.")
            docs_to_process = retriever.invoke(query)
        else:
            print(f"[INFO] Keyword query detected. Using precise code filtering on all {len(user_docs)} documents.")
            query_lower = query.lower()
            safe_query_re = re.escape(query_lower)
            
            exact_pattern = re.compile(r'\b' + safe_query_re + r'\b', re.IGNORECASE)
            starts_with_pattern = re.compile(r'\b' + safe_query_re, re.IGNORECASE)

            exact_matches = []
            starts_with_matches = []
            
            for doc in user_docs:
                if exact_pattern.search(doc.page_content):
                    exact_matches.append(doc)
                elif starts_with_pattern.search(doc.page_content):
                    starts_with_matches.append(doc)
            
            docs_to_process = exact_matches if exact_matches else starts_with_matches
        
        if not docs_to_process:
            return "No relevant data found for the query."

        print(f"[INFO] Found {len(docs_to_process)} documents to process.")
        extracted_data_dict = extract_all_lists_from_docs_no_mapping(docs_to_process)
        
        if "ERROR" in extracted_data_dict:
            return extracted_data_dict["ERROR"]
            
        output_lines = []
        for col_name, data_list in extracted_data_dict.items():
            safe_col_name = col_name.replace(' ', '_').upper()
            output_lines.append(f"{safe_col_name}_LIST={data_list}")
        
        return "\n".join(output_lines)
    
    qa_tool = Tool(
        name="CSVQA",
        func=csvqa_list_extractor_tool_func, 
        description=(
            "Use this tool to search and retrieve data from a pre-loaded dataset. "
            "The input to this tool should be a query describing the data, e.g., 'all relevant revenue data' or a specific product keyword like '4U'."
        ),
    )
    
    prefix = (
        f"{few_shot_block}\n\n"
        "You MUST follow EXACTLY the pattern below for the **FIRST** response.\n"
        "Output **ONLY** the three lines for the Action, nothing before or after it:\n\n"
        f"Thought: I must first use the CSVQA tool to retrieve necessary data based on the extracted keyword: '{action_input_kw}'.\n"
        "Action: CSVQA\n"
        f"Action Input: {action_input_kw}\n\n"
        "Begin! Now process the human input:\n"
        "{input}"
    )
    suffix = (
        "### INSTRUCTIONS – FINAL STEP\n"
        "The Observation contains the necessary structured lists (R_LIST, D_LIST, I_LIST) OR a failure signal.\n"
        "You MUST now output **exactly two lines** to finalize the process. Do **not** add any other text.\n"
        "Thought: I have successfully received the R_LIST, D_LIST, and I_LIST. I must now integrate these lists with the NRM optimization structure and output the final model.\n" 
        "Final Answer:FINAL_MODEL_OUTPUT:<LaTeX optimization model followed by R_LIST/D_LIST/I_LIST data>."
    )

    agent = initialize_agent(
        tools=[qa_tool], llm=llm2, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": prefix, "suffix": suffix, "input_variables": ["input"]},
        verbose=True, handle_parsing_errors=True, max_iterations=5,
    )

    escaped_query = user_query.replace('{', '{{').replace('}', '}}')
    final_answer = agent.run({"input": escaped_query})

    return final_answer

# ### RA


# In[14]:

from typing import List, Dict, Any
from langchain_core.documents import Document

def extract_data_dynamically(docs: List[Document]) -> Dict[str, List[Any]]:
    """
    A robust two-pass parser that handles documents with different columns, followed by a
    cleaning step to ensure data integrity.

    - Pass 1: Discovers all unique column names from all documents.
    - Pass 2: Extracts values for all discovered columns, using None for missing values.
    - Pass 3 (Cleaning): Removes any records that are missing essential data (e.g., Value or Weight).
    """
    if not docs:
        return {"ERROR": "No documents to process."}

    all_column_names = set()
    parsed_rows = []

    for doc in docs:
        content = doc.page_content.strip()
        if not content or '=' not in content:
            continue
        
        row_data = {}
        pairs = [p.strip() for p in content.split(',')]
        for pair in pairs:
            try:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                all_column_names.add(key)
                row_data[key] = value
            except ValueError:
                continue 
        
        parsed_rows.append(row_data)

    if not all_column_names:
        return {"ERROR": "Could not find any valid key-value pairs in the documents."}

    initial_lists = {col: [] for col in all_column_names}

    for row_data in parsed_rows:
        for col_name in all_column_names:
            value = row_data.get(col_name, None)
            
            if isinstance(value, str):
                try:
                    # Attempt to convert to float/int
                    value = float(value)
                except (ValueError, TypeError):
                    pass # Keep as string if it's not a number
            
            initial_lists[col_name].append(value)
            
    essential_keys = ['Value', 'Weight']
    if not all(key in initial_lists for key in essential_keys):
        return initial_lists

    indices_to_remove = {
        i for i, (v, w) in enumerate(zip(initial_lists['Value'], initial_lists['Weight'])) 
        if v is None or w is None
    }

    if not indices_to_remove:
        return initial_lists # No cleaning needed, return the complete lists

    cleaned_lists = {}
    for col_name, data_list in initial_lists.items():
        cleaned_list = [
            item for i, item in enumerate(data_list) 
            if i not in indices_to_remove
        ]
        cleaned_lists[col_name] = cleaned_list

    return cleaned_lists

def get_RA_response(user_query: str, dataset_address: str) -> str:
    """
    Optimization: Max iterations increased for robustness.
    """
    few_shot_block,label= build_few_shot(ra_store, user_query, k=2)

    user_docs = process_dataset_address_RA(dataset_address)
    if not user_docs:
        return "Final Answer: [Could not process dataset. Please check file paths.]"
        
    user_store = FAISS.from_documents(user_docs, embeddings)
    retriever = user_store.as_retriever(search_kwargs={"k": 400}) # OPTIMIZED: k=20

    keyword = extract_retrieval_keyword(user_query)
    print(f"[DEBUG]: keyword is {keyword}\n")

    action_input_kw = keyword if keyword else "all data"
    def csvqa(q: str) -> str:
        system_prompt = (
            "You are a silent, precise data extraction robot. Your only function is to convert the Context into a string containing multiple Python-style lists. Follow these rules with absolute precision:\n"
            "1.  **Discover Fields:** Scan the entire Context to identify all unique data fields (e.g., 'ProductName', 'Value', 'Weight', 'Capacity').\n"
            "2.  **Create Clean Lists:** For each field, create a list containing ONLY the actual values found for that field in the context.\n"
            "3.  **CRITICAL RULE: Do NOT pad lists with `None` to make them the same length. Each list should have its own natural length. This means lists for different data sources WILL have different lengths.**\n"
            "4.  **Format Output String:** Construct the final output string. Each list must be on a new line in the format: `fieldname_list = [...]`.\n"
            "\n"
            "---CRITICAL OUTPUT RULES---\n"
            "-   Your response MUST start immediately with the first list. DO NOT include any other text, explanations, or markdown.\n"
            "-   Strings in lists must use single quotes (').\n"
            "-   Numbers must be written as numbers (e.g., 8372.0).\n"
            "\n"
            "---EXAMPLE---\n"
            "Context: ProductName = Truck, Value = 8372, Weight = 36\nCapacity = 1576\nProductName = Sedan, Value = 1752, Weight = 15\n"
            "Correct Output:\n"
            "productname_list = ['Truck', 'Sedan']\n"
            "value_list = [8372.0, 1752.0]\n"
            "weight_list = [36.0, 15.0]\n"
            "capacity_list = [1576.0]\n"
            "---END EXAMPLE---\n"
            "\n"
            "Now, process the following context.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm2, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        formatted_string_from_llm = qa_chain.invoke({"input": 'all data'})['answer']
        return formatted_string_from_llm


    CSVQA_TOOL = Tool(
        name="CSVQA",
        func=csvqa,
        description="Return rows from the user's CSV files that best match the given keyword/phrase.",
    )

    prefix = (
        f"{few_shot_block}\n\n"
        "You are an autonomous operations research analyst. Your sole purpose is to formulate a mathematical optimization model based on the user's problem description.\n"
        "You MUST follow EXACTLY the pattern below for the **FIRST** response.\n"
        "Your first and only action is to use the `CSVQA` tool to retrieve the structured data.\n"
        "Output **ONLY** the three lines for the Action, nothing before or after it:\n\n"
        f"Thought: I must first use the CSVQA tool to retrieve necessary data based on the extracted keyword: '{action_input_kw}'.\n"
        "Action: CSVQA\n"
        f"Action Input: {action_input_kw}\n\n"
        "Begin! Now process the human input:\n"
        "{input}"
    )

    suffix = (
        "### FINAL INSTRUCTIONS ###\n"
        "The Observation contains the clean, structured data lists needed for the model.\n"
        "You MUST now formulate the final mathematical model in MarkDown.\n"
        "You MUST output **exactly two lines**: a 'Thought' and a 'Final Answer'.\n\n"
        "Thought: I have received the clean data lists. My plan is as follows:\n"
        "1.  **Parameters:** I will identify all numerical parameters from the Observation.\n"
        "2.  **Assemble and Expand Markdown:** I will write out every single term of the objective function and all constraints in the Markdown model, using the identified parameters.\n"
        "Now I will execute this plan to create the final Markdown model.\n\n"
        "Final Answer: FINAL_MODEL_OUTPUT: <The complete and fully expanded Markdown optimization model, based on the data from the Observation. Do not use placeholders like '...'.>"
    )
    agent = initialize_agent(
        tools=[CSVQA_TOOL],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5, 
    )
    escaped_query = user_query.replace('{', '{{').replace('}', '}}')
    final_answer = agent.run({"input": escaped_query})

    return final_answer

# ### AP


# In[16]:

def get_AP_response(user_query: str, dataset_address: str) -> str:
    """
    Optimization: Max iterations increased for robustness.
    """
    few_shot_block,label= build_few_shot(ap_store, user_query, k=1)

    user_docs = process_dataset_address_vector_AP(dataset_address)
    if not user_docs:
        return "Final Answer: [Could not process dataset. Please check file paths.]"
        
    user_store = FAISS.from_texts(user_docs, embeddings)
    retriever = user_store.as_retriever(search_kwargs={"k": 400}) # OPTIMIZED: k=20

    keyword = extract_retrieval_keyword(user_query)
    print(f"[DEBUG]: keyword is {keyword}\n")

    action_input_kw = keyword if keyword else "all relevant data"

    def csvqa(q: str) -> str:
        system_prompt = (
        "Retrieve the documents in order from top to bottom. Use the retrieved context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, retrieve all the data instead."
        "Context: {context}"
    )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm2, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        return qa_chain.invoke({"input": q})['answer']


    CSVQA_TOOL = Tool(
        name="CSVQA",
        func=csvqa,
        description="Return rows from the user's CSV files that best match the given keyword/phrase.",
    )

    prefix = (
        f"{few_shot_block}\n\n"
        "You MUST follow EXACTLY the pattern below for the **FIRST** response.\n"
        "Output **ONLY** the three lines for the Action, nothing before or after it:\n\n"
        f"Thought: I must first use the CSVQA tool to retrieve necessary data based on the extracted keyword: '{action_input_kw}'.\n"
        "Action: CSVQA\n"
        f"Action Input: {action_input_kw}\n\n"
        "Begin! Now process the human input:\n"
        "{input}"
    )

    suffix = (
    "### FINAL INSTRUCTIONS ###\n"
    "The Observation contains the structured data in JSON format.\n"
    "Your task is to create the final, fully expanded optimization model.\n"
    "You MUST output **exactly two lines**: a 'Thought' and a 'Final Answer'.\n\n"

    "Thought: I have the structured data. My plan is as follows:\n"
    "1.  **Parameters:** I will identify all numerical parameters (objective coefficients, constraint coefficients, and limits) from the Observation JSON.\n"
    "2.  **Objective & Constraints:** I will define the objective function and all constraints using these specific numbers.\n"
    "3.  **Assemble and Fully Expand:** I will now combine all parts into a single, complete LaTeX model. I will write out every single term in the summations explicitly, without using any summary notation.\n"

    "Final Answer: FINAL_MODEL_OUTPUT: <\n"
    "---CRITICAL INSTRUCTION---\n"
    "The model MUST be fully expanded. Do NOT use summary notation like Σ or placeholders like '...'. Write out every term using the actual numbers from the data.\n\n"
    "**CORRECT FORMAT EXAMPLE:** `$$max \\sum_{{i=1}}^{{3}} p_i x_i, p=[24,66,57[$$`\n"
    ">"
    )
    agent = initialize_agent(
        tools=[CSVQA_TOOL],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    escaped_query = user_query.replace('{', '{{').replace('}', '}}')
    final_answer = agent.run({"input": escaped_query})

    return final_answer

# ### FLP


# In[18]:

def get_FLP_response(user_query: str, dataset_address: str) -> str:
    """
    Optimization: Max iterations increased for robustness.
    """
    few_shot_block,label= build_few_shot(flp_store, user_query, k=1) # Corrected store to flp_store

    user_docs = process_dataset_address_vector_FLP(dataset_address)
    if not user_docs:
        return "Final Answer: [Could not process dataset. Please check file paths.]"
        
    user_store = FAISS.from_texts(user_docs, embeddings)
    retriever = user_store.as_retriever(search_kwargs={"k": 400}) # OPTIMIZED: k=20

    keyword = extract_retrieval_keyword(user_query)
    print(f"[DEBUG]: keyword is {keyword}\n")

    action_input_kw = keyword if keyword else "all relevant data"

    def csvqa(q: str) -> str:
        system_prompt = (
        "Retrieve the documents in order. Use the given context to answer the question. If mention a certain kind of product, retrieve all the relavant product information detail judging by its product name. If not mention a certain kind of product, make sure that all the data is retrieved."
        "Context: {context}"
    )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm2, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        return qa_chain.invoke({"input": q})['answer']


    CSVQA_TOOL = Tool(
        name="CSVQA",
        func=csvqa,
        description="Return rows from the user's CSV files that best match the given keyword/phrase.",
    )

    prefix = (
        f"{few_shot_block}\n\n"
        "You MUST follow EXACTLY the pattern below for the **FIRST** response.\n"
        "Output **ONLY** the three lines for the Action, nothing before or after it:\n\n"
        f"Thought: If the observation is null, I must first use the CSVQA tool to retrieve necessary data based on the extracted keyword: '{action_input_kw}'. If I have the observation data, I should output the final model instead of using any tool.\n"
        "Action: CSVQA\n"
        f"Action Input: {dataset_address}\n\n"
        "Begin! Now process the human input:\n"
        "{input}"
    )
    suffix = (
        "### INSTRUCTIONS – FINAL STEP\n"
        "The Observation contains the necessary structured lists OR a failure signal.\n"
        "You MUST now output **exactly two lines** to finalize the process. Do **not** add any other text.\n"
        "Thought: I have successfully received the necessary data. I must now integrate these lists with the transportation optimization structure and output the final model.\n" 
        "Final Answer:FINAL_MODEL_OUTPUT:<LaTeX optimization model followed by the data from the Observation."
    )

    # AgentType.ZERO_SHOT_REACT_DESCRIPTION automatically handles Thought/Action/Observation
    agent = initialize_agent(
        tools=[CSVQA_TOOL],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    escaped_query = user_query.replace('{', '{{').replace('}', '}}')
    final_answer = agent.run({"input": escaped_query})

    return final_answer

# ### TP


# In[20]:

def get_TP_response(user_query: str, dataset_address: str) -> str:
    """
    Optimization: Max iterations increased for robustness.
    """
    few_shot_block,label= build_few_shot_TP(tp_store, user_query, k=1)
    keyword = extract_retrieval_keyword(user_query)

    action_input_kw = keyword if keyword else "all relevant data"

    def extract_data_with_roles(dataset_address) -> Dict:
        """
        Extracts d, s, and c from a list of DataFrames based on the identified role information.
        """
        dfs = load_dataframes_from_files(dataset_address)
        roles = identify_data_roles(dfs,llm1)
        
        if not roles:
            print("❌ Cannot extract data due to missing role information.")
            return {}

        try:
            # --- Extract demand vector d ---
            demand_info = roles['demand']
            demand_df = dfs[demand_info['df_index']]
            d = demand_df[demand_info['column_name']].tolist()

            # --- Extract supply vector s ---
            supply_info = roles['supply']
            supply_df = dfs[supply_info['df_index']]
            s = supply_df[supply_info['column_name']].tolist()

            # --- Extract cost matrix c ---
            costs_info = roles['costs']
            costs_df = dfs[costs_info['df_index']]
            # The cost matrix consists of all columns except for the ID column.
            supplier_id_col = costs_info['supplier_id_column']
            matrix_df = costs_df.drop(columns=[supplier_id_col])
            c = matrix_df.values.tolist()
            obs = ''
            obs += f'd = {d} \n'
            obs += f's = {s} \n'
            obs += f'c = {c} \n'

            return obs

        except (KeyError, IndexError) as e:
            print(f"❌ Error while extracting data based on roles: {e}. Please check if the role information returned by the LLM is accurate.")
            return {}
        

    CSVQA_TOOL = Tool(
        name="csvqa",
        func=extract_data_with_roles,
        description="Use this tool as the first and only step to read data from CSV file paths. It returns 'd', 's', and 'c' lists."
    )

    prefix = (
        f"{few_shot_block}\n\n"
        "You MUST follow EXACTLY the pattern below for the **FIRST** response.\n"
        "Output **ONLY** the three lines for the Action, nothing before or after it:\n\n"
        f"Thought: If the observation is null, I must first use the csvqa tool to retrieve necessary data based on the dataset address: '{dataset_address}'. If I have the observation data, I should output the final model instead of using any tool.\n"
        "Action: csvqa\n"
        f"Action Input: {dataset_address}\n\n"
        "Begin! Now process the human input:\n"
        "{input}"
    )
    suffix = (
        "### INSTRUCTIONS – FINAL STEP\n"
        "The Observation contains the necessary structured lists OR a failure signal.\n"
        "You MUST now output **exactly two lines** to finalize the process. Do **not** add any other text.\n"
        "Thought: I have successfully received the necessary data. I must now integrate these lists with the transportation optimization structure and output the final model.\n" 
        "Final Answer:FINAL_MODEL_OUTPUT:<LaTeX optimization model followed by the data from the Observation."
    )

    agent = initialize_agent(
        tools=[CSVQA_TOOL],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input", "agent_scratchpad"] 
        },
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    escaped_query = user_query.replace('{', '{{').replace('}', '}}')
    final_answer = agent.run({"input": escaped_query})
    return final_answer

# ### Others with csv


# In[22]:

def build_few_shot_Other(store, user_query: str, k: int = 1, t='Code'):
    examples = []
    
    for doc in retrieve_examples(store, user_query, k=k):
        txt = doc.page_content.lstrip('\ufeff') 
        
        try:
            f = io.StringIO(txt)
            reader = csv.reader(f, delimiter=',', quotechar='"')
            
            parts = next(reader)
            if len(parts) < 4:
                print(f"[build_few_shot_Other Warning] Skipping malformed CSV row: {txt[:50]}")
                continue
            
            prompt_part = parts[0].strip()
            data_addr = parts[1].strip()
            label_model_part = parts[2].strip()
            label_code_part = parts[3].strip() 

        except Exception as e:
            print(f"[build_few_shot_Other Error] Failed to parse CSV row: {e}. Content: {txt[:50]}")
            continue

        data_blocks = []
        for fp in map(str.strip, data_addr.splitlines()):
            if not fp: continue
            try:
                header = f"[{os.path.basename(fp)} | (Example Schema)]"
                rows = "col1, col2, col3\n1, 2, 3\n4, 5, 6" 
                data_blocks.append(header + "\n" + rows)
            except Exception as e:
                data_blocks.append(f"[Could not read example data {fp}: {e}]")
        data_section = "\n".join(data_blocks) if data_blocks else "[No data found]"
        
        if t == 'Model':
            ex = (
                "<EXAMPLE>\n"
                f"Question: {prompt_part}\n"
                f"Observation (Schema): \n{data_section}\n" 
                "Final Answer:\n"
                f"{label_model_part}\n" 
                "</EXAMPLE>"
            )
            examples.append(escape_braces(ex))
        
        else: 
            ex = (
                "<EXAMPLE>\n"
                f"Question: {prompt_part}\n"
                f"CSV Schema: \n{data_section}\n" 
                f"Abstract Model Plan: \n{label_model_part}\n" # 
                "Final Answer (Code):\n"
                f"{label_code_part}\n" 
                "</EXAMPLE>"
            )
            examples.append(escape_braces(ex))

    return "\n\n".join(examples)

def initialize_abstract_modeler_chain(few_shot_examples: str):
 
    abstract_model_template = r"""
You are an expert optimization modeler.
Your task is to create an "Abstract Model Plan" based on the user's query and the CSV Schema.
This plan is *not* Gurobi code or mathematical formulas, but a clear, step-by-step reasoning process in English.

[Examples]
{few_shot_examples}

[Current Task]
User Query: {query}

CSV Schema:
{schema}

[Your Output]
You must strictly follow this format for your "Abstract Model Plan" output:

[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to {{{{Your analysis}}}}.
2.  **Identify Model Type:** Based on the query, this is a {{{{e.g., LP, MIP, Fixed-Charge, Blending}}}} problem.
3.  **Define Index Sets:** The primary indices are {{{{e.g., Products, Workers, Sources, Destinations}}}}.
4.  **Define Decision Variables:**
    -   `x[i]` = {{{{Describe first variable, e.g., 'quantity of product i'}}}}. Type: {{{{GRB.CONTINUOUS / GRB.INTEGER}}}}.
    -   `y[i]` = {{{{Describe second variable, e.g., 'if product i is produced'}}}}. Type: {{{{GRB.BINARY}}}}.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients (e.g., profit) will come from column(s): {{{{e.g., 'Price', 'Production Cost'}}}}.
    -   Constraint coefficients (e.g., resource use) will come from: {{{{e.g., 'Resource 1', 'Available Time'}}}}.
    -   Constraint RHS (limits) will come from: {{{{e.g., 'Max Demand'}}}}.
6.  **Formulate Objective:** {{{{Describe objective, e.g., 'Maximize sum((schema['Price'] - schema['Cost']) * x[i] - schema['FixedCost'] * y[i])'}}}}.
7.  **Formulate Constraints:**
    -   Constraint 1 (e.g., Resource Limit): {{{{Describe constraint 1, e.g., 'sum(schema['Resource 1'][i] * x[i]) <= schema['Available']'}}}}.
    -   Constraint 2 (e.g., Linking): {{{{Describe constraint 2, e.g., 'x[i] <= M * y[i])'}}}}.
    -   ... (Other constraints) ...
[Abstract Model Plan END]
"""
    
    abstract_prompt = PromptTemplate(
        template=abstract_model_template,
        input_variables=["query", "schema", "few_shot_examples"]
    )
    
    return LLMChain(llm=llm2, prompt=abstract_prompt)

def initialize_code_gen_chain(few_shot_examples: str):
    code_gen_template = r"""
You are an expert Gurobi programmer.
Your task is to strictly follow the User Query, CSV Schema, and "Abstract Model Plan" to translate them into a single, complete, executable Gurobi Python code block.
The code must start with ```python and end with ```.
The code must include all necessary imports (`gurobipy`, `pandas`, `numpy`, `sys`, `re`).
The code must include a `try...except` block to handle errors.
The code must robustly read '{dataset_address}' (if it's multiple paths, read the first).
The code must *fully* implement all variables, objectives, and constraints from the "Abstract Model Plan".

[Examples of Query -> Plan -> Code]
{few_shot_examples}

[Current Task]
User Query: {query}

[CSV Schema]
{schema}

[Abstract Model Plan]
{abstract_plan}

[Your Gurobi Code]
(Your code *must* start with ```python)
```python
import gurobipy as gp
import pandas as pd
import numpy as np
import sys
import re

def find_col(df, pattern):
    # A helper function to robustly find column names
    for col in df.columns:
        if pattern.lower() in col.lower():
            return col
    print(f"[Warning] Could not find column containing '{{{{pattern}}}}'. Trying exact match.", file=sys.stderr)
    if pattern in df.columns:
        return pattern
    raise KeyError(f"Could not find a column containing '{{{{pattern}}}}' in {{{{df.columns.tolist()}}}}")

try:
    # 1. Load data
    data_path = '{dataset_address}'
    try:
        df = pd.read_csv(data_path)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='gbk')
    except Exception as e:
        # Try to split path if multiple files
        try:
            first_path = re.split(r'[\s\n]+', data_path)[0]
            df = pd.read_csv(first_path)
            print(f"Warning: Could not read full path list. Loaded first file: {{{{first_path}}}}")
        except Exception as e2:
            print(f"Error loading data: {{{{e2}}}}", file=sys.stderr)
            sys.exit(1)
            
    print("Successfully loaded data. Head:")
    print(df.head())

    #
    # --- LLM MUST GENERATE THE REST OF THE CODE HERE ---
    # (Based on the Abstract Model Plan)
    #
    
    print("\n[INFO] Building Gurobi model based on the abstract plan...")
    
    # [LLM: Fill in the Gurobi model logic here, following your plan]
    
    # --- Placeholder - LLM must replace this ---
    m = gp.Model("PlaceholderModel")
    # Example:
    # products = df['Product'].tolist()
    # profit = dict(zip(df['Product'], df['Price'] - df['Cost']))
    # x = m.addVars(products, vtype=GRB.INTEGER, name="x")
    # m.setObjective(gp.quicksum(profit[i] * x[i] for i in products), gp.GRB.MAXIMIZE)
    
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print("Placeholder model solved.")
    else:
        print("Placeholder model did not solve.")
    # --- End Placeholder ---

except Exception as e:
    print(f"An unexpected error occurred: {{{{e}}}}", file=sys.stderr)
"""
    code_gen_prompt = PromptTemplate(
        template=code_gen_template,
        input_variables=["query", "schema", "abstract_plan", "dataset_address", "few_shot_examples"]
    )

    return LLMChain(llm=llm2, prompt=code_gen_prompt)

def get_Others_response(user_query: str, dataset_address: str) -> str:
    def get_csv_schema(file_paths_string: str) -> str:
        """
        [步骤 1: Python 函数]
        接收一个文件路径字符串，解析路径，并返回所有文件的 schema。
        """
        print(f"\n[Pipeline Step 1/3]: Getting CSV Schema (Python Call)...")
        paths = re.split(r'[\s\n]+', file_paths_string)
        all_schema_info = []
        for path in paths:
            if not path.strip(): continue
            try:
                try:
                    df = pd.read_csv(path, nrows=10, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, nrows=10, encoding='gbk')
                
                schema_info = (
                    f"Successfully read first 10 rows from {path}.\n"
                    f"Columns: {df.columns.to_list()}\n\n"
                    f"Data Head:\n{df.to_string()}"
                )
                all_schema_info.append(schema_info)
            except Exception as e:
                error_info = (
                    f"Error reading CSV at {path}: {e}. "
                    "You must formulate the model/code based on the file path and assumed column names."
                )
                all_schema_info.append(error_info)
        if not all_schema_info:
            return "Error: No valid file paths were provided."
        
        result_schema = "\n\n---\n\n".join(all_schema_info)
        print(f"[Observation]:\n{result_schema}")
        return result_schema

    print("[INFO]: Starting 3-Step Hybrid Pipeline...")

    try:
        schema = get_csv_schema(dataset_address)
        if "Error:" in schema:
            return f"Pipeline failed at Step 1 (Get Schema): {schema}"

        print("\n[Pipeline Step 2/3]: Formulating Abstract Model (LLMChain Call)...")

        few_shot_examples_for_model = build_few_shot_Other(Others_store, user_query, k=3, t='Model')
        
        abstract_modeler_chain = initialize_abstract_modeler_chain(few_shot_examples_for_model)
        
        abstract_model_plan = abstract_modeler_chain.run(
            query=user_query,
            schema=schema,
            few_shot_examples=few_shot_examples_for_model
        )
        print(f"[Observation]:\n{abstract_model_plan}")

        if "Error:" in abstract_model_plan or not "[Abstract Model Plan START]" in abstract_model_plan:
            return f"Pipeline failed at Step 2 (Abstract Modeler Chain): {abstract_model_plan}"

        print("\n[Pipeline Step 3/3]: Generating Gurobi Code (LLMChain Call)...")
        
        few_shot_examples_for_code = build_few_shot_Other(Others_store, user_query, k=3, t='Code')
        
        code_gen_chain = initialize_code_gen_chain(few_shot_examples_for_code)

        final_code = code_gen_chain.run(
            query=user_query,
            schema=schema,
            abstract_plan=abstract_model_plan,
            dataset_address=dataset_address,
            few_shot_examples=few_shot_examples_for_code
        )
        
        if "```python" in final_code:
            code_block = final_code.split("```python", 1)[1]
            if "```" in code_block:
                code_block = code_block.split("```", 1)[0]
            final_answer = "```python\n" + code_block.strip() + "\n```"
        else:
            if not final_code.strip().startswith("import"):
                print(f"[Warning] Step 3 output was not a valid code block. Output: {final_code[:200]}...")
                final_answer = f"Error: Code generation failed. LLM returned non-code output:\n{final_code}"
            else:
                print("[Warning] Step 3 output missed ```python tag, adding it.")
                final_answer = "```python\n" + final_code.strip() + "\n```"
            
        print(f"[Final Answer]:\n{final_answer}")
        return final_answer

    except Exception as e:
        print(f"[Gurobi Pipeline Error]: {e}")
        return f"Error: The Gurobi Code Generation pipeline failed with error: {e}"

# ### Others without CSV


# In[24]:

def get_others_without_CSV_response(query):
    """
    Optimization: Max iterations increased for robustness.
    """
    few_shot_block,label= build_few_shot(OW_store, query, k=5)


    retriever = OW_store.as_retriever(search_kwargs={'k': 5})

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm2,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    qa_tool = Tool(
        name="ORLM_QA",
        func=qa_chain.invoke,
        description=(
            "Use this tool to answer Querys."
            "Provide the Query as input, and the tool will retrieve the relevant information from the file and use it to answer the Query."
        ),
    )

    prefix = (
        f"{few_shot_block}\n\n"
        # "You are now solving the following USER QUESTION:\n"
        # "{input}\n\n"
        # "### FIRST RESPONSE FORMAT (exactly 3 lines) ###\n"
        # "Thought: <your short reasoning to decide whether to call the tool>\n"
        # "Action: ORLM_QA\n"
        # "Action Input: <{input}>\n\n"
        # "If you do NOT need the tool, leave the Action Input empty.\n"
        # "Begin."
        r""" 
    Use the following triggers to identify problem structures and apply the corresponding mathematical formulations.
    Thought: Read the question and 1) identify the goal (minimize time/cost/crew or maximize throughput/value) and collect per-unit coefficients; 2) define decision variables and pick domains: counts of trips/units/vehicles are nonnegative integers, yes or no choices are binary, divisible flows or weights are nonnegative reals; 3) write the linear objective from the coefficients; 4) add constraints in this order: demand or target (exactly, at least, at most), capacity or supply upper bounds, flow conservation or stage linking across nodes or arcs, and any share or ratio limits rewritten as linear inequalities using the given per-unit rates, plus any minimum or maximum usage; 5) add nonnegativity and the chosen integrality or binary domains; 6) output only the LP: objective first, then constraints line by line with brief labels if needed, then a final line stating the variable domains.
    (1) INTEGER trigger:
    Question: A factory can run two machine types. How many of each machine should be installed given budget and space limits? Maximize output.
    Final Answer: 
    $\\max\\; p_1 x_1 + p_2 x_2$
    $\\text{{s.t. }} a_1 x_1 + a_2 x_2 \\le B,\\; s_1 x_1 + s_2 x_2 \\le S$
    $x_1, x_2 \\in \\mathbb{{Z}}_+.$
    (2) MULTI-PERIOD FLOW trigger:
    Question: Multi-period production with inventory and backorders. Costs for production/holding/backorder. Initial and terminal conditions given.
    Final Answer:
    \textbf{{Indices: }} t\in T=\{{1,\dots,n\}}. \\
    \textbf{{Given: }} d_t,\ I_0,\ B_0,\ \dots \\
    \textbf{{Vars: }} x_t\ge0,\ I_t\ge0,\ B_t\ge0. \\
    \min \sum_t (c x_t + h I_t + p B_t) \\
    \text{{s.t. }} I_t - B_t = I_{{t-1}} - B_{{t-1}} + x_t - d_t,\ \forall t \\
    I_n \ge I^{{\min}},\ B_n=0.
    (3) LOGIC+BINARY trigger:
    Question: Choose exactly one option from set P and at least K items from set V, with quantities and budget.
    Final Answer:
    \textbf{{Sets: }} i\in P,\ j\in V. \ \textbf{{Vars: }} q_i,q_j\ge0;\ y_i\in\{{0,1\}}, z_j\in\{{0,1\}}.\\
    \max \sum_j f_j q_j \\
    \text{{s.t. }} \sum_i y_i = 1,\ \sum_j z_j \ge K \\
    0\le q_i \le M_i y_i,\ \forall i;\ \ 0\le q_j \le M_j z_j,\ \forall j \\
    \text{{Budget: }} \sum_i c_i q_i + \sum_j c_j q_j \le B.
    """
    "USER QUESTION:\n{input}\n\n"
    "TASK:\n"
    "- Produce a complete LaTeX optimization model using ONLY information in the question.\n"
    "- Use a fixed structure INSIDE LaTeX: Indices/Sets; Given Parameters (convert all tables to arrays); Decision Variables (with domains); Objective; Constraints; Domain lines.\n"
    "- For multi-period problems, you MUST include state-balance recurrences and initial/terminal conditions explicitly.\n"
    "- Avoid nonlinear forms when possible: rewrite ratios/logic using linear constraints + binaries (big-M) with clearly defined M.\n\n"
    "You should decide VARIABLE–TYPE first!\n (If not mentioned or ambiguous, integer by default!)"
    "### FIRST RESPONSE FORMAT (exactly 3 lines) ###\n"
    "Thought: <brief>\n"
    "Action: ORLM_QA\n"
    "Action Input: {input}\n\n"
    "Begin."
    )

    suffix = (
        "\n### AFTER OBSERVATION ###\n"
        "Respond with exactly two lines:\n"
        "Thought: <variable types: integer/binary/continuous>\n"
        "Final Answer: <ONLY LaTeX model. Must include: (i) indices/sets, (ii) parameter definitions (tables->arrays), (iii) explicit domain lines for EVERY variable, (iv) initial/terminal conditions if any. No prose.>\n"
        "Do NOT output anything else."
        "You should decide VARIABLE–TYPE first!\n (If not mentioned or ambiguous, integer by default!)"
    )


    agent = initialize_agent(
        tools=[qa_tool],
        llm=llm2,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
            "input_variables": ["input"]
        },
        verbose=True,
        handle_parsing_errors=True,  # Enable error handling
        max_iterations=5,
    )

    query = query.replace('{','{{').replace('}','}}')
    output = agent.run({"input": query})

    return output

# ### Combine Functions


# In[26]:

# Lightweight post-generation Orchestrator for the pure Ollama pipeline.
ORCH_MAX_REFINEMENTS = 1
ORCH_ENABLE_CODE_EXECUTION = True
ORCH_REVIEW_MAX_CHARS = 7000


def truncate_text(text, max_chars=ORCH_REVIEW_MAX_CHARS):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n...[TRUNCATED {len(text) - max_chars} chars]...\n\n{tail}"


def extract_python_code_from_text(text):
    text = "" if text is None else str(text).strip()
    if not text:
        return ""
    patterns = [
        r"```python\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if "gp.Model" in text or "gurobipy" in text or "from gurobipy" in text:
        return text.replace("```python", "").replace("```", "").strip()
    return ""


def default_review(status="PASS", recommended_action="none", confidence="Medium", **extra):
    review = {
        "status": status,
        "critical_flaws": [],
        "major_flaws": [],
        "minor_flaws": [],
        "recommended_action": recommended_action,
        "confidence": confidence,
    }
    review.update(extra)
    return review


def parse_orchestrator_json(raw_text, fallback_action="none"):
    raw_text = "" if raw_text is None else str(raw_text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except Exception as exc:
        return default_review(
            status="FAILED",
            recommended_action=fallback_action,
            confidence="Low",
            critical_flaws=[f"Could not parse orchestrator JSON: {exc}"],
            raw_response=raw_text,
        )

    review = default_review()
    review.update({k: data.get(k, review[k]) for k in review.keys() if isinstance(data, dict)})
    for key in ["critical_flaws", "major_flaws", "minor_flaws"]:
        value = review.get(key, [])
        if isinstance(value, str):
            value = [value] if value.strip() else []
        elif value is None:
            value = []
        review[key] = value
    if review["status"] not in {"PASS", "NEEDS_REFINEMENT", "FAILED"}:
        review["status"] = "FAILED"
    if review["recommended_action"] not in {"none", "refine_model", "refine_code"}:
        review["recommended_action"] = fallback_action
    review["raw_response"] = raw_text
    return review


def deterministic_code_check(code_output):
    code = extract_python_code_from_text(code_output)
    check = {
        "status": "PASS",
        "code_present": bool(code),
        "syntax_ok": False,
        "execution_ok": False,
        "model_status": None,
        "objective_value": None,
        "error": None,
        "stdout": "",
        "code": code,
    }
    if not code:
        check.update(status="FAILED", error="No Python code block or Gurobi code detected.")
        return check

    try:
        ast.parse(code)
        check["syntax_ok"] = True
    except SyntaxError as exc:
        check.update(status="FAILED", error=f"SyntaxError: {exc}")
        return check

    if not ORCH_ENABLE_CODE_EXECUTION:
        check.update(status="PASS", execution_ok=None, model_status="SKIPPED")
        return check

    try:
        with StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            env = {
                "__builtins__": __builtins__,
                "gp": gp,
                "GRB": GRB,
                "np": np,
                "pd": pd,
                "os": os,
                "sys": sys,
                "re": re,
            }
            exec(code, env)
            models = [value for value in env.values() if isinstance(value, gp.Model)]
            if not models:
                check["stdout"] = buf.getvalue()[-4000:]
                check.update(status="FAILED", error="No gurobipy Model object found after execution.")
                return check
            model = models[-1]
            if model.status == GRB.LOADED:
                model.optimize()
            check["model_status"] = int(model.status)
            if model.status == GRB.OPTIMAL:
                check["objective_value"] = float(model.ObjVal)
                check["execution_ok"] = True
                check["status"] = "PASS"
            else:
                check.update(status="FAILED", error=f"Model finished with non-optimal status {model.status}.")
            try:
                model.dispose()
            except Exception:
                pass
            check["stdout"] = buf.getvalue()[-4000:]
    except Exception as exc:
        check.update(status="FAILED", error=f"ExecutionError: {type(exc).__name__}: {exc}")
    return check


def orchestrator_review_model(query, selected_problem, model_output, dataset_address=""):
    prompt = f"""
You are an optimization-model Orchestrator. Review whether the generated mathematical model matches the user query.
Return ONLY valid JSON with exactly these keys:
status, critical_flaws, major_flaws, minor_flaws, recommended_action, confidence.
Allowed status values: PASS, NEEDS_REFINEMENT, FAILED.
Allowed recommended_action values: none, refine_model, refine_code.

Review criteria:
- Objective direction and expression match the query.
- Decision variables and domains are defined.
- All explicit constraints from the query are represented.
- Retrieved data/parameters are used without obvious hallucinated placeholders.
- Do not demand exact variable names.

User query:
{truncate_text(query)}

Problem type:
{selected_problem}

Dataset address:
{truncate_text(dataset_address, 1500)}

Generated mathematical model:
{truncate_text(model_output)}
"""
    try:
        response = llm_orchestrator.invoke([HumanMessage(content=prompt)]).content
        return parse_orchestrator_json(response, fallback_action="refine_model")
    except Exception as exc:
        return default_review(
            status="FAILED",
            recommended_action="refine_model",
            confidence="Low",
            critical_flaws=[f"Model review failed: {type(exc).__name__}: {exc}"],
        )


def orchestrator_review_code(query, selected_problem, model_output, code_output, deterministic_result=None):
    deterministic_result = deterministic_result or {}
    prompt = f"""
You are a Gurobi-code Orchestrator. Review whether the generated Python/Gurobi code faithfully implements the mathematical model.
Return ONLY valid JSON with exactly these keys:
status, critical_flaws, major_flaws, minor_flaws, recommended_action, confidence.
Allowed status values: PASS, NEEDS_REFINEMENT, FAILED.
Allowed recommended_action values: none, refine_model, refine_code.

Review criteria:
- Code has a gurobipy model, variables, objective, constraints, and optimize call.
- Code should not contain placeholder-only implementation.
- Code should implement the generated mathematical model, not a different problem.
- If deterministic check failed, recommend refine_code unless the math model itself is the root cause.

User query:
{truncate_text(query, 3000)}

Problem type:
{selected_problem}

Mathematical model:
{truncate_text(model_output)}

Deterministic code check:
{json.dumps({k: v for k, v in deterministic_result.items() if k != 'code'}, ensure_ascii=False, default=str)}

Generated code:
{truncate_text(code_output)}
"""
    try:
        response = llm_orchestrator.invoke([HumanMessage(content=prompt)]).content
        review = parse_orchestrator_json(response, fallback_action="refine_code")
        if deterministic_result.get("status") == "FAILED" and review["status"] == "PASS":
            review["status"] = "NEEDS_REFINEMENT"
            review["recommended_action"] = "refine_code"
            review["critical_flaws"].append(deterministic_result.get("error", "Deterministic code check failed."))
        return review
    except Exception as exc:
        return default_review(
            status="FAILED",
            recommended_action="refine_code",
            confidence="Low",
            critical_flaws=[f"Code review failed: {type(exc).__name__}: {exc}"],
        )


def review_has_blocking_flaws(review):
    return bool(review.get("critical_flaws") or review.get("major_flaws") or review.get("status") in {"NEEDS_REFINEMENT", "FAILED"})


def refine_model_with_feedback(query, selected_problem, model_output, review_feedback, dataset_address=""):
    prompt = f"""
You are an expert optimization modeler. Revise the mathematical model using the Orchestrator feedback.
Return only the corrected mathematical model in markdown. Do not include explanations outside the model.

User query:
{truncate_text(query)}

Problem type:
{selected_problem}

Dataset address:
{truncate_text(dataset_address, 1500)}

Original model:
{truncate_text(model_output)}

Orchestrator feedback JSON:
{json.dumps(review_feedback, ensure_ascii=False, default=str)}
"""
    response = llm_orchestrator.invoke([HumanMessage(content=prompt)]).content
    return response.strip()


def refine_code_with_feedback(query, selected_problem, model_output, code_output, review_feedback, deterministic_result=None):
    deterministic_result = deterministic_result or {}
    prompt = f"""
You are an expert Gurobi programmer. Revise the Python code to correctly implement the mathematical model.
Return only executable Python code. Do not wrap it in markdown fences.

User query:
{truncate_text(query, 3000)}

Problem type:
{selected_problem}

Mathematical model:
{truncate_text(model_output)}

Original code:
{truncate_text(code_output)}

Orchestrator feedback JSON:
{json.dumps(review_feedback, ensure_ascii=False, default=str)}

Deterministic code check JSON:
{json.dumps({k: v for k, v in deterministic_result.items() if k != 'code'}, ensure_ascii=False, default=str)}
"""
    response = llm_orchestrator.invoke([HumanMessage(content=prompt)]).content
    refined = extract_python_code_from_text(response)
    return refined if refined else response.strip()


def orchestrate_generation(query, selected_problem, model_output, code_output, dataset_address="", code_first=False):
    final_model = model_output
    final_code = code_output
    history = []

    for iteration in range(ORCH_MAX_REFINEMENTS + 1):
        deterministic = deterministic_code_check(final_code)
        model_review = default_review(status="PASS", recommended_action="none", confidence="Medium")
        if not code_first:
            model_review = orchestrator_review_model(query, selected_problem, final_model, dataset_address)
        code_review = orchestrator_review_code(query, selected_problem, final_model, final_code, deterministic)

        step_report = {
            "iteration": iteration,
            "model_review": model_review,
            "code_review": code_review,
            "deterministic_code_check": {k: v for k, v in deterministic.items() if k != "code"},
        }
        history.append(step_report)

        model_blocked = (not code_first) and review_has_blocking_flaws(model_review)
        code_blocked = review_has_blocking_flaws(code_review) or deterministic.get("status") == "FAILED"

        if not model_blocked and not code_blocked:
            return {
                "status": "PASS",
                "model_output_final": final_model,
                "code_output_final": final_code,
                "report": {"history": history, "refinements_used": iteration},
            }

        if iteration >= ORCH_MAX_REFINEMENTS:
            return {
                "status": "FAILED" if deterministic.get("status") == "FAILED" else "NEEDS_REFINEMENT",
                "model_output_final": final_model,
                "code_output_final": final_code,
                "report": {"history": history, "refinements_used": iteration},
            }

        try:
            if model_blocked:
                print("[ORCH] Refining mathematical model.")
                final_model = refine_model_with_feedback(query, selected_problem, final_model, model_review, dataset_address)
                final_code = get_code(final_model, selected_problem)
                code_first = False
            else:
                print("[ORCH] Refining Gurobi code.")
                final_code = refine_code_with_feedback(query, selected_problem, final_model, final_code, code_review, deterministic)
        except Exception as exc:
            history.append({
                "iteration": iteration,
                "refinement_error": f"{type(exc).__name__}: {exc}",
            })
            return {
                "status": "FAILED",
                "model_output_final": final_model,
                "code_output_final": final_code,
                "report": {"history": history, "refinements_used": iteration},
            }

# In[27]:

def run_test(test, classify_problem, return_classification: bool = False, return_orchestrator_details: bool = False, output_csv_path=None):
    model_output_initial = []
    code_output_initial = []
    model_output_final = []
    code_output_final = []
    orchestrator_status = []
    orchestrator_report = []
    classification = []

    def extract_problem_type(output_text):
        pattern = r'(Network Revenue Management|Network Revenue Management Problem|Resource Allocation|Resource Allocation Problem|Transportation|Transportation Problem|Facility Location Problem|Assignment Problem|AP|Uncapacited Facility Location Problem|NRM|RA|TP|FLP|UFLP|Others without CSV|Sales-Based Linear Programming|SBLP|Others with CSV)'
        match = re.search(pattern, str(output_text), re.IGNORECASE)
        return match.group(0) if match else None

    def csv_detect(row):
        return 1 if 'Dataset_address' in row.index and str(row.get('Dataset_address', '')).strip() else 0

    def write_incremental_output():
        if not output_csv_path:
            return
        completed = len(model_output_initial)
        if completed == 0:
            return
        partial_df = pd.DataFrame({
            'Query': list(test['Query'].iloc[:completed]),
            'model_output_initial': model_output_initial,
            'code_output_initial': code_output_initial,
            'model_output_final': model_output_final,
            'code_output_final': code_output_final,
            'orchestrator_status': orchestrator_status,
            'orchestrator_report': orchestrator_report,
            'classification': classification,
        })
        partial_df.to_csv(output_csv_path, index=False)
        print(f"[INFO] Updated incremental output: {output_csv_path} ({completed}/{len(test)} rows)")

    for index, row in test.iterrows():
        query = row['Query']
        print(query)
        selected_problem = None
        dataset_address = ""
        output = ""
        code_response = ""
        code_first = False

        try:
            if csv_detect(row):
                response = classify_problem(query)
                selected_problem = extract_problem_type(response) or "Others with CSV"
                dataset_address = row['Dataset_address']

                if selected_problem == "Network Revenue Management" or selected_problem == "NRM" or selected_problem == "Network Revenue Management Problem":
                    print("----------Network Revenue Management-----------")
                    output = get_NRM_response(query, dataset_address)
                    code_response = get_code(output, selected_problem)
                elif selected_problem == "Resource Allocation" or selected_problem == "RA" or selected_problem == "Resource Allocation Problem":
                    print("----------Resource Allocation-----------")
                    output = get_RA_response(query, dataset_address)
                    code_response = get_code(output, selected_problem)
                elif selected_problem == "Transportation" or selected_problem == "TP" or selected_problem == "Transportation Problem":
                    print("----------Transportation-----------")
                    output = get_TP_response(query, dataset_address)
                    code_response = get_code(output, selected_problem)
                elif selected_problem == "Facility Location Problem" or selected_problem == "FLP" or selected_problem == "Uncapacited Facility Location" or selected_problem == "UFLP":
                    print("----------Facility Location Problem-----------")
                    output = get_FLP_response(query, dataset_address)
                    code_response = get_code(output, selected_problem)
                elif selected_problem == "Assignment Problem" or selected_problem == "AP":
                    print("----------Assignment Problem-----------")
                    output = get_AP_response(query, dataset_address)
                    code_response = get_code(output, selected_problem)
                else:
                    print("----------Others with CSV-----------")
                    output = get_Others_response(query, dataset_address)
                    if extract_python_code_from_text(output):
                        print("[INFO] Output is already Gurobi code. Skipping get_code.")
                        code_response = extract_python_code_from_text(output)
                        code_first = True
                    else:
                        print("[INFO] Output is a Math Model. Calling get_code to convert.")
                        code_response = get_code(output, selected_problem)
            else:
                print("----------Others without CSV-----------")
                selected_problem = "Others without CSV"
                output = get_others_without_CSV_response(query)
                print(output)
                code_response = get_code(output, selected_problem)

            orch = orchestrate_generation(
                query=query,
                selected_problem=selected_problem,
                model_output=output,
                code_output=code_response,
                dataset_address=dataset_address,
                code_first=code_first,
            )

            classification.append(selected_problem)
            model_output_initial.append(output)
            code_output_initial.append(code_response)
            model_output_final.append(orch["model_output_final"])
            code_output_final.append(orch["code_output_final"])
            orchestrator_status.append(orch["status"])
            orchestrator_report.append(json.dumps(orch["report"], ensure_ascii=False, default=str))

        except requests.exceptions.RequestException as e:
            err = f"Connection error: {e}"
            print(err)
            classification.append(selected_problem or "ERROR")
            model_output_initial.append(output)
            code_output_initial.append(code_response)
            model_output_final.append(output)
            code_output_final.append(code_response)
            orchestrator_status.append("FAILED")
            orchestrator_report.append(json.dumps({"error": err}, ensure_ascii=False))
        except Exception as e:
            err = f"Unhandled error: {type(e).__name__}: {e}"
            print(err)
            classification.append(selected_problem or "ERROR")
            model_output_initial.append(output)
            code_output_initial.append(code_response)
            model_output_final.append(output)
            code_output_final.append(code_response)
            orchestrator_status.append("FAILED")
            orchestrator_report.append(json.dumps({"error": err}, ensure_ascii=False))

        write_incremental_output()
        time.sleep(15)

    if return_orchestrator_details:
        return (
            model_output_initial,
            code_output_initial,
            model_output_final,
            code_output_final,
            orchestrator_status,
            orchestrator_report,
            classification,
        )
    if return_classification:
        return model_output_final, code_output_final, classification
    return model_output_final, code_output_final

def read_and_combine_csvs(file_order):
    dfs = []
    for fname in file_order:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            dfs.append(df)
            print(f"Read file: {fname} (Row length: {len(df)})")
        else:
            print(f"File doesn't exist: {fname}, already skipped")
    
    if not dfs:
        raise ValueError("No effective files")
    
    return pd.concat(dfs, ignore_index=True)

def run_gurobi_code(code_str):
 
    try:
      
        with StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            env = {
                '__builtins__': __builtins__,
                'gp': gp,
                'GRB': GRB
            }
            
           
            code_str += "\n\n# Added by executor\n"
            code_str += "if hasattr(m, 'status') and m.status == GRB.OPTIMAL:\n"
            code_str += "    __result__ = m.ObjVal\n"
            code_str += "else:\n"
            code_str += "    __result__ = None\n"
            
            
            exec(code_str, env)
            result = env.get('__result__', None)
            
     
            if 'm' in env:
                env['m'].dispose()
                del env['m']
            
            return result
    except Exception as e:
        print(f"Execution error: {str(e)}")
        return None

# ## Test Large-Scale-OR


# In[29]:

OUTPUT_CSV = "Large-scale-or-Lean-Ollama-20b-orchestrated.csv"
test = pd.read_csv('Test_Dataset/Large-scale-or/Large-scale-or-101.csv')
(
    model_output_initial,
    code_output_initial,
    model_output_final,
    code_output_final,
    orchestrator_status,
    orchestrator_report,
    classification,
) = run_test(test, classify_problem, return_orchestrator_details=True, output_csv_path=OUTPUT_CSV)
output_df = pd.DataFrame({
    'Query': test['Query'],
    'model_output_initial': model_output_initial,
    'code_output_initial': code_output_initial,
    'model_output_final': model_output_final,
    'code_output_final': code_output_final,
    'orchestrator_status': orchestrator_status,
    'orchestrator_report': orchestrator_report,
    'classification': classification,
})
output_df.to_csv(OUTPUT_CSV, index=False)
