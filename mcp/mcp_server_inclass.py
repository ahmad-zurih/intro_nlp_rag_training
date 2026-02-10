import os
from mcp.server.fastmcp import FastMCP

# 1. Create an MCP server instance
mcp = FastMCP("MCP server example]")

# 2. Define a tool using the @mcp.tool() decorator. What else is there beside @mcp.tool?
#@mcp.tool()
#def read_file(file_path: str) -> str:
#   pass


@mcp.tool()
def calculator(num1: float, num2: float, operator: str) -> float:
    """
    A simple calculator function that performs basic arithmetic operations.

    Parameters
    ----------
    num1 : float
        The first number in the calculation.
    num2 : float
        The second number in the calculation.
    operator : str
        The arithmetic operation to perform. 
        Supported values are:
        - "+" : addition
        - "-" : subtraction
        - "*" : multiplication
        - "/" : division

    Returns
    -------
    float
        The result of the calculation.

    Raises
    ------
    ValueError
        If the operator is not supported or if division by zero is attempted.
    """
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    elif operator == "/":
        if num2 == 0:
            raise ValueError("Division by zero is not allowed.")
        return num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}. Use one of '+', '-', '*', '/'.")
        

# 3. Main entry point to run the server
if __name__ == "__main__":
    print("--- MCP Tool Server starting over stdio... ---")
    # This runs the server, communicating over standard input/output
    # It will wait for a client to connect.
    mcp.run(transport="stdio")