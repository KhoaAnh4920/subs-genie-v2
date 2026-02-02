from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("transcriber")

@mcp.tool()
def ping() -> str:
    """Ping the agent to check connectivity."""
    return "pong"

if __name__ == "__main__":
    mcp.run()
