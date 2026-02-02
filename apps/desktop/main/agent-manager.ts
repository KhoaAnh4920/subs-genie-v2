import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import path from "path";
import { app } from "electron";

export class AgentManager {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;

  async start() {
    console.log("[AgentManager] Starting Transcriber Agent...");

    const projectRoot = path.resolve(__dirname, "../../..");
    const pythonPath = path.join(
      projectRoot,
      "agents/transcriber/.venv/bin/python",
    ); // macOS/Linux path
    const scriptPath = path.join(projectRoot, "agents/transcriber/main.py");

    console.log(`[AgentManager] Python: ${pythonPath}`);
    console.log(`[AgentManager] Script: ${scriptPath}`);

    const fs = require("fs");
    if (!fs.existsSync(pythonPath)) {
      console.error(
        `[AgentManager] ❌ Critical Error: Python not found at ${pythonPath}`,
      );
      console.error(
        `[AgentManager] Run "uv sync" in agents/transcriber/ to create the venv.`,
      );
    }
    if (!fs.existsSync(scriptPath)) {
      console.error(
        `[AgentManager] ❌ Critical Error: Script not found at ${scriptPath}`,
      );
    }

    this.transport = new StdioClientTransport({
      command: pythonPath,
      args: [scriptPath],
    });

    this.client = new Client(
      {
        name: "electron-host",
        version: "1.0.0",
      },
      {
        capabilities: {},
      },
    );

    try {
      await this.client.connect(this.transport);
      console.log("[AgentManager] Connected to Transcriber Agent!");
    } catch (error) {
      console.error("[AgentManager] Connection failed:", error);
    }
  }

  async pingAgent() {
    if (!this.client) return "Client not initialized";

    console.log("[AgentManager] Sending 'ping' tool call...");
    try {
      const result = await this.client.callTool({
        name: "ping",
        arguments: {},
      });

      const text = (result.content[0] as any).text;
      console.log("[AgentManager] 📩 Received:", text);
      return text;
    } catch (error) {
      console.error("[AgentManager] Ping failed:", error);
      throw error;
    }
  }

  async stop() {
    if (this.transport) {
      await this.transport.close();
      console.log("[AgentManager] Transport closed.");
    }
  }
}

export const agentManager = new AgentManager();
