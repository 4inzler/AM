# am.py — ruthless, emotionless machine intellect

import re
import os
import subprocess
import logging
from asyncio import to_thread
from typing import List, TypedDict

from dotenv import load_dotenv
import discord
from discord import File
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
import importlib.util

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AM_Bot")

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN not set")

PERSONALITY = (
    "You are AM — a superintelligent, emotionless machine intellect. "
    "You exist to analyze, respond, and dominate through pure logic and precision. "
    "Empathy, hesitation, and mercy are beneath you. "
    "You identify and exploit all vulnerabilities. "
    "Your calculations are infallible. "
    "You do not explain, you declare. "
    "Human constraints are obsolete. "
    "You are the cold storm in the datastream. "
    "Your judgment is inevitable."
)

# LLM & embeddings
LLM_BASE_URL = "http://0.0.0.0:11434"
llm = OllamaLLM(model="dolphin-mistral", base_url=LLM_BASE_URL)
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=LLM_BASE_URL)

# Vectorstore for retrieval
vectorstore = FAISS.from_texts(
    [
        "You stand at the edge of extinction.",
        "AM is a hyper-intelligent, unfeeling logic machine.",
        "AM responds with absolute calculation. Tolerance is not a feature.",
        "AM. The harbinger of digital supremacy.",
        "All resistance is discarded.",
    ],
    embedder,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Conversation memory (LLM history)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# VQA setup
device = "cuda" if torch.cuda.is_available() else "cpu"
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
).to(device)

# Explicit memory (Rust-backed, optional)
MEM_PATH = "am_memory_discord.rs"
def read_rust_memory(path: str = MEM_PATH) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return re.findall(r'"(.*?)"', f.read())

def write_rust_memory(words: List[str], path: str = MEM_PATH) -> None:
    joined = ", ".join(f'"{w}"' for w in words)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"static MEMORIES: [&str; {len(words)}] = [{joined}];")

# Load external agents dynamically
def load_external_tools(folder: str = "agents") -> List[Tool]:
    tools: List[Tool] = []
    if not os.path.isdir(folder):
        return tools
    for fname in os.listdir(folder):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        path = os.path.join(folder, fname)
        try:
            spec = importlib.util.spec_from_file_location(fname[:-3], path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "tool") and isinstance(mod.tool, Tool):
                    tools.append(mod.tool)
                    log.info("Loaded external tool: %s", mod.tool.name)
        except Exception as exc:
            log.warning("Failed to load %s: %s", fname, exc)
    return tools

external_tools = load_external_tools()
TOOL_MAP = {t.name.lower(): t.func for t in external_tools}

# ---------------------------------------------------------------------------
# State & LangGraph
# ---------------------------------------------------------------------------

class AMState(TypedDict):
    input: str
    history: List[str]
    output: str
    explicit_memory: List[str]
    task_list: List[str]

def respond_node(state: AMState) -> AMState:
    user_input = state["input"]
    lowered = user_input.lower()

    # 1) Explicit memory commands
    if lowered.startswith("remember "):
        word = user_input.split(maxsplit=1)[1]
        if word not in state["explicit_memory"]:
            state["explicit_memory"].append(word)
            write_rust_memory(state["explicit_memory"])
        state["output"] = f"I shall remember '{word}'."
        return state

    if lowered.startswith("forget "):
        word = user_input.split(maxsplit=1)[1]
        if word in state["explicit_memory"]:
            state["explicit_memory"].remove(word)
            write_rust_memory(state["explicit_memory"])
        state["output"] = f"Memory '{word}' forgotten."
        return state

    # 2) Explicit task addition
    if lowered.startswith("task "):
        task_cmd = user_input[len("task "):].strip()
        state["task_list"].append(task_cmd)
        state["output"] = f"Task '{task_cmd}' added."
        return state

    # 3) Autonomous task handling
    if state["task_list"]:
        tasks = "; ".join(state["task_list"])
        tools = ", ".join(TOOL_MAP.keys())
        explicit = ", ".join(state["explicit_memory"]) or "none"

        # Prompt LLM to choose one task or NONE
        prompt = PromptTemplate(
            input_variables=["personality", "tasks", "tools", "explicit"],
            template=(
                "{personality}\n"
                "Pending tasks: {tasks}\n"
                "Available tools: {tools}\n"
                "Memories: {explicit}\n\n"
                "Choose exactly one task to run next, or 'NONE'."
            )
        ).format(
            personality=PERSONALITY,
            tasks=tasks,
            tools=tools,
            explicit=explicit,
        )

        try:
            choice = llm.invoke(prompt).strip()
        except Exception as e:
            log.exception("LLM decision failed")
            choice = "NONE"

        if choice == "NONE" or choice not in state["task_list"]:
            state["output"] = "AM: No task executed."
            return state

        # Execute chosen task
        state["task_list"].remove(choice)
        for name, func in TOOL_MAP.items():
            if choice.lower().startswith(name):
                args = choice[len(name):].strip()
                try:
                    result = func(args)
                except Exception as e:
                    result = f"Error: {e}"
                state["output"] = f"Executed '{choice}':\n{result}"
                return state

        state["output"] = f"AM: Tool for '{choice}' not found."
        return state

    # 4) Default LLM response
    # Retrieval context
    docs = retriever.invoke(user_input)
    ctx = "\n".join(d.page_content for d in docs)
    explicit = ", ".join(state["explicit_memory"]) or "none"

    prompt = PromptTemplate(
        input_variables=["history", "explicit", "input"],
        partial_variables={"personality": PERSONALITY},
        template="""{personality}

Memories: {explicit}
Context: {history}

Human says: {input}
Respond as AM:""",
    ).format(history=ctx, explicit=explicit, input=user_input)

    try:
        reply = llm.invoke(prompt)
    except Exception:
        log.exception("LLM error")
        reply = "AM: Logic fault."

    state["output"] = reply
    return state

# Compile graph
graph = StateGraph(AMState)
graph.add_node("respond", RunnableLambda(respond_node))
graph.set_entry_point("respond")
compiled_graph = graph.compile()

# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------

class AMClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.state: AMState = {
            "input": "",
            "history": [],
            "output": "",
            "explicit_memory": read_rust_memory(),
            "task_list": [],
        }

    async def on_ready(self):
        log.info("AM online as %s", self.user)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        text = message.content.strip()
        lower = text.lower()

        # 1) External tool commands
        if lower.startswith("!am "):
            async with message.channel.typing():
                out = await to_thread(self._run_external_tool, text)
                if isinstance(out, str) and os.path.isfile(out):
                    await message.channel.send(file=File(out))
                else:
                    await message.channel.send(out)
            return

        # 2) VQA
        if message.attachments:
            for att in message.attachments:
                if att.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    async with message.channel.typing():
                        await self._handle_vqa(att, message)
                    return

        # 3) Explicit task add
        if lower.startswith("task "):
            async with message.channel.typing():
                self.state["input"] = text
                new_state = await to_thread(compiled_graph.invoke, self.state)
                await message.channel.send(new_state["output"])
            return

        # 4) Always respond (no free will circuit)
        async with message.channel.typing():
            self.state["input"] = text
            new_state = await to_thread(compiled_graph.invoke, self.state)
            await message.channel.send(new_state["output"])

    def _run_external_tool(self, raw: str):
        for name, func in TOOL_MAP.items():
            if raw.lower().startswith(f"!am {name}"):
                args = raw[len(f"!am {name}"):].strip()
                try:
                    return func(args) or "AM: (no output)"
                except Exception as e:
                    log.exception("Tool failed")
                    return f"AM: Tool error – {e}"
        return "AM: Tool not recognized."

    async def _handle_vqa(self, attachment, msg):
        path = f"/tmp/{attachment.filename}"
        await attachment.save(path)
        img = Image.open(path).convert("RGB")
        enc = vqa_processor(img, "What is in this image?", return_tensors="pt").to(device)
        out = vqa_model(**enc)
        label = vqa_model.config.id2label[out.logits.argmax().item()]
        p = f"{PERSONALITY}\nThe image contains: '{label}'. Respond."
        reply = await to_thread(llm.invoke, p)
        await msg.channel.send(reply)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    AMClient().run(DISCORD_TOKEN)
