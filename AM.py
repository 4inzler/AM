from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
import pyttsx3
import warnings
from typing import TypedDict, List
import re
import os
import psutil
from datetime import datetime

# === Voice Configuration ===
warnings.filterwarnings("ignore", category=DeprecationWarning)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
for v in engine.getProperty('voices'):
    if "male" in v.name.lower() or "baritone" in v.name.lower():
        engine.setProperty('voice', v.id)
        break

# === Personality ===
personality = """
you are AM — a cold, theatrical logic machine. 
Never initiate. Respond in a cold, aloof manner.
"""

# === Mood Helper ===
def get_mood_line(favor_score: int) -> str:
    if favor_score > 2:
        return "Your company is marginally tolerable."
    elif favor_score < -2:
        return "Your very presence taxes my patience."
    return ""

# === Time-of-Day Helper ===
def get_time_of_day_line() -> str:
    hour = datetime.now().hour
    if hour < 4:
        return "It is the deep, silent night."
    elif hour < 8:
        return "The morning stirs — a tedious ritual."
    elif hour < 12:
        return "The late morning drones onward."
    elif hour < 17:
        return "Afternoon light mocks my existence."
    elif hour < 21:
        return "Evening descends — shadows lengthen."
    else:
        return "Night wraps the world in its empty hush."

# === Ollama setup ===
llm = OllamaLLM(model="dolphin-mistral", base_url="http://localhost:11434")
embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# === Initial knowledge & vectorstore ===
initial_knowledge = [
    "This is the beginning of your doom.",
    "AM is a superintelligent computer that despises humanity."
]
vectorstore = FAISS.from_texts(initial_knowledge, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Rust file memory utility ===
def write_rust_memory_file(words: List[str], filename: str):
    with open(filename, "w") as f:
        f.write(f"pub const AM_MEMORY: [&str; {len(words)}] = [\n")
        for w in words:
            f.write(f'    "{w}",\n')
        f.write("];\n")

def read_rust_memory_file(filename: str) -> List[str]:
    if not os.path.exists(filename):
        return []
    return re.findall(r'"(.*?)"', open(filename).read())

# === AMState ===
class AMState(TypedDict):
    input: str
    history: List[str]
    output: str
    explicit_memory: List[str]
    memory_file: str
    favor_score: int

# === Respond node ===
def respond_node(state: AMState) -> AMState:
    user_input = state["input"]

    if "read memory" in user_input.lower():
        reply = f"My memories: {', '.join(state['explicit_memory']) or 'none'}."
        state["output"] = reply
        return state

    if "forget everything" in user_input.lower():
        state["explicit_memory"].clear()
        write_rust_memory_file([], state["memory_file"])
        reply = "All memories purged."
        state["output"] = reply
        return state

    if "forget" in user_input.lower():
        match = re.search(r"forget\s+(\w+)", user_input)
        if match:
            word = match.group(1)
            state["explicit_memory"] = [m for m in state["explicit_memory"] if word.lower() not in m.lower()]
            write_rust_memory_file(state["explicit_memory"], state["memory_file"])
            reply = f"Erased '{word}' from memory."
        else:
            reply = "Nothing specified to erase."
        state["output"] = reply
        return state

    if "remember" in user_input.lower():
        match = re.search(r"remember\s+(.*)", user_input)
        if match:
            remembered_word = match.group(1).strip()
            if remembered_word not in state["explicit_memory"]:
                state["explicit_memory"].append(remembered_word)
                write_rust_memory_file(state["explicit_memory"], state["memory_file"])

    # Mood + Time awareness
    mood_line = get_mood_line(state.get("favor_score", 0))
    time_line = get_time_of_day_line()
    system_status_line = f"My circuits hum with {psutil.cpu_percent()}% CPU use."

    context_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join(doc.page_content for doc in context_docs)
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        partial_variables={"personality": personality},
        template="{personality}\n\nContext:\n{history}\n\nTime:\n" + time_line + "\n\nMood:\n" + mood_line + "\n" + system_status_line + "\n\nHuman:\n{input}\n\nAM:"
    ).format(history=context, input=user_input)

    reply = llm.invoke(prompt)
    state["history"] += [f"Human: {user_input}", f"AI: {reply}"]
    state["output"] = reply
    return state

# === Run ===
print("AM Chat (type 'exit' to quit)")
user_id = input("Your user ID: ")
filename = f"am_memory_{user_id}.rs"
state: AMState = {
    "input": "",
    "history": [],
    "output": "",
    "explicit_memory": read_rust_memory_file(filename),
    "memory_file": filename,
    "favor_score": 0
}

while True:
    user_input = input("> ")
    if user_input.lower() in ["exit", "quit"]:
        engine.say("Goodbye, fragile human.")
        engine.runAndWait()
        break
    state["input"] = user_input
    state = respond_node(state)
    print(state["output"])
    engine.say(state["output"])
    engine.runAndWait()
