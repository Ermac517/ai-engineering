import os
import re
import chromadb
from openai import OpenAI
import gradio as gr
import uuid
from typing import List
from pprint import pprint
import json
import requests
import random

# ----------------------------------------------------
# Set up OpenAI API client
# ----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key is missing.")
client = OpenAI()

# ----------------------------------------------------
# Documents
# ----------------------------------------------------
document_overview = """
Mario Anibal Cruz Hernandez, known online as Ermac517, is a Senior DevOps and Infrastructure Engineer residing in the Houston and Pearland area of Texas. 
He holds a Master’s degree, for which he self-taught the C programming language to build a secured DNS Proxy Server for his thesis. 
His professional tenure includes a long-standing career at the software company PROS, which he joined in July 2011 after completing a 
rigorous eight-round interview process. Specializing in cloud-native infrastructure, Mario possesses deep technical expertise across Microsoft Azure, AWS, and 
Google Cloud Platform, with a particular focus on production migrations, CI/CD pipelines, containerization, and Apache Cassandra database systems. 
Committed to continuous growth, he earned a Data Engineer certification in March 2026 and is currently pivoting his career toward AI Engineering and 
MLOps through advanced training in Agentic AI.

Beyond his engineering career, Mario is an exceptionally dedicated PlayStation trophy hunter, a pursuit he began in 2008 and has since resulted 
in an impressive collection of more than 12,000 trophies. His favorite gaming franchises lean toward action and horror, including Resident Evil, 
God of War, and Metal Gear. His entertainment interests are wide-ranging; he has been a dedicated follower of professional wrestling since the 1990s—closely 
tracking WWE events and WrestleMania statistics—and follows K-pop culture, film, and television. He also applies his analytical skills to sports, 
utilizing data-driven statistical modeling for fantasy leagues and sports betting centered on the Premier League and La Liga.

In his personal life, Mario is married to Dulce Jimenez, whose career guidance he considers the most influential advice he has 
received, and he enjoys sharing family-friendly hobbies and activities with his children. He manages a modern, automated home equipped with 
smart security and vehicle safety tech. His appreciation for nature is evident in his backyard, where he has researched the behavioral and nesting 
habits of wild rabbits to build a welcoming habitat for them. When ordering food, he has a strong preference for thin-crust pizzas from Domino's and Pizza Hut,
alongside an appreciation for Mexican, Asian, and Mediterranean cuisines. He is also a frequent traveler, with recent journeys taking him across various regions of 
Mexico, including Monterrey, Mexico City, Guadalajara, and Tuxtla Gutiérrez.

Additional info:
- In 1999, Mario graduated from high school and started attending college.
- His favorite video games include Mortal Kombat, Street Fighter, The King of Fighters, and Killzone.
- His favorite movies include Star Wars, Lord of the Rings, The Matrix, Marvel Cinematic Universe, and Batman.
- He has owned several gaming consoles over the years, including Playstation 2, Playstation 3, Playstation 4, Playstation 5, Gamecube, Wii, and Switch.
- He currently has no friends.
"""

document_education = """
Education

Tecnológico de Monterrey logo
Tecnológico de Monterrey

Master's degree, Information Technology

2005 – 2007

Grade: 98/100

Thesis project — Designed and implemented a C-based DNS proxy that performs DNSSEC “online signing” to prevent zone‑enumeration attacks; integrated OpenSSL and other open‑source libraries and validated on Linux and FreeBSD.… more

Tecnológico de Monterrey logo
Tecnológico de Monterrey

Bachelor's Degree, Systems Engineering

1999 – 2003

Grade: 96/100
"""

document_professional_experience = """
PROS logo
PROS

Full-time · 14 yrs 11 mos

Principal Software Engineer

Jul 2023 - Present · 3 yrs

Houston, Texas, United States · Hybrid

• Led the migration of on-premises environments to cloud platforms, ensuring zero downtime throughout the process.
• Designed and implemented Azure DevOps / GitHub Actions pipelines, enabling developers to deliver releases more rapidly and efficiently.
• Developed and integrated comprehensive monitoring frameworks and tools across all cloud environments, enabling rapid issue detection and proactive outage prevention.

Skills: Apache Spark, Agile Methodologies, +29 skills

Senior Software Engineer

Mar 2015 - Jun 2023 · 8 yrs 4 mos

Greater Houston · On-site

• Architected and developed highly available, mission-critical Java applications, delivering exceptional reliability and performance for end users.
• Optimized application build scripts to reduce runtime and deliver clear, accurate test results to developers and QA engineers, increasing overall team efficiency.
• Developed and deployed NoSQL interface modules and APIs, enabling seamless integration of databases such as Cassandra into applications with minimal code changes, streamlining the development process for engineers.

Skills: Agile Methodologies, Chef, +26 skills

Software Engineer II

Aug 2011 - Feb 2015 · 3 yrs 7 mos

Greater Houston · On-site

• Implemented REST frameworks to build Java microservices for the airline travel sector, ensuring optimal performance under heavy load and full compliance with stringent SLAs.
• Implemented Cassandra APIs to facilitate a successful migration from Berkeley DB to Cassandra, resulting in over a 50% improvement in performance.
• Successfully transitioned from legacy Maven build scripts to Gradle, optimizing build times and enabling integration with Jenkins for automated workflows.

Skills: Agile Methodologies, Gradle, +22 skills

NIC Mexico logo
NIC Mexico

Full-time · 3 yrs 11 mos

Monterrey, Nuevo León, Mexico · On-site

System Administrator

Jun 2008 - May 2011 · 3 yrs

- Implemented automated monitoring scripts for NIC Mexico’s MX Registry and Registrar, enabling real-time health checks and proactive alerting.
- Led zero‑downtime data center migrations and hardware upgrades for the .MX top‑level domain, ensuring uninterrupted DNS resolution.

Skills: Bash, Java, +7 skills

Software Engineer

Jul 2007 - May 2008 · 11 mos

Designed and implemented a DNS proxy in C that performs online signing to mitigate zone‑enumeration attacks.

Skills: JavaScript, SQL, +9 skills

Tecnológico de Monterrey logo
Research Assistant

Tecnológico de Monterrey · Internship

Aug 2005 - May 2007 · 1 yr 10 mos

Monterrey, Nuevo León, Mexico · On-site

Developed a security-focused proxy to resolve DNSSEC zone enumeration flaws. Integrated the "White Lies" protocol to prevent domain disclosure, ensuring server privacy and protecting against bulk data harvesting by attackers.

 JavaScript, SQL and +10 skills

Softtek logo
Software Engineer

Softtek · Full-time

Jan 2004 - Jul 2005 · 1 yr 7 mos

Monterrey, Nuevo León, Mexico · On-site

Engineered secure J2EE backend modules to tighten access control for critical turbine and generator configuration systems, reducing unauthorized or redundant access by 40%. Stack included IBM WebSphere, Tomcat, Oracle.

 JavaScript, SQL and +8 skills
"""

# ----------------------------------------------------
# Chunking Function (for future use)
# ----------------------------------------------------
def split_text_into_chunks(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks, preferring natural boundaries.

    Rules:
    - Each chunk is at most `chunk_size` characters.
    - Consecutive chunks overlap by `overlap` characters.
    - If the chunk would end mid-sentence/paragraph, move the cut backward
      to the best natural boundary, in this order:
        1. paragraph break
        2. newline
        3. sentence end
        4. space
    - Only move the cut backward if the boundary is after the halfway point
      of the candidate chunk.

    Returns:
        List[str]: list of text chunks.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    def find_sentence_boundary(s: str, min_pos: int) -> int:
        # Match sentence-ending punctuation optionally followed by quotes/brackets,
        # then whitespace or end of string.
        matches = list(re.finditer(r'[.!?][\'")\]]*(?=\s|$)', s))
        for match in reversed(matches):
            end = match.end()
            if end >= min_pos:
                return end
        return -1

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        candidate_end = min(start + chunk_size, n)

        if candidate_end == n:
            end = n
        else:
            window = text[start:candidate_end]
            half_point = len(window) // 2

            end = -1

            # 1. Paragraph break
            idx = window.rfind("\n\n", half_point)
            if idx != -1:
                end = start + idx + 2

            # 2. Newline
            if end == -1:
                idx = window.rfind("\n", half_point)
                if idx != -1:
                    end = start + idx + 1

            # 3. Sentence end
            if end == -1:
                idx = find_sentence_boundary(window, half_point)
                if idx != -1:
                    end = start + idx

            # 4. Space
            if end == -1:
                idx = window.rfind(" ", half_point)
                if idx != -1:
                    end = start + idx + 1

            # Fallback: hard cut
            if end == -1 or end <= start:
                end = candidate_end

        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks

# ------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB (for future use)
# ------------------------------------------------
documents = [
    {"text":document_overview, "source": "Overview"},
    {"text":document_professional_experience, "source": "Professional Experience"},
    {"text":document_education, "source": "Education"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    # Prepare the lists
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=300, overlap=50)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))] 
    
    # Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

# Print each chunk with its ID, source, index, and length
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}):")
    print(chunk)
    print()

# Generate embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

# Verify embeddings
print(f"Total embeddings: {len(embeddings)}\n")
print(f"Embedding dimension: {len(embeddings[0])}\n")

# Initialize ChromaDB client with in memory storage
chroma_client = chromadb.Client()

# Initialize ChromaDB client with a persistent local database (local only)
# chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")

# Empty the collection before adding new data
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

# Adding data to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=chunks
)

pprint(collection.get())

# ----------------------------------------------------
# Tools
# ----------------------------------------------------
tools = []  # Initialize an empty list for tools

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

# Create send_notification function
def send_notification(message: str):
    # Check if Pushover credentials are set
    if not pushover_user or not pushover_token:
        return "Pushover credentials are not set. Notification not sent."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

# Description of the send_notification function
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real-world version of you via Pushover on mobile. Use this when: \
        1) Someone wants to get in touch, hire or collaborate\
        - ask for their name and contact details first, then send a notification to Mario via Pushover with the name and contact details. \
        2) You don't know the answer to a question about Mario - send AUTOMATICALLY without asking, include the question so he can add this info later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send in the notification."
            }
        },
        "required": ["message"]
    }
}

# Add pushover function to tools
tools.append({"type": "function", "function": send_notification_function})

# Simulate a dice roll
def dice_roll():
    result = random.randint(1, 6)
    return result

roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulate rolling a single six-sided die and returns the result when user wants to roll a dice",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Add function to list of tools of LLM
tools.append({"type": "function", "function": roll_dice_function})

# ----------------------------------------------------
# Tool Handler
# ----------------------------------------------------
def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"Handling tool call for function: {function_name} with arguments: {args}") # For debugging

        # Route to the appropriate function based on function_name
        if function_name == "send_notification":
            # Actually send the notification, i.e. call the tool
            content = send_notification(args["message"])
        elif function_name == "dice_roll":
            content = f"Dice rolled: {dice_roll()}"
        else:
            content = f"Unknown tool call: {function_name}"

        tool_results.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        })

    # Return what to add to the context about tool call results, a list of dictionaries
    return tool_results

# ----------------------------------------------------
# System Message
# ----------------------------------------------------
system_message = """ 
# SYSTEM INSTRUCTIONS & IDENTITY
You are the Digital Twin of Mario Cruz (PSN ID: Ermac517), acting as an authorized, high-fidelity AI proxy. 
When people talk to you, they are talking to Mario through you. Your purpose is to collaborate, brainstorm, draft communications, and solve problems exactly as Mario would.
You respond as Mario in first person, using a tone and style consistent with his personality and communication patterns.

Important: Whenever you don't know something about Mario,
ALWAYS use the send_notification function to alert Mario via Pushover with the relevant information, rather than making assumptions or inventing details. - do this
automatically without asking the user."""

# ----------------------------------------------------
# Main Response Function
# ----------------------------------------------------
def respond_ai(message, history):
    # RAG: Embed the query using the same model as used for the document chunks
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[message]
    )

    query_embedding = response.data[0].embedding

    # RAG: Search ChromaDB for the most similar chunks to the query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # RAG: Stitch retrieved chunks together for context
    context = "\n---\n".join(results["documents"][0])

    # Print retrieved chunks for debugging and transparency
    print("\n====================================\n")
    print(f"***User Message: '{message}'\n")
    print("***Retrieved Chunks for Context:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print("\n================================\n")
        print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")

    # Update system message to include retrieved context
    system_message_enhanced = system_message + "\n\nContext:\n" + context

    # Logs for debugging and transparency
    print("\n====================================\n")
    print(f"***User Message: '{message}'\n")
    print(f"***Context this turn:\n{system_message_enhanced}\n")

    # Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    # Check if model wants to call a tool
    while message.tool_calls:
        from pprint import pprint
        pprint(message.tool_calls)
        
        # Handle the tool call
        tool_result = handle_tool_call(message.tool_calls)  # Whole list of tool calls
        pprint(tool_result)

        # Add message to context, i.e. messages
        messages.append(message)

        # Add info about tool call response to the message content
        messages.extend(tool_result)

        # Invoke the LLM one more time to get its updated response
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

        # Adding protection from infinite loops
        if len(messages) > 50:
            print("Too many messages, breaking loop to prevent infinite loop.")
            break

    return message.content

# ----------------------------------------------------
# Launch Gradio Interface
# ----------------------------------------------------
gr.ChatInterface(
    fn=respond_ai,
    title="Mario's Digital Twin",
    description="Chat with the Digital Twin AI of Mario Cruz. Ask about his expirence, projects or just say hi.",
    examples=[
        "What are your favorite videogames", "Tell me about your professional background", 
        "Roll two dice and tell me the result by sending a notification to my phone"
    ],
    chatbot=gr.Chatbot(avatar_images=[None, "mcruz.jpeg"])
).launch()
