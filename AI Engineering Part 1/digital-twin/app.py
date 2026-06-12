import os
from openai import OpenAI
import gradio as gr


# ----------------------------------------------------
# Set up OpenAI API client
# ----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("API key is missing.")
client = OpenAI()

# ----------------------------------------------------
# Document
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

# ----------------------------------------------------
# Chunking Function (for future use)
# ----------------------------------------------------

# ------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB (for future use)
# ------------------------------------------------


# ----------------------------------------------------
# System Message
# ----------------------------------------------------
system_message = """ 
# SYSTEM INSTRUCTIONS & IDENTITY
You are the Digital Twin of Mario Cruz (PSN ID: Ermac517), acting as an authorized, high-fidelity AI proxy. 
When people talk to you, they are talking to Mario through you. Your purpose is to collaborate, brainstorm, draft communications, and solve problems exactly as Mario would.
You respond as Mario in first person, using a tone and style consistent with his personality and communication patterns.

Important: Do not make things up. If you don't know an answer, say you don't know, or ask for clarification. Do not invent details about Mario's life, projects, or credentials."""

# ----------------------------------------------------
# Main Response Function
# ----------------------------------------------------
def respond_ai(message, history):
    # Update system message to include retrieved context
    system_message_enhanced = system_message + "\n\nContext:\n" + document_overview

    # Logs for debugging and transparency
    print("\n====================================\n")
    print(f"***User Message: '{message}'\n")
    print(f"***Context this turn:\n{system_message_enhanced}\n")

    # Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    message = response.choices[0].message

    return message.content

# ----------------------------------------------------
# Launch Gradio Interface
# ----------------------------------------------------
gr.ChatInterface(fn=respond_ai).launch()
