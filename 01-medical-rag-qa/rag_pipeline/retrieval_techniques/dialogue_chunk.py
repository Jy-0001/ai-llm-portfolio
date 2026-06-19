import os

dialogue = [
    "Alice: Hi, I'm having trouble with my order.",
    "Bot: I can help with that. What's your order number?",
    "Alice: It's 12345.",
    "Alice: I haven't received any shipping updates.",
    "Bot: Let me check... It seems your order was shipped yesterday.",
    "Alice: Oh, great! Thank you.",
]


def chunk_dialogue(dialogue_lines, max_turns_per_chunk=3):
    chunks = []
    for i in range(0, len(dialogue_lines), max_turns_per_chunk):
        chunk = "\n".join(dialogue_lines[i: i + max_turns_per_chunk])
        chunks.append(chunk)
    return chunks


chunks = chunk_dialogue(dialogue)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} ---")
    print(chunk)
