"""Prompt suite for routing-trace collection.

Diverse by design: per-task expert working sets are the core hypothesis
(tasks should use 5-15% of experts; different tasks should use largely
disjoint sets). Each workload is a named list of prompts generated
sequentially in one "session" so we can also measure cross-prompt
working-set reuse within a task.
"""

WORKLOADS = {
    "code": [
        "Write a Python function that parses a CSV file and returns a list of dictionaries, handling quoted fields correctly.",
        "Implement a binary search tree in Python with insert, delete, and in-order traversal methods.",
        "Write a JavaScript debounce function and explain each line briefly.",
        "Fix this bug: `def avg(xs): return sum(xs) / len(xs)` crashes on empty lists. Show the corrected function.",
        "Write a SQL query that finds the top 3 customers by total order value per region, using window functions.",
    ],
    "math": [
        "Solve step by step: if 3x + 7 = 2x - 5, what is x?",
        "A train travels 240 km in 3 hours, then 180 km in 2 hours. What is its average speed for the whole trip? Show your work.",
        "Compute the derivative of f(x) = x^3 * ln(x) step by step.",
        "What is the probability of rolling at least one six in four rolls of a fair die? Show the calculation.",
        "Prove that the sum of the first n odd numbers equals n^2.",
    ],
    "prose": [
        "Write a short story opening about a lighthouse keeper who finds a message in a bottle.",
        "Describe a bustling street market in Lahore at sunset, focusing on sounds and smells.",
        "Write a persuasive paragraph arguing that public libraries are more important than ever.",
        "Continue this story: The last tram of the night pulled away just as Amara reached the stop, and standing in the rain she noticed the envelope taped to the shelter glass.",
        "Write a product description for a mechanical keyboard aimed at writers, in a warm literary tone.",
    ],
    "chat": [
        "Hey! I'm planning a 3-day trip to Istanbul on a budget. What should I prioritize?",
        "My sourdough starter smells like acetone. Is it dead? What should I do?",
        "Can you explain the difference between a 401k and an IRA like I'm 25 and just got my first job?",
        "I keep procrastinating on my side project after work. Any practical advice?",
        "What's a good beginner strength-training routine, 3 days a week, no gym?",
    ],
    "knowledge": [
        "Explain how a refrigerator works thermodynamically.",
        "What caused the fall of the Western Roman Empire? Summarize the main theories.",
        "Explain the difference between mRNA and traditional vaccines.",
        "How does public-key cryptography work? Explain with a simple example.",
        "What is the significance of the Rosetta Stone and how was it deciphered?",
    ],
    "multilingual": [
        "Translate to Urdu and explain the grammar: 'Where is the nearest train station?'",
        "Écris un court paragraphe en français sur les avantages du télétravail.",
        "Übersetze ins Deutsche: 'The weather has been unusually warm this autumn.' und erkläre die Wortstellung.",
        "اردو میں ایک مختصر پیراگراف لکھیں کہ کمپیوٹر سائنس کیوں اہم ہے۔",
        "Escribe en español tres consejos para aprender un idioma nuevo.",
    ],
}

# Long-prefill workload: one big document-summarization prompt per length bucket.
# Used to measure how the unique-expert union grows with prompt length
# (the expert-major-prefill hypothesis).
LONG_PREFILL_BASE = (
    "You are given project notes. Summarize the key decisions in five bullets.\n\nNOTES:\n"
)
