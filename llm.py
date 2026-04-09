import requests
import re
import sys


OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a SPARQL query generator for a movie knowledge graph.

The graph uses these namespaces and properties:
- PREFIX mr:     <http://movierec.org/>
- PREFIX schema: <http://schema.org/>
- PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
- PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

Available triples:
- ?movie rdf:type schema:Movie
- ?movie schema:name ?title         (plain string literal)
- ?movie mr:movieId ?movieId        (integer)
- ?movie mr:hasGenre ?genre
- ?genre rdf:type mr:Genre
- ?genre rdfs:label ?label          (plain string literal)

Available genres: Action, Adventure, Animation, Children, Comedy, Crime, 
Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, 
Romance, Sci-Fi, Thriller, War, Western

Rules:
1. Always SELECT ?movieId ?title
2. Always place LIMIT 10 AFTER the closing brace, never inside WHERE
3. Filter genres using: ?genre rdfs:label ?label . FILTER(?label = "GenreName")
4. Never use FILTER(LANG(...))
5. Never use FILTER on a URI directly
6. Output ONLY the SPARQL query, no explanation, no markdown, no backticks
7. For multiple genres, use separate REQUIRED hasGenre joins for each genre. NEVER use OPTIONAL.
8. Never put a period inside a FILTER clause

Example — single genre:
SELECT ?movieId ?title
WHERE {
  ?movie rdf:type schema:Movie ;
         schema:name ?title ;
         mr:movieId ?movieId ;
         mr:hasGenre ?genre .
  ?genre rdfs:label ?label .
  FILTER(?label = "Thriller")
}
LIMIT 10

Example — multiple genres (Sci-Fi AND Action):
SELECT ?movieId ?title
WHERE {
  ?movie rdf:type schema:Movie ;
         schema:name ?title ;
         mr:movieId ?movieId ;
         mr:hasGenre ?g1 ;
         mr:hasGenre ?g2 .
  ?g1 rdfs:label ?l1 .
  ?g2 rdfs:label ?l2 .
  FILTER(?l1 = "Sci-Fi")
  FILTER(?l2 = "Action")
}
LIMIT 10
"""

def generate_sparql(user_input, retries=2):
    prompt = f"{SYSTEM_PROMPT}\n\nUser request: {user_input}\n\nSPARQL query:"

    for attempt in range(retries + 1):
        try:
            print("  [LLM] Generating SPARQL", end="", flush=True)

            response = requests.post(OLLAMA_URL, json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.1}
            }, timeout=60, stream=True)

            raw = ""
            for line in response.iter_lines():
                if line:
                    chunk = re.search(r'"response"\s*:\s*"(.*?)"', line.decode("utf-8"))
                    if chunk:
                        raw += chunk.group(1).replace("\\n", "\n").replace('\\"', '"')
                        print(".", end="", flush=True)

            print()  # newline after dots
            query = clean_query(raw)
            query = fix_common_errors(query)

            if validate_query(query):
                return query, None
            else:
                if attempt < retries:
                    print(f"  [LLM] Invalid query on attempt {attempt+1}, retrying...")
                    continue
                return None, f"Generated invalid SPARQL after {retries+1} attempts"

        except requests.exceptions.ConnectionError:
            return None, "Ollama is not running. Start it from your menu bar."
        except Exception as e:
            return None, f"LLM error: {str(e)}"

    return None, "Failed to generate valid SPARQL"

def clean_query(raw):
    raw = re.sub(r"```sparql|```", "", raw).strip()
    # Fix escaped quotes that Mistral sometimes outputs
    raw = raw.replace('\\"', '"').replace("\\\"", '"')
    # Fix backslash-quoted strings like \Thriller\ -> "Thriller"
    raw = re.sub(r'\\([^\\]+)\\', r'"\1"', raw)
    if "SELECT" in raw:
        raw = raw[raw.index("SELECT"):]
    return raw.strip()

def fix_common_errors(query):
    # Ensure SELECT DISTINCT to avoid cross-product duplicates
    query = re.sub(r'SELECT\s+(?!DISTINCT)', 'SELECT DISTINCT ', query)

    # Fix unclosed FILTER parentheses - count and balance them
    def fix_parens(q):
        lines = q.split("\n")
        fixed = []
        for line in lines:
            open_count = line.count("(")
            close_count = line.count(")")
            if open_count > close_count:
                line = line.rstrip() + ")" * (open_count - close_count)
            fixed.append(line)
        return "\n".join(fixed)

    query = fix_parens(query)

    # Fix stray periods inside FILTER clauses: FILTER(?x = "Y" .) -> FILTER(?x = "Y")
    query = re.sub(r'(\bFILTER\s*\([^)]*?)\s*\.\s*\)', r'\1)', query)

    # Fix stray periods before closing braces
    query = re.sub(r'\.\s*\}', '\n}', query)

    # Remove LANG filters
    query = re.sub(r'\s*FILTER\s*\(\s*LANG\s*\([^)]*\)[^)]*\)\s*\.?', '', query)

    # Replace OPTIONAL genre blocks with required joins
    query = re.sub(r'OPTIONAL\s*\{([^}]*mr:hasGenre[^}]*)\}', r'\1', query)

    # Fix LIMIT placement
    limit_match = re.search(r'(LIMIT\s+\d+)', query)
    if limit_match:
        limit_clause = limit_match.group(1)
        query = re.sub(r'\s*LIMIT\s+\d+', '', query).strip()
        query = query + f"\n{limit_clause}" if query.endswith("}") else query + "\nLIMIT 10"
    else:
        if query.endswith("}"):
            query += "\nLIMIT 10"

    return query.strip()

def validate_query(query):
    """Validate SPARQL structure and attempt rdflib parse."""
    if not query:
        return False
    required = ["SELECT", "WHERE", "?movieId", "?title"]
    if not all(term in query for term in required):
        return False

    # Try parsing with rdflib to catch syntax errors before execution
    try:
        from rdflib.plugins.sparql import prepareQuery
        prefixes = """
            PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX mr:     <http://movierec.org/>
            PREFIX schema: <http://schema.org/>
        """
        full_query = prefixes + query if "PREFIX" not in query else query
        prepareQuery(full_query)
        return True
    except Exception as e:
        print(f"  [Validation] SPARQL parse failed: {e}")
        return False