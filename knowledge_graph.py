from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD
import pandas as pd
import urllib.parse


# ── Namespaces ───────────────────────────────────────────────────
MR   = Namespace("http://movierec.org/")
SCHEMA = Namespace("http://schema.org/")


class MovieKnowledgeGraph:
    def __init__(self):
        self.g = Graph()
        self.g.bind("mr", MR)
        self.g.bind("schema", SCHEMA)

    def build(self, movies):
        print("Building RDF knowledge graph...")

        for _, row in movies.iterrows():
            movie_id  = row['movieId']
            title     = row['title']
            genres    = row['genres']

            movie_uri = MR[f"movie/{movie_id}"]

            # Type and title
            self.g.add((movie_uri, RDF.type,       SCHEMA.Movie))
            self.g.add((movie_uri, SCHEMA.name,    Literal(title, datatype=XSD.string)))
            self.g.add((movie_uri, MR.movieId,     Literal(movie_id, datatype=XSD.integer)))

            # Genres
            if genres and genres != "(no genres listed)":
                for genre in genres.split("|"):
                    genre_uri = MR[f"genre/{urllib.parse.quote(genre)}"]
                    self.g.add((genre_uri, RDF.type,    MR.Genre))
                    self.g.add((genre_uri, RDFS.label,  Literal(genre, datatype=XSD.string)))
                    self.g.add((movie_uri, MR.hasGenre, genre_uri))

        print(f"KG built: {len(self.g)} triples across {len(movies)} movies")

    def query_by_genre(self, genre, limit=10):
        genre_uri = MR[f"genre/{urllib.parse.quote(genre)}"]
        sparql = f"""
            PREFIX mr:     <http://movierec.org/>
            PREFIX schema: <http://schema.org/>

            SELECT ?movieId ?title WHERE {{
                ?movie rdf:type        schema:Movie ;
                       mr:hasGenre     <{genre_uri}> ;
                       schema:name     ?title ;
                       mr:movieId      ?movieId .
            }}
            LIMIT {limit}
        """
        results = self.g.query(sparql)
        return [(int(row.movieId), str(row.title)) for row in results]

    def query_movies_by_multiple_genres(self, genres, limit=10):
        genre_filters = "\n".join(
            f"?movie mr:hasGenre <{MR[f'genre/{urllib.parse.quote(g)}']}>  ."
            for g in genres
        )
        sparql = f"""
            PREFIX mr:     <http://movierec.org/>
            PREFIX schema: <http://schema.org/>

            SELECT ?movieId ?title WHERE {{
                {genre_filters}
                ?movie schema:name  ?title ;
                       mr:movieId   ?movieId .
            }}
            LIMIT {limit}
        """
        results = self.g.query(sparql)
        return [(int(row.movieId), str(row.title)) for row in results]

    def get_genres(self):
        sparql = """
            PREFIX mr:    <http://movierec.org/>
            PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?label WHERE {
                ?genre rdf:type   mr:Genre ;
                       rdfs:label ?label .
            }
            ORDER BY ?label
        """
        results = self.g.query(sparql)
        return [str(row.label) for row in results]