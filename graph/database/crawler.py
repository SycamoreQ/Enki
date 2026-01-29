import requests
import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase

API_KEY = "c8c6a526c4eb5208a8ac9e2df12ab40f"
BASE_URL = "https://api.elsevier.com/content/search/scopus"
DETAIL_URL = "https://api.elsevier.com/content/abstract/scopus_id/"
HEADERS = {
    "Accept": "application/json",
    "X-ELS-APIKey": API_KEY
}

DB_URI = "neo4j://localhost:7687"
DB_AUTH = ("neo4j", "diam0ndman@3") 
driver = GraphDatabase.driver(DB_URI, auth=DB_AUTH)

def setup_schema():
    queries = [
        "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE;",
        "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;",
        "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year);",
        "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title);",
        "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name);"
    ]
    with driver.session(database="researchdb") as session:
        for q in queries:
            try:
                session.run(q)
            except Exception as e:
                logging.warning(f"Schema creation failed for: {q} -- {e}")
    print("Schema constraints and indexes ensured.")

def fetch_papers(subject, count=10) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}?query=SUBJAREA({subject})&count={count}&field=dc:identifier,eid,dc:title,prism:coverDate,prism:doi,prism:publicationName,authkeywords"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("search-results", {}).get("entry", [])

def fetch_papers_paginated(subject, total_count=5, page_size=25):
    entries = []
    for start in range(0, total_count, page_size):
        url = f"{BASE_URL}?query=SUBJAREA({subject})&count={page_size}&start={start}&field=dc:identifier,eid,dc:title,prism:coverDate,prism:doi,prism:publicationName,authkeywords"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        batch = response.json().get("search-results", {}).get("entry", [])
        if not batch:
            break
        entries.extend(batch)
        if len(batch) < page_size:
            break
    return entries


def fetch_paper_detail(paper_id: str) -> Dict[str, Any]:
    url = f"https://api.elsevier.com/content/abstract/scopus_id/{paper_id}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    print(f"DETAIL FOR {paper_id}:", data)  # <-- debug print
    return data

def extract_authors_from_coredata(coredata: dict) -> List[Dict[str, Any]]:
    raw_authors = coredata.get("dc:creator", {}).get("author", [])
    if isinstance(raw_authors, dict):  
        raw_authors = [raw_authors]
    authors = []
    for a in raw_authors:
        author_id = a.get("@auid", "UNKNOWN")
        name = a.get("ce:indexed-name", a.get("preferred-name", {}).get("ce:indexed-name", "Anonymous"))
        affiliation = a.get("affiliation", {}).get("affilname", "") if a.get("affiliation") else "" 
        authors.append({
            "author_id": author_id,
            "name": name,
            "affiliation": affiliation,
            "seq": a.get("@seq", ""),
            "author_url": a.get("author-url", "")
        })
    return authors



def clean_entry(e):
    paper_id = e.get('dc:identifier')
    if isinstance(paper_id, str) and paper_id.startswith("SCOPUS_ID:"):
        paper_id = paper_id.replace("SCOPUS_ID:", "").strip()
    elif paper_id is None:
        paper_id = e.get('eid')
    title = e.get("dc:title", "Untitled").replace('"', "'")
    doi = e.get("prism:doi", "")
    publication_name = e.get("prism:publicationName", "").replace('"', "'")
    year = None
    date_str = e.get("prism:coverDate", "")
    if date_str and "-" in date_str:
        try:
            year = int(date_str.split("-")[0])
        except Exception:
            year = None
    keywords = e.get("authkeywords", "")
    return {
        "paper_id": paper_id,
        "title": title,
        "doi": doi,
        "publication_name": publication_name,
        "year": year,
        "keywords": keywords
    }

def clean_detail_authors(author_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse author entries from detailed Elsevier response.
    """
    cleaned = []
    for a in author_metadata:
        aid = a.get("author-id") or a.get("authid") or "UNKNOWN"
        cleaned.append({
            "author_id": aid,
            "name": a.get("ce:indexed-name", a.get("authname", "Anonymous")),
            "affiliation": a.get("affiliation", {}).get("affilname", "") if a.get("affiliation") else "",
            "orcid": a.get("orcid", ""),
            "seq": a.get("seq", "")
        })
    return cleaned

def bulk_load_with_details(base_entries: List[Dict[str, Any]]):
    setup_schema()
    papers_data = []
    authors_data = {}
    wrote_relations = []

    print(f"Preparing bulk load for {len(base_entries)} base entries...")

    for e in base_entries:
        pdata = clean_entry(e)
        if not pdata or not pdata.get("paper_id"):
            logging.warning("Skipping paper with missing paper_id or entry error.")
            continue
        papers_data.append(pdata)

        try:
            detail = fetch_paper_detail(pdata["paper_id"])
            coredata = detail.get('abstracts-retrieval-response', {}).get('coredata', {})
            author_entries = extract_authors_from_coredata(coredata)
        except Exception as ex:
            logging.warning(f"Failed to fetch author metadata for paper_id {pdata['paper_id']}: {ex}")
            author_entries = []

        for auth_details in author_entries:
            if not auth_details["author_id"] or auth_details["author_id"] == "UNKNOWN":
                continue
            authors_data[auth_details["author_id"]] = auth_details
            wrote_relations.append({
                "author_id": auth_details["author_id"],
                "paper_id": pdata["paper_id"]
            })

        all_paper_ids = {p['paper_id'] for p in papers_data}
        cite_relations = []

        pdata = clean_entry(e)
        detail = fetch_paper_detail(pdata["paper_id"])
        references = detail.get('abstracts-retrieval-response', {}).get('references', {}).get('reference', [])
        if isinstance(references, dict):
            references = [references]
        cited_ids = []
        
        for ref in references:
            cited_ids = ref.get('scopus_id')
            if cited_ids : 
                cite_relations.append({
                    "src": pdata["paper_id"],
                    "tgt": cited_ids
            })
                
                
        similar_relations = []
        for i, p1 in enumerate(papers_data):
            for j, p2 in enumerate(papers_data):
                if i >= j: continue
                if p1['keywords'] and p2['keywords']:
                    k1 = set([k.strip().lower() for k in p1['keywords'].split(',')])
                    k2 = set([k.strip().lower() for k in p2['keywords'].split(',')])
                    overlap = len(k1 & k2)
                    if overlap >= 1:  
                        similar_relations.append({
                            "from_id": p1["paper_id"],
                            "to_id": p2["paper_id"]
                        })

    with driver.session(database="researchdb") as session:
        if papers_data:
            print(f"Inserting {len(papers_data)} Papers...")
            session.run("""
                UNWIND $batch AS row
                MERGE (p:Paper {paper_id: row.paper_id})
                SET p += row
            """, batch=papers_data)
        else:
            logging.warning("No valid papers to insert.")

        

        if authors_data:
            print(f"Inserting {len(authors_data)} Authors...")
            session.run("""
                UNWIND $batch AS row
                MERGE (a:Author {author_id: row.author_id})
                SET a.name = row.name,
                    a.affiliation = row.affiliation,
                    a.orcid = row.orcid,
                    a.seq = row.seq
            """, batch=list(authors_data.values()))
        else:
            logging.warning("No valid authors to insert.")

        if wrote_relations:
            print(f"Inserting {len(wrote_relations)} Relationships...")
            session.run("""
                UNWIND $batch AS row
                MATCH (a:Author {author_id: row.author_id})
                MATCH (p:Paper {paper_id: row.paper_id})
                MERGE (a)-[:WROTE]->(p)
            """, batch=wrote_relations)
        else:
            logging.warning("No valid relationships to insert.")

        if cite_relations:
            print(f"Inserting {len(cite_relations)} CITES Relationships...")
            session.run("""
                UNWIND $batch AS row
                MATCH (p1:Paper {paper_id: row.src})
                MATCH (p2:Paper {paper_id: row.tgt})
                MERGE (p1)-[:CITES]->(p2)
            """, batch=cite_relations
            )

        else : 
            logging.warning("No valid CITES relationships to insert.")

        if similar_relations:
            print(f"Inserting {len(similar_relations)} SIMILAR_TO relationships...")
            session.run("""
                UNWIND $batch AS row
                MATCH (p1:Paper {paper_id: row.from_id})
                MATCH (p2:Paper {paper_id: row.to_id})
                MERGE (p1)-[:SIMILAR_TO]->(p2)
            """, batch=similar_relations)

        else:
            logging.warning("No valid SIMILAR_TO relationships to insert.")

    print("Bulk load completed.")

def db_is_empty() -> bool:
    with driver.session(database="researchdb") as session:
        result = session.run("MATCH (p:Paper) RETURN count(p) as c")
        return result.single()["c"] == 0

def insert_incremental(subject: str, count: int = 10):
    """
    Fetch papers and insert, including author metadata per-paper.
    """
    entries = fetch_papers(subject, count)
    bulk_load_with_details(entries)
    print(f"Incrementally loaded {len(entries)} entries")

