import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_CONFIG = {
    'uri' : 'neo4j://192.168.110.247:7687',
    'auth' : (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    'encrypted' : False 
    }