# -*- coding: utf-8 -*-
"""Neo4j 连接配置：走环境变量（见仓库根 .env.example），不再硬编码口令。"""
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_CONFIG = {
    'uri': os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    'auth': (os.getenv("NEO4J_USER", "neo4j"),
             os.getenv("NEO4J_PASSWORD", "neo4j")),
    'encrypted': False,
}

uri = NEO4J_CONFIG['uri']
auth = NEO4J_CONFIG['auth']
user = NEO4J_CONFIG['auth'][0]
password = NEO4J_CONFIG['auth'][1]
