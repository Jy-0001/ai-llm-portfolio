import subprocess
NEO4J_CONFIG = {
    'uri' : 'neo4j://localhost:7687',
    'auth' : ('neo4j', 'neo4jneo4j'),
    'encrypted' : False 
    }
# NEO4J_CONFIG = {
#     'uri' : 'neo4j://192.168.110.247:7687',
#     'auth' : ('neo4j', 'neo4jneo4j'),
#     'encrypted' : False 
#     }
uri = NEO4J_CONFIG['uri']
auth = NEO4J_CONFIG['auth']
user = NEO4J_CONFIG['auth'][0]
password = NEO4J_CONFIG['auth'][1]


