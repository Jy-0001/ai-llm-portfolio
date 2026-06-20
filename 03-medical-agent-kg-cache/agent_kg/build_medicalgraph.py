"""Build the medical knowledge graph in Neo4j from the structured dataset."""
import os
import json
import logging

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

NEO4J_CONFIG = {
    'uri': 'neo4j://localhost:7687',
    'auth': (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    'encrypted': False,
}

driver = GraphDatabase.driver(**NEO4J_CONFIG)

class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical.json')
        self.driver = GraphDatabase.driver( **NEO4J_CONFIG)
    # 读取文件
    def read_nodes(self):
        drugs = []        
        foods = []        
        diseases = []        
        symptoms = []        

        rels_recommandeat = []
        rels_recommanddrug = []
        rels_symptom = []
        count = 0
        for data in open(self.data_path):
            disease_dict = {}
            count += 1
            if count % 10 == 0:
                logger.info("Parsed %d records", count)

            data_json = json.loads(data)
            disease = data_json['name']
            diseases.append(disease)

            if 'symptom' in data_json:
                symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    rels_symptom.append([disease, symptom])
            
            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels_recommanddrug.append([disease, drug])
            
            if 'recommand_eat' in data_json:
                recommand_eat = data_json['recommand_eat']
                
                for _recommand in recommand_eat:
                    rels_recommandeat.append([disease, _recommand])
                foods += recommand_eat

        return set(drugs), set(foods), set(symptoms), set(diseases), rels_recommandeat, rels_recommanddrug, rels_symptom
    
    # 创建知识图谱疾病相关的节点, 疾病, 症状, 药品, ⻝品
    def create_graphnodes_and_graphrels(self):
        Drugs, Foods, Symptoms, Diseases, rels_recommandeat, rels_recommanddrug, rels_symptom = self.read_nodes()

        logger.info(
            "Nodes - Drugs: %d, Foods: %d, Symptoms: %d, Diseases: %d",
            len(Drugs), len(Foods), len(Symptoms), len(Diseases),
        )
        logger.info(
            "Relations - eat: %d, drug: %d, symptom: %d",
            len(rels_recommandeat), len(rels_recommanddrug), len(rels_symptom),
        )

        driver = GraphDatabase.driver(**NEO4J_CONFIG)

        with driver.session() as session:
            logger.info("Creating Disease nodes")
            for d in Diseases:
                cypher = 'merge (a:Disease{name:%r}) return a' % d
                session.run(cypher)

            logger.info("Creating Drug nodes")
            for n in Drugs:
                cypher = 'merge (a:Drug{name:%r}) return a' % n
                session.run(cypher)

            logger.info("Creating Food nodes")
            for n in Foods:
                cypher = 'merge (a:Food{name:%r}) return a' % n
                session.run(cypher)

            logger.info("Creating Symptom nodes")
            for n in Symptoms:
                cypher = 'merge (a:Symptom{name:%r}) return a' % n
                session.run(cypher)

        # 创建实体关系边
        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '推荐药品')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        num_edges = len(set(set_edges))
        logger.info("Creating %d %s relationships", num_edges, rel_name)

        driver = GraphDatabase.driver(**NEO4J_CONFIG)

        with driver.session() as session:
            for edge in set(set_edges):
                edge = edge.split('###')
                p = edge[0]
                q = edge[1]
                cypher = "match(p:%s), (q:%s) where p.name = '%s'and q.name='%s' merge(p)-[rel:%s{name:'%s'}]->(q)" % (start_node, end_node, p, q, rel_type, rel_name)

                try:
                    session.run(cypher)
                except Exception as e:
                    logger.error("Failed to create relationship: %s", e)
        return


if __name__ == '__main__':
    mg = MedicalGraph()
    logger.info("Building knowledge graph nodes and relationships")
    mg.create_graphnodes_and_graphrels()
        


            


