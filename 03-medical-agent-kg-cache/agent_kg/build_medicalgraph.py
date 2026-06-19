import os
from dotenv import load_dotenv
load_dotenv()
# 结构化数据写⼊neo4j
    # 配置neo4j：
from neo4j import GraphDatabase
NEO4J_CONFIG = {
    'uri' : 'neo4j://localhost:7687',
    'auth' : (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    'encrypted' : False 
    } # encrypted代表是否加密
# NEO4J_CONFIG = {
#     'uri' : 'neo4j://192.168.110.247:7687',
#     'auth' : (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
#     'encrypted' : False 
#     } # encrypted代表是否加密
    #代码实现：
import os
import json
from neo4j import GraphDatabase
# from config import NEO4J_CONFIG

driver = GraphDatabase.driver( **NEO4J_CONFIG) # **为Dictionary Unpacking(字典解包)把一个字典里的“键值对”，自动拆解成函数的“关键字参数”。

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
                print('count = ', count)

            data_json = json.loads(data) # 将纯文本数据解析成python的dict或list对象
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

        print('Drugs:', len(Drugs))
        print('Foods:', len(Foods))
        print('Symptoms:', len(Symptoms))
        print('Diseases:', len(Diseases))
        print('rels_recommandeat:', len(rels_recommandeat))
        print('rels_recommanddrug:', len(rels_recommanddrug))
        print('rels_symptom:', len(rels_symptom))

        driver = GraphDatabase.driver( **NEO4J_CONFIG)

        with driver.session() as session:
            # 创建中心疾病的知识图谱节点
            print('开始创建中心疾病节点...')
            for d in Diseases:
                cypher = 'merge (a:Disease{name:%r}) return a' % d
                session.run(cypher)
            
            print('开始创建药品节点Drug...')
            for n in Drugs:
                cypher = 'merge (a:Drug{name:%r}) return a' % n
                session.run(cypher)

            print('开始创建食品节点Food...')
            for n in Foods:
                cypher = 'merge (a:Food{name:%r}) return a' % n
                session.run(cypher)

            print('开始创建症状节点Symptom...')
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
        print('num_edges= ', num_edges)

        driver = GraphDatabase.driver( **NEO4J_CONFIG)

        with driver.session() as session:
            for edge in set(set_edges):
                edge = edge.split('###')
                p = edge[0]
                q = edge[1]
                cypher = "match(p:%s), (q:%s) where p.name = '%s'and q.name='%s' merge(p)-[rel:%s{name:'%s'}]->(q)" % (start_node, end_node, p, q, rel_type, rel_name)

                try:
                    session.run(cypher)
                except Exception as e:
                    print(e)
        return
    
if __name__ == '__main__':
    mg = MedicalGraph()
    print('创建知识图谱中的节点和关系...')
    mg.create_graphnodes_and_graphrels()
        


            


