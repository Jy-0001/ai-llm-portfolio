import os
from dotenv import load_dotenv
load_dotenv()
# 结构化数据写⼊neo4j
    # 配置neo4j：
from neo4j import GraphDatabase
NEO4J_CONFIG = {
    'uri' : 'neo4j://192.168.110.247:7687',
    'auth' : (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    'encrypted' : False 
    } # encrypted代表是否加密
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
        checks = []
        departments = []
        producers = []
        disease_infos = []

        rels_recommandeat = []
        rels_recommanddrug = []
        rels_symptom = []
        rels_department = []
        rels_noteat = []
        rels_doeat = []
        rels_commondrug = []
        rels_check = []
        rels_drug_producer = []
        rels_acompany = []
        rels_category = []

        count = 0
        for data in open(self.data_path):
            disease_dict = {}
            count += 1
            if count % 500 == 0:
                print('count = ', count)

            data_json = json.loads(data) # 将纯文本数据解析成python的dict或list对象
            disease = data_json['name']
            diseases.append(disease)

            #为不需要建立节点的疾病属性做准备
            disease_dict['name'] = disease
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''


            if 'symptom' in data_json:
                symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    rels_symptom.append([disease, symptom])

            if 'acompany' in data_json:
                for acompany in data_json['acompany']:
                    rels_acompany.append([disease, acompany])

            if 'desc' in data_json:
                disease_dict['desc'] = data_json['desc']
            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']
            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']
            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']
            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']
            if 'cure_lasttime' in data_json:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']
            if 'cured_prob' in data_json:
                disease_dict['cured_prob'] = data_json['cured_prob']
            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']


            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                    rels_category.append([disease, cure_department[0]])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    rels_department.append([small, big])
                    rels_category.append([disease, small])
                
                disease_dict['cure_department'] = cure_department
                departments += cure_department
            
            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    rels_commondrug.append([disease, drug])
                drugs += common_drug

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels_recommanddrug.append([disease, drug])

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    rels_noteat.append([disease, _not])

                foods += not_eat

            if 'do_eat' in data_json:
                do_eat = data_json['do_eat']
                for _do in do_eat:
                    rels_doeat.append([disease, _do])

                foods += do_eat
            
            if 'recommand_eat' in data_json:
                recommand_eat = data_json['recommand_eat']
                for _recommand in recommand_eat:
                    rels_recommandeat.append([disease, _recommand])
                foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']
                for _check in check:
                    rels_check.append([disease, _check])
                    checks += check

            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')]for i in drug_detail]
                producers += producer

            disease_infos.append(disease_dict)

        return set(drugs), set(foods), set(checks), set(departments), set(producers), set(symptoms), set(diseases), disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commondrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category
    
    # 创建知识图谱疾病相关的节点, 疾病, 症状, 药品, ⻝品
    def create_graphnodes_and_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commondrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category= self.read_nodes()

        print('Drugs: ', len(Drugs))
        print('Foods: ', len(Foods))
        print('Checks: ', len(Checks))
        print('Departments: ', len(Departments))
        print('Producers: ', len(Producers))
        print('Symtptoms: ', len(Symptoms))
        print('Diseases: ', len(Diseases))
        print('-------------------------------------------------------------')
        print('rels_check: ', len(rels_check))
        print('rels_recommandeat: ', len(rels_recommandeat))
        print('rels_noteat: ', len(rels_noteat))
        print('rels_doeat: ', len(rels_doeat))
        print('rels_department: ', len(rels_department))
        print('rels_commanddrug: ', len(rels_commondrug))
        print('rels_drug_producer: ', len(rels_drug_producer))
        print('rels_recommanddrug: ', len(rels_recommanddrug))
        print('rels_symptom: ', len(rels_symptom))
        print('rels_acompany: ', len(rels_acompany))
        print('rels_category: ', len(rels_category))

        driver = GraphDatabase.driver( **NEO4J_CONFIG)

        with driver.session() as session:
            # 创建中心疾病的知识图谱节点
            print('开始创建中心疾病节点...')
            # for d in Diseases:
            #     cypher = 'merge (a:Disease{name:%r}) return a' % d
            #     session.run(cypher)
            
            n = 0
            m = 0
            for d in disease_infos:
                cypher = "merge (a:Disease{name:%r, desc:%r, prevent:%r, cause:%r, easy_get:%r, cure_lasttime:%r, cure_department:%r, cure_way:%r, cure_prob:%r})" % (d['name'], d['desc'], d['prevent'], d['cause'], d['easy_get'], d['cure_lasttime'], d['cure_department'], d['cure_way'], d['cured_prob'])

                try:
                    session.run(cypher)
                except:
                    m += 1
                    pass
                n += 1
                if n % 500 == 0:
                    print('n = ', n)
            
            print('疾病节点写⼊neo4j完毕, 共计{}个, ERROR{}个.'.format(n, m))
            print('=====================$$$$$$$$$$$========================')

            # 创建"药品", "⻝品", "症状", "检查", "科室", "⽣产商"的知识图谱节
            count, err = 0, 0
            print('开始创建药品节点Drug...')
            for n in Drugs:
                cypher = 'merge (a:Drug{name:%r}) return a' % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                    pass

            print('药品Drug, count={}, error={}'.format(count, err))
            print('---------------------------------------------')

            count, err = 0, 0
            print('开始创建食品节点Food...')
            for n in Foods:
                cypher = 'merge (a:Food{name:%r}) return a' % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                    pass
            print('食品Food, count={}, error={}'.format(count, err))
            print('---------------------------------------------')

            count, err = 0, 0
            print('开始创建症状节点Symptom...')
            for n in Symptoms:
                cypher = 'merge (a:Symptom{name:%r}) return a' % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                    pass
            print('症状Symptom, count={}, error={}'.format(count, err))
            print('---------------------------------------------')

            count, err = 0, 0
            print('开始创建检查节点Check......')
            for n in Checks:
                cypher = "MERGE (a:Check{name:%r}) RETURN a" % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                pass
            print('检查Check, count={}, error={}'.format(count, err))
            print('-----------------------------------------------')

            count, err = 0, 0
            print('开始创建科室节点Department......')
            for n in Departments:
                cypher = "MERGE (a:Department{name:%r}) RETURN a" % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                    pass
            print('科室Department, count={}, error={}'.format(count, err))
            print('-----------------------------------------------')

            count, err = 0, 0
            print('开始创建⽣产商节点Producer......')
            for n in Producers:
                cypher = "MERGE (a:Producer{name:%r}) RETURN a" % n
                count += 1
                try:
                    session.run(cypher)
                except:
                    err += 1
                    pass
            print('⽣产商Producer, count={}, error={}'.format(count, err))
            print('-----------------------------------------------')

        # 创建实体关系边
        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '推荐药品')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Food', rels_noteat, 'no_eat', '忌吃')
        self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
        self.create_relationship('Disease', 'Drug', rels_commondrug, 'command_drug', '常⽤药品')
        self.create_relationship('Disease', 'Drug', rels_drug_producer, 'drugs_of', '药品⼚商')
        self.create_relationship('Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Department', rels_category, 'belongs_to', '所属科室')

    # 编写类内函数，创建实体关系边
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        num_edges = len(set(set_edges))
        print('创建关系{}, num_edges = {}'.format(rel_name, num_edges))

        # 实例化图数据库驱动对象
        driver = GraphDatabase.driver( **NEO4J_CONFIG)

        with driver.session() as session:
            n, m = 0, 0
            for edge in set(set_edges):
                edge = edge.split('###')
                p = edge[0]
                q = edge[1]
                cypher = "match(p:%s), (q:%s) where p.name = '%s'and q.name='%s' merge(p)-[rel:%s{name:'%s'}]->(q)" % (start_node, end_node, p, q, rel_type, rel_name)

                try:
                    n += 1
                    session.run(cypher)
                except Exception as e:
                    m += 1
                
                if n % 100 == 0:
                    print('n = ', n)
            print('当前关系：{}，处理关系数量：{}，ERROR数量： {}'.format(rel_name, n, m))
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        return
    
if __name__ == '__main__':
    mg = MedicalGraph()
    print('创建知识图谱中的节点和关系...')
    mg.create_graphnodes_and_graphrels()
        


            


