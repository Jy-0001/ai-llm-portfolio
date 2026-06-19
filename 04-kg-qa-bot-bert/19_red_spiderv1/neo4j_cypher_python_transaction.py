# neo4j:可视化spo关系
    # 命令：
        # create命令：
            # 创建节点：
                # 例：CREATE (e:Employee{id:222, name:'Bob', salary:6000, deptnp:12})
            # 创建关系：
                # 例：CREATE (p1:Profile1)-[r:Buy]->(p2:Profile2)
        # match命令：匹配(查询)已有数据：必须是已经有的数据
            # 例：MATCH (e:Employee) RETURN e.id, e.name, e.salary, e.deptno
            # 查询所有节点：MATCH (n) RETURN n
        # merge命令：若节点存在, 则等效与match命令; 节点不存在, 则等效于create命令.
            # 例：MERGE (e:Employee {id:146, name:'Lucer', salary:3500, deptno:16})
            # merge创建命令：可以创建有/⽆⽅向性的关系.
                # 例：MERGE (p1:Profile1)-[r:miss]-(p2:Profile2)
        # where命令：类似于SQL中的添加查询条件
            # 例：MATCH (e:Employee) WHERE e.id=123 RETURN e
        # delete命令：删除节点/关系及其关联的属性
            # 例：MATCH (p1:Profile1)-[r]-(p2:Profile2) DELETE p1, r, p2
        # (sort)命令：Cypher命令中的排序使⽤的是order by.
            # 例：MATCH (e:Employee) RETURN e.id, e.name, e.salary, e.deptno ORDER BY e.id
            # 例：MATCH (e:Employee) RETURN e.id, e.name, e.salary, e.deptno ORDER BY e.id DESC 是降序
        # 字符串函数：
            # toUpper()函数：将⼀个输⼊字符串转换为⼤写字⺟.
                # 例：MATCH (e:Employee) RETURN e.id, toUpper(e.name)
            # toLower()函数：将⼀个输⼊字符串转换为⼩写字⺟.
                # 例：MATCH (e:Employee) RETURN e.id, toLower(e.name)
            # substring()函数：返回⼀个⼦字符串：substring(input_str, start_index, end_index)
                # 例：MATCH (e:Employee) RETURN e.id, substring(e.name,0,2)
            # replace()函数：替换掉⼦字符串.replace(input_str, origin_str, new_str)
                # 例：MATCH (e:Employee) RETURN e.id, replace(e.name,e.name,e.name + "_HelloWorld"), e.salary,
        # 聚合函数：
            # count()函数：返回由match命令匹配成功的条数.
                # 例：MATCH (e:Employee) RETURN count(*)
            # max()函数：返回由match命令匹配成功的记录中的最⼤值.
                # MATCH (e:Employee) RETURN max(e.salary)
            # min()函数：返回由match命令匹配成功的记录中的最⼩值.
                # MATCH (e:Employee) RETURN min(e.salary)
            # sum()函数：返回由match命令匹配成功的记录中某字段的全部加和值.
                # MATCH (e:Employee) RETURN sum(e.salary)
            # avg()函数：返回由match命令匹配成功的记录中某字段的平均值.
                # MATCH (e:Employee) RETURN avg(e.salary)
        # 索引index：使⽤create index on来创建索引.
            # 例：CREATE INDEX employee_id_index FOR (e:Employee) ON (e.id);
            #删除索引：使⽤drop index on来删除索引.
                # 例：DROP INDEX employee_id_index;

# 在Python中使⽤neo4j
    # neo4j/neo4j-driver库：python中的neo4j驱动
        # 使用：
from class19_red_spider.build_medicalgraph import GraphDatabase

uri = 'bolt://192.168.110.247:7687' # 5.x版本用这个
# uri = 'bolt://0.0.0.0:7687' # 3.x版本用这个

driver = GraphDatabase.driver(uri, auth=('neo4j', 'neo4jneo4j'), max_connection_lifetime=100) # 创建驱动实例

with driver.session() as session:
    cypher = "create(p:Company) set p.name='算法工程师' return p.name"
    record = session.run(cypher)
    record_list = list(record)
    result = list(map(lambda x: x[0], record))
    print('result', result)
    print(record_list)

    # transaction事务：如果⼀组数据库操作要么全部发⽣要么⼀步也不执⾏, 我们称该组处理步骤为⼀个事务, 它是数据库⼀致性的保证.
        # 
from class19_red_spider.build_medicalgraph import GraphDatabase

def _some_operations(tx, cat_name, mouse_name):
    query = (
                'merge(a:cat {name:$cat_name})'
                'merge(b:mouse {name:$mouse_name})'
                'merge(a)-[r:And]-(b)'
            )
    tx.run(query, cat_name=cat_name, mouse_name=mouse_name)

uri = 'neo4j://192.168.110.247:7687'

driver = GraphDatabase.driver(uri, auth=('neo4j', 'neo4jneo4j'))

with driver.session() as session:
    session.execute_write(_some_operations, cat_name='tom', mouse_name='jerry')
    print('transaction compelete')
