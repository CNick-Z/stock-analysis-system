# get_notion_database_info.py
import requests
import configparser

class NotionDatabaseManager:
    def __init__(self):
    # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read('./conf/config.cfg')
        # 获取配置信息
        self.database_id = self.config['notion']['database_id']
        self.token = self.config['notion']['token']
        self.base_url = self.config['notion']['base_url']
        self.headers = {
            "Authorization": f'Bearer {self.token}',
            "Notion-Version": "2022-06-28",  # 使用最新的 API 版本
            "Content-Type": "application/json"
            }

    def query_notion_database(self,date):
        '''
        # 构造查询数据库信息的 URL
        query_url = f"{base_url}{database_id}"
        # 发起请求获取数据库信息
        response = requests.get(query_url, headers=headers)
        database_info = response.json()
        print(database_info)
        '''
        # 构造查询数据库的 URL
        target_date = date
        query_url = f"{self.base_url}{self.database_id}/query"
        
        filter_data = {
                        "filter": {
                            "and": [
                            # 原有的日期过滤条件
                            {
                                "property": "Siu|",
                                "date": {
                                    "equals": date
                                }
                            },
                            # 新增的类型排除条件
                            {
                                "property": "=xdT",  # 类型字段ID
                                "select": {
                                "does_not_equal": "ETF"  # 精确排除ETF类型
                                }
                            }
                            ]
                        }
                        }
        '''
        filter_data = {
                        "filter": {
                            # 原有的日期过滤条件
                                "property": "Siu|",
                                "date": {
                                    "equals": date
                                }
                        }
                    }
        '''
        response = requests.post(query_url, headers=self.headers,json=filter_data)
        database_info = response.json()
        buylist=[]
        selllist=[]
        for key,item in enumerate(database_info['results']):
            date=item['properties']['交易日期']['date']['start']
            price=item['properties']['成交价格']['number']
            pos=item['properties']['交易股数']['number']
            symbol=item['properties']['股票代码']['title'][0]['text']['content']
            if pos>0:
                buylist.append([date,symbol,price,pos])
            else:
                selllist.append([date,symbol,price,abs(pos)])
        return buylist,selllist

if __name__ == "__main__":
    # 创建 NotionDatabaseManager 实例
    db_manager = NotionDatabaseManager()
    # 获取数据库信息
    buylist,selllist=db_manager.query_notion_database('2025-03-19')
