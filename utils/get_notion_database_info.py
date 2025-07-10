# get_notion_database_info.py
import configparser
from notion_client import Client
from datetime import datetime, timedelta
import logging
import time

class NotionDatabaseManager:
    def __init__(self):
    # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read('./conf/config.cfg')
        # 获取配置信息
        self.database_id = self.config['notion']['database_id']
        self.task_database_id = self.config['notion']['task_database_id']
        self.stock_database_id = self.config['notion']['stock_database_id']
        self.notion = Client(auth=self.config['notion']['token'])
        self.userid = self.config['notion']['user_id']

    def query_notion_database(self,date):
        '''
        # 修改为使用notion-client库查询数据库
        '''
        # 构造查询数据库的 URL     
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
        database_info = self.notion.databases.query(database_id=self.database_id, **filter_data)
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

    def update_task_database(self,date,advice,action='buy'):
        # 构造查询数据库的 URL        
        filter_data = {
                        "filter": {
                            # 原有的日期过滤条件
                                "property": "日期",
                                "date": {
                                   "equals": date
                                }
                        }
                        }
        database_info = self.notion.databases.query(database_id=self.task_database_id, **filter_data)
        if len(database_info['results'])==0:
            # 准备属性
            title = 'Today Advice'
            # 构造要插入的数据
            new_page = {
                "parent": {
                    "database_id": self.task_database_id
                },
                "properties": {
                    "名称": {
                        "title": [
                            {
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    },
                    "日期": {
                        "date": {
                            "start": date,
                            "end": None
                        }
                    },
                    "状态": {
                        "status": {
                            "name": "未开始"
                        }
                    },
                    "负责人": {
                        "people": [
                            {
                                "object": "user",
                                "id": self.userid
                            }
                        ]
                    },
                    "文本": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": advice
                                }
                            }
                        ]
                    }
                }
            }
            try:
                response = self.notion.pages.create(**new_page)
                logging.info("页面创建成功！")
                logging.info(response)
            except Exception as e:
                logging.error(f"页面创建失败！{e}")
        else:
            return
    def update_stock_database(self,data_dict):
        database_info = self.notion.databases.query(database_id=self.stock_database_id,).get("results")
        for page in database_info:
            try:
                # 获取当前记录的股票代码
                symbol = page["properties"]["股票代码"]["title"][0]["plain_text"]
                
                # 如果该股票在价格字典中存在
                if symbol in data_dict:
                    new_price = data_dict[symbol]
                    
                    # 更新Notion页面
                    self.notion.pages.update(
                        page_id=page["id"],
                        properties={
                            "当前价格": {"number": new_price}
                        }
                    )
                    
                    logging.info(f" 已更新 {symbol}: {new_price}")
                    time.sleep(0.3)  # 控制请求频率
                
            except Exception as e:
                logging.error(f" 更新失败 {symbol}: {str(e)}")




if __name__ == "__main__":
    # 创建 NotionDatabaseManager 实例
    db_manager = NotionDatabaseManager()
    # 获取数据库信息
    buylist,selllist=db_manager.query_notion_database('2025-07-08')



