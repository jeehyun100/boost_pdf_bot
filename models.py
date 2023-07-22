from sqlalchemy import create_engine, Column, Integer, String, DateTime, UnicodeText, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
import datetime
import os

DeclarativeBase = declarative_base()


def db_connect():
    # return create_engine(url=URL.create(drivername='mysql+mysqldb',
    #                                     username=os.environ.get('DAUM_NEWS_CRAWLER_DB_USER'),
    #                                     password=os.environ.get('DAUM_NEWS_CRAWLER_DB_PW'),
    #                                     host=os.environ.get('DAUM_NEWS_CRAWLER_DB_HOST'),
    #                                     port=os.environ.get('DAUM_NEWS_CRAWLER_DB_PORT'),
    #                                     database=os.environ.get('DAUM_NEWS_CRAWLER_DB')),
    #                      connect_args={'charset': 'utf8mb4',
    #                                    'use_unicode': 'True'})
    # return create_engine(url=URL.create(drivername='mysql+mysqldb',
    #                                 username='psnr_user',
    #                                 password='11111111',
    #                                 host='localhost',
    #                                 port='3306',
    #                                 database='PNSR',
    #                     connect_args={'charset': 'utf8mb4',
    #                                 'use_unicode': 'True'}))
    return create_engine(url=URL.create(drivername='mysql+mysqldb',
                                    username='pnsr_user',
                                    password='Audgk13%',
                                    host='localhost',
                                    port='3306',
                                    database='PNSR'))


def create_tables(engine):
    DeclarativeBase.metadata.create_all(engine, checkfirst=True)


class Article(DeclarativeBase):
    __tablename__ = 'stock_reports'
    
    # id = Field()
    # stocks = Field()
    # title = Field()
    # source = Field()
    # created_date = Field()


    # stocks_url = li.xpath('div[1]/strong/a/@href').extract()
    # id_temp = li.xpath('div[2]/p/a/@href').extract()[0]
    # id = re.sub(r'[^0-9]', '', id_temp)
    # stocks_name = li.xpath('div[1]/strong/a/text()').extract()
    # title = li.xpath('div[2]/p/a/text()').extract()
    # prefer_price = ' '.join(li.xpath('div[3]/text()').extract()).replace("\t","").replace("\r","").replace("\n","")
    # invest_opinion = ' '.join(li.xpath('div[4]/text()').extract()).replace("\t","").replace("\r","").replace("\n","")
    # source = li.xpath('div[5]/text()').extract()[0]
    # created_dt = li.xpath('div[6]/text()').extract()[0]
    


    id = Column('id', String(50), primary_key=True)
    stocks = Column('stocks', String(100), nullable=False, server_default='')
    title = Column('title', String(1000), nullable=False, server_default='')
    prefer_price = Column('prefer_price', String(1000), nullable=False, server_default='')
    invest_opinion = Column('invest_opinion', String(1000), nullable=False, server_default='')
    source = Column('source', String(100), nullable=False, server_default='')
    created_dt = Column('created_dt', DateTime, nullable=False, server_default=str(datetime.datetime.min))
    attachment_file = Column('attachment_file', String(1000), nullable=False, server_default='')
    
    
    

    def __init__(self):
        
        self.id = ''
        self.stocks = ''
        self.title = ''
        self.prefer_price = ''
        self.invest_opinion = ''
        self.source = ''
        self.download_url = ''
        self.created_dt = datetime.datetime.min
        self.attachment_file = ''


