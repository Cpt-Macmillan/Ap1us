#### 语音消息输出
import pyttsx3
def saya(words):
    engine = pyttsx3.init()
    engine.say(words)
    engine.runAndWait()

def bot_format(message, i=0):
    if(message == None):
        print("ERROR: No message for output.")
        return
    else:
        print("BOT: {}".format(message))
        if(i==0):
            saya(message)

#### 借助rasa完成意图识别
#### 通过rasa来意图识别
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

intent_trainer = Trainer(config.load("config_spacy.yml"))
intent_training_data = load_data('intent_recognize.json')
intent_interpreter = intent_trainer.train(intent_training_data)

def rasa_intent_recognize(message):
    message_low = message.lower()
    if("price" in message_low):
        return "get_price"
    elif("descri" in message_low):
        return "get_description"
    elif("volume" in message_low):
        return "get_volume"
    elif("plot" in message_low or "chart" in message_low or 'performance' in message_low):
        return "get_plot"
    intent = intent_interpreter.parse(message)
    #print(intent)
    
    if(intent['intent']['confidence'] > 0.4):
        return (intent['intent']['name'])
    return 'chitchat'


#### 使用spacy完成公司名称提取 
#### 自动补句号 但没有实现大小写猜测
import spacy
nlp = spacy.load("en_core_web_md")
embedding_dim = nlp.vocab.vectors_length
include_entities = ['DATE','ORG']

#### 从公司名称映射股票代码
company_to_code = {
    'apple':"AAPL",
    'facebook':'FB',
    'google':'GOOG',
    'tesla':'TSLA',
    'amazon':'AMZN',
    'microsoft':'MSFT',
    'twitter':'TWTR',
    'hp':'HPQ',
    'oracle':'ORCL'
}

#### Spacy Part2
#### 使用spacy完成对公司名字的提取 自动补句号 但没有实现大小写猜测
def extract_entities(message):
    if(message[-1]!='.' and message[-1]!='?'):
        message = message + "."
    ents = dict.fromkeys(include_entities)
    doc = nlp(message)
    #print(doc.ents)
    for ent in doc.ents:
        if(ent.label_)in include_entities:
            if(ents[ent.label_] == None):
                ents[ent.label_] = []
            ents[ent.label_].append(ent.text)
    return ents


def extract_code(message,ents):
    pattern = re.compile("(\s|^)[A-Z]{2,4}(\s|$)")
    pattern2 = re.compile("[A-Z]{2,4}")
    start = 0
    target = re.findall(r"[A-Z]{2,4}",message)
    for e in target:
        if(ents['ORG']==None):
            ents['ORG'] = [e]
        elif(e not in ents['ORG']):
            ents['ORG'].append(e)
    return ents

def extract_corp(message,ents):
    doc = nlp(message)
    for e in doc:
        e = e.text
        if(e.lower() in company_to_code):
            if(ents['ORG']==None):
                ents['ORG'] = [e]
            elif(e not in ents['ORG']):
                ents['ORG'].append(e)
    return ents


#### 基于chunk的否定识别，这里只能够排除对公司的否定
def exclude_negated(message,ents):
    ORG = ents['ORG']
    if(ORG == None):
        return ents
    ends = sorted([message.index(e) + len(e) for e in ORG])
    start = 0
    chunks = []
    for end in ends:
        chunks.append(message[start:end])
        start = end
    for chunk in chunks:
        for ent in ORG:
            if ent in chunk:
                if "not" in chunk or "n't" in chunk or "instead of" in chunk or ent=='Goodjob':
                    ents['ORG'].remove(ent)
    return ents


#### Iex Cloud

from iexfinance.stocks import Stock,get_historical_data
import matplotlib.pyplot as plt
from datetime import datetime,date,timedelta

my_token = "pk_ac7cc798df634b338460eaf71b72a40f"

def get_stock_price(code):
    return_value = "code_error"
    try:
        stock = Stock(code,token = my_token)
        bot_format("On my way...")
        quote = stock.get_quote()
        
        if(quote == None):
            return_value = "connection_failed"
        else:
            return_value = quote['latestPrice']
    finally:
        return return_value

    
def get_stock_volume(code):
    return_value = "code_error"
    try:
        stock = Stock(code,token = my_token)
        bot_format("Copy that.")
        bot_format("I'm on the way...",1)
        quote = stock.get_quote()
        if(quote == None):
            return_value = "connection_failed"
        else:
            if(quote['latestVolume'] != None):
                return_value = quote['latestVolume']
            else:
                return_value = quote['previousVolume']
    finally:
        return return_value
    
def show_stock_description(code):
    return_value = "code_error"
    try:
        stock = Stock(code,token = my_token)
        bot_format("On my way...")
        quote = stock.get_company()
        bot_format("The stock: \"{}\" belongs to the company\"{}\". About {}".format(quote['symbol'],quote['companyName'],quote['description']))
        if(quote == None):
            return_value = "connection_failed"
        else:
            return_value = 'done'
    finally:
        return return_value
def show_stock_plot(code):
    return_value = "code_error" 
    try:
        stock = Stock(code,token = my_token)
        bot_format("Loading",1)
        today = datetime.today()
        start = today + timedelta(days = -7)
        df = get_historical_data(code,start,today,output_format = 'pandas',token = my_token)
        df.drop(['volume'],axis = 1).plot()
        plt.show()
        return_value = "done"
    finally:
        return return_value
    
    

#### 使用rasa 识别 chitchat
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

chitchat_trainer = Trainer(config.load("/config_spacy.yml"))
chitchat_training_data = load_data('/chitchat.json')
chitchat_interpreter = chitchat_trainer.train(chitchat_training_data)

def rasa_chitchat_recognize(message):
    chitchat = chitchat_interpreter.parse(message)
    if('thank' in message):
        return 'thank'
    if(chitchat['intent']['confidence'] > 0.2):
        return (chitchat['intent']['name'])
    return 'default'

#### 判断股票代码
def is_stockCode(string):
    pattern = re.compile("[A-Z]{2,4}")
    if(re.match(pattern,string)):
        return True
    else:
        return False

#### 主要处理环节

#状态标志符
wait_for_affirm = 0
user_intent = 0
stock_company = 0
stock_code = 0

def respond(message):
    global wait_for_affirm, user_intent, stock_company, stock_code
    update_Info = 0
    has_intent = 1#默认每一句话都有意图
    
    #常规意图提取
    intent = rasa_intent_recognize(message)
    if(intent == 'chitchat'):#识别出的意图是chitchat(即没有意图)
        has_intent = 0
    elif(intent!='affirm' and intent!='deny'):#识别出动意图是 是/否
        bot_format("Well, I see.")
        user_intent = intent #更新user_intent
    
    #信息提取
    ents = extract_entities(message)
    ents = extract_code(message,ents)
    ents = extract_corp(message,ents)
    ents_excluded = exclude_negated(message,ents)
    
    
    if(ents_excluded['ORG'] != None):
        update_Info = 1
        for org in ents_excluded['ORG']:
            if(is_stockCode(org) == False):#该组织是公司名或其他组织
                if(stock_company==0 or org.lower()!=stock_company.lower):
                    stock_company = org
                    stock_code = 0
            else:#该组织名是股票代码
                stock_code = org
                stock_company = 0
    
    #闲聊意图提取
    chitchat_intent = rasa_chitchat_recognize(message)
    
    
    
    #开始回复信息
    
    if(intent=='affirm'):
        #第一类情况：在询问是否需要之后 回答“是”
        if(wait_for_affirm == 1):
            for comp,code in company_to_code.items():
                if(comp==stock_company.lower()):
                    #print('found')
                    stock_code = code
                    wait_for_affirm = 0
                    #继续按照之前的意图执行任务
                    break
            #print("error: not found in dict.")
        #第二类情况：莫名其妙地回答一个“是”
        else:
            bot_format("I'm sorry, I don't know what you are talking about.")
            return
    elif(intent=='deny'):
        #第三类情况：在询问是否需要之后 回答“否”
        if(wait_for_affirm == 1):
            bot_format("Sorry for that. :( Please provide me with the stock code.")
            wait_for_affirm = 0
            return
        #第四类情况：莫名其妙地回答一个“否”
        else:
            bot_format("I'm sorry, I don't know what you are talking.")
            return

    #print("u_intent:",user_intent," wfa:",wait_for_affirm," s_company:",stock_company," s_code:",stock_code)
    #print("update_Info: ",update_Info," has_intent:",has_intent,"chitchat_intent:",chitchat_intent)
        #第五类情况:闲聊
    if(has_intent == 0):
        if(chitchat_intent == 'scold' and update_Info ==0):
            bot_format(r"Sorry for not satisfying you :( I'm still learning to progress.")
        elif(chitchat_intent == 'greet' and update_Info==0):
            bot_format("Nice to meet you.")
        elif(chitchat_intent == 'goodbye'):
            bot_format("Looking forward to see you again :)")
            return
        elif(chitchat_intent == 'praise'):
            bot_format("Thank u. I'm still progressing.")
        elif(chitchat_intent=='thanks'):
            bot_format("You are welcome.")
            
        if(update_Info==0):
            return
        
    

        #第6.5类情况：用户更新了别的东西“非闲聊”但是没有确认是/否
    if(wait_for_affirm == 1 and intent!='affirm'):
        bot_format("Well, I still need you to confirm the code of the stock.")
    
        #到这里为止,已经排除所有闲聊的情况了，但是依然有可能没有给出意图
    
        #第七类情况：已经有明确的意图(无论是否是本次输入更新的），但没公司名字，也没有股票代码
        #此类情况有一种可能：1）本轮更新了意图 
    if(user_intent!=0 and stock_company==0 and stock_code==0):
        bot_format("I will do that for you, but which company?")
        return
    
        #第八类情况：还不知道用户的意图，但是已经确认了是哪一支股票
    if(user_intent==0 and stock_code!=0):
        bot_format("Well, what do you want to know about this?")
        return
    
        #第九类情况：无论有没有明确的意图 这里只有公司名字 不确认股票代码
    if(stock_company!=0 and stock_code==0):
        if stock_company.lower() in company_to_code:
            bot_format(r"For '{}',do you mean its stock '{}' ?".format(stock_company,company_to_code[stock_company.lower()]))
            wait_for_affirm = 1
            return
        else:
            
            bot_format(r"Sorry, I don't know the stock code of that orgnization '{}'".format(stock_company))
            stock_company = 0
            return

    
    if(stock_code!=0 and user_intent!=0):
        if(user_intent=='get_price'):
            price = get_stock_price(stock_code)
            if(price == 'code_error'):
                bot_format("Sorry, there's something wrong with that code: {}.".format(stock_code))
            elif(price == 'connection_failed'):
                bot_format("Sorry, query failed due to network error.")
            else:
                bot_format("The latest price of the stock is {}.".format(price))
            return
        elif(user_intent == 'get_volume'):
            volume = get_stock_volume(stock_code)
            if(volume == 'code_error'):
                bot_format("Sorry, there's something wrong with that code: {}.".format(stock_code))
            elif(volume == 'connection_failed'):
                bot_format("Sorry, query failed due to network error.")
            else:
                bot_format("The latest volume of the stock is {}.".format(volume))
            return
        elif(user_intent == 'get_plot'):
            plot = show_stock_plot(stock_code)
            if(plot!='done'):
                bot_format("Sorry, I can't find the historical data at this moment.")
            else:
                bot_format("Here comes the plot")
        elif(user_intent == 'get_description'):
            bot_format("loading",1)
            show_stock_description(stock_code)


def main():
    for i in range(100):
        user_input = input()
        if(user_input=='quit()'):
            return
        
        respond(user_input)

main()
