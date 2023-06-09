import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
from tqdm import tqdm
import datetime
#=====定义函数====
#from HTMLTable import HTMLTable
import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.font_manager import FontProperties  
#import matplotlib.text.Text
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from dingtalkchatbot.chatbot import DingtalkChatbot
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=69d2f134c31ced0426894ed975f29b519c1a8bd163a808840ef5812c5a0477a1'
from qiniu import Auth, put_file, etag
def gmt_img_url(key=None,local_file=None,**kwargs):
    # refer:https://developer.qiniu.com/kodo/sdk/1242/python
    # key:上传后保存的文件名；
    # local_file:本地图片路径，fullpath
    # 遗留问题：如果服务器图片已存在，需要对保存名进行重命名

    #需要填写你的 Access Key 和 Secret Key
    access_key = 'svjFs68isTvptqveLl9xBADP9v8s0jZdUzoGe0-U'
    secret_key = 'XRqt6RgoeK9-hZmKyPjPuFQkeYcU0cPNVgKWEl7l'

    #构建鉴权对象
    q = Auth(access_key, secret_key)

    #要上传的空间
    bucket_name = 'carsonlee'

    #生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key)

    #要上传文件的本地路径
    ret, info = put_file(token, key, local_file)

    base_url = 'http://ruusug320.hn-bkt.clouddn.com'    #七牛测试url
    url = base_url + '/' + key
    #private_url = q.private_download_url(url)

    return url
#fpath = os.path.join(rcParams["datapath"], "fonts/ttf/cmr10.ttf")
prop = fm.FontProperties(fname='/root/chain_every_day/SimHei.ttf')

# ======= 正式开始执行

def cal(x):
    if x>= pd.to_datetime('2013-01-01') and x<= pd.to_datetime('2016-10-31'):
        y = 'Second cycle'
    elif x>= pd.to_datetime('2016-11-01') and x<= pd.to_datetime('2020-04-30'):
        y = 'Third cycle'
    else:
        y = 'Fourth cycle'
    return y
url_address = ['https://api.glassnode.com/v1/metrics/indicators/puell_multiple',
                'https://api.glassnode.com/v1/metrics/indicators/sopr_adjusted',
                'https://api.glassnode.com/v1/metrics/market/mvrv_z_score',
                'https://api.glassnode.com/v1/metrics/indicators/rhodl_ratio',
                'https://api.glassnode.com/v1/metrics/indicators/net_realized_profit_loss',
                'https://api.glassnode.com/v1/metrics/market/price_usd_close',
                'https://api.glassnode.com/v1/metrics/supply/profit_relative',
                'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_sum',
                'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_from_exchanges_sum']
url_name = ['Puell Multiple', 'aSOPR','MVRV Z-Score','RHODL Ratio','Net Realized Profit/Loss','Price','Percent Supply in Profit','in_exchanges', 'out_exchanges']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2012-10-01')]
last_data = last_data.sort_values(by=['date'])
last_data = last_data.reset_index(drop=True)
date = []
pm = []
mvrv = []
rhold = []
net = []
price = []
sopr_7 = []
sopr_50 = []
supply = []
for j in range(len(last_data)-49):
    ins = last_data[j:j+50]
    ins = ins.reset_index(drop=True)
    date.append(ins['date'][49])
    pm.append(ins['Puell Multiple'][49])
    mvrv.append(ins['MVRV Z-Score'][49])
    rhold.append(ins['RHODL Ratio'][49])
    sopr_50.append(np.mean(ins['aSOPR']))
    supply.append(np.mean(ins['Percent Supply in Profit'][-40:]))
    price.append(ins['Price'][49])
    #短期指标
    net.append(np.mean(ins['Net Realized Profit/Loss'][-7:]))
    sopr_7.append(np.mean(ins['aSOPR'][-7:]))
res_df = pd.DataFrame({'date':date,'Puell Multiple':pm,'MVRV Z-Score':mvrv,'RHODL Ratio':rhold,'Net Realized Profit/Loss':net,'Price':price,'Percent Supply in Profit':supply,'7MA aSOPR':sopr_7,'50MA aSOPR':sopr_50})
res_df = res_df[(res_df.date>='2013-01-01')]
res_df['cycle'] = res_df['date'].apply(lambda x:cal(x))
res_df['log(BTC price)'] = np.log(res_df['Price'])
res_df['log(RHODL Ratio)'] = np.log(res_df['RHODL Ratio'])
res_df['x1'] = 7
res_df['x2'] = 0
res_df['y1'] = 4
res_df['y2'] = 0.5
res_df['z1'] = np.log(49000)
res_df['z2'] = np.log(350)
res_df['w'] = 1
res_df['p1'] = 0.9
res_df['p2'] = 0.5

url_address = ['https://api.glassnode.com/v1/metrics/market/mvrv_z_score',
                'https://api.glassnode.com/v1/metrics/market/price_usd_close']
url_name = ['MVRV Z-Score','Price']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'ETH', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2015-01-01')]
last_data = last_data.sort_values(by=['date'])
last_data = last_data.reset_index(drop=True)
date = []
mvrv = []
price = []
for j in range(len(last_data)-49):
    ins = last_data[j:j+50]
    ins = ins.reset_index(drop=True)
    date.append(ins['date'][49])
    mvrv.append(ins['MVRV Z-Score'][49])
    price.append(ins['Price'][49])
eth_df = pd.DataFrame({'date':date,'MVRV Z-Score':mvrv,'Price':price})
eth_df = eth_df[(eth_df.date>='2015-01-01')]
eth_df['cycle'] = eth_df['date'].apply(lambda x:cal(x))
eth_df['log(ETH price)'] = np.log(eth_df['Price'])
eth_df['x1'] = 7
eth_df['x2'] = 0

url_address = ['https://api.glassnode.com/v1/metrics/market/price_usd_close']
url_name = ['Price']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2010-01-01')]
from dateutil.relativedelta import relativedelta 
#last_data['new_date'] = last_data['date'].apply(lambda x:x + relativedelta(years=1))
last_data = last_data.sort_values(by=['date'])
last_data = last_data.reset_index(drop=True)
date = []
price_raw = []
price_ma120 = []
price_ma200 = []
price_ma1y = []
price_ma4y = []
price_ma3_5y = []
price_ma1_2y = []
for j in range(len(last_data)-1824):
    ins = last_data[j:j+1825]
    ins = ins.sort_values(by='date')
    ins = ins.reset_index(drop=True)
    date.append(ins['date'][1824])
    price_raw.append(ins['Price'][1824])
    price_ma3_5y.append(np.mean(ins['Price'][0:730]))
    price_ma1_2y.append(np.mean(ins['Price'][-730:-365]))
    price_ma4y.append(np.mean(ins['Price'][-1459:]))
    price_ma1y.append(np.mean(ins['Price'][-364:]))
    price_ma200.append(np.mean(ins['Price'][-199:]))
    price_ma120.append(np.mean(ins['Price'][-119:]))
jun_df = pd.DataFrame({'date':date,'price_raw':price_raw,'price_ma120':price_ma120,'price_ma200':price_ma200,'price_ma1y':price_ma1y,'price_ma4y':price_ma4y,'price_ma3_5y':price_ma3_5y,'price_ma1_2y':price_ma1_2y})
jun_df = jun_df[(jun_df.date>='2018-12-01')]
jun_df['cycle'] = jun_df['date'].apply(lambda x:cal(x))

# 表格
date_value = eth_df['date'][len(eth_df)-1] #+ datetime.timedelta(days=1)

jun_df = jun_df.sort_values(by='date')
jun_df = jun_df.reset_index(drop=True)
sub_jun_df = jun_df[['date','price_raw','price_ma120','price_ma200','price_ma4y']][-5:-1]
sub_jun_df = sub_jun_df.set_index('date')
col_name = []
for ele in list(sub_jun_df.index):
    col_name.append(str(ele)[0:10])
sub_jun_df_T = pd.DataFrame(sub_jun_df.values.T,columns=col_name,index=['price_close','price_ma120','price_ma200','price_ma4y'])
sub_jun_df_T = sub_jun_df_T.round(0)
res_df = res_df.sort_values(by='date')
res_df = res_df.reset_index(drop=True)
sub_res_df = res_df[['date','Puell Multiple','MVRV Z-Score','RHODL Ratio','Net Realized Profit/Loss','Percent Supply in Profit','7MA aSOPR','50MA aSOPR']][-4:]
sub_res_df = sub_res_df.set_index('date')
sub_res_df_T = pd.DataFrame(sub_res_df.values.T,columns=col_name,index=['Puell Multiple','BTC MVRV Z-Score','RHODL Ratio','Net Realized Profit/Loss','Percent Supply in Profit','7MA aSOPR','50MA aSOPR'])
eth_df = eth_df.sort_values(by='date')
eth_df = eth_df.reset_index(drop=True)
sub_eth_df = eth_df[['date','MVRV Z-Score']][-4:]
sub_eth_df = sub_eth_df.set_index('date')
sub_eth_df_T = pd.DataFrame(sub_eth_df.values.T,columns=col_name,index=['ETH MVRV Z-Score'])
sub_eth_df_T = sub_eth_df_T.round(4)
combine_df = pd.concat([sub_res_df_T,sub_eth_df_T,sub_jun_df_T])
combine_df = combine_df.applymap(lambda x: format(x, '.4'))
combine_df = combine_df.reset_index(drop=True)


def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation
chart_width = 0.88
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']
#fm.fontManager.addfont('/home/tione/notebook/picture/SimHei.ttf')
##  图表布局规划
fig_new = plt.figure(figsize=(12, 20), facecolor=(0.82, 0.83, 0.85))



title_asset_pool = str(datetime.datetime.now().strftime('%Y-%m-%d'))
plt.suptitle(f'比特币数据日报:  {title_asset_pool}',
             fontsize=20,
             fontweight=10,
             fontproperties = prop)

# 投资回测结果的评价指标全部被打印在图表上，所有的指标按照表格形式打印
# 为了实现表格效果，指标的标签和值分成两列打印，每一列的打印位置相同
# ===================================


url_address = ['https://api.glassnode.com/v1/metrics/market/price_usd_close',
               'https://api.glassnode.com/v1/metrics/mining/revenue_sum',
               'https://api.glassnode.com/v1/metrics/addresses/min_1k_count',
               'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net']
url_name = ['Price','revenue_sum','min_1k_count','net_volume']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
btc_data = result_data[(result_data.date>='2012-10-01')]
btc_data = btc_data.sort_values(by=['date'])
btc_data = btc_data.reset_index(drop=True)

url_address = ['https://api.glassnode.com/v1/metrics/market/price_usd_close']
url_name = ['Price']
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'ETH', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
eth_data = result_data[(result_data.date>='2012-10-01')]
eth_data = eth_data.sort_values(by=['date'])
eth_data = eth_data.reset_index(drop=True)

combine_data = eth_data.merge(btc_data,how='left',on=['date'])
combine_data['per'] = combine_data['Price_x']/combine_data['Price_y']

fig_new.text(0.4, 0.90, f'昨日ETH/BTC价格比:\n'
                     f'昨日矿工总收入7日环比增加:\n'
                     f'昨日持币大于1000地址数7日环比增加:\n'
                     f'昨日交易所BTC净流入:\n'
                     f'交易所BTC7日总净流入:\n', ha='right',fontproperties = prop)

b1 = combine_data['per'][len(combine_data)-1]
b2 = (combine_data['revenue_sum'][len(combine_data)-1]-combine_data['revenue_sum'][len(combine_data)-8])/combine_data['revenue_sum'][len(combine_data)-8]
b3 = (combine_data['min_1k_count'][len(combine_data)-1]-combine_data['min_1k_count'][len(combine_data)-8])/combine_data['min_1k_count'][len(combine_data)-8]
b4 = combine_data['net_volume'][len(combine_data)-1]
b5 = np.sum(combine_data['revenue_sum'][len(combine_data)-8:len(combine_data)-1])

fig_new.text(0.41, 0.90, f'{b1:.2}    \n'
                     f'{b2: .2%}    \n'
                     f'{b3:.2%}    \n'
                     f'{round(b4,2)}个\n'
                     f'{round(b5,2)}个 \n',fontproperties = prop)

# ===================================


url_address = ['https://api.glassnode.com/v1/metrics/supply/current',
               'https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net']
url_name = ['shizhi','net_volume']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'USDT', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
usdt_data = result_data[(result_data.date>='2012-10-01')]
usdt_data = usdt_data.sort_values(by=['date'])
usdt_data = usdt_data.reset_index(drop=True)

url_address = ['https://api.glassnode.com/v1/metrics/distribution/balance_us_government']
url_name = ['balance_us']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
balance_us_data = result_data[(result_data.date>='2012-10-01')]
balance_us_data = balance_us_data.sort_values(by=['date'])
balance_us_data = balance_us_data.reset_index(drop=True)

c1 = usdt_data['shizhi'][len(usdt_data)-1]/100000000
c2 = (usdt_data['shizhi'][len(usdt_data)-1]- usdt_data['shizhi'][len(usdt_data)-8])/ usdt_data['shizhi'][len(usdt_data)-8]
c3 = usdt_data['net_volume'][len(usdt_data)-1]/100000000
c4 = np.sum(usdt_data['net_volume'][len(usdt_data)-8:len(usdt_data)-1])/100000000
c5 = balance_us_data['balance_us'][len(balance_us_data)-1]

fig_new.text(0.77, 0.90, f'昨日USDT总市值:\n'
                     f'USDT市值7日环比增加:\n'
                     f'昨日交易所USDT净流入:\n'
                     f'交易所USDT7日总流入:\n'
                     f'美国政府持有BTC数量:\n', ha='right',fontproperties = prop)

fig_new.text(0.78, 0.90, f'{round(c1,2)}亿美金 \n'
                     f'{c2: .2%}    \n'
                     f'{round(c3,2)}亿    \n'
                     f'{round(c4,2)}亿\n'
                     f'{round(c5,2)}个 \n',fontproperties = prop)

ax1 = fig_new.add_axes([0.05, 0.75, 0.31, 0.13])
ax2 = fig_new.add_axes([0.395, 0.75, 0.31, 0.13])
ax3 = fig_new.add_axes([0.74, 0.75, 0.31, 0.13])
ax4 = fig_new.add_axes([0.05, 0.60, 0.37, 0.13])
ax5 = fig_new.add_axes([0.45, 0.60, 0.60, 0.13])
ax6 = fig_new.add_axes([0.05, 0.45, 0.21, 0.13])
ax7 = fig_new.add_axes([0.31, 0.45, 0.21, 0.13])
ax8 = fig_new.add_axes([0.58, 0.45, 0.21, 0.13])
ax9 = fig_new.add_axes([0.84, 0.45, 0.21, 0.13])
ax10 = fig_new.add_axes([0.05, 0.39, 1, 0.08])
ax11 = fig_new.add_axes([0.05, 0.30, 1, 0.08])
ax12 = fig_new.add_axes([0.05, 0.21, 1, 0.08])
ax13 = fig_new.add_axes([0.05, 0.12, 1, 0.08])
ax14 = fig_new.add_axes([0.05, 0.03, 1, 0.08])
ax5.axis('off')
#======制定第1区域
# 标签
labels = ['繁荣期','衰退期','萧条期','复苏期']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#007A00','#D1D1D1','#EB90FA']
# 指针指向
arrow = 3
# 标题
title = '比特币第四周期'
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax1.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax1.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax1.add_patch(r)
ax1.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax1.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax1.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax1.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax1.set_frame_on(False)
ax1.axes.set_xticks([])
ax1.axes.set_yticks([])
ax1.axis('equal')
plt.tight_layout()
#======制定第2区域
url_address = ['https://api.glassnode.com/v1/metrics/derivatives/futures_liquidated_volume_long_relative']
url_name = ['duo_kong']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2010-01-01')]
last_data = last_data.reset_index(drop=True)
duokong = last_data['duo_kong'][len(last_data)-1]
# 标签
labels = ['多头优势','空头优势']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#007A00']
# 指针指向
if duokong > 0.5:
    arrow = 1
else:
    arrow = 2
# 标题
title = '比特币多头清算占比：'+str("%.1f%%" % (round(duokong * 100, 2)))
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图

#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax2.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax2.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax2.add_patch(r)
ax2.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax2.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax2.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax2.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax2.set_frame_on(False)
ax2.axes.set_xticks([])
ax2.axes.set_yticks([])
ax2.axis('equal')
plt.tight_layout()
#======制定第3区域
headers = {

    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"

}#设置头部信息,伪装浏览器

response = requests.get( "https://history.btc126.com/zhishu/" , headers=headers )  #get方法访问,传入headers参数，
api_res_a = response.text
#api_res_a = json.dumps(api_res_a)
value = api_res_a.split('\t\t\t\tvalue:')[1][0:3]
value = int(value)
# 标签
labels = ['极度恐惧','恐惧','贪婪','极度贪婪']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#007A00','#0063BF','#EB90FA','#ED1C24']
# 指针指向
if value <=  25:
    arrow = 1
elif value > 25 and value < 50:
    arrow = 2
elif value >= 50 and value < 75:
    arrow = 3
else:
    arrow = 4
# 标题
title = '比特币恐惧和贪婪指数:' + str(' ') + str(round(value,0))

# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax3.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax3.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax3.add_patch(r)
ax3.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax3.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax3.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax3.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax3.set_frame_on(False)
ax3.axes.set_xticks([])
ax3.axes.set_yticks([])
ax3.axis('equal')
plt.tight_layout()

#======制定第4区域
import akshare as ak
plt.rcParams['figure.figsize'] = (5, 5)
usa_interest_rate = ak.macro_bank_usa_interest_rate()
usa_interest_rate_date = usa_interest_rate['日期'][len(usa_interest_rate)-1]
usa_interest_rate_date = str(pd.to_datetime(usa_interest_rate_date))[0:10]
# 柱状图
name = ['add 20%','not add']
count = [0.174,0.826]

ax4.bar(name,count,0.6,color='lightskyblue')
for i in range(len(name)):
    # 金牌
    ax4.text(name[i],count[i], "%.1f%%" % (round(float(count[i]) * 100, 2)),va="bottom",ha="center",fontsize=12,fontproperties=prop)
# plt.tick_params(axis='both', labelsize=14)
#ax4.set_xticks(fontsize=14,fontproperties=prop)
ax4.set_xlabel(usa_interest_rate_date+"加息幅度",fontsize=14,fontproperties=prop)
ax4.set_ylabel("概率",fontsize=14,fontproperties=prop)
ax4.set_title("CME美联储加息预测",fontproperties=prop,fontsize=14)

#======制定第5区域
eco_df = pd.read_csv('us_ec_data.csv',encoding='utf-8-sig')
#plt.rcParams['font.sans-serif'] = prop
#plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

col_width=3.0
row_height=0.625
font_size=14
header_color='#40466e'
row_colors=['#f1f1f2', 'w']
edge_color='w'
bbox=[0, 0, 1, 1]
header_columns=0
size = (np.array(eco_df.shape[::-1]) + np.array([0, 1])
        ) * np.array([col_width, row_height])
fig, ax = plt.subplots(figsize=size)
ax.axis('off')

# 计算每一列的最大宽度
colWidths = []
for col in eco_df.columns:
    colWidths.append(max(max(eco_df[col].map(lambda x: len(str(x).encode("gbk")))), len(str(col).encode("gbk"))))
maxCol = max(colWidths)
for i in range(len(colWidths)):
    colWidths[i] = colWidths[i] / maxCol      

mpl_table = ax5.table(cellText=eco_df.values, bbox=bbox,
                     colLabels=eco_df.columns, 
                     colWidths=colWidths)
mpl_table.auto_set_font_size(False)
#mpl_table.set_fontsize()
for k, cell in mpl_table._cells.items():
    cell.set_edgecolor(edge_color)
    #if k[0] == 0 or k[1] < header_columns:
    cell.set_text_props(weight='bold', color='w',fontproperties=prop)
    cell.set_facecolor(header_color)
ax5 = ax
#======制定第6区域
import requests #先导入爬虫的库，不然调用不了爬虫的函数
import json
#下面是可以正常爬取的区别，更改了User-Agent字段

headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
}#设置头部信息,伪装浏览器

response = requests.get( "https://api.btc126.com/coinglass.php?leibie=ahr999" , headers=headers )  #get方法访问,传入headers参数，
api_res_a = response.content.decode('utf-8')
api_res_a = json.loads(api_res_a)
api_value_a = api_res_a['data']
api_value_b = api_value_a[len(api_value_a)-1]
# 标签
labels = ['抄底区','定投区','观察区']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#EB90FA','#D1D1D1']
# 指针指向
ahr999 = api_value_b['ahr999']
if ahr999 < 0.25:
    arrow = 1
elif ahr999 >= 0.25 and ahr999 < 1:
    arrow = 2
else:
    arrow = 3
# 标题
title = 'AHR999:' + str(' ') + str(round(ahr999,2))
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax6.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax6.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax6.add_patch(r)
ax6.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax6.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax6.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax6.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax6.set_frame_on(False)
ax6.axes.set_xticks([])
ax6.axes.set_yticks([])
ax6.axis('equal')
plt.tight_layout()
#======制定第7区域
sub_res_df_mvrv = res_df['MVRV Z-Score'][len(res_df)-1]
# 标签
labels = ['抄底区','观察区','逃顶区']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#D1D1D1','#007A00']

if sub_res_df_mvrv < 0:
    arrow = 1
elif sub_res_df_mvrv >= 0 and sub_res_df_mvrv < 7:
    arrow = 2
else:
    arrow = 3
# 标题
title = 'MVRV Z-Score:' + str(' ') + str(round(sub_res_df_mvrv,2))
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax7.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax7.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax7.add_patch(r)
ax7.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax7.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax7.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax7.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax7.set_frame_on(False)
ax7.axes.set_xticks([])
ax7.axes.set_yticks([])
ax7.axis('equal')
plt.tight_layout()
#======制定第8区域
net = res_df['Net Realized Profit/Loss'][len(res_df)-1]
# 标签
labels = ['抄底区','观察区']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#D1D1D1']

if net < 0:
    arrow = 1
else:
    arrow = 2
# 标题
title = '7MA NRPL:' + str(' ') + str(round(net,2))
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax8.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax8.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax8.add_patch(r)
ax8.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax8.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax8.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax8.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax8.set_frame_on(False)
ax8.axes.set_xticks([])
ax8.axes.set_yticks([])
ax8.axis('equal')
plt.tight_layout()
#======制定第9区域
sopr = res_df['7MA aSOPR'][len(res_df)-1]
# 标签
labels = ['抄底区','观察区']
# 颜色（可以在Excel中查看颜色对应的十六进制码）
colors = ['#ED1C24','#D1D1D1']

if sopr < 0:
    arrow = 1
else:
    arrow = 2
# 标题
title = '7MA aSOPR:' + str(' ') + str(round(sopr,2))
# 调用仪表图
# 首先进行一些完整性检查
N = len(labels)
if arrow > N: 
    raise Exception("\n\nThe category ({}) is greated than \
    the length\nof the labels ({})".format(arrow, N))


# 如果colors是一个字符串，我们假设它是一个matplotlib colormap，并将其离散为N个离散颜色 
if isinstance(colors, str):
    cmap = cm.get_cmap(colors, N)
    cmap = cmap(np.arange(N))
    colors = cmap[::-1,:].tolist()
if isinstance(colors, list): 
    if len(colors) == N:
        colors = colors[::-1]
    else: 
        raise Exception("\n\nnumber of colors {} not equal \
        to number of categories{}\n".format(len(colors), N))

# 开始绘图
#fig, ax = plt.subplots()
ang_range, mid_points = degree_range(N)
labels = labels[::-1]

# 绘制扇形和弧线
patches = []
for ang, c in zip(ang_range, colors): 
    # 扇形
    patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
    # 弧线
    patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
[ax9.add_patch(p) for p in patches]


# 设置标签
for mid, lab in zip(mid_points, labels): 
    ax9.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
        horizontalalignment='center', verticalalignment='center', fontsize=10, \
        fontweight='bold', rotation = rot_text(mid),fontproperties = prop)

# 设置底部和标题
r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
ax9.add_patch(r)
ax9.text(0, -0.05, title, horizontalalignment='center', \
     verticalalignment='center', fontsize=10, fontweight='bold',fontproperties=prop)

# 画出箭头
pos = mid_points[abs(arrow - N)]
ax9.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
ax9.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
ax9.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

# 移除框架和刻度，并使轴相等和紧实
ax9.set_frame_on(False)
ax9.axes.set_xticks([])
ax9.axes.set_yticks([])
ax9.axis('equal')
plt.tight_layout()

#=============================================
# 绘画折线图
ax10_fu1 = ax10.twinx()
sns.lineplot(x="date", y="x1", data=res_df, color = 'green',linestyle='--',ax=ax10_fu1)
sns.lineplot(x="date", y="x2", data=res_df, color='red',linestyle='--',ax=ax10_fu1)
sns.lineplot(x="date", y="MVRV Z-Score",color='black',data=res_df,ax=ax10_fu1)
sns.lineplot(x="date", y="log(BTC price)",hue = 'cycle', data=res_df,ax=ax10)
ax10.tick_params(labelsize=10)
#plt.title('MVRV Z-Score —— log(BTC price)', fontsize=10) 
ax10.legend(loc='upper left', fontsize=5)
ax10.set_ylabel("MVRV Z-Score",fontsize=10)
#plt.show()
#plt.savefig('MVRV Z-Score.png') 
# 绘画折线图
ax11_fu2 = ax11.twinx()
sns.lineplot(x="date", y="y1", data=res_df, color = 'green',linestyle='--', ax=ax11_fu2)
sns.lineplot(x="date", y="y2", data=res_df, color='red', linestyle='--',ax=ax11_fu2)
sns.lineplot(x="date", y="Puell Multiple",color='black',data=res_df, ax=ax11_fu2)
sns.lineplot(x="date", y="log(BTC price)",hue = 'cycle', data=res_df, ax=ax11)
ax11.tick_params(labelsize=10)
#plt.title('Puell Multiple —— log(BTC price)', fontsize=10) 
ax11.legend(loc='upper left', fontsize=5)
ax11.set_ylabel("Puell Multiple",fontsize=10)
#plt.show()
#plt.savefig('Puell.png')
#plt.close()
# 绘画折线图
ax12_fu3 = ax12.twinx()
sns.lineplot(x="date", y="z1", data=res_df, color = 'green',linestyle='--',ax=ax12_fu3)
sns.lineplot(x="date", y="z2", data=res_df, color='red', linestyle='--',ax=ax12_fu3)
sns.lineplot(x="date", y="log(RHODL Ratio)",color='black',data=res_df, ax=ax12_fu3)
sns.lineplot(x="date", y="log(BTC price)",hue = 'cycle', data=res_df, ax=ax12)
ax12.tick_params(labelsize=10)
#plt.title('log(RHODL Ratio) —— log(BTC price)', fontsize=10) 
ax12.legend(loc='upper left', fontsize=5)
ax12.set_ylabel("RHODL Ratio",fontsize=10)
#plt.show()
#plt.savefig('RHODL.png')
#plt.close()

# 绘画折线图
sub_res_df = res_df[res_df.date>='2022-01-01']

ax13_fu5 = ax13.twinx()
sns.lineplot(x="date", y="w", data=sub_res_df, color='red',linestyle='--',  ax=ax13_fu5)
sns.lineplot(x="date", y="Net Realized Profit/Loss",color='black',data=sub_res_df, ax=ax13_fu5)
sns.lineplot(x="date", y="log(BTC price)",hue = 'cycle', data=sub_res_df, ax=ax13)
ax13.tick_params(labelsize=10)
#plt.title('7MA Net Realized Profit/Loss —— log(BTC price)', fontsize=10) 
ax13.legend(loc='upper left', fontsize=5)
ax13.set_ylabel("7MA NRPL",fontsize=10)
ax14_fu6 = ax14.twinx()
sns.lineplot(x="date", y="w", data=sub_res_df, color='red',linestyle='--', ax=ax14_fu6)
sns.lineplot(x="date", y="7MA aSOPR",color='black',data=sub_res_df, ax=ax14_fu6)
sns.lineplot(x="date", y="log(BTC price)",hue = 'cycle', data=sub_res_df, ax=ax14)
ax14.tick_params(labelsize=10)
ax14.legend(loc='upper left', fontsize=5)
ax14.set_ylabel("7MA aSOPR",fontsize=10)

fig_new.text(0.1, 0.8, '0xCarson出品，lian xi+V：shengxiaruchu009',
        fontsize=50, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30',fontproperties = prop)

fig_new.text(0.5, 0.5, '0xCarson出品，lian xi+V：shengxiaruchu009',
        fontsize=50, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30',fontproperties = prop)

fig_new.text(0.7, 0.3, '0xCarson出品，lian xi+V：shengxiaruchu009',
        fontsize=50, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30',fontproperties = prop)

fig_new.text(0.3, 0.3, '0xCarson出品，lian xi+V：shengxiaruchu009',
        fontsize=50, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30',fontproperties = prop)

fig_new.text(0.9, 0.8, '0xCarson出品，lian xi+V：shengxiaruchu009',
        fontsize=50, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30',fontproperties = prop)

fig_name = str(title_asset_pool) + '比特币链上数据一览图' + '.png'
fig_new.savefig(fig_name, dpi=200, bbox_inches='tight')
plt.close()


import telegram
content = '/root/chain_every_day/' + fig_name
bot = telegram.Bot(token='6219784883:AAE3YXlXvxNArWJu-0qKpKlhm4KaTSHcqpw')

eco_df = eco_df.reset_index(drop=True)
eco_df['event'] = eco_df.iloc[:,0]
eco_df['importent'] = eco_df.iloc[:,1]
eco_df['next_time'] = eco_df.iloc[:,2]
eco_df['predict'] = eco_df.iloc[:,3]
eco_df['front'] = eco_df.iloc[:,4]

flag = 0
for i in range(len(eco_df)):
    data = eco_df['next_time'][i]
    if str(data) == str(title_asset_pool):
        flag = 1
        event = eco_df['event'][i]
        importent = eco_df['importent'][i]
        predict = eco_df['predict'][i]
        front = eco_df['front'][i]
    else:
        continue

sub_res_df = res_df[['date','Puell Multiple','MVRV Z-Score','RHODL Ratio','Net Realized Profit/Loss','Percent Supply in Profit','7MA aSOPR','50MA aSOPR']][-4:]
sub_res_df = sub_res_df.reset_index(drop=True)
v1= sub_res_df['7MA aSOPR'][1]
v2= sub_res_df['7MA aSOPR'][2]
v3= sub_res_df['7MA aSOPR'][3]


url_address = ['https://api.glassnode.com/v1/metrics/indicators/puell_multiple',
                'https://api.glassnode.com/v1/metrics/indicators/sopr_adjusted']
url_name = ['Puell Multiple', 'aSOPR']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': 'BTC', 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    ins['date'] =  ins['t']
    ins[name] =  ins['v']
    ins = ins[['date',name]]
    data_list.append(ins)

result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2012-10-01')]
last_data = last_data.sort_values(by=['date'])
last_data = last_data.reset_index(drop=True)
v4 = last_data['aSOPR'][len(last_data)-1]

if v1 < 1 and v2 < 1 and v3 < 1:
    celue = '抄底'
elif v4 < 1:
    celue = '定投'
else:
    celue = '观察'


if flag == 1:
    text = '今日重要提示：%s，重要程度：%s，前值：%s，预测值：%s，数据公布前后比特币价格波动很大，注意控制合约风险。'%(event,importent,front,predict)
else:
    text = '【重要提示】今日比特币现货买入策略：%s'%(celue)

bot.sendDocument(chat_id='-840309715', document=open(content, 'rb'))
bot.sendMessage(chat_id='-840309715', text=text)


#推送钉钉群

time_str = str(time.time())[0:10]
key = 'blockchain_' + time_str + '.png'
img_url = gmt_img_url(key=key, local_file=fig_name)

xiaoding = DingtalkChatbot(webhook)
if flag == 1:
    txt = '【今日重要提示】 @所有人\n' \
          '> %s，重要程度：%s，前值：%s，预测值：%s，数据公布前后大饼价格波动很大，注意控制合约风险。\n\n' \
          '> ![数据监控结果](%s)\n'\
          '> ###### 币coin搜索0xCarson,关注OKX实盘。 \n'%(event,importent,front,predict,img_url)
else:
    txt = '【今日重要提示】 @所有人\n' \
          '> 大饼现货买入策略：%s \n\n'\
          '> ![数据监控结果](%s)\n'\
          '> ###### 币coin搜索0xCarson,关注OKX实盘。 \n'%(celue,img_url)
xiaoding.send_markdown(title='数据监控', text=txt);






