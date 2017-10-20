
# coding: utf-8

# In[118]:

import networkx as nx
import matplotlib.pyplot as plt
import community
import numpy as np
import urllib.request 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from collections import Counter


# # Load networks from book1 and book5
# 

# In[3]:

def createGraph(fileUrl) :
    df = pd.read_csv(fileUrl,sep=',') #,error_bad_lines=False
    G = nx.from_pandas_dataframe(df,source='Source',target='Target',edge_attr='weight')
    return G

fileUrls = ["https://raw.githubusercontent.com/mathbeveridge/asoiaf/master/data/asoiaf-book1-edges.csv", 
            "https://raw.githubusercontent.com/mathbeveridge/asoiaf/master/data/asoiaf-book5-edges.csv"]

G1,G5= [createGraph(x) for x in fileUrls]


# # Find communities from the network

# In[136]:

get_ipython().magic('matplotlib notebook')
def find_community(G,network_id,ax):
    partition = community.best_partition(G,weight='weight')
    communities = set(partition.values())
    num_comm = len(communities)
    
    # details
    print("\n\bFrom network {}, {} communities are discovered.".format(network_id,num_comm))
    
    temp = []
    for i in range(num_comm):
        a = [temp.append(['Community '+str(i),name]) for name, comm in partition.items() if comm == i]
    df = pd.DataFrame(temp,columns=['Community #','Characters'])
    print(df.describe())
        
    
    # plot   
    plt.hist(list(partition.values()),bins=range(num_comm+1),align='left',rwidth=0.8,alpha=0.5,picker=5)
    plt.ylabel("Number of characters")
    labels = [ "Community "+str(i+1) for i in range(num_comm)]
    plt.xticks(range(num_comm),labels,rotation=90)
    
    # twin x
    ax_x = ax.twinx()
    _ = Counter(list(partition.values()))
    s2 = [_[key]/df.shape[0] for key in sorted(_.keys())]
    ax_x.plot(range(num_comm), s2, '-ro')
    ax_x.set_ylabel('%', color='r')
    ax_x.tick_params('y', colors='r')

    
    return partition,num_comm,df


fig = plt.figure(figsize=(10,8)) 
ax1 = plt.subplot(121)
partition1,num_comm1,df1 = find_community(G1,1,ax1)
ax1.axis([-1,10,0,80])
ax2 = plt.subplot(122)
partition5,num_comm5,df5 = find_community(G5,5,ax2)
ax2.axis([-1,10,0,80])

'''
# Interactivity
def onclick(event):
    index_bar = np.round(event.xdata)
    
    if event.inaxes == ax1: 
        names = [name for name, comm in partition1.items() if comm == index_bar]
        ax1.set_title('aaa {}'.format( ','.join(names)))
    else:
        names = [name for name, comm in partition5.items() if comm == index_bar]
        ax2.set_title('aaa {}'.format(','.join(names)))
        
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
''' 
fig.tight_layout()    
plt.show()



# # ****** Analysis of communities *****
# In Book5, more characters and communities are involved. 
# However, considering the size of communities, only 4 to 5 communites are in dominance. 
# 
# Around half of the characters (77/187) are survived, while 110 people in Book 1 died. 
# 

# In[144]:

from matplotlib_venn import venn2, venn2_circles

plt.figure()
venn2([set(df1['Characters']), set(df5['Characters'])], ('Characters from Book 1', 'Characters from Book 2'), alpha= 0.4)
plt.show()


# In[295]:


def plot_merge_books():
    combined_df = pd.merge(df1,df5,how='outer',on='Characters')

    # count the components in communities in Book5 ## - add new characters in Book 5
    import re
    p = re.compile(r'Community ')

    counts = np.zeros((num_comm1+1,num_comm5))
    temp = combined_df[  (combined_df["Community #_x"].isnull()) ]["Community #_y"].value_counts()   
    temp2 = np.zeros((num_comm5))
    for j in temp.index:
        num = int(p.sub('',j))
        temp2[num] = temp.loc[j]
    counts[0,]=temp2

    for i in range(num_comm1):
        temp = combined_df[ (combined_df["Community #_x"]==('Community '+str(i))) ]["Community #_y"].value_counts()
        temp2 = np.zeros((num_comm5))
        for j in temp.index:
            num = int(p.sub('',j))
            temp2[num] = temp.loc[j]
        counts[i+1,]=(temp2)

    stacked_counts = np.cumsum(counts,axis=0)   



    # plot
    fig = plt.figure()

    from cycler import cycler
    plt.gca().set_prop_cycle(cycler('color', [plt.cm.RdBu(i) for i in np.linspace(0, 1, num_comm1+1)]))

    plt.bar(range(num_comm5), counts[0])

    for index,i in enumerate(counts):
        if index==0: 
            continue
        plt.bar(range(num_comm5), i, bottom=stacked_counts[index-1,]) #stacked=True,align='left',rwidth=0.8,alpha=0.5

    plt.ylabel("Number of characters")
    labels = [ "Community "+str(i+1) for i in range(num_comm5)]
    plt.xticks(range(num_comm5),labels,rotation=90)
    legend_labels = ["Community "+str(i+1)+" in Book 1" for i in range(num_comm1-1,-1,-1) ] 
    legend_labels.append('New')
    legned_labels=legend_labels.reverse()
    plt.legend(legend_labels)
    plt.axis([-1,11,0,80])
    plt.show()



# Communities 3 and 6 top the two of the largest communites in Book 1. However, they are seperated in Book 5. The majority of Community 6 stay in Community 4, the thrid largest group in Book 5.
# 
# 
# Communities 4 in Book1 is splited into two parts that most of the members move to Commnunity 3 in Book 5 firming the larget community. 
# 
# 
# Community 2 keeps most of its integraty and increases to a more powerful group. So does Community 5.
# 
# 
# 

# # Centrality

# In[273]:

def centrality(G):
    degree = nx.degree_centrality(G)
    weighted_degree = nx.degree(G, nbunch=None, weight='weight')
    eig = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight='weight')
    page = nx.pagerank(G)
    between = nx.betweenness_centrality(G,weight='weight')
    closeness = nx.closeness_centrality(G)
    
    te=dict()
    for key,value in degree.items():     
        te[key]=[degree[key],weighted_degree[key],eig[key],page[key],between[key],closeness[key]]
    
    cen = pd.DataFrame.from_dict(te,orient='index')
    cen = cen.rename(columns={0:'Degree',1:'Weighted_degree',2:'Eigen',3:'Pagerank',4:'Betweenness',5:'Closeness'})
    return cen

cen1 = centrality(G1)
cen5 = centrality(G5)
cen1.head()


# # Plot the network

# In[308]:

# plot function
import matplotlib as mpl


def plot_network(G,num_comm,partition,centrality_type,cen):
    plt.figure(figsize=(8,5))
    _G = G.copy()
    
    # remove edges between communities
    removed_edges = []
    for i in _G.edges():
        if partition[i[0]] != partition[i[1]]:
            _G.remove_edge(i[0],i[1])
            removed_edges.append(i) 
    
    # layout
    pos = nx.spring_layout(_G)
    color_list =  plt.cm.RdBu_r(np.linspace(0, 1, num_comm))
    cmap = mpl.colors.ListedColormap(color_list, name='my_name')
    
    for i in range(num_comm):
        characters = [name for name, comm in partition.items() if comm == i]
        ns1 = [ np.log10(cen.loc[n][centrality_type]+1)*40  for n in characters ]
        nx.draw_networkx_nodes(_G, pos, nodelist=characters, node_size=ns1, node_color=cmap(i), alpha=0.6)
        _edges = _G.edges(characters)
        _weight = [ _G[u][v]['weight']/10 for (u,v) in _edges ]
        nx.draw_networkx_edges(_G, pos, edgelist=_edges, width=_weight, alpha=0.5)    
        # only label points with high values: above 95% or top 3
        # top 3
        #ort(ns1)
        # percentil
        for k in characters:
            if np.log10(cen.loc[k][centrality_type]+1)*30 > np.percentile(ns1,80):
                x,y = pos[k]
                plt.text(x,y,s=k,fontsize=8)

    plt.title('Click node to show the character name')
    # Interactivity
    def onclick(event):
        dis = [np.sqrt((event.ydata-value[1])**2+(event.xdata-value[0])**2) for key,value in pos.items()]
        if np.min(dis)<0.1:
            selected_ind = np.argmin(dis)
            char = list(pos.keys())[selected_ind]
            plt.gca().set_title("Selected character: {} -- Community {}".format(char,partition[char]+1))
            
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    # plot removed edges between communities
    _weight = [ G[u][v]['weight']/10 for (u,v) in removed_edges ]
    nx.draw_networkx_edges(_G,pos,edgelist=removed_edges,width=_weight, alpha=0.5, edge_color='lightgrey')   
    
    plt.tick_params(axis='both', which='both', bottom='off', top='off',  labelbottom='off',  labelleft='off', left='off')
    plt.show()
  


# In[309]:

# !!! remove attributes that have high correlations    
# only keep the columns with low coef: Eigen  Pagerank  Betweenness Closeness
df_corr = pd.DataFrame(cen1.corr(),columns=cen1.columns)
print(df_corr.head())


centrality_type = 'Weighted_degree'
plot_network(G1,num_comm1,partition1,centrality_type,cen1)




# # *** Network plot -Book 1
# 
# In Book 1, ordered by the size of communities
# 
# <li> Community 3: Cersei, Eddard, and Robert
# <li> Community 6: House Stark 
# <li> Community 1: Lannisters 
# <li> Community 4: Jon-Snow 
# <li> Community 5: Joffrey and Stark sisters 
# <li> Community 2: Daenerys Targarye 
# <li> Community 7: - 
# <li> Community 8: -
# 
# Compared to Book 5, altough the sizes of communities are smaller in Book 1, there are more connections between groups. These connections are believed to drive the changes of powers which will eventually decide who will take the throne. 
# 
# Cersei, Eddard, and Robert are the centers in the early story that they form the largest group, Community 3 in Book 1 followed by House Stark from Community 6. 
# 
# Here Jon-Snow is a bastered child who is isolated from the family as the center of Community 4. This is the exact group whose power is well preserved and grows into the most powerful one in Book 5 (Community 3). 
# 
# Lannisters are watching by side of the situations to take the throne. They are the third larget group in Book 1. Unfortunately, their power is splitted into two parts in Book 5 (Community 2 and 6). 
# 
# In the very beginning, Daenerys Targaryen's influence, in Community 2, is not fully developed. 
# 
# In summary, it is still too early to say who will be the winner.
# 
# 
# 

# In[297]:

plot_network(G5,num_comm5,partition5,centrality_type,cen5)


# # Network plot -Book 5
# 
# In Book 5, ordered by the size of communities
# 
# <li> Community 3: Jon-Snow
# <li> Community 1: Daenerys Targarye
# <li> Community 4: Theon-Greyjoy 
# <li> Community 6: Cersei 
# <li> Community 2: Tyrion 
# <li> Community 7: Asha-Greyjoy 
# <li> Community 5: Victarion-Greyjoy
# <li> Community 8: Bran-Stark
# <li> Community 9: -
# <li> Community 10: -
# 
# 
# 
# After the death of Eddard and Robert, the King's power is splited apart. Community 2 cerntered at Tyrion and 6 centered at Cersei preserve most of the former powers.  
# 
# Meanwhile, the Starks lost their land, but are rising up in Community 4 and 8. They also keep close connections with Jon-Snow in Community 3. 
# 
# House Lannisters is seperated and gather around two centers: Cersei and Tyrion. 
#  
# Jon-Snow not only keeps his power but grow into the largest group. 
# 
# Daenerys Targarye is a big surprise. She turns into the second most powerful people. It shows a densy connections between her and Tyrion (Community 2).  
# 
# 
# 

# In[296]:

plot_merge_books()


# In[ ]:





###############################################################

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def get_indexes(keys, sorted_list):
    return [sorted_list.index(i) for i in keys]
    
def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(len(backitems)-1,-1,-1)]

###############################################################
# network analysis on Book 1
###########################################################

cen = cen1
num_comm = num_comm1
partition = partition1


sorted_name_degree = sort_by_value(cen[0])
sorted_value_degree = sorted(cen[0].values(),reverse=True)

sorted_name_weighted_degree = sort_by_value(cen[1])
sorted_value_weighted_degreee = sorted(cen[1].values(),reverse=True)

sorted_name_eig = sort_by_value(cen[2])
sorted_value_eig = sorted(cen[2].values(),reverse=True)

sorted_name_page = sort_by_value(cen[3])
sorted_value_page = sorted(cen[3].values(),reverse=True)

sorted_name_between = sort_by_value(cen[4])
sorted_value_between = sorted(cen[4].values(),reverse=True)

sorted_name_closeness = sort_by_value(cen[5])
sorted_value_closeness = sorted(cen[5].values(),reverse=True)


rank_all = {}
rank_all['degree']=sorted_name_degree
rank_all['weight_degree']=sorted_name_weighted_degree
rank_all['eig']=sorted_name_eig
rank_all['page']=sorted_name_page
rank_all['between']=sorted_name_between
rank_all['closeness']=sorted_name_closeness


#########################################################
#community 0-num_comm

for i in range(0,num_comm):
    people_in_community_0 = get_keys(partition, i) 
    
    rank_all = {}
            #{'degree':0,'weight_degree':0,'eig':0,'page':0,'between':0,'closeness':0,'community':0}
    rank_all['degree']=get_indexes(people_in_community_0,sorted_name_degree)
    rank_all['weight_degree']=get_indexes(people_in_community_0,sorted_name_weighted_degree)
    rank_all['eig']=get_indexes(people_in_community_0,sorted_name_eig)
    rank_all['page']=get_indexes(people_in_community_0,sorted_name_page)
    rank_all['between']=get_indexes(people_in_community_0,sorted_name_between)
    rank_all['closeness']=get_indexes(people_in_community_0,sorted_name_closeness)
    
    
    #plot line
    x=range(0,len(people_in_community_0))
    plt.figure(figsize=(20,20))
    plt.plot(x, rank_all['degree'], color = 'blue', linewidth = 2, marker = 'o',label='degree')
    plt.plot(x, rank_all['weight_degree'], color = 'red', linewidth = 2, marker = 'o',label='weight_degree')
    plt.plot(x, rank_all['eig'], color = 'yellow', linewidth = 2, marker = 'o',label='eig')
    plt.plot(x, rank_all['page'], color = 'black', linewidth = 2, marker = 'o',label='page')
    plt.plot(x, rank_all['between'], color = 'c', linewidth = 2, marker = 'o',label='between')
    plt.plot(x, rank_all['closeness'], color = 'm', linewidth = 2, marker = 'o',label='closeness')
    plt.ylabel('centrality rank')
    #plt.xlabel('book1_people_in_community_'+str(i))
    plt.xticks(x, people_in_community_0, rotation=90)  
    plt.legend(loc='lower right')
    plt.savefig("book1_people_in_community_"+str(i)+"_centrality.png")  
    plt.show()



###############################################################

cen = cen5
num_comm = num_comm5
partition = partition5


sorted_name_degree = sort_by_value(cen[0])
sorted_value_degree = sorted(cen[0].values(),reverse=True)

sorted_name_weighted_degree = sort_by_value(cen[1])
sorted_value_weighted_degreee = sorted(cen[1].values(),reverse=True)

sorted_name_eig = sort_by_value(cen[2])
sorted_value_eig = sorted(cen[2].values(),reverse=True)

sorted_name_page = sort_by_value(cen[3])
sorted_value_page = sorted(cen[3].values(),reverse=True)

sorted_name_between = sort_by_value(cen[4])
sorted_value_between = sorted(cen[4].values(),reverse=True)

sorted_name_closeness = sort_by_value(cen[5])
sorted_value_closeness = sorted(cen[5].values(),reverse=True)


rank_all = {}
rank_all['degree']=sorted_name_degree
rank_all['weight_degree']=sorted_name_weighted_degree
rank_all['eig']=sorted_name_eig
rank_all['page']=sorted_name_page
rank_all['between']=sorted_name_between
rank_all['closeness']=sorted_name_closeness


#########################################################
#community 0-num_comm

for i in range(0,num_comm):
    people_in_community_0 = get_keys(partition, i) 
    
    rank_all = {}
            #{'degree':0,'weight_degree':0,'eig':0,'page':0,'between':0,'closeness':0,'community':0}
    rank_all['degree']=get_indexes(people_in_community_0,sorted_name_degree)
    rank_all['weight_degree']=get_indexes(people_in_community_0,sorted_name_weighted_degree)
    rank_all['eig']=get_indexes(people_in_community_0,sorted_name_eig)
    rank_all['page']=get_indexes(people_in_community_0,sorted_name_page)
    rank_all['between']=get_indexes(people_in_community_0,sorted_name_between)
    rank_all['closeness']=get_indexes(people_in_community_0,sorted_name_closeness)
    
    
    #plot line
    x=range(0,len(people_in_community_0))
    plt.figure(figsize=(20,20))
    plt.plot(x, rank_all['degree'], color = 'blue', linewidth = 2, marker = 'o',label='degree')
    plt.plot(x, rank_all['weight_degree'], color = 'red', linewidth = 2, marker = 'o',label='weight_degree')
    plt.plot(x, rank_all['eig'], color = 'yellow', linewidth = 2, marker = 'o',label='eig')
    plt.plot(x, rank_all['page'], color = 'black', linewidth = 2, marker = 'o',label='page')
    plt.plot(x, rank_all['between'], color = 'c', linewidth = 2, marker = 'o',label='between')
    plt.plot(x, rank_all['closeness'], color = 'm', linewidth = 2, marker = 'o',label='closeness')
    plt.ylabel('centrality rank')
    #plt.xlabel('people_in_community_'+str(i))
    plt.xticks(x, people_in_community_0, rotation=90)  
    plt.legend(loc='lower right')
    plt.savefig("book5_people_in_community_"+str(i)+"_centrality.png")  
    plt.show()



#rank_all = {} #dict.fromkeys(G.nodes(),{})
#for item in G.nodes():
#    rank_all[item] = {'degree':0,'weight_degree':0,'eig':0,'page':0,'between':0,'closeness':0,'community':0}
#    new_index = sorted_name_degree.index(item)
#    rank_all[item]['degree']=new_index
#    new_index = sorted_name_weighted_degree.index(item)
#    rank_all[item]['weight_degree']=new_index
#    new_index = sorted_name_eig.index(item)
#    rank_all[item]['eig']=new_index
#    new_index = sorted_name_page.index(item)
#    rank_all[item]['page']=new_index
#    new_index = sorted_name_between.index(item)
#    rank_all[item]['between']=new_index
#    new_index = sorted_name_closeness.index(item)
#    rank_all[item]['closeness']=new_index
#    rank_all[item]['community']=partition[item]


##################################################################
# comparison
# Lannister-Tyrion, House Targaryen, Jon Snow, Lannister-Cersei, House Stark, House Greyjoy
#â€¢	The size of the community 41,61,78,45,8,77
#â€¢	The average weighted degree centrality 143.56,154.25,155.48,159.2,95.62,172.57
#â€¢	The average eigenvector centrality 87.61,52.03,189.68,157.08,261.87,225.23
#â€¢	The average PageRank centrality 153.98,170.10,160.14,140.4,127.37,164.12
#â€¢	The average betweenness centrality 144.22,168.72,154.35,161.46,165.5,154.77
#â€¢	The average closeness centrality 125.80,165.80,142.99,147.8,228.12,172.57
##################################################################


community_no = [1,0,2,5,7] #6,4,3

#1. Lannister-Tyrion
for i in community_no: 
    people_in_community_0 = get_keys(partition, i) 
    size_comm = len(get_keys(partition5, i)) 
    
    
    rank_all = {}
    rank_all['weight_degree']=get_indexes(people_in_community_0,sorted_name_weighted_degree)
    rank_all['eig']=get_indexes(people_in_community_0,sorted_name_eig)
    rank_all['page']=get_indexes(people_in_community_0,sorted_name_page)
    rank_all['between']=get_indexes(people_in_community_0,sorted_name_between)
    rank_all['closeness']=get_indexes(people_in_community_0,sorted_name_closeness)
    
    print(size_comm)
    print(np.average(rank_all['weight_degree']))
    print(np.average(rank_all['eig'])) 
    print(np.average(rank_all['page'])) 
    print(np.average(rank_all['between']))
    print(np.average(rank_all['closeness']))


    
#6. House Greyjoy #6,4,3
size_comm = len(get_keys(partition5, 6)) + len(get_keys(partition5, 4)) + len(get_keys(partition5, 3)) 


people_in_community_0 = get_keys(partition, 6) + get_keys(partition, 4) + get_keys(partition, 3)


rank_all = {}
rank_all['weight_degree']=get_indexes(people_in_community_0,sorted_name_weighted_degree)
rank_all['eig']=get_indexes(people_in_community_0,sorted_name_eig)
rank_all['page']=get_indexes(people_in_community_0,sorted_name_page)
rank_all['between']=get_indexes(people_in_community_0,sorted_name_between)
rank_all['closeness']=get_indexes(people_in_community_0,sorted_name_closeness)


np.average(rank_all['weight_degree']) 
np.average(rank_all['eig']) 
np.average(rank_all['page']) 
np.average(rank_all['between']) 
np.average(rank_all['closeness']) 


#################################

# Lannister-Tyrion, House Targaryen, Jon Snow, Lannister-Cersei, House Stark, House Greyjoy
#â€¢	The size of the community 41,61,78,45,8,77
#â€¢	The average weighted degree centrality 143.56,154.25,155.48,159.2,95.62,172.57
#â€¢	The average eigenvector centrality 87.61,52.03,189.68,157.08,261.87,225.23
#â€¢	The average PageRank centrality 153.98,170.10,160.14,140.4,127.37,164.12
#â€¢	The average betweenness centrality 144.22,168.72,154.35,161.46,165.5,154.77
#â€¢	The average closeness centrality 125.80,165.80,142.99,147.8,228.12,172.57


#plot line
x=range(0,6)
plt.plot(x, [41,61,78,45,8,77], color = 'blue', linewidth = 2, marker = 'o',label='size')
plt.plot(x, [143.56,154.25,155.48,159.2,95.62,172.57], color = 'red', linewidth = 2, marker = 'o',label='weight_degree')
plt.plot(x, [87.61,52.03,189.68,157.08,261.87,225.23], color = 'yellow', linewidth = 2, marker = 'o',label='eig')
plt.plot(x, [153.98,170.10,160.14,140.4,127.37,164.12], color = 'black', linewidth = 2, marker = 'o',label='page')
plt.plot(x, [144.22,168.72,154.35,161.46,165.5,154.77], color = 'c', linewidth = 2, marker = 'o',label='between')
plt.plot(x, [125.80,165.80,142.99,147.8,228.12,172.57], color = 'm', linewidth = 2, marker = 'o',label='closeness')
plt.ylabel('average rank')
plt.xticks(x, ['Lannister-Tyrion', 'House Targaryen', 'Jon Snow', 'Lannister-Cersei', 'House Stark', 'House Greyjoy'], rotation=90)  
plt.legend(loc='lower right')
plt.savefig("compare_ave_indexes.png")  
plt.show()

ave_size = [41,61,78,45,8,77]
ave_wdegree = [143.56,154.25,155.48,159.2,95.62,172.57]
ave_eig = [87.61,52.03,189.68,157.08,261.87,225.23]
ave_page = [153.98,170.10,160.14,140.4,127.37,164.12]
ave_between = [144.22,168.72,154.35,161.46,165.5,154.77]
ave_closeness = [125.80,165.80,142.99,147.8,228.12,172.57]

ave_size_nor = 1-(ave_size-np.min(ave_size))/(np.max(ave_size)-np.min(ave_size))
ave_wdegree_nor = (ave_wdegree-np.min(ave_wdegree))/(np.max(ave_wdegree)-np.min(ave_wdegree))
ave_eig_nor = (ave_eig-np.min(ave_eig))/(np.max(ave_eig)-np.min(ave_eig))
ave_page_nor = (ave_page-np.min(ave_page))/(np.max(ave_page)-np.min(ave_page))
ave_between_nor = (ave_between-np.min(ave_between))/(np.max(ave_between)-np.min(ave_between))
ave_closeness_nor = (ave_closeness-np.min(ave_closeness))/(np.max(ave_closeness)-np.min(ave_closeness))

sum_all = ave_size_nor+ave_wdegree_nor+ave_eig_nor+ave_page_nor+ave_between_nor+ave_closeness_nor

x=range(0,6)
plt.plot(x, sum_all, color = 'blue', linewidth = 2, marker = 'o',label='size')
plt.ylabel('sum normalized rank')
plt.xticks(x, ['Lannister-Tyrion', 'House Targaryen', 'Jon Snow', 'Lannister-Cersei', 'House Stark', 'House Greyjoy'], rotation=90)  
plt.legend(loc='lower right')
plt.savefig("compare_sum.png")  
plt.show()


#################################################
# Random Walk
#################################################
A = nx.to_numpy_matrix(G5)
for i in range(0,100):
    B=A/A.sum(axis=1)
    B=B**2
    A=B


Gnew=nx.from_numpy_matrix(A)
Gnew.nodes()

cen_new = centrality(Gnew)

partition_new = find_community(Gnew)
num_comm_new = len(set(partition_new.values()))

#the 0 has found 119
#the 1 has found 62
#the 2 has found 136 

cen = cen_new
num_comm = num_comm_new
partition = partition_new


sorted_name_degree = sort_by_value(cen[0])
sorted_value_degree = sorted(cen[0].values(),reverse=True)

sorted_name_weighted_degree = sort_by_value(cen[1])
sorted_value_weighted_degreee = sorted(cen[1].values(),reverse=True)

sorted_name_eig = sort_by_value(cen[2])
sorted_value_eig = sorted(cen[2].values(),reverse=True)

sorted_name_page = sort_by_value(cen[3])
sorted_value_page = sorted(cen[3].values(),reverse=True)

sorted_name_between = sort_by_value(cen[4])
sorted_value_between = sorted(cen[4].values(),reverse=True)

sorted_name_closeness = sort_by_value(cen[5])
sorted_value_closeness = sorted(cen[5].values(),reverse=True)


name_list = G5.nodes()
rank_all = {}
rank_all['degree']= [name_list[i] for i in sorted_name_degree] 
rank_all['weight_degree']=[name_list[i] for i in sorted_name_weighted_degree] 
rank_all['eig']=[name_list[i] for i in sorted_name_degree]
rank_all['page']=[name_list[i] for i in sorted_name_page]
rank_all['between']=[name_list[i] for i in sorted_name_between]
rank_all['closeness']=[name_list[i] for i in sorted_name_closeness]


for i in range(0,num_comm):
    people_in_community_0 = get_keys(partition, i) 
    
    rank_all = {}
            #{'degree':0,'weight_degree':0,'eig':0,'page':0,'between':0,'closeness':0,'community':0}
    rank_all['degree']=get_indexes(people_in_community_0,sorted_name_degree)
    rank_all['weight_degree']=get_indexes(people_in_community_0,sorted_name_weighted_degree)
    rank_all['eig']=get_indexes(people_in_community_0,sorted_name_eig)
    rank_all['page']=get_indexes(people_in_community_0,sorted_name_page)
    rank_all['between']=get_indexes(people_in_community_0,sorted_name_between)
    rank_all['closeness']=get_indexes(people_in_community_0,sorted_name_closeness)
    
     
    #plot line
    x=range(0,len(people_in_community_0))
    plt.figure(figsize=(20,20))
    plt.plot(x, rank_all['degree'], color = 'blue', linewidth = 2, marker = 'o',label='degree')
    plt.plot(x, rank_all['weight_degree'], color = 'red', linewidth = 2, marker = 'o',label='weight_degree')
    plt.plot(x, rank_all['eig'], color = 'yellow', linewidth = 2, marker = 'o',label='eig')
    plt.plot(x, rank_all['page'], color = 'black', linewidth = 2, marker = 'o',label='page')
    plt.plot(x, rank_all['between'], color = 'c', linewidth = 2, marker = 'o',label='between')
    plt.plot(x, rank_all['closeness'], color = 'm', linewidth = 2, marker = 'o',label='closeness')
    plt.ylabel('centrality rank')
    #plt.xlabel('people_in_community_'+str(i))
    plt.xticks(x, [name_list[ii] for ii in people_in_community_0], rotation=90)  
    plt.legend(loc='lower right')
    plt.savefig("randomWalk_people_in_community_"+str(i)+"_centrality.png")  
    plt.show()




# In[ ]:



