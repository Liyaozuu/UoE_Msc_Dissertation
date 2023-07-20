import numpy as np
import pandas as pd
import random
import math
import statistics as stat
import matplotlib.pyplot as plt
import itertools
import warnings
import os


def mmn_queueing_jsq(n,arrival_rate,service_rate,simulation_time=60,simuseed=math.nan,policy="jsq1"):
    '''
    Simulating the mmn queueing system using the method of JSQ 
    (namely join the shortest queue method), the user needs to provide
    the number of servers (n),arrival rate (lambda) and service rate (mu),
    in case of n >= 2, if mu is provided as a scaler, then all servers
    share the same mu, if mu is provided as a vector, then all servers may have different mu,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    policy:           there are different policies you may choose to implement the jsq system
    {
    jsq1: Join the shortest queue, if tied, choose randomly from the shortest queue (default)
    jsq2: Join the shortest queue, if tied, choose larger mu, if still tied,
          choose randomly from the shortest queue
    jiq1: Join the idle queue randomly, if no idle queue, choose a queue randomly
    jiq2: Join the idle queue with largest mu, if tied, choose randomly,
          if no idle queue, choose a queue randomly
    jiq3: Join the idle queue randomly, if no idle queue, choose a queue with largest mu,
          if tied, choose randomly
    jiq4: Join the idle queue with largest mu, if tied, choose randomly, if no idle queue,
          choose a queue with largest mu, if tied, choose randomly
    jswq: Join the smallest workload queue, based on the expected waiting time, if tied, choose randomly
    }
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''

    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    lamb =  arrival_rate
    
    if isinstance(service_rate, list):
        if len(service_rate) == n:
            mu = service_rate
        elif (n > 1 and len(service_rate) == 1):
            mu = n*service_rate
        else:
            raise Exception("Unmatched cardinality for service rate and n")
    elif isinstance(service_rate, int):
        if n == 1:
            mu = [service_rate]
        elif n > 1:
            mu = n*[service_rate]
    else:
        raise Exception("Please use int or list type for service rate")
    
    mu = np.array(mu)

    def jsq_policy(plc):
        if plc == "jsq1":
            #Join the shortest queue, if tied, choose randomly from the shortest queue (default)
            sa = random.choice(np.where(event_calendar[-1,total_col] == \
                                   min(event_calendar[-1,total_col]))[0])
        elif plc == "jsq2":
            #Join the shortest queue, if tied, choose larger mu, if still tied,
            #choose randomly from the shortest queue
            shortest_queues = np.where(event_calendar[-1,total_col] == \
                                   min(event_calendar[-1,total_col]))[0]
            if len(shortest_queues) == 1:
                sa = shortest_queues[0]
            elif len(shortest_queues) > 1:
                sa = shortest_queues[random.choice(np.where(mu[shortest_queues] == max(mu[shortest_queues]))[0])]
            
        elif plc == "jiq1":
            #Join the idle queue randomly, if no idle queue, choose a queue randomly
            idle_queues = np.where(event_calendar[-1,total_col] == 0)[0]
            if len(idle_queues) > 0:
                sa = random.choice(idle_queues)
            elif len(idle_queues) == 0:
                sa = random.choice(np.arange(n))
            
        elif plc == "jiq2":
            #Join the idle queue with largest mu, if tied, choose randomly,
            #if no idle queue, choose a queue randomly
            idle_queues = np.where(event_calendar[-1,total_col] == 0)[0]
            if len(idle_queues) > 0:
                sa = idle_queues[random.choice(np.where(mu[idle_queues] == max(mu[idle_queues]))[0])]
            elif len(idle_queues) == 0:
                sa = random.choice(np.arange(n))
            
        elif plc == "jiq3":
            #Join the idle queue randomly, if no idle queue, choose a queue with largest mu,
            #if tied, choose randomly
            idle_queues = np.where(event_calendar[-1,total_col] == 0)[0]
            if len(idle_queues) > 0:
                sa = random.choice(idle_queues)
            elif len(idle_queues) == 0:
                sa = random.choice(np.where(mu == max(mu))[0])
            
        elif plc == "jiq4":
            #Join the idle queue with largest mu, if tied, choose randomly, if no idle queue,
            #choose a queue with largest mu, if tied, choose randomly
            idle_queues = np.where(event_calendar[-1,total_col] == 0)[0]
            if len(idle_queues) > 0:
                sa = idle_queues[random.choice(np.where(mu[idle_queues] == max(mu[idle_queues]))[0])]
            elif len(idle_queues) == 0:
                sa = random.choice(np.where(mu == max(mu))[0])
            
        elif plc == "jswq":
            #Join the smallest workload queue, based on the expected waiting time,
            #if tied, choose randomly
            exp_time = (1/mu)*event_calendar[-1,total_col]
            sa = random.choice(np.where(exp_time == min(exp_time))[0])
        
        return sa
    
    # Record the start/queue/finish timing for each arrival
    for i in range(1,n+1):
        exec(f'serverlog_{i} = []')
    timing_table = np.zeros(5)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1)  # keep track of the number customers
    no_completed_service = np.zeros(n+1)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            server_assignment = jsq_policy(policy)
            no_customers[server_assignment] += 1
            arrival_label += 1
            
            eval(f'serverlog_{server_assignment+1}').append(arrival_label)
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            timing_table[-1,4] = server_assignment+1
            
            # If the server is not occupied
            if event_calendar[-1,table_col[server_assignment]] == 0:
                event_calendar[-1,table_col[server_assignment]] += 1
                event_calendar[-1,avail_col[server_assignment]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[server_assignment])
                event_calendar[-1,total_col[server_assignment]] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
            
            # Else if the server is occupied
            elif event_calendar[-1,table_col[server_assignment]] == 1:
                event_calendar[-1,queue_col[server_assignment]] += 1
                event_calendar[-1,total_col[server_assignment]] += 1
            
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            job_completion = eval(f'serverlog_{server_completion+1}')[0]
            no_completed_service[server_completion] += 1
            
            event_calendar[-1,-1] += 1
            
            del eval(f'serverlog_{server_completion+1}')[0]
            timing_table[job_completion,3] = event_calendar[-1,0]
            
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[server_completion])
                event_calendar[-1,queue_col[server_completion]] -= 1
                event_calendar[-1,total_col[server_completion]] -= 1
                
                timing_table[eval(f'serverlog_{server_completion+1}')[0],2] = event_calendar[-1,0]
    
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) +\
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "mu": mu,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat









def queue_length_time_graph(event_calendar,other_stat,gtype="queue",sq_no=math.nan):
    '''
    A function to plot the queue length time graph, provided with an event calendar
    and other_stat which are produced by the mmn_queueing_jsq function
    
    There are x inputs in this function,
    event_calendar:   the event calendar that was produced by the mmn_queueing_jsq function
    other_stat:       the dictionary other_stat that was produced by the mmn_queueing_jsq function
    gtype:            choose the type of graph you need, there are two options, the "queue" type
                      is by default, and it generates the queue length versus time for
                      each server. The "server" type generates the number of servers versus time
                      for each queue length.
    sq_no:            by default, only the server1 or the queue length of 0 will be plotted,
                      depending on the gtype. Unless specify it here, only scaler or list type input
                      are accepted.
                      
    The outputs are the corresponding requested graphs.
    
    code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    # Preprocessing
    if isinstance(sq_no, int):
        if gtype == "queue":
            if sq_no > other_stat["n"] or sq_no < 1:
                raise Exception("request out of range")
            else:
                sq_no = [sq_no]
        elif gtype == "server":
            if sq_no < 0:
                raise Exception("only positive queue length exists")
            else:
                sq_no = [sq_no]
        else:
            raise TypeError("Please use queue or server for gtype")
    elif isinstance(sq_no, list) or isinstance(sq_no, np.ndarray):
        if gtype == "queue":
            if max(sq_no) > other_stat["n"] or min(sq_no) < 1:
                raise Exception("request out of range")
            else:
                pass
        elif gtype == "server":
            if min(sq_no) < 0:
                raise Exception("only positive queue length exists")
            else:
                pass
        else:
            raise TypeError("Please use queue or server for gtype")
    elif math.isnan(sq_no) == True:
        if gtype == "queue":
            sq_no = [1]
        elif gtype == "server":
            sq_no = [0]
        else:
            raise TypeError("Please use queue or server for gtype")
    else:
        raise Exception("Please use int, list or np.array type for sq_no")
    
    # If gtype = "queue"
    if gtype == "queue":
        for i in sq_no:
            plt.figure(i)
            plt.plot(event_calendar["Time"],event_calendar[f"Server{i} Queue"])
            plt.title(f"Server{i}")
            plt.xlabel("Time")
            plt.ylabel("Queue length")
    else:
        for i in sq_no:
            plt.figure(i)
            plt.plot(event_calendar["Time"],(event_calendar.iloc[:,other_stat["queue_col"]] == i).sum(axis=1).to_numpy())
            plt.title(f"Queue length of {i}")
            plt.xlabel("Time")
            plt.ylabel("The number of servers")






def mmn_queueing_jsq_weibull(n,arrival_rate,scale,shape,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the method of JSQ 
    (namely join the shortest queue method), the user needs to provide
    the number of servers (n),arrival rate (lambda) and scale/shape,
    in case of n >= 2, if scale/shape is provided as a scaler, then all servers
    share the same scale/shape, if scale/shape is provided as a vector, then all servers may have different scale/shape,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    arrival_rate:     exponential rate of arrival
    scale:            weibull distribution scale
    shape:            weibull distribution shape
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''

    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    lamb =  arrival_rate
    
    if isinstance(scale, list):
        if len(scale) == n:
            scale = scale
        elif (n > 1 and len(scale) == 1):
            scale = n*scale
        else:
            raise Exception("Unmatched cardinality for scale and n")
    elif isinstance(scale, int):
        if n == 1:
            scale = [scale]
        elif n > 1:
            scale = n*[scale]
    else:
        raise Exception("Please use int or list type for scale")
        
    if isinstance(shape, list):
        if len(shape) == n:
            shape = shape
        elif (n > 1 and len(shape) == 1):
            shape = n*shape
        else:
            raise Exception("Unmatched cardinality for shape and n")
    elif isinstance(shape, int):
        if n == 1:
            shape = [shape]
        elif n > 1:
            shape = n*[shape]
    else:
        raise Exception("Please use int or list type for shape")
    
    # Record the start/queue/finish timing for each arrival
    for i in range(1,n+1):
        exec(f'serverlog_{i} = []')
    timing_table = np.zeros(5)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1)  # keep track of the number customers
    no_completed_service = np.zeros(n+1)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            server_assignment = random.choice(np.where(event_calendar[-1,total_col] == \
                                   min(event_calendar[-1,total_col]))[0])
            no_customers[server_assignment] += 1
            arrival_label += 1
            
            eval(f'serverlog_{server_assignment+1}').append(arrival_label)
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            timing_table[-1,4] = server_assignment+1
            
            # If the server is not occupied
            if event_calendar[-1,table_col[server_assignment]] == 0:
                event_calendar[-1,table_col[server_assignment]] += 1
                event_calendar[-1,avail_col[server_assignment]] = event_calendar[-1,0] + \
                                                        scale[server_assignment]*np.random.weibull(shape[server_assignment])
                event_calendar[-1,total_col[server_assignment]] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
            
            # Else if the server is occupied
            elif event_calendar[-1,table_col[server_assignment]] == 1:
                event_calendar[-1,queue_col[server_assignment]] += 1
                event_calendar[-1,total_col[server_assignment]] += 1
            
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            job_completion = eval(f'serverlog_{server_completion+1}')[0]
            no_completed_service[server_completion] += 1
            
            event_calendar[-1,-1] += 1
            
            del eval(f'serverlog_{server_completion+1}')[0]
            timing_table[job_completion,3] = event_calendar[-1,0]
            
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        scale[server_completion]*np.random.weibull(shape[server_completion])
                event_calendar[-1,queue_col[server_completion]] -= 1
                event_calendar[-1,total_col[server_completion]] -= 1
                
                timing_table[eval(f'serverlog_{server_completion+1}')[0],2] = event_calendar[-1,0]
    
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "scale": scale,
        "shape": shape,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat







def mmn_queueing_redundancy_dos(n,d,arrival_rate,service_rate,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and service rate (mu),
    in case of n >= 2, if mu is provided as a scaler, then all servers
    share the same mu, if mu is provided as a vector, then all servers may have different mu,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on start method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(service_rate, list):
        if len(service_rate) == n:
            mu = service_rate
        elif (n > 1 and len(service_rate) == 1):
            mu = n*service_rate
        else:
            raise Exception("Unmatched cardinality for service rate and n")
    elif isinstance(service_rate, int):
        if n == 1:
            mu = [service_rate]
        elif n > 1:
            mu = n*[service_rate]
    else:
        raise Exception("Please use int or list type for service rate")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(5)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            
            # If all servers are occupied
            if sum(redun_server[copy_list]==0) == 0:
                event_calendar[-1,queue_col[copy_list]] += 1
                event_calendar[-1,total_col[copy_list]] += 1
                for i in copy_list:
                    eval(f'redun_queue_{i+1}').append(arrival_label)
                new_history = pd.DataFrame(copy_list).T
                new_history.index = [f"{arrival_label}"]
                copyhistory = copyhistory.append(new_history)
            
            # If only one server is available
            elif sum(redun_server[copy_list]==0) == 1:
                avail_server = copy_list[np.where(redun_server[copy_list]==0)[0][0]]
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[avail_server])
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
                
            # If more than one servers are available
            else:
                avail_server = random.choice(copy_list[np.where(redun_server[copy_list]==0)[0]])
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[avail_server])
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
            
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            event_calendar[-1,-1] += 1
            
            timing_table[redun_server[server_completion],3] = event_calendar[-1,0]
            
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[server_completion])
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                for i in copyhistory.loc[f"{redun_server[server_completion]}"]:
                    event_calendar[-1,queue_col[i]] -= 1
                    event_calendar[-1,total_col[i]] -= 1
                    eval(f'redun_queue_{i+1}').remove(redun_server[server_completion])
                copyhistory = copyhistory.drop(f"{redun_server[server_completion]}")
                no_customers[server_completion] += 1
                
                timing_table[redun_server[server_completion],2] = event_calendar[-1,0]
                timing_table[redun_server[server_completion],4] = server_completion+1
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "mu": mu,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat






def mmn_queueing_redundancy_doc(n,d,arrival_rate,service_rate,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and service rate (mu),
    in case of n >= 2, if mu is provided as a scaler, then all servers
    share the same mu, if mu is provided as a vector, then all servers may have different mu,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on completion method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(service_rate, list):
        if len(service_rate) == n:
            mu = service_rate
        elif (n > 1 and len(service_rate) == 1):
            mu = n*service_rate
        else:
            raise Exception("Unmatched cardinality for service rate and n")
    elif isinstance(service_rate, int):
        if n == 1:
            mu = [service_rate]
        elif n > 1:
            mu = n*[service_rate]
    else:
        raise Exception("Please use int or list type for service rate")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(5)
    queue_record = np.zeros(n)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            new_history = pd.DataFrame(copy_list).T
            new_history.index = [f"{arrival_label}"]
            copyhistory = copyhistory.append(new_history)
            
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            queue_record = np.vstack([queue_record,np.zeros(n)])
            
            for i in copy_list:
                no_customers[i] += 1
                
                # If the server is idle
                if event_calendar[-1,table_col[i]] == 0:
                    event_calendar[-1,table_col[i]] += 1
                    event_calendar[-1,avail_col[i]] = event_calendar[-1,0]+random.expovariate(mu[i])
                    event_calendar[-1,total_col[i]] += 1
                    redun_server[i] = arrival_label
                    
                    queue_record[arrival_label,i] = event_calendar[-1,0]
                    
                # If the server is occupied
                elif event_calendar[-1,table_col[i]] == 1:
                    event_calendar[-1,queue_col[i]] += 1
                    event_calendar[-1,total_col[i]] += 1
                    eval(f'redun_queue_{i+1}').append(arrival_label)
        
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            job_completion = redun_server[server_completion]
            event_calendar[-1,-1] += 1
            
            timing_table[job_completion,2] = queue_record[job_completion,server_completion]
            timing_table[job_completion,3] = event_calendar[-1,0]
            timing_table[job_completion,4] = server_completion+1
            
            # For the completion server
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        random.expovariate(mu[server_completion])
                event_calendar[-1,queue_col[server_completion]] -= 1
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                eval(f'redun_queue_{server_completion+1}').remove(redun_server[server_completion])
            
                queue_record[redun_server[server_completion],server_completion] = event_calendar[-1,0]
            
            # For the other servers in the copylist
            for i in copyhistory.loc[f"{job_completion}"]:
                if i != server_completion:
                    no_customers[i] -= 1
                
                    # If the completed job is undertaken
                    if redun_server[i] == job_completion:
                        
                        # If there is no queue behind
                        if event_calendar[-1,queue_col[i]] == 0:
                            event_calendar[-1,table_col[i]] -= 1
                            event_calendar[-1,avail_col[i]] = math.nan
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = 0

                        # Else if there is a queue behind
                        elif event_calendar[-1,queue_col[i]] > 0:
                            event_calendar[-1,avail_col[i]] = event_calendar[-1,0] + \
                                                                    random.expovariate(mu[i])
                            event_calendar[-1,queue_col[i]] -= 1
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = eval(f'redun_queue_{i+1}')[0]
                            eval(f'redun_queue_{i+1}').remove(redun_server[i])
                            
                            queue_record[redun_server[i],i] = event_calendar[-1,0]
                    
                    # Else if the completed job is not undertaken
                    elif redun_server[i] != job_completion:

                        event_calendar[-1,queue_col[i]] -= 1
                        event_calendar[-1,total_col[i]] -= 1
                        eval(f'redun_queue_{i+1}').remove(job_completion)

            copyhistory = copyhistory.drop(f"{job_completion}")


                
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "mu": mu,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat





def mmn_queueing_redundancy_dos_identical(n,d,arrival_rate,service_rate,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and service rate (mu),
    in case of n >= 2, if mu is provided as a scaler, then all servers
    share the same mu, if mu is provided as a vector, then all servers may have different mu,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on start method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(service_rate, list):
        if len(service_rate) == n:
            mu = service_rate
        elif (n > 1 and len(service_rate) == 1):
            mu = n*service_rate
        else:
            raise Exception("Unmatched cardinality for service rate and n")
    elif isinstance(service_rate, int):
        if n == 1:
            mu = [service_rate]
        elif n > 1:
            mu = n*[service_rate]
    else:
        raise Exception("Please use int or list type for service rate")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(6)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            
            timing_table = np.vstack([timing_table,np.zeros(6)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            timing_table[-1,5] = random.expovariate(1)
            
            # If all servers are occupied
            if sum(redun_server[copy_list]==0) == 0:
                event_calendar[-1,queue_col[copy_list]] += 1
                event_calendar[-1,total_col[copy_list]] += 1
                for i in copy_list:
                    eval(f'redun_queue_{i+1}').append(arrival_label)
                new_history = pd.DataFrame(copy_list).T
                new_history.index = [f"{arrival_label}"]
                copyhistory = copyhistory.append(new_history)
            
            # If only one server is available
            elif sum(redun_server[copy_list]==0) == 1:
                avail_server = copy_list[np.where(redun_server[copy_list]==0)[0][0]]
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        timing_table[arrival_label,5]/mu[avail_server]
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
                
            # If more than one servers are available
            else:
                avail_server = random.choice(copy_list[np.where(redun_server[copy_list]==0)[0]])
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        timing_table[arrival_label,5]/mu[avail_server]
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
            
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            event_calendar[-1,-1] += 1
            
            timing_table[redun_server[server_completion],3] = event_calendar[-1,0]
            
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        timing_table[redun_server[server_completion],5]/mu[server_completion]
                for i in copyhistory.loc[f"{redun_server[server_completion]}"]:
                    event_calendar[-1,queue_col[i]] -= 1
                    event_calendar[-1,total_col[i]] -= 1
                    eval(f'redun_queue_{i+1}').remove(redun_server[server_completion])
                copyhistory = copyhistory.drop(f"{redun_server[server_completion]}")
                no_customers[server_completion] += 1
                
                timing_table[redun_server[server_completion],2] = event_calendar[-1,0]
                timing_table[redun_server[server_completion],4] = server_completion+1
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server','Base_exp1']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]
    
    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "mu": mu,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat






def mmn_queueing_redundancy_doc_identical(n,d,arrival_rate,service_rate,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and service rate (mu),
    in case of n >= 2, if mu is provided as a scaler, then all servers
    share the same mu, if mu is provided as a vector, then all servers may have different mu,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on completion method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(service_rate, list):
        if len(service_rate) == n:
            mu = service_rate
        elif (n > 1 and len(service_rate) == 1):
            mu = n*service_rate
        else:
            raise Exception("Unmatched cardinality for service rate and n")
    elif isinstance(service_rate, int):
        if n == 1:
            mu = [service_rate]
        elif n > 1:
            mu = n*[service_rate]
    else:
        raise Exception("Please use int or list type for service rate")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(6)
    queue_record = np.zeros(n)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            new_history = pd.DataFrame(copy_list).T
            new_history.index = [f"{arrival_label}"]
            copyhistory = copyhistory.append(new_history)
            
            timing_table = np.vstack([timing_table,np.zeros(6)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            timing_table[-1,5] = random.expovariate(1)
            queue_record = np.vstack([queue_record,np.zeros(n)])
            
            for i in copy_list:
                no_customers[i] += 1
                
                # If the server is idle
                if event_calendar[-1,table_col[i]] == 0:
                    event_calendar[-1,table_col[i]] += 1
                    event_calendar[-1,avail_col[i]] = event_calendar[-1,0]+timing_table[arrival_label,5]/mu[i]
                    event_calendar[-1,total_col[i]] += 1
                    redun_server[i] = arrival_label
                    
                    queue_record[arrival_label,i] = event_calendar[-1,0]
                    
                # If the server is occupied
                elif event_calendar[-1,table_col[i]] == 1:
                    event_calendar[-1,queue_col[i]] += 1
                    event_calendar[-1,total_col[i]] += 1
                    eval(f'redun_queue_{i+1}').append(arrival_label)
        
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            job_completion = redun_server[server_completion]
            event_calendar[-1,-1] += 1
            
            timing_table[job_completion,2] = queue_record[job_completion,server_completion]
            timing_table[job_completion,3] = event_calendar[-1,0]
            timing_table[job_completion,4] = server_completion+1
            
            # For the completion server
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,queue_col[server_completion]] -= 1
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                eval(f'redun_queue_{server_completion+1}').remove(redun_server[server_completion])
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        timing_table[redun_server[server_completion],5]/mu[server_completion]
            
                queue_record[redun_server[server_completion],server_completion] = event_calendar[-1,0]
            
            # For the other servers in the copylist
            for i in copyhistory.loc[f"{job_completion}"]:
                if i != server_completion:
                    no_customers[i] -= 1
                
                    # If the completed job is undertaken
                    if redun_server[i] == job_completion:
                        
                        # If there is no queue behind
                        if event_calendar[-1,queue_col[i]] == 0:
                            event_calendar[-1,table_col[i]] -= 1
                            event_calendar[-1,avail_col[i]] = math.nan
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = 0

                        # Else if there is a queue behind
                        elif event_calendar[-1,queue_col[i]] > 0:
                            event_calendar[-1,queue_col[i]] -= 1
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = eval(f'redun_queue_{i+1}')[0]
                            eval(f'redun_queue_{i+1}').remove(redun_server[i])
                            event_calendar[-1,avail_col[i]] = event_calendar[-1,0] + \
                                                                    timing_table[redun_server[i],5]/mu[i]
                            
                            queue_record[redun_server[i],i] = event_calendar[-1,0]
                    
                    # Else if the completed job is not undertaken
                    elif redun_server[i] != job_completion:

                        event_calendar[-1,queue_col[i]] -= 1
                        event_calendar[-1,total_col[i]] -= 1
                        eval(f'redun_queue_{i+1}').remove(job_completion)

            copyhistory = copyhistory.drop(f"{job_completion}")


                
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server','Base_exp1']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "mu": mu,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat





def mmn_queueing_redundancy_dos_weibull(n,d,arrival_rate,scale,shape,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and scale/shape,
    in case of n >= 2, if scale/shape is provided as a scaler, then all servers
    share the same scale/shape, if scale/shape is provided as a vector, then all servers may have different scale/shape,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on start method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    scale:            weibull distribution scale
    shape:            weibull distribution shape
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(scale, list):
        if len(scale) == n:
            scale = scale
        elif (n > 1 and len(scale) == 1):
            scale = n*scale
        else:
            raise Exception("Unmatched cardinality for scale and n")
    elif isinstance(scale, int):
        if n == 1:
            scale = [scale]
        elif n > 1:
            scale = n*[scale]
    else:
        raise Exception("Please use int or list type for scale")
        
    if isinstance(shape, list):
        if len(shape) == n:
            shape = shape
        elif (n > 1 and len(shape) == 1):
            shape = n*shape
        else:
            raise Exception("Unmatched cardinality for shape and n")
    elif isinstance(shape, int):
        if n == 1:
            shape = [shape]
        elif n > 1:
            shape = n*[shape]
    else:
        raise Exception("Please use int or list type for shape")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(5)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            
            # If all servers are occupied
            if sum(redun_server[copy_list]==0) == 0:
                event_calendar[-1,queue_col[copy_list]] += 1
                event_calendar[-1,total_col[copy_list]] += 1
                for i in copy_list:
                    eval(f'redun_queue_{i+1}').append(arrival_label)
                new_history = pd.DataFrame(copy_list).T
                new_history.index = [f"{arrival_label}"]
                copyhistory = copyhistory.append(new_history)
            
            # If only one server is available
            elif sum(redun_server[copy_list]==0) == 1:
                avail_server = copy_list[np.where(redun_server[copy_list]==0)[0][0]]
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        scale[avail_server]*np.random.weibull(shape[avail_server])
                                                        
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
                
            # If more than one servers are available
            else:
                avail_server = random.choice(copy_list[np.where(redun_server[copy_list]==0)[0]])
                event_calendar[-1,table_col[avail_server]] += 1
                event_calendar[-1,avail_col[avail_server]] = event_calendar[-1,0] + \
                                                        scale[avail_server]*np.random.weibull(shape[avail_server])
                event_calendar[-1,total_col[avail_server]] += 1
                redun_server[avail_server] = arrival_label
                no_customers[avail_server] += 1
                
                timing_table[-1,2] = event_calendar[-1,0]
                timing_table[-1,4] = avail_server+1
            
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            event_calendar[-1,-1] += 1
            
            timing_table[redun_server[server_completion],3] = event_calendar[-1,0]
            
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                        scale[server_completion]*np.random.weibull(shape[server_completion])
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                for i in copyhistory.loc[f"{redun_server[server_completion]}"]:
                    event_calendar[-1,queue_col[i]] -= 1
                    event_calendar[-1,total_col[i]] -= 1
                    eval(f'redun_queue_{i+1}').remove(redun_server[server_completion])
                copyhistory = copyhistory.drop(f"{redun_server[server_completion]}")
                no_customers[server_completion] += 1
                
                timing_table[redun_server[server_completion],2] = event_calendar[-1,0]
                timing_table[redun_server[server_completion],4] = server_completion+1
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "scale": scale,
        "shape": shape,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat





def mmn_queueing_redundancy_doc_weibull(n,d,arrival_rate,scale,shape,simulation_time=60,simuseed=math.nan):
    '''
    Simulating the mmn queueing system using the redundancy method, the user needs to provide
    the number of servers (n), the number of copies to send (d), arrival rate (lambda) and scale/shape,
    in case of n >= 2, if scale/shape is provided as a scaler, then all servers
    share the same scale/shape, if scale/shape is provided as a vector, then all servers may have different scale/shape,
    the simulation time is by default 60 units, and for reproducibility purposes, the user
    can define simulation seed, by default the seed value is not set.
    
    Using the delete on completion method.
    
    There are 6 inputs in this function,
    n:                the number of servers, a positive finite integer
    d:                the number of copies to send
    arrival_rate:     exponential rate of arrival
    service_rate:     exponential rate of service time
    simulation_time:  the simulation time, a positive finite number
    simuseed:         the simulation seed number
    
    
    There are 8 outputs in this function,
    event_calendar:   the event calendar for the whole simulation, containing all important
                      information needed (current time, next customer arrival time, finish time,
                      server occupancy variable, server next available time, server queue length,
                      server total, grand queue length and grand total)
    utiliz:           the utilization for each server
    ans:              the average number in the system for each server and grand average
    aql:              the average queue length for each server and grand average
    awt:              the average waiting time for each customer and grand average
    act:              the average cycle time for each customer and grand average
    no_cust_arr_comp: the table illustrating the number of arrived customers and the
                      the number of completed services for each server and grand total
    other_stat:       return other calculated statistics, for plotting purposes
    
    Code writer: Yaozu Li
    uoe S.No: 2298215
    
    '''
    
    if math.isnan(simuseed) == True:
        pass
    else:
        random.seed(simuseed)
        np.random.seed(simuseed)

    if not type(n) is int:
        raise TypeError("n needs to be integer")
    if n < 0:
        raise Exception("n needs to be positive")
    
    if d < 1 or d > n:
        raise Exception("d needs to be in the range 1 <= d <= n")
    
    lamb =  arrival_rate
    
    if isinstance(scale, list):
        if len(scale) == n:
            scale = scale
        elif (n > 1 and len(scale) == 1):
            scale = n*scale
        else:
            raise Exception("Unmatched cardinality for scale and n")
    elif isinstance(scale, int):
        if n == 1:
            scale = [scale]
        elif n > 1:
            scale = n*[scale]
    else:
        raise Exception("Please use int or list type for scale")
        
    if isinstance(shape, list):
        if len(shape) == n:
            shape = shape
        elif (n > 1 and len(shape) == 1):
            shape = n*shape
        else:
            raise Exception("Unmatched cardinality for shape and n")
    elif isinstance(shape, int):
        if n == 1:
            shape = [shape]
        elif n > 1:
            shape = n*[shape]
    else:
        raise Exception("Please use int or list type for shape")
    
    # Simulating the Redundancy queues
    redun_server = np.zeros(n,dtype=int)
    for i in range(1,n+1):
        exec(f'redun_queue_{i} = []')
    copyhistory = pd.DataFrame(np.zeros(d,dtype=int)).T
    
    # Record the start/queue/finish timing for each arrival
    timing_table = np.zeros(5)
    queue_record = np.zeros(n)
    
    # Simulating the event calendar
    table_col = np.arange(n)*4+3
    avail_col = np.arange(n)*4+4
    queue_col = np.arange(n)*4+5
    total_col = np.arange(n)*4+6
    compare_col = np.insert(avail_col,0,[1,2])
    
    event_calendar = np.zeros([2,4+4*n])
    event_calendar[-1,1] = random.expovariate(lamb)
    event_calendar[-1,2] = simulation_time
    event_calendar[-1,avail_col] = math.nan
    
    no_customers = np.zeros(n+1,dtype=int)  # keep track of the number customers
    no_completed_service = np.zeros(n+1,dtype=int)   # keep track of the number of completed services
    arrival_label = 0    # keep track on each arrival's label
    
    while event_calendar[-1,0] < simulation_time:
        event_calendar = np.vstack([event_calendar,event_calendar[-1,:]])
        event_calendar[-1,0] = min(event_calendar[-2,compare_col])
        
        # If the next event is an arrival
        if event_calendar[-1,0] == event_calendar[-2,1]:
            event_calendar[-1,1] = event_calendar[-1,0]+random.expovariate(lamb)
            copy_list = np.random.choice(np.arange(n),size=d,replace=False)
            arrival_label += 1
            new_history = pd.DataFrame(copy_list).T
            new_history.index = [f"{arrival_label}"]
            copyhistory = copyhistory.append(new_history)
            
            timing_table = np.vstack([timing_table,np.zeros(5)])
            timing_table[-1,0] = arrival_label
            timing_table[-1,1] = event_calendar[-1,0]
            queue_record = np.vstack([queue_record,np.zeros(n)])
            
            for i in copy_list:
                no_customers[i] += 1
                
                # If the server is idle
                if event_calendar[-1,table_col[i]] == 0:
                    event_calendar[-1,table_col[i]] += 1
                    event_calendar[-1,avail_col[i]] = event_calendar[-1,0] + \
                                                      scale[i]*np.random.weibull(shape[i])
                    event_calendar[-1,total_col[i]] += 1
                    redun_server[i] = arrival_label
                    
                    queue_record[arrival_label,i] = event_calendar[-1,0]
                    
                # If the server is occupied
                elif event_calendar[-1,table_col[i]] == 1:
                    event_calendar[-1,queue_col[i]] += 1
                    event_calendar[-1,total_col[i]] += 1
                    eval(f'redun_queue_{i+1}').append(arrival_label)
        
        # Else if the next event is the simulation termination
        elif event_calendar[-1,0] == simulation_time:
            pass
        
        # Else if the next event is a service completion
        else:
            server_completion = np.where(event_calendar[-2,avail_col] == event_calendar[-1,0])[0][0]
            no_completed_service[server_completion] += 1
            job_completion = redun_server[server_completion]
            event_calendar[-1,-1] += 1
            
            timing_table[job_completion,2] = queue_record[job_completion,server_completion]
            timing_table[job_completion,3] = event_calendar[-1,0]
            timing_table[job_completion,4] = server_completion+1
            
            # For the completion server
            # If there is no queue behind
            if event_calendar[-1,queue_col[server_completion]] == 0:
                event_calendar[-1,table_col[server_completion]] -= 1
                event_calendar[-1,avail_col[server_completion]] = math.nan
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = 0
            
            # Else if there is a queue behind
            elif event_calendar[-1,queue_col[server_completion]] > 0:
                event_calendar[-1,avail_col[server_completion]] = event_calendar[-1,0] + \
                                                    scale[server_completion]*np.random.weibull(shape[server_completion])
                event_calendar[-1,queue_col[server_completion]] -= 1
                event_calendar[-1,total_col[server_completion]] -= 1
                redun_server[server_completion] = eval(f'redun_queue_{server_completion+1}')[0]
                eval(f'redun_queue_{server_completion+1}').remove(redun_server[server_completion])
            
                queue_record[redun_server[server_completion],server_completion] = event_calendar[-1,0]
            
            # For the other servers in the copylist
            for i in copyhistory.loc[f"{job_completion}"]:
                if i != server_completion:
                    no_customers[i] -= 1
                
                    # If the completed job is undertaken
                    if redun_server[i] == job_completion:
                        
                        # If there is no queue behind
                        if event_calendar[-1,queue_col[i]] == 0:
                            event_calendar[-1,table_col[i]] -= 1
                            event_calendar[-1,avail_col[i]] = math.nan
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = 0

                        # Else if there is a queue behind
                        elif event_calendar[-1,queue_col[i]] > 0:
                            event_calendar[-1,avail_col[i]] = event_calendar[-1,0] + \
                                                                    scale[i]*np.random.weibull(shape[i])
                            event_calendar[-1,queue_col[i]] -= 1
                            event_calendar[-1,total_col[i]] -= 1
                            redun_server[i] = eval(f'redun_queue_{i+1}')[0]
                            eval(f'redun_queue_{i+1}').remove(redun_server[i])
                            
                            queue_record[redun_server[i],i] = event_calendar[-1,0]
                    
                    # Else if the completed job is not undertaken
                    elif redun_server[i] != job_completion:

                        event_calendar[-1,queue_col[i]] -= 1
                        event_calendar[-1,total_col[i]] -= 1
                        eval(f'redun_queue_{i+1}').remove(job_completion)

            copyhistory = copyhistory.drop(f"{job_completion}")


                
                
    no_customers[-1] = sum(no_customers[:-1])
    no_completed_service[-1] = sum(no_completed_service[:-1])
    event_calendar = pd.DataFrame(event_calendar).tail(-1)
    event_calendar.columns = ['Time','Next Customer','Finish Time'] + \
                    list(itertools.chain.from_iterable([[f"Server{i}", \
                    f"Server{i} Available Time",f"Server{i} Queue",f"Server{i} Total"] for i in range(1,n+1)])) + \
                    ['Live_track']
    event_calendar["Grand Queue"] = event_calendar.iloc[:,queue_col].sum(axis=1)
    event_calendar["Grand Total"] = event_calendar.iloc[:,total_col].sum(axis=1)
    
    timing_table = pd.DataFrame(timing_table)
    timing_table.columns = ['Arrival_label','Start','Queue','Finish','Server']
    timing_table = timing_table[timing_table['Finish'] != 0]
    
    # Concluding performance measures
    # Time dynamic
    time1 = event_calendar.iloc[:-1,0].to_numpy()
    time2 = event_calendar.iloc[1:,0].to_numpy()
    time3 = time2 - time1
    
    # Calculate the utilization for each server
    utiliz_server = event_calendar.iloc[:-1,table_col].to_numpy()
    utiliz_data = ((time3 @ utiliz_server)/simulation_time).round(3)
    utiliz = pd.DataFrame(utiliz_data).transpose()
    utiliz.columns = [f"Server{i} Utilization" for i in range(1,n+1)]
    
    # Calculating the average number in the systems for each server and grand total
    ans_server = event_calendar.iloc[:-1,np.append(total_col,event_calendar.shape[1]-1)].to_numpy()
    ans_data = ((time3 @ ans_server)/simulation_time).round(3)
    ans = pd.DataFrame(ans_data).transpose()
    ans.columns = [f"Server{i} Average number in the system" for i in range(1,n+1)]+["Grand Average number in the system"]
    
    # Calculating the average queue length
    aql_server = event_calendar.iloc[:-1,np.append(queue_col,event_calendar.shape[1]-2)].to_numpy()
    aql_data = ((time3 @ aql_server)/simulation_time).round(3)
    aql = pd.DataFrame(aql_data).transpose()
    aql.columns = [f"Server{i} Average queue length" for i in range(1,n+1)]+["Grand Average queue length"]
    
    # Average waiting time
    awt_data = np.array([])
    for i in range(1,n+1):
        awt_table = timing_table[timing_table['Server']==i]
        if awt_table.shape[0] != 0:
            awt_data = np.append(awt_data,stat.mean(awt_table['Queue']-awt_table['Start']))
        else:
            awt_data = np.append(awt_data,0)
    awt_data = np.append(awt_data,stat.mean(timing_table['Queue']-timing_table['Start']))
    awt = pd.DataFrame(awt_data.round(3)).transpose()
    awt.columns = [f"Server{i} Average waiting time" for i in range(1,n+1)]+["Grand Average waiting time"]
    
    # Average cycle time
    act_data = np.array([])
    for i in range(1,n+1):
        act_table = timing_table[timing_table['Server']==i]
        if act_table.shape[0] != 0:
            act_data = np.append(act_data,stat.mean(act_table['Finish']-act_table['Start']))
        else:
            act_data = np.append(act_data,0)
    act_data = np.append(act_data,stat.mean(timing_table['Finish']-timing_table['Start']))
    act = pd.DataFrame(act_data.round(3)).transpose()
    act.columns = [f"Server{i} Average cycle time" for i in range(1,n+1)]+["Grand Average cycle time"]

    # Show the number of customers arrived and completed
    no_cust_arr_comp = pd.DataFrame(np.vstack([no_customers,no_completed_service]))
    no_cust_arr_comp.columns = [f"Server{i}" for i in range(1,n+1)]+["Grand total"]
    no_cust_arr_comp.index = ["Arrived customers", "Completed services"]
    
    # Find the maximum queue length
    mql = event_calendar.iloc[:,queue_col].max()
    
    # For other statistics
    other_stat = {
        "n": n,
        "lamb": lamb,
        "scale": scale,
        "shape": shape,
        "simulation_time": simulation_time,
        "table_col": table_col,
        "avail_col": avail_col,
        "queue_col": queue_col,
        "total_col": total_col,
        "compare_col": compare_col,
        "time3": time3,
        "mql": mql
    }
    
    # Return the event calendar and all performance measures table
    return event_calendar, utiliz, ans, aql, awt, act, no_cust_arr_comp, other_stat





def weibull_scale_calculator(desired_mean,shape):
    '''
    Please enter the desired mean for each server
    '''
    return desired_mean/math.gamma(1+1/shape)



def system_compare_exp(system1,system2):
    '''
    Please input the outputs of the queueing function of the two systems.
    '''
    #sys1_time = np.arange(0,system1[7]['simulation_time']+1,system1[7]['simulation_time']/20)
    #sys2_time = np.arange(0,system2[7]['simulation_time']+1,system2[7]['simulation_time']/20)
    
    if system1[7]['simulation_time'] != system2[7]['simulation_time']:
        raise Exception("the two system should have same simulation time")

    aaa = system1[0].iloc[np.where(system1[0]["Time"][1:].to_numpy() != system1[0]["Next Customer"][:-1].to_numpy())[0][:-1]+1,:]
    bbb = system2[0].iloc[np.where(system2[0]["Time"][1:].to_numpy() != system2[0]["Next Customer"][:-1].to_numpy())[0][:-1]+1,:]

    plt.plot(aaa['Time'],aaa['Live_track'],label='system1')
    plt.plot(bbb['Time'],bbb['Live_track'],label='system2')
    plt.title('Comparison of the two systems')
    plt.xlabel('Time')
    plt.ylabel('# Customers')
    plt.legend()

    sys1_data = np.array([])
    for i in range(1,system1[7]['simulation_time']+1):
        sys1_data=np.append(sys1_data,aaa[aaa['Time']<i].iloc[-1,-3])
    sys2_data = np.array([])
    for i in range(1,system2[7]['simulation_time']+1):
        sys2_data=np.append(sys2_data,bbb[bbb['Time']<i].iloc[-1,-3])
    diff = sys1_data - sys2_data
    
    result = np.array([sum(diff>0)/system1[7]['simulation_time'],\
                       sum(diff==0)/system1[7]['simulation_time'],\
                       sum(diff<0)/system1[7]['simulation_time']])

    # Forward order
    if result[0] - result[2] > 0.75:
        print("System 1 is significantly better than system 2")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    elif 0.75 >= result[0] - result[2] > 0.55:
        print("System 1 is substantially better than system 2")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    elif 0.55 >= result[0] - result[2] > 0.35:
        print("System 1 is moderately better than system 2")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    elif 0.35 >= result[0] - result[2] > 0.15:
        print("System 1 is slightly better than system 2")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    
    # 5/5
    elif 0.15 >= result[0] - result[2] > -0.15:
        print("The 2 systems have almost the same performance")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    
    # reverse order
    elif -0.15 >= result[0] - result[2] > -0.35:
        print("System 2 is slightly better than system 1")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]} of the time")
    elif -0.35 >= result[0] - result[2] > -0.55:
        print("System 2 is moderately better than system 1")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    elif -0.55 >= result[0] - result[2] > -0.75:
        print("System 2 is substantially better than system 1")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
    elif -0.75 >= result[0] - result[2]:
        print("System 2 is significantly better than system 1")
        print(f"System 1 is better on {result[0]*100}% of the time")
        print(f"System 2 is better on {result[2]*100}% of the time")
























