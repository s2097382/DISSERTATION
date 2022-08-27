import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from intersect import intersection
import time



data = "Test Files/Veri_5.in"
N,M = 2,2
run(data,N,M)


def run(data,N,M):
    global BR,BK,BPA,TBR,FR,FK,FPA,Z
    global glb_sup_quantity, glb_dem_quantity, glb_sup_price, glb_dem_price, terminate, current_price, initial_surplus
    global hourly_supply_data, hourly_demand_data

    #Import data
    f = open(data, "r")
    lines = f.readlines()
    f.close()

    lines = [x for x in lines if x != '\n']
    #convery numbers in data to float
    # split_dat = [x[:len(x)-1].split(',') for x in lines[::2]] #for test data
    split_dat = [x[:len(x)-1].split(',') for x in lines]

    def maybe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return s
    data_int = [[maybe_float(i) for i in row] for row in split_dat]

    #extract hourly data
    hourly_data = [x for x in data_int if "S" in x ]
    block_data = [x for x in data_int if "B" in x ]
    flexible_data = [x for x in data_int if "F" in x ]

    #list of all hourly bids and time period they are in
    hourly_bids = sorted(list(set([(x[0],x[2]) for x in hourly_data])))
    #list of block bids
    block_bids = sorted(list(set([x[0] for x in block_data])))
    #list of flexible bids
    flexible_bids = sorted(list(set([x[0] for x in flexible_data])))

    #get supply and demand bids
    hourly_supply_bids = list(set([(x[0],x[2]) for x in hourly_data if x[4] < 0]))
    hourly_demand_bids = list(set([(x[0],x[2]) for x in hourly_data if x[4] > 0]))
    block_supply_bids = [x[0] for x in block_data if x[4] < 0]
    block_demand_bids = [x[0] for x in block_data if x[4] > 0]
    flex_supply_bids = [x[0] for x in flexible_data if x[4] < 0]
    flex_demand_bids = [x[0] for x in flexible_data if x[4] > 0]

    #get hourly supply and demand data
    hourly_supply_data = [ x for x in hourly_data if x[0] in [i[0] for i in hourly_supply_bids] ]
    hourly_demand_data = [ x for x in hourly_data if x[0] in [i[0] for i in hourly_demand_bids] ]

    #parent and child block bids
    child_bids = [x for x in block_data if x[7] != ""]
    child_supply_bids = [x[0] for x in child_bids if x[4] < 0]
    child_demand_bids = [x[0] for x in child_bids if x[4] > 0]

    #parent and child bids
    parent_bids = []
    child_parent = []
    for x in child_bids:
        parent = x[7]
        child_parent.append([x[0],parent])
        if parent not in parent_bids:
            parent_bids.append(x[7])


    ## STEP 0
    # Initialise parameters
    BR = block_bids.copy() #Set of rejected block bids (ALL STARTING OFF)
    BK = [] #Set of accepted block bids
    BPA = [] #Set of paradoxically accepted block bids
    TBR = [] #Set of temporarily rejected bids
    FR = flexible_bids.copy() #Set of rejected flexible bids
    FK = [] #Set of accepted flexible bids
    FPA = [] #Paradoxically accepted flexible bids
    iteration_limit = N
    grand_iteration_limit = M
    Z = list(set([int(x[2]) for x in hourly_data]))

    #global price quantity variables per hour
    glb_sup_quantity = [None] * 24
    glb_dem_quantity = [None] * 24
    glb_sup_price = [None] * 24
    glb_dem_price = [None] * 24
    terminate = False

    ## STEP 1
    # Aggregate curves for hourly plots to find equilibrium price
    ## STEP 2
    #get equilibrium point for each hour
    #set current price to this
    hourly_equil = [] #equilibrium price of each hour
    for z in Z:
        hourly_equil.append(plots(int(z),False,False,0,False)[1])

    current_price = hourly_equil


    #store market surplus before adding or rejecting supply d_bids
    initial_surplus = getinitialobj()


    grand_iteration_number = 0
    while terminate == False:
        ## STEP 3
        #calculate increase in surplus by accepting each bid

        iteration_number = 0
        repeat = True
        while repeat == True and iteration_number < iteration_limit:
            step3(block_data)
            iteration_number+=1
            if iteration_number < iteration_limit:
                repeat = False
                repeat = step4(block_data)

        #if BK not empty - check for paradox ically accepted bids
        if BK != []:
            step5(block_data)

        #if linked bids are in data check for these
        if parent_bids != []:
            step6(parent_bids, child_parent, block_data)
        #account for flexible bids
        if flexible_bids != []:
            step7(flexible_data, flex_demand_bids, flex_supply_bids)

        ## STEP 8
        grand_iteration_number += 1
        if grand_iteration_number > grand_iteration_limit:
            #Go to step 9
            step9(block_data, flexible_data, flex_demand_bids, flex_supply_bids, flexible_bids, child_bids, child_parent)


    final_obj = getfinalobj(block_data, flexible_data)

    return final_obj
def plots(h, print_plot, update, M, reject):
    global glb_sup_quantity, glb_sup_price, glb_dem_quantity, glb_dem_price

    #-------------------------------------------------------------------------------
    ## PLOTS

    #Input:
    #h - hour of day (starts qith 1)
    #print_plot - TRUE/FALSE print the plot image
    #update - TRUE if we want to update the existing plots
    #M - quantity of bid with largest impact value (used to update)

    #Return intersection point of supply and demand curve
    #-------------------------------------------------------------------------------
    #create list of supply break points for each hour (quantity price pairs)
    #supply_bp[hour][breakpoint]


    #create plot data if not already exiist
    if glb_sup_quantity[h-1] == None:
        qp_dataV2(h)

    #If plots need to be updated to account for block bids
    if update:

        if reject: #if rejected then move curve left else move right
            if M < 0: #if M negative (supply) then only shift supply curve (keep values positive)
                glb_sup_quantity[h-1] = [x+M for x in glb_sup_quantity[h-1]]
            elif M > 0: #if M positive (demand) then only shift demand curve
                glb_dem_quantity[h-1] = [x-M for x in glb_dem_quantity[h-1]]
        else:
            if M < 0: #if M negative (supply) then only shift supply curve (keep values positive)
                glb_sup_quantity[h-1] = [x-M for x in glb_sup_quantity[h-1]]
            elif M > 0: #if M positive (demand) then only shift demand curve
                glb_dem_quantity[h-1] = [x+M for x in glb_dem_quantity[h-1]]

    if print_plot:
        plt.figure(figsize=(10,8))
        plt.plot(glb_sup_quantity[h-1],glb_sup_price[h-1],'r',label="Supply", marker='.')
        plt.plot(glb_dem_quantity[h-1],glb_dem_price[h-1],'b',label="Demand")
        plt.xlabel("Quantity (MWh)")
        plt.ylabel("Price (TL/MWh)")
        plt.title("Hour "+str(h))
        plt.legend(loc="upper left")
        plt.show() # Display the graph


    a,b = intersection(glb_dem_quantity[h-1],glb_dem_price[h-1], glb_sup_quantity[h-1],glb_sup_price[h-1])

    #dont know why
    if len(a) > 1:
        inter = [float(a[0]),float(b[0])]
    elif a:
        inter = [float(a),float(b)]
    else:
        inter = 0

    return inter
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def getinitialobj():
    obj = []
    for z in Z:

        poly_x = []
        poly_y = []

        mcp = plots(z,False,False,0,False)
        poly_sup_q = [x for x in glb_sup_quantity[z-1] if x < mcp[0]]
        poly_sup_p = [x for x in glb_sup_price[z-1] if x < mcp[1]]
        poly_dem_q = [x for x in glb_dem_quantity[z-1] if x < mcp[0]]
        poly_dem_p = [x for x in glb_dem_price[z-1] if x > mcp[1]]

        #add in origin if not already there
        if [poly_sup_q[0],poly_sup_p[0]] != [0,0]:
            poly_x = [0]
            poly_y = [0]

        poly_x = poly_x + poly_sup_q + [mcp[0]] + poly_dem_q
        poly_y = poly_y + poly_sup_p + [mcp[1]] + poly_dem_p

        #account for space if demand curve doesnt go all the way to y axis
        if poly_x[-1] != 0:
            poly_x.append(0)
            poly_y.append(poly_y[-1])

        # obj += PolyArea(poly_x,poly_y)
        obj.append(PolyArea(poly_x,poly_y))
    return obj
def getfinalobj(block_data, flexible_data):

    block_term = 0
    for b in BK:
        bid_info = [ x for x in block_data if b == x[0] ][0]
        hours_covered = len(range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6])))
        block_term += hours_covered*bid_info[4]*bid_info[5]
    for b in BPA:
        bid_info = [ x for x in block_data if b == x[0] ][0]
        hours_covered = len(range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6])))
        block_term += hours_covered*bid_info[4]*bid_info[5]

    flexible_term = 0
    for b in FK:
        bid_info = [ x for x in flexible_data if b == x[0] ][0]
        flexible_term += bid_info[4]*bid_info[5]

    final_obj = sum(initial_surplus) + block_term + flexible_term

    return final_obj
def step3(block_data):
    global current_price, BR, BK

    go_s3 = True
    while(go_s3):
        go_s3 = False
        # Calculate the impact value of each bid in BR
        impact_value_b = []
        for b in BR:

            #bid info
            bid_info = [ x for x in block_data if b == x[0] ][0]
            hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))
            #calculate average clearing price of block bid
            avg_cp_b = avg_cp_fun(bid_info)

            #impact value of bid
            impact = (float(bid_info[5]) - avg_cp_b) * float(bid_info[4]) * len(hours_covered)
            impact_value_b.append([b,impact])

        # 3.1 Rank block bids in decreasing order of impact value
        impact_sorted = sorted(impact_value_b, key=lambda x: x[1], reverse=True)

        ## 3.2 Check if positive impact exists
        # If true -> 3.2.1, ELSE 3.2.2
        #Check for positive impact
        positive_impact = False
        for i in impact_sorted:
            if i[1] > 0:
                positive_impact = True
                largest_impact = i
                break

        #If positive then
        if positive_impact:
            #3.2.1
            #bid info of bid with largest impact
            largest_bid_info = [ x for x in block_data if largest_impact[0] == x[0] ][0]
            #hours covered by bid
            largest_hours_covered = range(int(largest_bid_info[2]), int(largest_bid_info[2])+int(largest_bid_info[6]))

            #Remove bid from BR and add to BK
            BR.remove(largest_impact[0])
            BK.append(largest_impact[0])
            for z in largest_hours_covered:
                c = plots(z,False,True,largest_bid_info[4], False)
                if c == 0:
                    print("No intersectin point")
                    print(largest_bid_info)
                    print("h: "+str(z))
                else:
                    current_price[z-1] = c[1]

            go_s3 = True
def step4(block_data):
    global current_price, BK, BR

    #impact of rejecting bids
    reject_impact_value = []
    for b in BK:
        bid_info = [ x for x in block_data if b == x[0] ][0] #all info on bid
        avg_cp_b = avg_cp_fun(bid_info) #average clearing price of bid
        hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))
        #impact value of rejective (NEGATIVE OF FORMULA 15)
        ##KEEPING THIS AS POSITIVE SIGN NOW
        impact = -(float(bid_info[5]) - avg_cp_b) * float(bid_info[4]) * len(hours_covered)
        reject_impact_value.append([b,impact])


    #4.1
    #rank bids in inreasing value
    rej_impact_sorted = sorted(reject_impact_value, key=lambda x: x[1], reverse=False)
    # if neg impact exists -> 4.2.1
    negative_impact = False
    for i in rej_impact_sorted:
        if i[1] < 0:
            negative_impact = True
            largest_neg_impact = i
            break

    #4.2.1
    #If negative impact then

    if negative_impact:

        ng_impact = True #one positive impact exists to keep the loop going

        largest_neg_info = [ x for x in block_data if largest_neg_impact[0] == x[0] ][0]
        largest_neg_hours = range(int(largest_neg_info[2]), int(largest_neg_info[2])+int(largest_neg_info[6]))

        BK.remove(largest_neg_impact[0])
        BR.append(largest_neg_impact[0])

        for z in largest_neg_hours:
            current_price[z-1] = plots(z,False,True,largest_neg_info[4],True)[1]

        repeat = False
    else:
        #GO TO STEP 3
        repeat = True

    return repeat
def step5(block_data):
    global BK, BPA
    for b in BK:
        bid_info = [ x for x in block_data if b == x[0] ][0]
        hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))
        avg_cp_b = avg_cp_fun(bid_info)
        impact = (float(bid_info[5]) - avg_cp_b) * float(bid_info[4]) * len(hours_covered)
        if impact < 0:
            BK.remove(b)
            BPA.append(b)
def step6(parent_bids, child_parent, block_data):
    global BR,BK,TBR,current_price
    # account for linked bids

    fully_searched = False
    parents_checked = []
    while fully_searched == False:

        #bids that havnt been cheked yet (sorted)
        rejected_parents = sorted([x for x in BR if x in parent_bids])
        tocheck = [x for x in rejected_parents if x not in parents_checked]

        if tocheck != []:
            block_checked = tocheck[0]
            #search for block bids in BK whose parent is block_checked
            block_children = [ x[0] for x in child_parent if x[1] == block_checked]

            #check if child is in BK
            for c in block_children:
                #6.3.1 if a child of block_checked is in BK then
                if c in BK:
                    bid_info = [ x for x in block_data if c == x[0] ][0]
                    hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))

                    BK.remove(c)
                    BR.append(c)

                    for z in hours_covered:
                        current_price[z-1] = plots(z,False,True,bid_info[4],True)[1]

            parents_checked.append(block_checked)
        if set(parents_checked) == set(rejected_parents):
            fully_searched = True
def step7(flexible_data, flex_demand_bids, flex_supply_bids):
    global FR,FK,current_price

    fully_searched = False

    while fully_searched == False: #while FR still has bids

        fully_searched = True
        #7.1 rank demand (supply) in increasing (decreading) order
        FR_data = [x for x in flexible_data if x[0] in FR]
        flex_dem_sorted = sorted([x for x in FR_data if x[0] in flex_demand_bids], key = lambda x: x[5], reverse = True)
        flex_sup_sorted = sorted([x for x in FR_data if x[0] in flex_supply_bids], key = lambda x: x[5])


        #7.2
        change_found = False
        if flex_dem_sorted != []:
            flex_dem_largest = flex_dem_sorted[0]
            min_curr_price = min(current_price)
            min_curr_hour = current_price.index(min(current_price))+1
            if flex_dem_largest[5] > min_curr_price: #if price of bid is larger that smallest current price
                #accept bid for that hour & update price
                FR.remove(flex_dem_largest[0])
                FK.append(flex_dem_largest[0])
                current_price[min_curr_hour-1] = plots(min_curr_hour,False,True,flex_dem_largest[4],False)[1]
                fully_searched = False
                change_found = True
        #7.3

        if flex_sup_sorted != [] and change_found == False:
            flex_sup_lowest = flex_sup_sorted[0]
            max_curr_price = max(current_price)
            max_curr_hour = current_price.index(max(current_price))+1
            if flex_sup_lowest[5] < max_curr_price: #if price of bid is larger that smallest current price
                #accept bid for that hour & update price
                FR.remove(flex_sup_lowest[0])
                FK.append(flex_sup_lowest[0])
                current_price[max_curr_hour-1] = plots(max_curr_hour,False,True,flex_sup_lowest[4],False)[1]
                fully_searched = False
def step9(block_data, flexible_data, flex_demand_bids, flex_supply_bids, flexible_bids, child_bids, child_parent):
    global BR,BK,FR,FK,current_price, terminate

    go_s9 = True
    while(go_s9):
        go_s9 = False
        impact_value_bf = []

        for b in BR:

            bid_info = [ x for x in block_data if b == x[0] ][0]
            hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))
            avg_cp_b = avg_cp_fun(bid_info)

            #impact value of bid
            impact = (bid_info[5] - avg_cp_b) * bid_info[4] * len(hours_covered)
            impact_value_bf.append([b,impact])

        for b in FR:

            bid_info = [ x for x in flexible_data if b == x[0] ][0]

            if b in flex_demand_bids:
                clearing_price = min(current_price)
                f_hour = current_price.index(clearing_price)
            elif b in flex_supply_bids:
                clearing_price = max(current_price)
                f_hour = current_price.index(clearing_price)

            #impact value of bid
            impact = (bid_info[5] - clearing_price) * bid_info[4]
            impact_value_bf.append([b,impact,f_hour])


        #9.1 Rank  bids in decreasing order of impact value
        impact_sorted = sorted(impact_value_bf, key=lambda x: x[1], reverse=True)

        #9.2 check if positive impact
        #If true -> 9.2.1 else .2.2
        #Check for positive impact
        positive_impact = [x for x in impact_sorted if x[1] > 0]

        #find positive impact that is either flexible bid or
        #has no parent or has parent that is accepted
        bid_found = False
        for p in positive_impact:
            if p[0] in flexible_bids:

                bid_info = [ x for x in flexible_data if p[0] == x[0] ][0]

                FR.remove(p[0])
                FK.append(p[0])

                current_price[p[2]-1] = plots(p[2],False,True,bid_info[4],False)[1]

                bid_found = True
                go_s9 = True
                break
            else:#if child bid check that its parent is accepted
                if p[0] in [x[0] for x in child_bids]:
                    parent = [x[1] for x in child_parent if x[0] == p[0]][0]
                    if parent not in BR:

                        bid_info = [ x for x in block_data if p[0] == x[0] ][0]
                        b_hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))

                        BR.remove(p[0])
                        BK.append(p[0])

                        for z in b_hours_covered:
                            current_price[z-1] = plots(z,False,True,bid_info[4],False)[1]

                        bid_found = True
                        go_s9 = True
                        break

                #else if has no parent
                else:

                    bid_info = [ x for x in block_data if p[0] == x[0] ][0]
                    b_hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6]))

                    BR.remove(p[0])
                    BK.append(p[0])

                    for z in b_hours_covered:
                        current_price[z-1] = plots(z,False,True,bid_info[4],False)[1]

                    bid_found = True
                    go_s9 = True
                    break

    terminate = True
def avg_cp_fun(bid_info):

    #calculate average clearing price of block bid
    hours_covered = range(int(bid_info[2]), int(bid_info[2])+int(bid_info[6])) #list of hours covered
    total_price = 0
    for h in hours_covered:
        total_price += current_price[h-1] #check this is current price and not hourly_equil
    avg_cp_b = total_price/len(hours_covered)

    return avg_cp_b
def qp_dataV2(h):

    h_sup = [x for x in hourly_supply_data if x[2] == h ]
    h_dem = [x for x in hourly_demand_data if x[2] == h ]

    #unique supply bids
    used_bids = set()
    sup_bids = [x[0] for x in h_sup if x[0] not in used_bids and (used_bids.add(x[0]) or True)]
    #unique demand bids
    used_bids = set()
    dem_bids = [x[0] for x in h_dem if x[0] not in used_bids and (used_bids.add(x[0]) or True)]

    #list of all price break points
    used = set()
    sup_bp = sorted([float(x[5]) for x in h_sup if x[5] not in used and (used.add(x[5]) or True)])
    used = set()
    dem_bp = sorted([float(x[5]) for x in h_dem if x[5] not in used and (used.add(x[5]) or True)])

    sup_qp_pair = []
    for b in sup_bids:
        dat = [x for x in h_sup if x[0] == b ] #data for this bid
        b_price = [x[5] for x in h_sup if x[0] == b]
        b_quantity = [abs(x[4]) for x in h_sup if x[0] == b]

        #for each break point check if that bp exists for current bid
        #if not then find quantity and add to list
        for i in sup_bp:
            if i in b_price: #if this bp already in bid then add
                sup_qp_pair.append([b_quantity[b_price.index(i)],i])
            else:
                sup_qp_pair.append([np.interp(i, b_price,b_quantity),i])


    dem_qp_pair = []
    for b in dem_bids:
        dat = [x for x in h_dem if x[0] == b ] #data for this bid
        b_price = [x[5] for x in h_dem if x[0] == b]
        b_quantity = [abs(x[4]) for x in h_dem if x[0] == b]

        #for each break point check if that bp exists for current bid
        #if not then find quantity and add to list
        for i in dem_bp:
            if i in b_price: #if this bp already in bid then add
                dem_qp_pair.append([b_quantity[b_price.index(i)],i])
            else:
                dem_qp_pair.append([np.interp(i, b_price,b_quantity),i])



    #horizontally add all demand together for each price break points
    f=lambda x: x[1]
    agg_sup = sorted([[sum(x[0] for x in g),k] for k, g in groupby(sorted(sup_qp_pair, key=f), key=f)], key=lambda x: x[1], reverse=False)
    agg_dem = sorted([[sum(x[0] for x in g),k] for k, g in groupby(sorted(dem_qp_pair, key=f), key=f)], key=lambda x: x[1], reverse=False)

    sup_q = [x[0] for x in agg_sup]
    sup_p = [x[1] for x in agg_sup]
    dem_q = [x[0] for x in agg_dem]
    dem_p = [x[1] for x in agg_dem]

    glb_sup_quantity[h-1] = sup_q
    glb_dem_quantity[h-1] = dem_q
    glb_sup_price[h-1] = sup_p
    glb_dem_price[h-1] = dem_p
